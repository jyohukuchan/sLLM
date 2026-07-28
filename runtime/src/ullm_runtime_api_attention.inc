ullm_status ullm_runtime_causal_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 causal attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (sequence_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 causal attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 causal attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 causal attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, v_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 causal attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 causal attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (q_heads > max_size / sequence_len) {
        set_error("f32 causal attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = sequence_len * q_heads;
    if (head_dim > max_size / q_head_sequence) {
        set_error("f32 causal attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;

    if (kv_heads > max_size / sequence_len) {
        set_error("f32 causal attention kv head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_sequence = sequence_len * kv_heads;
    if (head_dim > max_size / kv_head_sequence ||
        value_dim > max_size / kv_head_sequence) {
        set_error("f32 causal attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_sequence * head_dim;
    const size_t v_elements = kv_head_sequence * value_dim;
    if (value_dim > max_size / q_head_sequence) {
        set_error("f32 causal attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_head_sequence * value_dim;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 causal attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 causal attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->bytes < k_bytes) {
        set_error("f32 causal attention k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 causal attention v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 causal attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        causal_attn_f32_host(
            q,
            k,
            v,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (causal_attn_f32_hip_kernel(
            q_buffer,
            k_buffer,
            v_buffer,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 causal attention HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return causal_attn_f32_hip_staging(
        q_buffer,
        k_buffer,
        v_buffer,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_causal_attn_f32_flash2(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 causal flash2 attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (sequence_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 causal flash2 attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 causal flash2 attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 causal flash2 attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, v_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 causal flash2 attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 causal flash2 attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (q_heads > max_size / sequence_len) {
        set_error("f32 causal flash2 attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = sequence_len * q_heads;
    if (head_dim > max_size / q_head_sequence) {
        set_error("f32 causal flash2 attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;

    if (kv_heads > max_size / sequence_len) {
        set_error("f32 causal flash2 attention kv head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_sequence = sequence_len * kv_heads;
    if (head_dim > max_size / kv_head_sequence ||
        value_dim > max_size / kv_head_sequence) {
        set_error("f32 causal flash2 attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_sequence * head_dim;
    const size_t v_elements = kv_head_sequence * value_dim;
    if (value_dim > max_size / q_head_sequence) {
        set_error("f32 causal flash2 attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_head_sequence * value_dim;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 causal flash2 attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 causal flash2 attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->bytes < k_bytes) {
        set_error("f32 causal flash2 attention k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 causal flash2 attention v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 causal flash2 attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        causal_attn_f32_host(
            q,
            k,
            v,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (causal_attn_f32_flash2_hip_kernel(
            q_buffer,
            k_buffer,
            v_buffer,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_CAUSAL_ATTN_F32_FLASH2_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 causal flash2 attention HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return causal_attn_f32_hip_staging(
        q_buffer,
        k_buffer,
        v_buffer,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_causal_attn_batch_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t batch_count,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 batched causal attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (batch_count == 0 || sequence_len == 0 || q_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || value_dim == 0) {
        set_error("f32 batched causal attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 batched causal attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 batched causal attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, v_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 batched causal attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 batched causal attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (q_heads > max_size / sequence_len) {
        set_error("f32 batched causal attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence_per_batch = sequence_len * q_heads;
    if (kv_heads > max_size / sequence_len) {
        set_error("f32 batched causal attention kv head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_sequence_per_batch = sequence_len * kv_heads;

    if (head_dim > max_size / q_head_sequence_per_batch ||
        value_dim > max_size / q_head_sequence_per_batch) {
        set_error("f32 batched causal attention q/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements_per_batch = q_head_sequence_per_batch * head_dim;
    const size_t output_elements_per_batch = q_head_sequence_per_batch * value_dim;
    if (head_dim > max_size / kv_head_sequence_per_batch ||
        value_dim > max_size / kv_head_sequence_per_batch) {
        set_error("f32 batched causal attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements_per_batch = kv_head_sequence_per_batch * head_dim;
    const size_t v_elements_per_batch = kv_head_sequence_per_batch * value_dim;

    if (batch_count > max_size / q_elements_per_batch ||
        batch_count > max_size / k_elements_per_batch ||
        batch_count > max_size / v_elements_per_batch ||
        batch_count > max_size / output_elements_per_batch) {
        set_error("f32 batched causal attention batch element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = batch_count * q_elements_per_batch;
    const size_t k_elements = batch_count * k_elements_per_batch;
    const size_t v_elements = batch_count * v_elements_per_batch;
    const size_t output_elements = batch_count * output_elements_per_batch;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 batched causal attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 batched causal attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->bytes < k_bytes) {
        set_error("f32 batched causal attention k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 batched causal attention v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 batched causal attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        for (size_t batch = 0; batch < batch_count; ++batch) {
            causal_attn_f32_host(
                q + batch * q_elements_per_batch,
                k + batch * k_elements_per_batch,
                v + batch * v_elements_per_batch,
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                output + batch * output_elements_per_batch);
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (causal_attn_batch_f32_hip_kernel(
            q_buffer,
            k_buffer,
            v_buffer,
            batch_count,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 batched causal attention HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return causal_attn_batch_f32_hip_staging(
        q_buffer,
        k_buffer,
        v_buffer,
        batch_count,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_causal_attn_batch_f32_flash2(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t batch_count,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 batched causal flash2 attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (batch_count == 0 || sequence_len == 0 || q_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || value_dim == 0) {
        set_error("f32 batched causal flash2 attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 batched causal flash2 attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 batched causal flash2 attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, v_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 batched causal flash2 attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 batched causal flash2 attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (q_heads > max_size / sequence_len) {
        set_error("f32 batched causal flash2 attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence_per_batch = sequence_len * q_heads;
    if (kv_heads > max_size / sequence_len) {
        set_error("f32 batched causal flash2 attention kv head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_sequence_per_batch = sequence_len * kv_heads;

    if (head_dim > max_size / q_head_sequence_per_batch ||
        value_dim > max_size / q_head_sequence_per_batch) {
        set_error("f32 batched causal flash2 attention q/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements_per_batch = q_head_sequence_per_batch * head_dim;
    const size_t output_elements_per_batch = q_head_sequence_per_batch * value_dim;
    if (head_dim > max_size / kv_head_sequence_per_batch ||
        value_dim > max_size / kv_head_sequence_per_batch) {
        set_error("f32 batched causal flash2 attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements_per_batch = kv_head_sequence_per_batch * head_dim;
    const size_t v_elements_per_batch = kv_head_sequence_per_batch * value_dim;

    if (batch_count > max_size / q_elements_per_batch ||
        batch_count > max_size / k_elements_per_batch ||
        batch_count > max_size / v_elements_per_batch ||
        batch_count > max_size / output_elements_per_batch) {
        set_error("f32 batched causal flash2 attention batch element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = batch_count * q_elements_per_batch;
    const size_t k_elements = batch_count * k_elements_per_batch;
    const size_t v_elements = batch_count * v_elements_per_batch;
    const size_t output_elements = batch_count * output_elements_per_batch;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 batched causal flash2 attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 batched causal flash2 attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->bytes < k_bytes) {
        set_error("f32 batched causal flash2 attention k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 batched causal flash2 attention v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 batched causal flash2 attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        for (size_t batch = 0; batch < batch_count; ++batch) {
            causal_attn_f32_host(
                q + batch * q_elements_per_batch,
                k + batch * k_elements_per_batch,
                v + batch * v_elements_per_batch,
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                output + batch * output_elements_per_batch);
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (causal_attn_batch_f32_flash2_hip_kernel(
            q_buffer,
            k_buffer,
            v_buffer,
            batch_count,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_F32_FLASH2_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_CAUSAL_ATTN_F32_FLASH2_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "f32 batched causal flash2 attention HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return causal_attn_batch_f32_hip_staging(
        q_buffer,
        k_buffer,
        v_buffer,
        batch_count,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_decode_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 decode attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 decode attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 decode attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 decode attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 decode attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 decode attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (head_dim > max_size / q_heads) {
        set_error("f32 decode attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_heads * head_dim;

    if (kv_heads > max_size / cache_len) {
        set_error("f32 decode attention kv head-cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = cache_len * kv_heads;
    if (head_dim > max_size / kv_head_cache ||
        value_dim > max_size / kv_head_cache) {
        set_error("f32 decode attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_cache * head_dim;
    const size_t v_elements = kv_head_cache * value_dim;
    if (value_dim > max_size / q_heads) {
        set_error("f32 decode attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_heads * value_dim;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 decode attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 decode attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("f32 decode attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("f32 decode attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 decode attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k_cache = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const float *>(v_cache_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        decode_attn_f32_host(
            q,
            k_cache,
            v_cache,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (decode_attn_f32_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 decode attention HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return decode_attn_f32_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_cached_prefix_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 cached prefix attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (new_tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 cached prefix attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 cached prefix attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 cached prefix attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 cached prefix attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 cached prefix attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (new_tokens > max_size - cached_prefix_len) {
        set_error("f32 cached prefix attention total context overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + new_tokens;

    if (q_heads > max_size / new_tokens) {
        set_error("f32 cached prefix attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = new_tokens * q_heads;
    if (head_dim > max_size / q_head_sequence ||
        value_dim > max_size / q_head_sequence) {
        set_error("f32 cached prefix attention q/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;
    const size_t output_elements = q_head_sequence * value_dim;

    if (kv_heads > max_size / total_context) {
        set_error("f32 cached prefix attention kv head-context element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_context = total_context * kv_heads;
    if (head_dim > max_size / kv_head_context ||
        value_dim > max_size / kv_head_context) {
        set_error("f32 cached prefix attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_context * head_dim;
    const size_t v_elements = kv_head_context * value_dim;

    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 cached prefix attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 cached prefix attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("f32 cached prefix attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("f32 cached prefix attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 cached prefix attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k_cache = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const float *>(v_cache_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        cached_prefix_attn_f32_host(
            q,
            k_cache,
            v_cache,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (cached_prefix_attn_f32_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 cached prefix attention HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return cached_prefix_attn_f32_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_cached_prefix_attn_f32_flash2(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 cached prefix flash2 attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (new_tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 cached prefix flash2 attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 cached prefix flash2 attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 cached prefix flash2 attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 cached prefix flash2 attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 cached prefix flash2 attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (new_tokens > max_size - cached_prefix_len) {
        set_error("f32 cached prefix flash2 attention total context overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + new_tokens;

    if (q_heads > max_size / new_tokens) {
        set_error("f32 cached prefix flash2 attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = new_tokens * q_heads;
    if (head_dim > max_size / q_head_sequence ||
        value_dim > max_size / q_head_sequence) {
        set_error("f32 cached prefix flash2 attention q/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;
    const size_t output_elements = q_head_sequence * value_dim;

    if (kv_heads > max_size / total_context) {
        set_error("f32 cached prefix flash2 attention kv head-context element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_context = total_context * kv_heads;
    if (head_dim > max_size / kv_head_context ||
        value_dim > max_size / kv_head_context) {
        set_error("f32 cached prefix flash2 attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_context * head_dim;
    const size_t v_elements = kv_head_context * value_dim;

    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 cached prefix flash2 attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 cached prefix flash2 attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("f32 cached prefix flash2 attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("f32 cached prefix flash2 attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 cached prefix flash2 attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k_cache = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const float *>(v_cache_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        cached_prefix_attn_f32_host(
            q,
            k_cache,
            v_cache,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (cached_prefix_attn_f32_flash2_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "f32 cached prefix flash2 attention HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return cached_prefix_attn_f32_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("fp8 e4m3 cached prefix attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (new_tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("fp8 e4m3 cached prefix attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("fp8 e4m3 cached prefix attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error(
            "fp8 e4m3 cached prefix attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(k_scale) || k_scale <= 0.0f ||
        !std::isfinite(v_scale) || v_scale <= 0.0f) {
        set_error("fp8 e4m3 cached prefix attention scales must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error(
            "fp8 e4m3 cached prefix attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("fp8 e4m3 cached prefix attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (new_tokens > max_size - cached_prefix_len) {
        set_error("fp8 e4m3 cached prefix attention total context overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + new_tokens;

    if (q_heads > max_size / new_tokens) {
        set_error("fp8 e4m3 cached prefix attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = new_tokens * q_heads;
    if (head_dim > max_size / q_head_sequence ||
        value_dim > max_size / q_head_sequence) {
        set_error("fp8 e4m3 cached prefix attention q/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;
    const size_t output_elements = q_head_sequence * value_dim;

    if (kv_heads > max_size / total_context) {
        set_error("fp8 e4m3 cached prefix attention kv head-context element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_context = total_context * kv_heads;
    if (head_dim > max_size / kv_head_context ||
        value_dim > max_size / kv_head_context) {
        set_error("fp8 e4m3 cached prefix attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_context * head_dim;
    const size_t v_elements = kv_head_context * value_dim;

    if (q_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("fp8 e4m3 cached prefix attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements;
    const size_t v_bytes = v_elements;
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("fp8 e4m3 cached prefix attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("fp8 e4m3 cached prefix attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("fp8 e4m3 cached prefix attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("fp8 e4m3 cached prefix attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k_cache = static_cast<const uint8_t *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const uint8_t *>(v_cache_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        cached_prefix_attn_fp8_e4m3_host(
            q,
            k_cache,
            v_cache,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (cached_prefix_attn_fp8_e4m3_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "fp8 e4m3 cached prefix attention HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return cached_prefix_attn_fp8_e4m3_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        k_scale,
        v_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

static ullm_status cached_prefix_attn_fp8_e4m3_flash2_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    bool q_is_fp8,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("fp8 e4m3 cached prefix flash2 attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (new_tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("fp8 e4m3 cached prefix flash2 attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("fp8 e4m3 cached prefix flash2 attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error(
            "fp8 e4m3 cached prefix flash2 attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_is_fp8 && (!std::isfinite(q_scale) || q_scale <= 0.0f)) ||
        !std::isfinite(k_scale) || k_scale <= 0.0f ||
        !std::isfinite(v_scale) || v_scale <= 0.0f) {
        set_error("fp8 e4m3 cached prefix flash2 attention scales must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_is_fp8 && value_dim > 256) {
        set_error("fp8 e4m3 cached prefix flash2 fp8q attention value_dim exceeds 256");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error(
            "fp8 e4m3 cached prefix flash2 attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("fp8 e4m3 cached prefix flash2 attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (new_tokens > max_size - cached_prefix_len) {
        set_error("fp8 e4m3 cached prefix flash2 attention total context overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + new_tokens;

    if (q_heads > max_size / new_tokens) {
        set_error("fp8 e4m3 cached prefix flash2 attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = new_tokens * q_heads;
    if (head_dim > max_size / q_head_sequence ||
        value_dim > max_size / q_head_sequence) {
        set_error("fp8 e4m3 cached prefix flash2 attention q/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;
    const size_t output_elements = q_head_sequence * value_dim;

    if (kv_heads > max_size / total_context) {
        set_error("fp8 e4m3 cached prefix flash2 attention kv head-context element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_context = total_context * kv_heads;
    if (head_dim > max_size / kv_head_context ||
        value_dim > max_size / kv_head_context) {
        set_error("fp8 e4m3 cached prefix flash2 attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_context * head_dim;
    const size_t v_elements = kv_head_context * value_dim;

    if ((!q_is_fp8 && q_elements > max_size / sizeof(float)) ||
        output_elements > max_size / sizeof(float)) {
        set_error("fp8 e4m3 cached prefix flash2 attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_is_fp8 ? q_elements : q_elements * sizeof(float);
    const size_t k_bytes = k_elements;
    const size_t v_bytes = v_elements;
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("fp8 e4m3 cached prefix flash2 attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("fp8 e4m3 cached prefix flash2 attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("fp8 e4m3 cached prefix flash2 attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("fp8 e4m3 cached prefix flash2 attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        if (q_is_fp8) {
            cached_prefix_attn_fp8_e4m3_rocwmma_host(
                static_cast<const uint8_t *>(q_buffer->ptr),
                static_cast<const uint8_t *>(k_cache_buffer->ptr),
                static_cast<const uint8_t *>(v_cache_buffer->ptr),
                cached_prefix_len,
                new_tokens,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                q_scale,
                k_scale,
                v_scale,
                static_cast<float *>(output_buffer->ptr));
        } else {
            cached_prefix_attn_fp8_e4m3_host(
                static_cast<const float *>(q_buffer->ptr),
                static_cast<const uint8_t *>(k_cache_buffer->ptr),
                static_cast<const uint8_t *>(v_cache_buffer->ptr),
                cached_prefix_len,
                new_tokens,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                k_scale,
                v_scale,
                static_cast<float *>(output_buffer->ptr));
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (cached_prefix_attn_fp8_e4m3_flash2_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
            q_is_fp8,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        q_is_fp8 ||
        std::getenv("ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_FLASH2_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "fp8 e4m3 cached prefix flash2 attention HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return cached_prefix_attn_fp8_e4m3_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        k_scale,
        v_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return cached_prefix_attn_fp8_e4m3_flash2_impl(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        1.0f,
        k_scale,
        v_scale,
        false,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2_fp8q(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return cached_prefix_attn_fp8_e4m3_flash2_impl(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cached_prefix_len,
        new_tokens,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_scale,
        k_scale,
        v_scale,
        true,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3_rocwmma(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (new_tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((head_dim % 16) != 0 || (value_dim % 16) != 0) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention currently requires head_dim and value_dim to be multiples of 16");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (((q_heads / kv_heads) % 16) != 0) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention requires q_heads/kv_heads to be a multiple of 16");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(q_scale) || q_scale <= 0.0f ||
        !std::isfinite(k_scale) || k_scale <= 0.0f ||
        !std::isfinite(v_scale) || v_scale <= 0.0f) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention scales must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (new_tokens > max_size - cached_prefix_len) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention total context overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + new_tokens;

    if (q_heads > max_size / new_tokens) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = new_tokens * q_heads;
    if (head_dim > max_size / q_head_sequence ||
        value_dim > max_size / q_head_sequence) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention q/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;
    const size_t output_elements = q_head_sequence * value_dim;

    if (kv_heads > max_size / total_context) {
        set_error(
            "fp8 e4m3 cached prefix rocWMMA attention kv head-context element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_context = total_context * kv_heads;
    if (head_dim > max_size / kv_head_context ||
        value_dim > max_size / kv_head_context) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_context * head_dim;
    const size_t v_elements = kv_head_context * value_dim;

    if (output_elements > max_size / sizeof(float)) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements;
    const size_t k_bytes = k_elements;
    const size_t v_bytes = v_elements;
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("fp8 e4m3 cached prefix rocWMMA attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        cached_prefix_attn_fp8_e4m3_rocwmma_host(
            static_cast<const uint8_t *>(q_buffer->ptr),
            static_cast<const uint8_t *>(k_cache_buffer->ptr),
            static_cast<const uint8_t *>(v_cache_buffer->ptr),
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
            static_cast<float *>(output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (cached_prefix_attn_fp8_e4m3_rocwmma_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    set_error(
        hip_kernel_error.empty() ?
            "fp8 e4m3 cached prefix rocWMMA attention HIP kernel is unavailable" :
            hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status paged_decode_attn_f32_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 paged decode attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_len == 0 || block_size == 0 || cache_blocks == 0 || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 paged decode attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 paged decode attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 paged decode attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_buffer != nullptr && !buffers_share_backend(q_buffer, gate_buffer)) ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 paged decode attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 paged decode attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t block_table_entries = ((cache_len - 1) / block_size) + 1;
    if (cache_blocks > max_size / block_size) {
        set_error("f32 paged decode attention physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (block_table_entries > cache_blocks || cache_len > physical_tokens) {
        set_error("f32 paged decode attention cache_len exceeds physical cache block capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (head_dim > max_size / q_heads) {
        set_error("f32 paged decode attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_heads * head_dim;

    if (kv_heads > max_size / physical_tokens) {
        set_error("f32 paged decode attention kv head-cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = physical_tokens * kv_heads;
    if (head_dim > max_size / kv_head_cache ||
        value_dim > max_size / kv_head_cache) {
        set_error("f32 paged decode attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_cache * head_dim;
    const size_t v_elements = kv_head_cache * value_dim;
    if (value_dim > max_size / q_heads) {
        set_error("f32 paged decode attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_heads * value_dim;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float) ||
        block_table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("f32 paged decode attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t gate_bytes = gate_buffer == nullptr ? 0 : output_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 paged decode attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer != nullptr && gate_buffer->bytes < gate_bytes) {
        set_error("f32 paged decode attention gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("f32 paged decode attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("f32 paged decode attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_table_buffer->bytes < block_table_bytes) {
        set_error("f32 paged decode attention block table buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 paged decode attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *gate =
            gate_buffer == nullptr ? nullptr : static_cast<const float *>(gate_buffer->ptr);
        const auto *k_cache = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const float *>(v_cache_buffer->ptr);
        const auto *block_table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        for (size_t entry = 0; entry < block_table_entries; ++entry) {
            if (static_cast<size_t>(block_table[entry]) >= cache_blocks) {
                set_error("f32 paged decode attention block table contains an out-of-range block id");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        auto *output = static_cast<float *>(output_buffer->ptr);
        paged_decode_attn_f32_host(
            q,
            k_cache,
            v_cache,
            block_table,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        if (gate != nullptr) {
            for (size_t i = 0; i < output_elements; ++i) {
                const float gate_value = gate[i];
                const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
                output[i] *= sigmoid;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_decode_attn_f32_hip_kernel(
            q_buffer,
            gate_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 paged decode attention HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return paged_decode_attn_f32_hip_staging(
        q_buffer,
        gate_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        gate_bytes,
        k_bytes,
        v_bytes,
        block_table_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_decode_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return paged_decode_attn_f32_impl(
        q_buffer,
        nullptr,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_gemma_full_attn_batched_512_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t prefix_len,
    size_t query_rows,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    constexpr size_t q_heads = 8u;
    constexpr size_t width = 512u;
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || output_buffer == nullptr) {
        set_error("Gemma full batched attention received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (query_rows == 0 || query_rows > 128 || cache_len == 0 || block_size == 0 || cache_blocks == 0 ||
        !std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("Gemma full batched attention received invalid geometry or softmax scale");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (prefix_len > cache_len || query_rows > cache_len - prefix_len ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer) ||
        !stream_matches_buffer(output_buffer, stream)) {
        set_error("Gemma full batched attention buffers, stream, or causal prefix are invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = query_rows * q_heads * width;
    const size_t physical_tokens = cache_blocks * block_size;
    const size_t kv_elements = physical_tokens * width;
    const size_t table_entries = (cache_len - 1u) / block_size + 1u;
    if (q_elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        kv_elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        table_entries > std::numeric_limits<size_t>::max() / sizeof(std::uint32_t)) {
        set_error("Gemma full batched attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t kv_bytes = kv_elements * sizeof(float);
    const size_t table_bytes = table_entries * sizeof(std::uint32_t);
    if (q_buffer->bytes < q_bytes || k_cache_buffer->bytes < kv_bytes ||
        v_cache_buffer->bytes < kv_bytes || block_table_buffer->bytes < table_bytes ||
        output_buffer->bytes < q_bytes) {
        set_error("Gemma full batched attention buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v = static_cast<const float *>(v_cache_buffer->ptr);
        const auto *table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        for (size_t row = 0; row < query_rows; ++row) {
            paged_decode_attn_f32_host(
                q + row * q_heads * width, k, v, table, prefix_len + row + 1u,
                block_size, q_heads, 1u, width, width, softmax_scale,
                output + row * q_heads * width);
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string hip_error;
    if (gemma_full_attn_batched_512_f32_hip_kernel(
            q_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer, prefix_len,
            query_rows, cache_len, block_size, cache_blocks, softmax_scale, output_buffer,
            stream, &hip_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(hip_error.empty() ? "Gemma full batched attention HIP kernel is unavailable" : hip_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_gemma_sliding_attn_ring_batched_256_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_ring_buffer,
    const ullm_runtime_buffer *v_ring_buffer,
    const ullm_runtime_buffer *fresh_k_buffer,
    const ullm_runtime_buffer *fresh_v_buffer,
    size_t prefix_len,
    size_t history_rows,
    size_t query_rows,
    size_t ring_capacity,
    size_t q_heads,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    constexpr size_t width = 256u;
    if (q_buffer == nullptr || k_ring_buffer == nullptr || v_ring_buffer == nullptr ||
        fresh_k_buffer == nullptr || fresh_v_buffer == nullptr || output_buffer == nullptr ||
        query_rows == 0 || query_rows > 128 || history_rows > 511 || history_rows > prefix_len ||
        ring_capacity == 0 || q_heads == 0 || !std::isfinite(softmax_scale) || softmax_scale <= 0.0f ||
        !buffers_share_backend(q_buffer, k_ring_buffer) || !buffers_share_backend(q_buffer, v_ring_buffer) ||
        !buffers_share_backend(q_buffer, fresh_k_buffer) || !buffers_share_backend(q_buffer, fresh_v_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer) || !stream_matches_buffer(output_buffer, stream)) {
        set_error("Gemma sliding ring batched attention received invalid buffers or geometry");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = query_rows * q_heads * width;
    const size_t ring_elements = ring_capacity * width;
    const size_t fresh_elements = query_rows * width;
    if (q_elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        ring_elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        fresh_elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        q_buffer->bytes < q_elements * sizeof(float) || output_buffer->bytes < q_elements * sizeof(float) ||
        k_ring_buffer->bytes < ring_elements * sizeof(float) || v_ring_buffer->bytes < ring_elements * sizeof(float) ||
        fresh_k_buffer->bytes < fresh_elements * sizeof(float) || fresh_v_buffer->bytes < fresh_elements * sizeof(float)) {
        set_error("Gemma sliding ring batched attention buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *kr = static_cast<const float *>(k_ring_buffer->ptr);
        const auto *vr = static_cast<const float *>(v_ring_buffer->ptr);
        const auto *fk = static_cast<const float *>(fresh_k_buffer->ptr);
        const auto *fv = static_cast<const float *>(fresh_v_buffer->ptr);
        auto *out = static_cast<float *>(output_buffer->ptr);
        for (size_t row = 0; row < query_rows; ++row) for (size_t head = 0; head < q_heads; ++head) {
            std::vector<float> scores(history_rows + row + 1u);
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source = 0; source < scores.size(); ++source) {
                const float *key = source < history_rows
                    ? kr + ((prefix_len - history_rows + source) % ring_capacity) * width
                    : fk + (source - history_rows) * width;
                float score = 0.0f;
                for (size_t dim = 0; dim < width; ++dim) score += q[(row * q_heads + head) * width + dim] * key[dim];
                scores[source] = score * softmax_scale;
                max_score = std::max(max_score, scores[source]);
            }
            float denom = 0.0f;
            for (float score : scores) denom += std::exp(score - max_score);
            for (size_t dim = 0; dim < width; ++dim) {
                float value = 0.0f;
                for (size_t source = 0; source < scores.size(); ++source) {
                    const float *v = source < history_rows
                        ? vr + ((prefix_len - history_rows + source) % ring_capacity) * width
                        : fv + (source - history_rows) * width;
                    value += std::exp(scores[source] - max_score) * v[dim];
                }
                out[(row * q_heads + head) * width + dim] = value / denom;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string hip_error;
    if (gemma_sliding_attn_ring_batched_256_f32_hip_kernel(
            q_buffer, k_ring_buffer, v_ring_buffer, fresh_k_buffer, fresh_v_buffer,
            prefix_len, history_rows, query_rows, ring_capacity, q_heads, softmax_scale,
            output_buffer, stream, &hip_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(hip_error.empty() ? "Gemma sliding ring batched attention HIP kernel is unavailable" : hip_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

static ullm_status paged_decode_attn_typed_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    size_t q_bytes,
    size_t k_cache_bytes,
    size_t v_cache_bytes,
    size_t table_bytes,
    size_t k_scale_bytes,
    size_t v_scale_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<uint8_t> host_k(k_cache_bytes);
    std::vector<uint8_t> host_v(v_cache_bytes);
    std::vector<uint32_t> host_table(table_bytes / sizeof(uint32_t));
    std::vector<uint16_t> host_k_scales(k_scale_bytes / sizeof(uint16_t));
    std::vector<uint16_t> host_v_scales(v_scale_bytes / sizeof(uint16_t));
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;
    bool copied = hip_runtime().copy_async(
        host_q.data(), q_buffer->ptr, q_bytes, HIP_MEMCPY_DEVICE_TO_HOST, hip_stream, device_id) &&
        hip_runtime().copy_async(
            host_k.data(), k_cache_buffer->ptr, k_cache_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id) &&
        hip_runtime().copy_async(
            host_v.data(), v_cache_buffer->ptr, v_cache_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id) &&
        hip_runtime().copy_async(
            host_table.data(), block_table_buffer->ptr, table_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id);
    if (copied && k_scale_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            host_k_scales.data(), k_scale_buffer->ptr, k_scale_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id);
    }
    if (copied && v_scale_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            host_v_scales.data(), v_scale_buffer->ptr, v_scale_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id);
    }
    if (!copied || !synchronize_hip_staging(stream, device_id)) {
        set_error("failed to copy typed paged decode attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    for (uint32_t block_id : host_table) {
        if (static_cast<size_t>(block_id) >= cache_blocks) {
            set_error("typed paged decode attention block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!paged_decode_attn_typed_f32_host(
            host_q.data(),
            host_k.data(),
            host_v.data(),
            host_table.data(),
            k_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN ? host_k_scales.data() : nullptr,
            v_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN ? host_v_scales.data() : nullptr,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_dtype,
            v_dtype,
            host_output.data())) {
        set_error("typed paged decode attention encountered invalid cache values or scales");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr, host_output.data(), output_bytes, HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream, device_id) ||
        !synchronize_hip_staging(stream, device_id)) {
        set_error("failed to copy typed paged decode attention output back to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_paged_decode_attn_typed_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || output_buffer == nullptr) {
        set_error("typed paged decode attention received a null payload or block-table pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_len == 0 || block_size == 0 || cache_blocks == 0 || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || value_dim == 0 || (q_heads % kv_heads) != 0 ||
        !std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("typed paged decode attention dimensions or softmax scale are invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!kv_cache_dtype_is_supported(k_dtype) || !kv_cache_dtype_is_supported(v_dtype) ||
        (kv_cache_dtype_requires_scale(k_dtype) != (k_scale_buffer != nullptr)) ||
        (kv_cache_dtype_requires_scale(v_dtype) != (v_scale_buffer != nullptr))) {
        set_error("typed paged decode attention cache dtype and FP8 scale buffers disagree");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer) ||
        (k_scale_buffer != nullptr && !buffers_share_backend(q_buffer, k_scale_buffer)) ||
        (v_scale_buffer != nullptr && !buffers_share_backend(q_buffer, v_scale_buffer))) {
        set_error("typed paged decode attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream) ||
        (k_scale_buffer != nullptr && !stream_matches_buffer(k_scale_buffer, stream)) ||
        (v_scale_buffer != nullptr && !stream_matches_buffer(v_scale_buffer, stream))) {
        set_error("typed paged decode attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size || q_heads > max_size / head_dim) {
        set_error("typed paged decode attention physical cache or query size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    const size_t table_entries = (cache_len - 1u) / block_size + 1u;
    if (cache_len > physical_tokens || table_entries > cache_blocks ||
        kv_heads > max_size / physical_tokens || value_dim > max_size / q_heads) {
        set_error("typed paged decode attention cache capacity or output size is invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_values = q_heads * head_dim;
    const size_t output_values = q_heads * value_dim;
    const size_t physical_heads = physical_tokens * kv_heads;
    if (head_dim > max_size / physical_heads || value_dim > max_size / physical_heads) {
        set_error("typed paged decode attention cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_cache_values = physical_heads * head_dim;
    const size_t v_cache_values = physical_heads * value_dim;
    const size_t k_payload_width = kv_cache_dtype_payload_bytes(k_dtype);
    const size_t v_payload_width = kv_cache_dtype_payload_bytes(v_dtype);
    if (q_values > max_size / sizeof(float) || output_values > max_size / sizeof(float) ||
        k_cache_values > max_size / k_payload_width || v_cache_values > max_size / v_payload_width ||
        physical_heads > max_size / sizeof(uint16_t) || table_entries > max_size / sizeof(uint32_t)) {
        set_error("typed paged decode attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_values * sizeof(float);
    const size_t output_bytes = output_values * sizeof(float);
    const size_t k_cache_bytes = k_cache_values * k_payload_width;
    const size_t v_cache_bytes = v_cache_values * v_payload_width;
    const size_t table_bytes = table_entries * sizeof(uint32_t);
    const size_t scale_bytes = physical_heads * sizeof(uint16_t);
    if (q_buffer->bytes < q_bytes || k_cache_buffer->bytes < k_cache_bytes ||
        v_cache_buffer->bytes < v_cache_bytes || block_table_buffer->bytes < table_bytes ||
        output_buffer->bytes < output_bytes ||
        (k_scale_buffer != nullptr && k_scale_buffer->bytes < scale_bytes) ||
        (v_scale_buffer != nullptr && v_scale_buffer->bytes < scale_bytes)) {
        set_error("typed paged decode attention buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *table = static_cast<const uint32_t *>(block_table_buffer->ptr);
        for (size_t index = 0; index < table_entries; ++index) {
            if (static_cast<size_t>(table[index]) >= cache_blocks) {
                set_error("typed paged decode attention block table contains an out-of-range block id");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        if (!paged_decode_attn_typed_f32_host(
                static_cast<const float *>(q_buffer->ptr),
                k_cache_buffer->ptr,
                v_cache_buffer->ptr,
                table,
                k_scale_buffer == nullptr ? nullptr : static_cast<const uint16_t *>(k_scale_buffer->ptr),
                v_scale_buffer == nullptr ? nullptr : static_cast<const uint16_t *>(v_scale_buffer->ptr),
                cache_len,
                block_size,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                k_dtype,
                v_dtype,
                static_cast<float *>(output_buffer->ptr))) {
            set_error("typed paged decode attention encountered invalid cache values or scales");
            return ULLM_STATUS_RUNTIME_ERROR;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_decode_attn_typed_f32_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            k_scale_buffer,
            v_scale_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_dtype,
            v_dtype,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (std::getenv("ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL") != nullptr) {
        set_error(
            hip_kernel_error.empty() ? "typed paged decode attention HIP kernel is unavailable" :
                                        hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return paged_decode_attn_typed_f32_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        k_scale_buffer,
        v_scale_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        k_dtype,
        v_dtype,
        q_bytes,
        k_cache_bytes,
        v_cache_bytes,
        table_bytes,
        k_scale_buffer == nullptr ? 0u : scale_bytes,
        v_scale_buffer == nullptr ? 0u : scale_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_decode_attn_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr) {
        set_error("f32 paged decode attention sigmoid gate received a null gate buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return paged_decode_attn_f32_impl(
        q_buffer,
        gate_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}

ullm_status paged_decode_attn_split_f32_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || workspace_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 paged decode split attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_len == 0 || block_size == 0 || cache_blocks == 0 || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || value_dim == 0 || source_tile == 0) {
        set_error("f32 paged decode split attention dimensions and source_tile must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (head_dim > 256 || value_dim > 256) {
        set_error("f32 paged decode split attention head_dim/value_dim must not exceed 256");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 paged decode split attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error(
            "f32 paged decode split attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_buffer != nullptr && !buffers_share_backend(q_buffer, gate_buffer)) ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, workspace_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error(
            "f32 paged decode split attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 paged decode split attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t block_table_entries = ((cache_len - 1) / block_size) + 1;
    const size_t split_count = ((cache_len - 1) / source_tile) + 1;
    if (value_dim > max_size - 2) {
        set_error("f32 paged decode split attention workspace stride overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t workspace_stride = value_dim + 2;
    if (split_count > max_size / q_heads) {
        set_error("f32 paged decode split attention partial count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t partial_count = q_heads * split_count;
    if (workspace_stride > max_size / partial_count) {
        set_error("f32 paged decode split attention workspace element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t workspace_elements = partial_count * workspace_stride;
    if (workspace_elements > max_size / sizeof(float)) {
        set_error("f32 paged decode split attention workspace byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t workspace_bytes = workspace_elements * sizeof(float);
    if (partial_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        q_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        set_error("f32 paged decode split attention launch grid exceeds HIP grid limit");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_blocks > max_size / block_size) {
        set_error("f32 paged decode split attention physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (block_table_entries > cache_blocks || cache_len > physical_tokens) {
        set_error("f32 paged decode split attention cache_len exceeds physical cache block capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (head_dim > max_size / q_heads) {
        set_error("f32 paged decode split attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_heads * head_dim;
    if (kv_heads > max_size / physical_tokens) {
        set_error("f32 paged decode split attention kv head-cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = physical_tokens * kv_heads;
    if (head_dim > max_size / kv_head_cache || value_dim > max_size / kv_head_cache) {
        set_error("f32 paged decode split attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_cache * head_dim;
    const size_t v_elements = kv_head_cache * value_dim;
    if (value_dim > max_size / q_heads) {
        set_error("f32 paged decode split attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_heads * value_dim;
    if (q_elements > max_size / sizeof(float) || k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) || output_elements > max_size / sizeof(float) ||
        block_table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("f32 paged decode split attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t gate_bytes = gate_buffer == nullptr ? 0 : output_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);
    if (q_buffer->bytes < q_bytes || k_cache_buffer->bytes < k_bytes ||
        v_cache_buffer->bytes < v_bytes || block_table_buffer->bytes < block_table_bytes ||
        workspace_buffer->bytes < workspace_bytes || output_buffer->bytes < output_bytes ||
        (gate_buffer != nullptr && gate_buffer->bytes < gate_bytes)) {
        set_error("f32 paged decode split attention buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *gate =
            gate_buffer == nullptr ? nullptr : static_cast<const float *>(gate_buffer->ptr);
        const auto *k_cache = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const float *>(v_cache_buffer->ptr);
        const auto *block_table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        for (size_t entry = 0; entry < block_table_entries; ++entry) {
            if (static_cast<size_t>(block_table[entry]) >= cache_blocks) {
                set_error(
                    "f32 paged decode split attention block table contains an out-of-range block id");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        auto *output = static_cast<float *>(output_buffer->ptr);
        paged_decode_attn_f32_host(
            q,
            k_cache,
            v_cache,
            block_table,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        if (gate != nullptr) {
            for (size_t i = 0; i < output_elements; ++i) {
                const float gate_value = gate[i];
                const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
                output[i] *= sigmoid;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_decode_attn_split_f32_hip_kernel(
            q_buffer,
            gate_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            source_tile,
            split_count,
            workspace_buffer,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "f32 paged decode split attention HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return paged_decode_attn_f32_hip_staging(
        q_buffer,
        gate_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        gate_bytes,
        k_bytes,
        v_bytes,
        block_table_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_decode_attn_split_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return paged_decode_attn_split_f32_impl(
        q_buffer,
        nullptr,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        source_tile,
        workspace_buffer,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_decode_attn_split_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr) {
        set_error("f32 paged decode split attention sigmoid gate received a null gate buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return paged_decode_attn_split_f32_impl(
        q_buffer,
        gate_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        source_tile,
        workspace_buffer,
        output_buffer,
        stream);
}

static ullm_status paged_decode_attn_split_typed_f32_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (k_dtype == ULLM_KV_CACHE_DTYPE_F32 && v_dtype == ULLM_KV_CACHE_DTYPE_F32 &&
        k_scale_buffer == nullptr && v_scale_buffer == nullptr) {
        return paged_decode_attn_split_f32_impl(
            q_buffer,
            gate_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            source_tile,
            workspace_buffer,
            output_buffer,
            stream);
    }
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || workspace_buffer == nullptr || output_buffer == nullptr) {
        set_error("typed paged decode split attention received a null payload or workspace pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_len == 0u || block_size == 0u || cache_blocks == 0u || q_heads == 0u ||
        kv_heads == 0u || head_dim == 0u || value_dim == 0u || source_tile == 0u ||
        head_dim > 256u || value_dim > 256u || q_heads % kv_heads != 0u ||
        !std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("typed paged decode split attention dimensions or softmax scale are invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!kv_cache_dtype_is_supported(k_dtype) || !kv_cache_dtype_is_supported(v_dtype) ||
        (kv_cache_dtype_requires_scale(k_dtype) != (k_scale_buffer != nullptr)) ||
        (kv_cache_dtype_requires_scale(v_dtype) != (v_scale_buffer != nullptr))) {
        set_error("typed paged decode split attention cache dtype and FP8 scale buffers disagree");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_buffer != nullptr && !buffers_share_backend(q_buffer, gate_buffer)) ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, workspace_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer) ||
        (k_scale_buffer != nullptr && !buffers_share_backend(q_buffer, k_scale_buffer)) ||
        (v_scale_buffer != nullptr && !buffers_share_backend(q_buffer, v_scale_buffer))) {
        set_error("typed paged decode split attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream) ||
        !stream_matches_buffer(workspace_buffer, stream) ||
        (k_scale_buffer != nullptr && !stream_matches_buffer(k_scale_buffer, stream)) ||
        (v_scale_buffer != nullptr && !stream_matches_buffer(v_scale_buffer, stream))) {
        set_error("typed paged decode split attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size || q_heads > max_size / head_dim) {
        set_error("typed paged decode split attention physical cache or query size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    const size_t table_entries = (cache_len - 1u) / block_size + 1u;
    const size_t split_count = (cache_len - 1u) / source_tile + 1u;
    if (cache_len > physical_tokens || table_entries > cache_blocks ||
        kv_heads > max_size / physical_tokens || value_dim > max_size / q_heads ||
        value_dim > max_size - 2u || split_count > max_size / q_heads) {
        set_error("typed paged decode split attention capacity or workspace geometry is invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_values = q_heads * head_dim;
    const size_t output_values = q_heads * value_dim;
    const size_t physical_heads = physical_tokens * kv_heads;
    const size_t partial_count = q_heads * split_count;
    const size_t workspace_stride = value_dim + 2u;
    if (head_dim > max_size / physical_heads || value_dim > max_size / physical_heads ||
        workspace_stride > max_size / partial_count) {
        set_error("typed paged decode split attention element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_values = physical_heads * head_dim;
    const size_t v_values = physical_heads * value_dim;
    const size_t workspace_values = partial_count * workspace_stride;
    const size_t k_width = kv_cache_dtype_payload_bytes(k_dtype);
    const size_t v_width = kv_cache_dtype_payload_bytes(v_dtype);
    if (q_values > max_size / sizeof(float) || output_values > max_size / sizeof(float) ||
        k_values > max_size / k_width || v_values > max_size / v_width ||
        workspace_values > max_size / sizeof(float) || physical_heads > max_size / sizeof(uint16_t) ||
        table_entries > max_size / sizeof(uint32_t)) {
        set_error("typed paged decode split attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_values * sizeof(float);
    const size_t output_bytes = output_values * sizeof(float);
    const size_t k_bytes = k_values * k_width;
    const size_t v_bytes = v_values * v_width;
    const size_t table_bytes = table_entries * sizeof(uint32_t);
    const size_t scale_bytes = physical_heads * sizeof(uint16_t);
    const size_t workspace_bytes = workspace_values * sizeof(float);
    const size_t gate_bytes = gate_buffer == nullptr ? 0u : output_bytes;
    if (q_buffer->bytes < q_bytes || k_cache_buffer->bytes < k_bytes ||
        v_cache_buffer->bytes < v_bytes || block_table_buffer->bytes < table_bytes ||
        workspace_buffer->bytes < workspace_bytes || output_buffer->bytes < output_bytes ||
        (gate_buffer != nullptr && gate_buffer->bytes < gate_bytes) ||
        (k_scale_buffer != nullptr && k_scale_buffer->bytes < scale_bytes) ||
        (v_scale_buffer != nullptr && v_scale_buffer->bytes < scale_bytes)) {
        set_error("typed paged decode split attention buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *table = static_cast<const uint32_t *>(block_table_buffer->ptr);
        for (size_t index = 0; index < table_entries; ++index) {
            if (static_cast<size_t>(table[index]) >= cache_blocks) {
                set_error("typed paged decode split attention block table contains an out-of-range block id");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        if (!paged_decode_attn_typed_f32_host(
                static_cast<const float *>(q_buffer->ptr),
                k_cache_buffer->ptr,
                v_cache_buffer->ptr,
                table,
                k_scale_buffer == nullptr ? nullptr : static_cast<const uint16_t *>(k_scale_buffer->ptr),
                v_scale_buffer == nullptr ? nullptr : static_cast<const uint16_t *>(v_scale_buffer->ptr),
                cache_len,
                block_size,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                k_dtype,
                v_dtype,
                static_cast<float *>(output_buffer->ptr))) {
            set_error("typed paged decode split attention encountered invalid cache values or scales");
            return ULLM_STATUS_RUNTIME_ERROR;
        }
        if (gate_buffer != nullptr) {
            const auto *gate = static_cast<const float *>(gate_buffer->ptr);
            auto *output = static_cast<float *>(output_buffer->ptr);
            for (size_t index = 0; index < output_values; ++index) {
                output[index] *= 1.0f / (1.0f + std::exp(-gate[index]));
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string hip_kernel_error;
    if (paged_decode_attn_split_typed_f32_hip_kernel(
            q_buffer,
            gate_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            k_scale_buffer,
            v_scale_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            source_tile,
            split_count,
            k_dtype,
            v_dtype,
            workspace_buffer,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (std::getenv("ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_SPLIT_KERNEL") != nullptr) {
        set_error(hip_kernel_error.empty() ?
            "typed paged decode split attention HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    const ullm_status staged = paged_decode_attn_typed_f32_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        k_scale_buffer,
        v_scale_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        k_dtype,
        v_dtype,
        q_bytes,
        k_bytes,
        v_bytes,
        table_bytes,
        k_scale_buffer == nullptr ? 0u : scale_bytes,
        v_scale_buffer == nullptr ? 0u : scale_bytes,
        output_bytes,
        output_buffer,
        stream);
    if (staged != ULLM_STATUS_OK || gate_buffer == nullptr) {
        return staged;
    }
    return ullm_runtime_sigmoid_mul_f32(
        gate_buffer, output_buffer, output_values, output_buffer, stream);
}

ullm_status ullm_runtime_paged_decode_attn_split_typed_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return paged_decode_attn_split_typed_f32_impl(
        q_buffer, nullptr, k_cache_buffer, v_cache_buffer, block_table_buffer,
        k_scale_buffer, v_scale_buffer, cache_len, block_size, cache_blocks,
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, source_tile,
        k_dtype, v_dtype, workspace_buffer, output_buffer, stream);
}

ullm_status ullm_runtime_paged_decode_attn_split_sigmoid_gate_typed_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t source_tile,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr) {
        set_error("typed paged decode split attention sigmoid gate received a null gate buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return paged_decode_attn_split_typed_f32_impl(
        q_buffer, gate_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer,
        k_scale_buffer, v_scale_buffer, cache_len, block_size, cache_blocks,
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, source_tile,
        k_dtype, v_dtype, workspace_buffer, output_buffer, stream);
}

ullm_status ullm_runtime_paged_kv_write_f32(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    if (k_buffer == nullptr || v_buffer == nullptr || block_table_buffer == nullptr ||
        k_cache_buffer == nullptr || v_cache_buffer == nullptr) {
        set_error("f32 paged KV write received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_size == 0 || cache_blocks == 0 || kv_heads == 0 ||
        head_dim == 0 || value_dim == 0) {
        set_error("f32 paged KV write dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(k_buffer, v_buffer) ||
        !buffers_share_backend(k_buffer, block_table_buffer) ||
        !buffers_share_backend(k_buffer, k_cache_buffer) ||
        !buffers_share_backend(k_buffer, v_cache_buffer)) {
        set_error("f32 paged KV write buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(k_cache_buffer, stream) ||
        !stream_matches_buffer(v_cache_buffer, stream)) {
        set_error("f32 paged KV write stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size) {
        set_error("f32 paged KV write physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (cache_position >= physical_tokens) {
        set_error("f32 paged KV write cache_position exceeds physical cache capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t block_index = cache_position / block_size;
    const size_t block_table_entries = block_index + 1;

    if (head_dim > max_size / kv_heads || value_dim > max_size / kv_heads) {
        set_error("f32 paged KV write token element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_heads * head_dim;
    const size_t v_elements = kv_heads * value_dim;
    if (kv_heads > max_size / physical_tokens) {
        set_error("f32 paged KV write kv head-cache count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = physical_tokens * kv_heads;
    if (head_dim > max_size / kv_head_cache ||
        value_dim > max_size / kv_head_cache) {
        set_error("f32 paged KV write cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_cache_elements = kv_head_cache * head_dim;
    const size_t v_cache_elements = kv_head_cache * value_dim;
    if (k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        k_cache_elements > max_size / sizeof(float) ||
        v_cache_elements > max_size / sizeof(float) ||
        block_table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("f32 paged KV write byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t k_cache_bytes = k_cache_elements * sizeof(float);
    const size_t v_cache_bytes = v_cache_elements * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);

    if (k_buffer->bytes < k_bytes) {
        set_error("f32 paged KV write k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 paged KV write v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_table_buffer->bytes < block_table_bytes) {
        set_error("f32 paged KV write block table buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_cache_bytes) {
        set_error("f32 paged KV write k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_cache_bytes) {
        set_error("f32 paged KV write v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (k_buffer->backend == BackendKind::Cpu) {
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        const auto *block_table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        if (static_cast<size_t>(block_table[block_index]) >= cache_blocks) {
            set_error("f32 paged KV write block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        auto *k_cache = static_cast<float *>(k_cache_buffer->ptr);
        auto *v_cache = static_cast<float *>(v_cache_buffer->ptr);
        paged_kv_write_f32_host(
            k,
            v,
            block_table,
            cache_position,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            k_cache,
            v_cache);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_kv_write_f32_hip_kernel(
            k_buffer,
            v_buffer,
            block_table_buffer,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            k_cache_buffer,
            v_cache_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 paged KV write HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return paged_kv_write_f32_hip_staging(
        k_buffer,
        v_buffer,
        block_table_buffer,
        cache_position,
        block_size,
        cache_blocks,
        kv_heads,
        head_dim,
        value_dim,
        k_bytes,
        v_bytes,
        block_table_bytes,
        k_cache_bytes,
        v_cache_bytes,
        k_cache_buffer,
        v_cache_buffer,
        stream);
}

static ullm_status paged_kv_write_typed_f32_hip_staging(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    size_t k_bytes,
    size_t v_bytes,
    size_t table_bytes,
    size_t k_cache_bytes,
    size_t v_cache_bytes,
    size_t k_scale_bytes,
    size_t v_scale_bytes,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_buffer *k_scale_buffer,
    ullm_runtime_buffer *v_scale_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<std::uint32_t> host_table(table_bytes / sizeof(std::uint32_t));
    std::vector<uint8_t> host_k_cache(k_cache_bytes);
    std::vector<uint8_t> host_v_cache(v_cache_bytes);
    std::vector<uint16_t> host_k_scales(k_scale_bytes / sizeof(uint16_t));
    std::vector<uint16_t> host_v_scales(v_scale_bytes / sizeof(uint16_t));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = k_buffer->hip_device_id;

    bool copied = hip_runtime().copy_async(
        host_k.data(), k_buffer->ptr, k_bytes, HIP_MEMCPY_DEVICE_TO_HOST, hip_stream, device_id) &&
        hip_runtime().copy_async(
            host_v.data(), v_buffer->ptr, v_bytes, HIP_MEMCPY_DEVICE_TO_HOST, hip_stream, device_id) &&
        hip_runtime().copy_async(
            host_table.data(), block_table_buffer->ptr, table_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id) &&
        hip_runtime().copy_async(
            host_k_cache.data(), k_cache_buffer->ptr, k_cache_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id) &&
        hip_runtime().copy_async(
            host_v_cache.data(), v_cache_buffer->ptr, v_cache_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id);
    if (copied && k_scale_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            host_k_scales.data(), k_scale_buffer->ptr, k_scale_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id);
    }
    if (copied && v_scale_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            host_v_scales.data(), v_scale_buffer->ptr, v_scale_bytes, HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream, device_id);
    }
    if (!copied || !synchronize_hip_staging(stream, device_id)) {
        set_error("failed to copy typed paged KV write HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    const size_t block_index = cache_position / block_size;
    if (static_cast<size_t>(host_table[block_index]) >= cache_blocks) {
        set_error("typed paged KV write block table contains an out-of-range block id");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!paged_kv_write_typed_f32_host(
            host_k.data(),
            host_v.data(),
            host_table.data(),
            cache_position,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            k_dtype,
            v_dtype,
            host_k_cache.data(),
            host_v_cache.data(),
            k_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN ? host_k_scales.data() : nullptr,
            v_dtype == ULLM_KV_CACHE_DTYPE_FP8_E4M3FN ? host_v_scales.data() : nullptr)) {
        set_error("typed paged KV write encountered a non-finite or unrepresentable value");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    copied = hip_runtime().copy_async(
        k_cache_buffer->ptr, host_k_cache.data(), k_cache_bytes, HIP_MEMCPY_HOST_TO_DEVICE,
        hip_stream, device_id) &&
        hip_runtime().copy_async(
            v_cache_buffer->ptr, host_v_cache.data(), v_cache_bytes, HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream, device_id);
    if (copied && k_scale_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            k_scale_buffer->ptr, host_k_scales.data(), k_scale_bytes, HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream, device_id);
    }
    if (copied && v_scale_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            v_scale_buffer->ptr, host_v_scales.data(), v_scale_bytes, HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream, device_id);
    }
    if (!copied || !synchronize_hip_staging(stream, device_id)) {
        set_error("failed to copy typed paged KV write caches back to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_paged_kv_write_typed_f32(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_buffer *k_scale_buffer,
    ullm_runtime_buffer *v_scale_buffer,
    ullm_runtime_stream *stream) {
    if (k_buffer == nullptr || v_buffer == nullptr || block_table_buffer == nullptr ||
        k_cache_buffer == nullptr || v_cache_buffer == nullptr) {
        set_error("typed paged KV write received a null payload or block-table pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_size == 0 || cache_blocks == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("typed paged KV write dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!kv_cache_dtype_is_supported(k_dtype) || !kv_cache_dtype_is_supported(v_dtype)) {
        set_error("typed paged KV write received an unsupported cache dtype");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((kv_cache_dtype_requires_scale(k_dtype) != (k_scale_buffer != nullptr)) ||
        (kv_cache_dtype_requires_scale(v_dtype) != (v_scale_buffer != nullptr))) {
        set_error("typed paged KV write FP8 planes require exactly one FP16 scale buffer per plane");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(k_buffer, v_buffer) ||
        !buffers_share_backend(k_buffer, block_table_buffer) ||
        !buffers_share_backend(k_buffer, k_cache_buffer) ||
        !buffers_share_backend(k_buffer, v_cache_buffer) ||
        (k_scale_buffer != nullptr && !buffers_share_backend(k_buffer, k_scale_buffer)) ||
        (v_scale_buffer != nullptr && !buffers_share_backend(k_buffer, v_scale_buffer))) {
        set_error("typed paged KV write buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(k_cache_buffer, stream) ||
        !stream_matches_buffer(v_cache_buffer, stream) ||
        (k_scale_buffer != nullptr && !stream_matches_buffer(k_scale_buffer, stream)) ||
        (v_scale_buffer != nullptr && !stream_matches_buffer(v_scale_buffer, stream))) {
        set_error("typed paged KV write stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size) {
        set_error("typed paged KV write physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (cache_position >= physical_tokens || kv_heads > max_size / physical_tokens ||
        kv_heads > max_size / head_dim || kv_heads > max_size / value_dim) {
        set_error("typed paged KV write cache position or element count is invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t block_index = cache_position / block_size;
    const size_t table_entries = block_index + 1u;
    const size_t k_values = kv_heads * head_dim;
    const size_t v_values = kv_heads * value_dim;
    const size_t physical_heads = physical_tokens * kv_heads;
    if (head_dim > max_size / physical_heads || value_dim > max_size / physical_heads) {
        set_error("typed paged KV write cache payload element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_cache_values = physical_heads * head_dim;
    const size_t v_cache_values = physical_heads * value_dim;
    const size_t k_payload_width = kv_cache_dtype_payload_bytes(k_dtype);
    const size_t v_payload_width = kv_cache_dtype_payload_bytes(v_dtype);
    if (k_values > max_size / sizeof(float) || v_values > max_size / sizeof(float) ||
        k_cache_values > max_size / k_payload_width || v_cache_values > max_size / v_payload_width ||
        physical_heads > max_size / sizeof(uint16_t) || table_entries > max_size / sizeof(uint32_t)) {
        set_error("typed paged KV write byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_bytes = k_values * sizeof(float);
    const size_t v_bytes = v_values * sizeof(float);
    const size_t table_bytes = table_entries * sizeof(uint32_t);
    const size_t k_cache_bytes = k_cache_values * k_payload_width;
    const size_t v_cache_bytes = v_cache_values * v_payload_width;
    const size_t scale_bytes = physical_heads * sizeof(uint16_t);
    if (k_buffer->bytes < k_bytes || v_buffer->bytes < v_bytes ||
        block_table_buffer->bytes < table_bytes || k_cache_buffer->bytes < k_cache_bytes ||
        v_cache_buffer->bytes < v_cache_bytes ||
        (k_scale_buffer != nullptr && k_scale_buffer->bytes < scale_bytes) ||
        (v_scale_buffer != nullptr && v_scale_buffer->bytes < scale_bytes)) {
        set_error("typed paged KV write buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (k_buffer->backend == BackendKind::Cpu) {
        const auto *table = static_cast<const uint32_t *>(block_table_buffer->ptr);
        if (static_cast<size_t>(table[block_index]) >= cache_blocks) {
            set_error("typed paged KV write block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (!paged_kv_write_typed_f32_host(
                static_cast<const float *>(k_buffer->ptr),
                static_cast<const float *>(v_buffer->ptr),
                table,
                cache_position,
                block_size,
                kv_heads,
                head_dim,
                value_dim,
                k_dtype,
                v_dtype,
                k_cache_buffer->ptr,
                v_cache_buffer->ptr,
                k_scale_buffer == nullptr ? nullptr : static_cast<uint16_t *>(k_scale_buffer->ptr),
                v_scale_buffer == nullptr ? nullptr : static_cast<uint16_t *>(v_scale_buffer->ptr))) {
            set_error("typed paged KV write encountered a non-finite or unrepresentable value");
            return ULLM_STATUS_RUNTIME_ERROR;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_kv_write_typed_f32_hip_kernel(
            k_buffer,
            v_buffer,
            block_table_buffer,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            k_dtype,
            v_dtype,
            k_cache_buffer,
            v_cache_buffer,
            k_scale_buffer,
            v_scale_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (std::getenv("ULLM_REQUIRE_HIP_TYPED_PAGED_KV_WRITE_KERNEL") != nullptr) {
        set_error(
            hip_kernel_error.empty() ? "typed paged KV write HIP kernel is unavailable" :
                                        hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return paged_kv_write_typed_f32_hip_staging(
        k_buffer,
        v_buffer,
        block_table_buffer,
        cache_position,
        block_size,
        cache_blocks,
        kv_heads,
        head_dim,
        value_dim,
        k_dtype,
        v_dtype,
        k_bytes,
        v_bytes,
        table_bytes,
        k_cache_bytes,
        v_cache_bytes,
        k_scale_buffer == nullptr ? 0u : scale_bytes,
        v_scale_buffer == nullptr ? 0u : scale_bytes,
        k_cache_buffer,
        v_cache_buffer,
        k_scale_buffer,
        v_scale_buffer,
        stream);
}

ullm_status ullm_runtime_paged_kv_write_chunk_f32(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_start,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    if (k_buffer == nullptr || v_buffer == nullptr || block_table_buffer == nullptr ||
        k_cache_buffer == nullptr || v_cache_buffer == nullptr) {
        set_error("f32 paged KV write chunk received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || m > 4096 || block_size == 0 || cache_blocks == 0 || kv_heads == 0 ||
        head_dim == 0 || value_dim == 0) {
        set_error("f32 paged KV write chunk width must be in 1..4096 and dimensions nonzero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(k_buffer, v_buffer) ||
        !buffers_share_backend(k_buffer, block_table_buffer) ||
        !buffers_share_backend(k_buffer, k_cache_buffer) ||
        !buffers_share_backend(k_buffer, v_cache_buffer)) {
        set_error("f32 paged KV write chunk buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(k_cache_buffer, stream) ||
        !stream_matches_buffer(v_cache_buffer, stream)) {
        set_error("f32 paged KV write chunk stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size) {
        set_error("f32 paged KV write chunk physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (cache_start > max_size - m || cache_start + m > physical_tokens) {
        set_error("f32 paged KV write chunk logical range exceeds physical cache capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t end = cache_start + m;
    const size_t table_entries = (end - 1) / block_size + 1;
    if (kv_heads > max_size / head_dim || kv_heads > max_size / value_dim) {
        set_error("f32 paged KV write chunk token element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_row = kv_heads * head_dim;
    const size_t v_row = kv_heads * value_dim;
    if (m > max_size / k_row || m > max_size / v_row ||
        physical_tokens > max_size / kv_heads) {
        set_error("f32 paged KV write chunk element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = m * k_row;
    const size_t v_elements = m * v_row;
    const size_t cache_heads = physical_tokens * kv_heads;
    if (head_dim > max_size / cache_heads || value_dim > max_size / cache_heads) {
        set_error("f32 paged KV write chunk cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_cache_elements = cache_heads * head_dim;
    const size_t v_cache_elements = cache_heads * value_dim;
    if (k_elements > max_size / sizeof(float) || v_elements > max_size / sizeof(float) ||
        k_cache_elements > max_size / sizeof(float) ||
        v_cache_elements > max_size / sizeof(float) ||
        table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("f32 paged KV write chunk byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t k_cache_bytes = k_cache_elements * sizeof(float);
    const size_t v_cache_bytes = v_cache_elements * sizeof(float);
    const size_t table_bytes = table_entries * sizeof(std::uint32_t);
    if (k_buffer->bytes < k_bytes || v_buffer->bytes < v_bytes ||
        block_table_buffer->bytes < table_bytes || k_cache_buffer->bytes < k_cache_bytes ||
        v_cache_buffer->bytes < v_cache_bytes) {
        set_error("f32 paged KV write chunk buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->backend == BackendKind::Cpu) {
        const auto *table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        for (size_t index = 0; index < table_entries; ++index) {
            if (static_cast<size_t>(table[index]) >= cache_blocks) {
                set_error("f32 paged KV write chunk block table contains an out-of-range block id");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        paged_kv_write_chunk_f32_host(
            static_cast<const float *>(k_buffer->ptr),
            static_cast<const float *>(v_buffer->ptr),
            table,
            cache_start,
            m,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            static_cast<float *>(k_cache_buffer->ptr),
            static_cast<float *>(v_cache_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }
    const ullm_status table_status = validate_paged_chunk_block_table_hip(
        block_table_buffer,
        table_entries,
        cache_blocks,
        stream,
        "f32 paged KV write chunk");
    if (table_status != ULLM_STATUS_OK) {
        return table_status;
    }
    std::string hip_error;
    if (paged_kv_write_chunk_f32_hip_kernel(
            k_buffer,
            v_buffer,
            block_table_buffer,
            cache_start,
            m,
            block_size,
            cache_blocks,
            table_entries,
            kv_heads,
            head_dim,
            value_dim,
            k_cache_buffer,
            v_cache_buffer,
            stream,
            &hip_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(hip_error.empty() ? "f32 paged KV write chunk HIP kernel is unavailable" : hip_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_paged_kv_write_chunk_typed_f32(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_start,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_buffer *k_scale_buffer,
    ullm_runtime_buffer *v_scale_buffer,
    ullm_runtime_stream *stream) {
    if (k_dtype == ULLM_KV_CACHE_DTYPE_F32 && v_dtype == ULLM_KV_CACHE_DTYPE_F32 &&
        k_scale_buffer == nullptr && v_scale_buffer == nullptr) {
        return ullm_runtime_paged_kv_write_chunk_f32(
            k_buffer, v_buffer, block_table_buffer, cache_start, m, block_size, cache_blocks,
            kv_heads, head_dim, value_dim, k_cache_buffer, v_cache_buffer, stream);
    }
    if (k_buffer == nullptr || v_buffer == nullptr || block_table_buffer == nullptr ||
        k_cache_buffer == nullptr || v_cache_buffer == nullptr) {
        set_error("typed paged KV write chunk received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0u || m > 4096u || block_size == 0u || cache_blocks == 0u || kv_heads == 0u ||
        head_dim == 0u || value_dim == 0u || !kv_cache_dtype_is_supported(k_dtype) ||
        !kv_cache_dtype_is_supported(v_dtype) ||
        (kv_cache_dtype_requires_scale(k_dtype) != (k_scale_buffer != nullptr)) ||
        (kv_cache_dtype_requires_scale(v_dtype) != (v_scale_buffer != nullptr))) {
        set_error("typed paged KV write chunk geometry, dtype, or FP8 scale buffers are invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(k_buffer, v_buffer) ||
        !buffers_share_backend(k_buffer, block_table_buffer) ||
        !buffers_share_backend(k_buffer, k_cache_buffer) ||
        !buffers_share_backend(k_buffer, v_cache_buffer) ||
        (k_scale_buffer != nullptr && !buffers_share_backend(k_buffer, k_scale_buffer)) ||
        (v_scale_buffer != nullptr && !buffers_share_backend(k_buffer, v_scale_buffer)) ||
        !stream_matches_buffer(k_cache_buffer, stream) ||
        !stream_matches_buffer(v_cache_buffer, stream) ||
        (k_scale_buffer != nullptr && !stream_matches_buffer(k_scale_buffer, stream)) ||
        (v_scale_buffer != nullptr && !stream_matches_buffer(v_scale_buffer, stream))) {
        set_error("typed paged KV write chunk buffers or stream belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size || cache_start > max_size - m) {
        set_error("typed paged KV write chunk physical cache geometry overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    const size_t end = cache_start + m;
    if (end > physical_tokens || kv_heads > max_size / head_dim ||
        kv_heads > max_size / value_dim || physical_tokens > max_size / kv_heads) {
        set_error("typed paged KV write chunk logical range or element geometry is invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_row = kv_heads * head_dim;
    const size_t v_row = kv_heads * value_dim;
    const size_t cache_heads = physical_tokens * kv_heads;
    if (m > max_size / k_row || m > max_size / v_row ||
        head_dim > max_size / cache_heads || value_dim > max_size / cache_heads) {
        set_error("typed paged KV write chunk element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_values = m * k_row;
    const size_t v_values = m * v_row;
    const size_t k_cache_values = cache_heads * head_dim;
    const size_t v_cache_values = cache_heads * value_dim;
    const size_t table_entries = (end - 1u) / block_size + 1u;
    const size_t k_width = kv_cache_dtype_payload_bytes(k_dtype);
    const size_t v_width = kv_cache_dtype_payload_bytes(v_dtype);
    if (k_values > max_size / sizeof(float) || v_values > max_size / sizeof(float) ||
        k_cache_values > max_size / k_width || v_cache_values > max_size / v_width ||
        cache_heads > max_size / sizeof(uint16_t) || table_entries > max_size / sizeof(uint32_t)) {
        set_error("typed paged KV write chunk byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_bytes = k_values * sizeof(float);
    const size_t v_bytes = v_values * sizeof(float);
    const size_t k_cache_bytes = k_cache_values * k_width;
    const size_t v_cache_bytes = v_cache_values * v_width;
    const size_t table_bytes = table_entries * sizeof(uint32_t);
    const size_t scale_bytes = cache_heads * sizeof(uint16_t);
    if (k_buffer->bytes < k_bytes || v_buffer->bytes < v_bytes ||
        block_table_buffer->bytes < table_bytes || k_cache_buffer->bytes < k_cache_bytes ||
        v_cache_buffer->bytes < v_cache_bytes ||
        (k_scale_buffer != nullptr && k_scale_buffer->bytes < scale_bytes) ||
        (v_scale_buffer != nullptr && v_scale_buffer->bytes < scale_bytes)) {
        set_error("typed paged KV write chunk buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->backend != BackendKind::Hip) {
        set_error("typed paged KV write chunk currently requires the native HIP kernel");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    const ullm_status table_status = validate_paged_chunk_block_table_hip(
        block_table_buffer, table_entries, cache_blocks, stream, "typed paged KV write chunk");
    if (table_status != ULLM_STATUS_OK) return table_status;
    std::string hip_error;
    if (paged_kv_write_chunk_typed_f32_hip_kernel(
            k_buffer, v_buffer, block_table_buffer, cache_start, m, block_size, cache_blocks,
            table_entries, kv_heads, head_dim, value_dim, k_dtype, v_dtype, k_cache_buffer,
            v_cache_buffer, k_scale_buffer, v_scale_buffer, stream, &hip_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(hip_error.empty() ? "typed paged KV write chunk HIP kernel is unavailable" : hip_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

static ullm_status paged_causal_gqa_chunk_f32_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || output_buffer == nullptr || q_buffer == output_buffer ||
        (gate_buffer != nullptr &&
         (gate_buffer == output_buffer || gate_buffer == q_buffer))) {
        set_error("f32 paged causal GQA chunk received a null or aliased pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || m > 128 || block_size == 0 || cache_blocks == 0 || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 paged causal GQA chunk width must be in 1..128 and dimensions nonzero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (head_dim > 256 || value_dim > 256) {
        set_error("f32 paged causal GQA chunk head_dim/value_dim must be at most 256");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer != nullptr && head_dim != value_dim) {
        set_error("f32 paged causal GQA chunk gated ABI requires head_dim == value_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 paged causal GQA chunk q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 paged causal GQA chunk softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_buffer != nullptr && !buffers_share_backend(q_buffer, gate_buffer)) ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 paged causal GQA chunk buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 paged causal GQA chunk stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cached_prefix_len > max_size - m || cache_blocks > max_size / block_size) {
        set_error("f32 paged causal GQA chunk context or physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + m;
    const size_t physical_tokens = cache_blocks * block_size;
    if (total_context > physical_tokens) {
        set_error("f32 paged causal GQA chunk context exceeds physical cache capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t table_entries = (total_context - 1) / block_size + 1;
    if (m > max_size / q_heads || q_heads > max_size / head_dim ||
        q_heads > max_size / value_dim || kv_heads > max_size / physical_tokens) {
        set_error("f32 paged causal GQA chunk element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_heads_per_chunk = m * q_heads;
    if (q_heads_per_chunk > max_size / head_dim ||
        q_heads_per_chunk > max_size / value_dim) {
        set_error("f32 paged causal GQA chunk element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_heads_per_chunk * head_dim;
    const size_t output_elements = q_heads_per_chunk * value_dim;
    const size_t cache_heads = physical_tokens * kv_heads;
    if (head_dim > max_size / cache_heads || value_dim > max_size / cache_heads) {
        set_error("f32 paged causal GQA chunk cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = cache_heads * head_dim;
    const size_t v_elements = cache_heads * value_dim;
    if (q_elements > max_size / sizeof(float) || output_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) || v_elements > max_size / sizeof(float) ||
        table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("f32 paged causal GQA chunk byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t gate_bytes = output_bytes;
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t table_bytes = table_entries * sizeof(std::uint32_t);
    if (q_buffer->bytes < q_bytes || k_cache_buffer->bytes < k_bytes ||
        v_cache_buffer->bytes < v_bytes || block_table_buffer->bytes < table_bytes ||
        output_buffer->bytes < output_bytes ||
        (gate_buffer != nullptr && gate_buffer->bytes < gate_bytes)) {
        set_error("f32 paged causal GQA chunk buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        for (size_t index = 0; index < table_entries; ++index) {
            if (static_cast<size_t>(table[index]) >= cache_blocks) {
                set_error("f32 paged causal GQA chunk block table contains an out-of-range block id");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        paged_causal_gqa_chunk_f32_host(
            static_cast<const float *>(q_buffer->ptr),
            static_cast<const float *>(k_cache_buffer->ptr),
            static_cast<const float *>(v_cache_buffer->ptr),
            table,
            cached_prefix_len,
            m,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            static_cast<float *>(output_buffer->ptr));
        if (gate_buffer != nullptr) {
            const auto *gate = static_cast<const float *>(gate_buffer->ptr);
            auto *output = static_cast<float *>(output_buffer->ptr);
            for (size_t index = 0; index < output_elements; ++index) {
                output[index] *= 1.0f / (1.0f + std::exp(-gate[index]));
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    const ullm_status table_status = validate_paged_chunk_block_table_hip(
        block_table_buffer,
        table_entries,
        cache_blocks,
        stream,
        "f32 paged causal GQA chunk");
    if (table_status != ULLM_STATUS_OK) {
        return table_status;
    }
    std::string hip_error;
    if (paged_causal_gqa_chunk_f32_hip_kernel(
            q_buffer,
            gate_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cached_prefix_len,
            m,
            block_size,
            cache_blocks,
            table_entries,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(hip_error.empty() ? "f32 paged causal GQA chunk HIP kernel is unavailable" : hip_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return paged_causal_gqa_chunk_f32_impl(
        q_buffer,
        nullptr,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cached_prefix_len,
        m,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr) {
        set_error("f32 paged causal GQA chunk sigmoid gate received a null gate buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return paged_causal_gqa_chunk_f32_impl(
        q_buffer,
        gate_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cached_prefix_len,
        m,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}

static ullm_status paged_causal_gqa_chunk_typed_f32_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (k_dtype == ULLM_KV_CACHE_DTYPE_F32 && v_dtype == ULLM_KV_CACHE_DTYPE_F32 &&
        k_scale_buffer == nullptr && v_scale_buffer == nullptr) {
        return paged_causal_gqa_chunk_f32_impl(
            q_buffer, gate_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer,
            cached_prefix_len, m, block_size, cache_blocks, q_heads, kv_heads, head_dim,
            value_dim, softmax_scale, output_buffer, stream);
    }
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || output_buffer == nullptr || q_buffer == output_buffer ||
        (gate_buffer != nullptr &&
         (gate_buffer == output_buffer || gate_buffer == q_buffer))) {
        set_error("typed paged causal GQA chunk received a null or aliased pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0u || m > 128u || block_size == 0u || cache_blocks == 0u || q_heads == 0u ||
        kv_heads == 0u || head_dim == 0u || value_dim == 0u || head_dim > 256u ||
        value_dim > 256u || q_heads % kv_heads != 0u ||
        (gate_buffer != nullptr && head_dim != value_dim) ||
        !std::isfinite(softmax_scale) || softmax_scale <= 0.0f ||
        !kv_cache_dtype_is_supported(k_dtype) || !kv_cache_dtype_is_supported(v_dtype) ||
        (kv_cache_dtype_requires_scale(k_dtype) != (k_scale_buffer != nullptr)) ||
        (kv_cache_dtype_requires_scale(v_dtype) != (v_scale_buffer != nullptr))) {
        set_error("typed paged causal GQA chunk geometry, dtype, gate, or FP8 scales are invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_buffer != nullptr && !buffers_share_backend(q_buffer, gate_buffer)) ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer) ||
        (k_scale_buffer != nullptr && !buffers_share_backend(q_buffer, k_scale_buffer)) ||
        (v_scale_buffer != nullptr && !buffers_share_backend(q_buffer, v_scale_buffer)) ||
        !stream_matches_buffer(output_buffer, stream) ||
        (k_scale_buffer != nullptr && !stream_matches_buffer(k_scale_buffer, stream)) ||
        (v_scale_buffer != nullptr && !stream_matches_buffer(v_scale_buffer, stream))) {
        set_error("typed paged causal GQA chunk buffers or stream belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cached_prefix_len > max_size - m || cache_blocks > max_size / block_size) {
        set_error("typed paged causal GQA chunk context or physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + m;
    const size_t physical_tokens = cache_blocks * block_size;
    if (total_context > physical_tokens || m > max_size / q_heads ||
        q_heads > max_size / head_dim || q_heads > max_size / value_dim ||
        kv_heads > max_size / physical_tokens) {
        set_error("typed paged causal GQA chunk capacity or element geometry is invalid");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t query_heads = m * q_heads;
    if (query_heads > max_size / head_dim || query_heads > max_size / value_dim) {
        set_error("typed paged causal GQA chunk element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_values = query_heads * head_dim;
    const size_t output_values = query_heads * value_dim;
    const size_t cache_heads = physical_tokens * kv_heads;
    if (head_dim > max_size / cache_heads || value_dim > max_size / cache_heads) {
        set_error("typed paged causal GQA chunk cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_values = cache_heads * head_dim;
    const size_t v_values = cache_heads * value_dim;
    const size_t table_entries = (total_context - 1u) / block_size + 1u;
    const size_t k_width = kv_cache_dtype_payload_bytes(k_dtype);
    const size_t v_width = kv_cache_dtype_payload_bytes(v_dtype);
    if (q_values > max_size / sizeof(float) || output_values > max_size / sizeof(float) ||
        k_values > max_size / k_width || v_values > max_size / v_width ||
        cache_heads > max_size / sizeof(uint16_t) || table_entries > max_size / sizeof(uint32_t)) {
        set_error("typed paged causal GQA chunk byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_values * sizeof(float);
    const size_t output_bytes = output_values * sizeof(float);
    const size_t k_bytes = k_values * k_width;
    const size_t v_bytes = v_values * v_width;
    const size_t table_bytes = table_entries * sizeof(uint32_t);
    const size_t scale_bytes = cache_heads * sizeof(uint16_t);
    if (q_buffer->bytes < q_bytes || k_cache_buffer->bytes < k_bytes ||
        v_cache_buffer->bytes < v_bytes || block_table_buffer->bytes < table_bytes ||
        output_buffer->bytes < output_bytes ||
        (gate_buffer != nullptr && gate_buffer->bytes < output_bytes) ||
        (k_scale_buffer != nullptr && k_scale_buffer->bytes < scale_bytes) ||
        (v_scale_buffer != nullptr && v_scale_buffer->bytes < scale_bytes)) {
        set_error("typed paged causal GQA chunk buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->backend != BackendKind::Hip) {
        set_error("typed paged causal GQA chunk currently requires the native HIP kernel");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    const ullm_status table_status = validate_paged_chunk_block_table_hip(
        block_table_buffer, table_entries, cache_blocks, stream, "typed paged causal GQA chunk");
    if (table_status != ULLM_STATUS_OK) return table_status;
    std::string hip_error;
    if (paged_causal_gqa_chunk_typed_f32_hip_kernel(
            q_buffer, gate_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer,
            k_scale_buffer, v_scale_buffer, cached_prefix_len, m, block_size, cache_blocks,
            table_entries, q_heads, kv_heads, head_dim, value_dim, softmax_scale,
            k_dtype, v_dtype, output_buffer, stream, &hip_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(hip_error.empty() ? "typed paged causal GQA chunk HIP kernel is unavailable" : hip_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_typed_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return paged_causal_gqa_chunk_typed_f32_impl(
        q_buffer, nullptr, k_cache_buffer, v_cache_buffer, block_table_buffer,
        k_scale_buffer, v_scale_buffer, cached_prefix_len, m, block_size, cache_blocks,
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, k_dtype, v_dtype,
        output_buffer, stream);
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_sigmoid_gate_typed_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    const ullm_runtime_buffer *k_scale_buffer,
    const ullm_runtime_buffer *v_scale_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_kv_cache_dtype k_dtype,
    ullm_kv_cache_dtype v_dtype,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr) {
        set_error("typed paged causal GQA chunk sigmoid gate received a null gate buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return paged_causal_gqa_chunk_typed_f32_impl(
        q_buffer, gate_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer,
        k_scale_buffer, v_scale_buffer, cached_prefix_len, m, block_size, cache_blocks,
        q_heads, kv_heads, head_dim, value_dim, softmax_scale, k_dtype, v_dtype,
        output_buffer, stream);
}

// This remains intentionally separate from paged_causal_gqa_chunk_f32_impl: the scalar reader
// keeps its CPU fallback and HIP module unchanged while the production WMMA reader loads its own
// gfx1201 rocWMMA module for the validated Qwen3.5-9B M=1..=128 prefill shapes.
static ullm_status paged_causal_gqa_chunk_wmma_f32_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || output_buffer == nullptr || q_buffer == output_buffer ||
        (gate_buffer != nullptr &&
         (gate_buffer == output_buffer || gate_buffer == q_buffer))) {
        set_error("paged causal GQA WMMA received a null or aliased pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0u || m > 128u || cache_blocks == 0u) {
        set_error("paged causal GQA WMMA requires M in 1..=128 and nonzero cache blocks");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer != nullptr && head_dim != value_dim) {
        set_error("paged causal GQA WMMA gated ABI requires head_dim == value_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("paged causal GQA WMMA softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_buffer != nullptr && !buffers_share_backend(q_buffer, gate_buffer)) ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("paged causal GQA WMMA buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("paged causal GQA WMMA stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_buffer->backend != BackendKind::Hip) {
        set_error("paged causal GQA WMMA requires a HIP gfx1201 backend");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!paged_causal_gqa_chunk_wmma_shape_is_supported(
            hip_runtime().device_arch_name(q_buffer->hip_device_id),
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim)) {
        set_error(
            "paged causal GQA WMMA requires exact gfx1201, block=256, Q=16, KV=4, and head/value dim=256");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cached_prefix_len > max_size - m || cache_blocks > max_size / block_size) {
        set_error("paged causal GQA WMMA context or physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t total_context = cached_prefix_len + m;
    const size_t physical_tokens = cache_blocks * block_size;
    if (total_context > physical_tokens) {
        set_error("paged causal GQA WMMA context exceeds physical cache capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t table_entries = (total_context - 1u) / block_size + 1u;
    if (m > max_size / q_heads || q_heads > max_size / head_dim ||
        q_heads > max_size / value_dim || kv_heads > max_size / physical_tokens) {
        set_error("paged causal GQA WMMA element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_heads_per_chunk = m * q_heads;
    if (q_heads_per_chunk > max_size / head_dim ||
        q_heads_per_chunk > max_size / value_dim) {
        set_error("paged causal GQA WMMA element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_heads_per_chunk * head_dim;
    const size_t output_elements = q_heads_per_chunk * value_dim;
    const size_t cache_heads = physical_tokens * kv_heads;
    if (head_dim > max_size / cache_heads || value_dim > max_size / cache_heads) {
        set_error("paged causal GQA WMMA cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = cache_heads * head_dim;
    const size_t v_elements = cache_heads * value_dim;
    if (q_elements > max_size / sizeof(float) || output_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) || v_elements > max_size / sizeof(float) ||
        table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("paged causal GQA WMMA byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t table_bytes = table_entries * sizeof(std::uint32_t);
    if (q_buffer->bytes < q_bytes || k_cache_buffer->bytes < k_bytes ||
        v_cache_buffer->bytes < v_bytes || block_table_buffer->bytes < table_bytes ||
        output_buffer->bytes < output_bytes ||
        (gate_buffer != nullptr && gate_buffer->bytes < output_bytes)) {
        set_error("paged causal GQA WMMA buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const ullm_status table_status = validate_paged_chunk_block_table_hip(
        block_table_buffer,
        table_entries,
        cache_blocks,
        stream,
        "paged causal GQA WMMA");
    if (table_status != ULLM_STATUS_OK) {
        return table_status;
    }
    std::string hip_error;
    if (paged_causal_gqa_chunk_wmma_f32_hip_kernel(
            q_buffer,
            gate_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cached_prefix_len,
            m,
            block_size,
            cache_blocks,
            table_entries,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    set_error(
        hip_error.empty() ? "paged causal GQA WMMA HIP kernel is unavailable" :
                            hip_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return paged_causal_gqa_chunk_wmma_f32_impl(
        q_buffer,
        nullptr,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cached_prefix_len,
        m,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr) {
        set_error("paged causal GQA WMMA sigmoid gate received a null gate buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return paged_causal_gqa_chunk_wmma_f32_impl(
        q_buffer,
        gate_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cached_prefix_len,
        m,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_prototype_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return ullm_runtime_paged_causal_gqa_chunk_wmma_f32(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cached_prefix_len,
        m,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_prototype_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    return ullm_runtime_paged_causal_gqa_chunk_wmma_sigmoid_gate_f32(
        q_buffer,
        gate_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cached_prefix_len,
        m,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output_buffer,
        stream);
}
