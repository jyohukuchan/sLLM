ullm_status ullm_runtime_linear_attn_gate_beta_f32(
    const ullm_runtime_buffer *a_buffer,
    const ullm_runtime_buffer *b_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t sequence_len,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    if (a_buffer == nullptr || b_buffer == nullptr || a_log_buffer == nullptr ||
        dt_bias_buffer == nullptr || gate_output_buffer == nullptr ||
        beta_output_buffer == nullptr) {
        set_error("f32 linear attention gate beta received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (heads == 0 || sequence_len == 0) {
        set_error("f32 linear attention gate beta heads and sequence length must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(a_buffer, b_buffer) ||
        !buffers_share_backend(a_buffer, a_log_buffer) ||
        !buffers_share_backend(a_buffer, dt_bias_buffer) ||
        !buffers_share_backend(a_buffer, gate_output_buffer) ||
        !buffers_share_backend(a_buffer, beta_output_buffer)) {
        set_error("f32 linear attention gate beta buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(gate_output_buffer, stream) ||
        !stream_matches_buffer(beta_output_buffer, stream)) {
        set_error("f32 linear attention gate beta stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (sequence_len > max_size / heads) {
        set_error("f32 linear attention gate beta element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t sequence_elements = heads * sequence_len;
    if (sequence_elements > max_size / sizeof(float) ||
        heads > max_size / sizeof(float)) {
        set_error("f32 linear attention gate beta byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t sequence_bytes = sequence_elements * sizeof(float);
    const size_t head_bytes = heads * sizeof(float);
    if (a_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta a buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (b_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta b buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_log_buffer->bytes < head_bytes) {
        set_error("f32 linear attention gate beta A_log buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (dt_bias_buffer->bytes < head_bytes) {
        set_error("f32 linear attention gate beta dt_bias buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_output_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (beta_output_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta beta output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (a_buffer->backend == BackendKind::Cpu) {
        const auto *a = static_cast<const float *>(a_buffer->ptr);
        const auto *b = static_cast<const float *>(b_buffer->ptr);
        const auto *a_log = static_cast<const float *>(a_log_buffer->ptr);
        const auto *dt_bias = static_cast<const float *>(dt_bias_buffer->ptr);
        auto *gate = static_cast<float *>(gate_output_buffer->ptr);
        auto *beta = static_cast<float *>(beta_output_buffer->ptr);
        linear_attn_gate_beta_f32_host(
            a,
            b,
            a_log,
            dt_bias,
            heads,
            sequence_len,
            gate,
            beta);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_gate_beta_f32_hip_kernel(
            a_buffer,
            b_buffer,
            a_log_buffer,
            dt_bias_buffer,
            heads,
            sequence_len,
            gate_output_buffer,
            beta_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "f32 linear attention gate beta HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_gate_beta_f32_hip_staging(
        a_buffer,
        b_buffer,
        a_log_buffer,
        dt_bias_buffer,
        heads,
        sequence_len,
        sequence_bytes,
        head_bytes,
        gate_output_buffer,
        beta_output_buffer,
        stream);
}

ullm_status ullm_runtime_linear_attn_recurrent_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *beta_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    ullm_runtime_buffer *state_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr ||
        gate_buffer == nullptr || beta_buffer == nullptr || state_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 linear attention recurrent received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads == 0 || value_heads == 0 || sequence_len == 0 || key_dim == 0 ||
        value_dim == 0) {
        set_error(
            "f32 linear attention recurrent key_heads, value_heads, sequence length, key_dim, and value_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (value_heads % key_heads != 0) {
        set_error("f32 linear attention recurrent value_heads must be a multiple of key_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, v_buffer) ||
        !buffers_share_backend(q_buffer, gate_buffer) ||
        !buffers_share_backend(q_buffer, beta_buffer) ||
        !buffers_share_backend(q_buffer, state_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 linear attention recurrent buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(state_buffer, stream) ||
        !stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 linear attention recurrent stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (sequence_len > max_size / key_heads) {
        set_error("f32 linear attention recurrent key head sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t key_head_sequence_elements = key_heads * sequence_len;
    if (sequence_len > max_size / value_heads) {
        set_error("f32 linear attention recurrent value head sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t value_head_sequence_elements = value_heads * sequence_len;
    if (key_head_sequence_elements > max_size / key_dim) {
        set_error("f32 linear attention recurrent q/k element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qk_elements = key_head_sequence_elements * key_dim;
    if (value_head_sequence_elements > max_size / value_dim) {
        set_error("f32 linear attention recurrent v/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t v_elements = value_head_sequence_elements * value_dim;
    if (value_heads > max_size / key_dim) {
        set_error("f32 linear attention recurrent state key element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t state_key_elements = value_heads * key_dim;
    if (state_key_elements > max_size / value_dim) {
        set_error("f32 linear attention recurrent state element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t state_elements = state_key_elements * value_dim;
    if (qk_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        value_head_sequence_elements > max_size / sizeof(float) ||
        state_elements > max_size / sizeof(float)) {
        set_error("f32 linear attention recurrent byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qk_bytes = qk_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t gate_beta_bytes = value_head_sequence_elements * sizeof(float);
    const size_t state_bytes = state_elements * sizeof(float);

    if (q_buffer->bytes < qk_bytes) {
        set_error("f32 linear attention recurrent q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->bytes < qk_bytes) {
        set_error("f32 linear attention recurrent k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 linear attention recurrent v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer->bytes < gate_beta_bytes) {
        set_error("f32 linear attention recurrent gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (beta_buffer->bytes < gate_beta_bytes) {
        set_error("f32 linear attention recurrent beta buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (state_buffer->bytes < state_bytes) {
        set_error("f32 linear attention recurrent state buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < v_bytes) {
        set_error("f32 linear attention recurrent output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        const auto *gate = static_cast<const float *>(gate_buffer->ptr);
        const auto *beta = static_cast<const float *>(beta_buffer->ptr);
        auto *state = static_cast<float *>(state_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        linear_attn_recurrent_f32_host(
            q,
            k,
            v,
            gate,
            beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            state,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_recurrent_f32_hip_kernel(
            q_buffer,
            k_buffer,
            v_buffer,
            gate_buffer,
            beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            state_buffer,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "f32 linear attention recurrent HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_recurrent_f32_hip_staging(
        q_buffer,
        k_buffer,
        v_buffer,
        gate_buffer,
        beta_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        qk_bytes,
        v_bytes,
        gate_beta_bytes,
        state_bytes,
        state_buffer,
        output_buffer,
        stream);
}
