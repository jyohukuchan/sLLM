ullm_status ullm_runtime_sq_fp8_matvec_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (payload_buffer == nullptr || scale_buffer == nullptr || input_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("SQ FP8 matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("SQ FP8 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_kind > 2u) {
        set_error("SQ FP8 matvec scale kind must be tensor(0), row(1), or row_block(2)");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_kind == 2u && scale_block_cols == 0) {
        set_error("SQ FP8 matvec row_block scale_block_cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(payload_buffer, scale_buffer) ||
        !buffers_share_backend(payload_buffer, input_buffer) ||
        !buffers_share_backend(payload_buffer, output_buffer)) {
        set_error("SQ FP8 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("SQ FP8 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("SQ FP8 matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    size_t scale_values = 1;
    if (scale_kind == 1u) {
        scale_values = rows;
    } else if (scale_kind == 2u) {
        const size_t blocks_per_row = (cols + scale_block_cols - 1) / scale_block_cols;
        if (blocks_per_row > max_size / rows) {
            set_error("SQ FP8 matvec row_block scale count overflows");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        scale_values = rows * blocks_per_row;
    }
    if (cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float) ||
        scale_values > max_size / sizeof(float)) {
        set_error("SQ FP8 matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_payload_bytes = matrix_elements;
    const size_t required_scale_bytes = scale_values * sizeof(float);
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    if (payload_buffer->bytes < required_payload_bytes) {
        set_error("SQ FP8 matvec payload buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < required_scale_bytes) {
        set_error("SQ FP8 matvec scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("SQ FP8 matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("SQ FP8 matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (payload_buffer->backend == BackendKind::Cpu) {
        const auto *payload = static_cast<const uint8_t *>(payload_buffer->ptr);
        const auto *scales = static_cast<const float *>(scale_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!sq_fp8_matvec_f32_host(
                payload,
                scales,
                input,
                rows,
                cols,
                scale_kind,
                1,
                scale_block_cols,
                output)) {
            set_error("SQ FP8 matvec host path rejected invalid values");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (sq_fp8_matvec_f32_hip_kernel(
            payload_buffer,
            scale_buffer,
            input_buffer,
            rows,
            cols,
            scale_kind,
            1,
            scale_block_cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "SQ FP8 matvec HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return sq_fp8_matvec_f32_hip_staging(
        payload_buffer,
        scale_buffer,
        input_buffer,
        rows,
        cols,
        scale_kind,
        scale_block_cols,
        required_payload_bytes,
        required_scale_bytes,
        required_input_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_sq_fp8_matvec_batch_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (payload_buffer == nullptr || scale_buffer == nullptr || input_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("SQ FP8 matvec batch received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0 || batch_count == 0) {
        set_error("SQ FP8 matvec batch rows, cols, and batch count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_kind > 2u) {
        set_error("SQ FP8 matvec batch scale kind must be tensor(0), row(1), or row_block(2)");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_kind == 2u && scale_block_cols == 0) {
        set_error("SQ FP8 matvec batch row_block scale_block_cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(payload_buffer, scale_buffer) ||
        !buffers_share_backend(payload_buffer, input_buffer) ||
        !buffers_share_backend(payload_buffer, output_buffer)) {
        set_error("SQ FP8 matvec batch buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("SQ FP8 matvec batch stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("SQ FP8 matvec batch matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    size_t scale_values = 1;
    if (scale_kind == 1u) {
        scale_values = rows;
    } else if (scale_kind == 2u) {
        const size_t blocks_per_row = (cols + scale_block_cols - 1) / scale_block_cols;
        if (blocks_per_row > max_size / rows) {
            set_error("SQ FP8 matvec batch row_block scale count overflows");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        scale_values = rows * blocks_per_row;
    }
    if (batch_count > max_size / cols) {
        set_error("SQ FP8 matvec batch input element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_elements = batch_count * cols;
    if (batch_count > max_size / rows) {
        set_error("SQ FP8 matvec batch output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = batch_count * rows;
    if (input_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float) ||
        scale_values > max_size / sizeof(float)) {
        set_error("SQ FP8 matvec batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_payload_bytes = matrix_elements;
    const size_t required_scale_bytes = scale_values * sizeof(float);
    const size_t required_input_bytes = input_elements * sizeof(float);
    const size_t required_output_bytes = output_elements * sizeof(float);
    if (payload_buffer->bytes < required_payload_bytes) {
        set_error("SQ FP8 matvec batch payload buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < required_scale_bytes) {
        set_error("SQ FP8 matvec batch scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("SQ FP8 matvec batch input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("SQ FP8 matvec batch output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (payload_buffer->backend == BackendKind::Cpu) {
        const auto *payload = static_cast<const uint8_t *>(payload_buffer->ptr);
        const auto *scales = static_cast<const float *>(scale_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!sq_fp8_matvec_batch_f32_host(
                payload,
                scales,
                input,
                rows,
                cols,
                scale_kind,
                1,
                scale_block_cols,
                batch_count,
                output)) {
            set_error("SQ FP8 matvec batch host path rejected invalid values");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (sq_fp8_matvec_batch_f32_hip_kernel(
            payload_buffer,
            scale_buffer,
            input_buffer,
            rows,
            cols,
            scale_kind,
            1,
            scale_block_cols,
            batch_count,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "SQ FP8 matvec batch HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return sq_fp8_matvec_batch_f32_hip_staging(
        payload_buffer,
        scale_buffer,
        input_buffer,
        rows,
        cols,
        scale_kind,
        scale_block_cols,
        batch_count,
        required_payload_bytes,
        required_scale_bytes,
        required_input_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

static ullm_status sq_fp8_matvec_block2d_impl(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t scale_block_rows,
    size_t scale_block_cols,
    size_t batch_count,
    bool use_batch_kernel,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    ullm_sq_fp8_execution_path *execution_path) {
    if (payload_buffer == nullptr || scale_buffer == nullptr || input_buffer == nullptr ||
        output_buffer == nullptr || execution_path == nullptr) {
        set_error("SQ FP8 block2d matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0 || batch_count == 0) {
        set_error("SQ FP8 block2d matvec rows, cols, and batch count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_block_rows == 0 || scale_block_cols == 0) {
        set_error("SQ FP8 block2d matvec scale block rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(payload_buffer, scale_buffer) ||
        !buffers_share_backend(payload_buffer, input_buffer) ||
        !buffers_share_backend(payload_buffer, output_buffer)) {
        set_error("SQ FP8 block2d matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("SQ FP8 block2d matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("SQ FP8 block2d matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    const size_t scale_block_row_count = 1 + (rows - 1) / scale_block_rows;
    const size_t scale_block_col_count = 1 + (cols - 1) / scale_block_cols;
    if (scale_block_col_count > max_size / scale_block_row_count) {
        set_error("SQ FP8 block2d matvec scale count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t scale_values = scale_block_row_count * scale_block_col_count;
    if (cols > max_size / batch_count) {
        set_error("SQ FP8 block2d matvec input element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_elements = batch_count * cols;
    if (rows > max_size / batch_count) {
        set_error("SQ FP8 block2d matvec output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = batch_count * rows;
    if (scale_values > max_size / sizeof(float) ||
        input_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("SQ FP8 block2d matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_scale_bytes = scale_values * sizeof(float);
    const size_t required_input_bytes = input_elements * sizeof(float);
    const size_t required_output_bytes = output_elements * sizeof(float);
    if (payload_buffer->bytes < matrix_elements) {
        set_error("SQ FP8 block2d matvec payload buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < required_scale_bytes) {
        set_error("SQ FP8 block2d matvec scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("SQ FP8 block2d matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("SQ FP8 block2d matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (payload_buffer->backend == BackendKind::Cpu) {
        const auto *payload = static_cast<const uint8_t *>(payload_buffer->ptr);
        const auto *scales = static_cast<const float *>(scale_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        const bool succeeded = use_batch_kernel ?
            sq_fp8_matvec_batch_f32_host(
                payload,
                scales,
                input,
                rows,
                cols,
                2u,
                scale_block_rows,
                scale_block_cols,
                batch_count,
                output) :
            sq_fp8_matvec_f32_host(
                payload,
                scales,
                input,
                rows,
                cols,
                2u,
                scale_block_rows,
                scale_block_cols,
                output);
        if (!succeeded) {
            set_error("SQ FP8 block2d matvec CPU reference rejected invalid values");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        *execution_path = ULLM_SQ_FP8_EXECUTION_PATH_CPU_REFERENCE;
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    const bool succeeded = use_batch_kernel ?
        sq_fp8_matvec_batch_f32_hip_kernel(
            payload_buffer,
            scale_buffer,
            input_buffer,
            rows,
            cols,
            2u,
            scale_block_rows,
            scale_block_cols,
            batch_count,
            output_buffer,
            stream,
            &hip_kernel_error) :
        sq_fp8_matvec_f32_hip_kernel(
            payload_buffer,
            scale_buffer,
            input_buffer,
            rows,
            cols,
            2u,
            scale_block_rows,
            scale_block_cols,
            output_buffer,
            stream,
            &hip_kernel_error);
    if (!succeeded) {
        set_error(
            hip_kernel_error.empty() ? "SQ FP8 block2d matvec HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    *execution_path = ULLM_SQ_FP8_EXECUTION_PATH_HIP_KERNEL;
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_sq_fp8_matvec_block2d_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t scale_block_rows,
    size_t scale_block_cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    ullm_sq_fp8_execution_path *execution_path) {
    return sq_fp8_matvec_block2d_impl(
        payload_buffer,
        scale_buffer,
        input_buffer,
        rows,
        cols,
        scale_block_rows,
        scale_block_cols,
        1,
        false,
        output_buffer,
        stream,
        execution_path);
}

ullm_status ullm_runtime_sq_fp8_matvec_block2d_batch_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t scale_block_rows,
    size_t scale_block_cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    ullm_sq_fp8_execution_path *execution_path) {
    return sq_fp8_matvec_block2d_impl(
        payload_buffer,
        scale_buffer,
        input_buffer,
        rows,
        cols,
        scale_block_rows,
        scale_block_cols,
        batch_count,
        true,
        output_buffer,
        stream,
        execution_path);
}

ullm_status ullm_runtime_sq_fp8_matvec_pair_f32(
    const ullm_runtime_buffer *left_payload_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    uint32_t left_scale_kind,
    size_t left_scale_block_cols,
    const ullm_runtime_buffer *right_payload_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    uint32_t right_scale_kind,
    size_t right_scale_block_cols,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream) {
    if (left_payload_buffer == nullptr || left_scale_buffer == nullptr ||
        right_payload_buffer == nullptr || right_scale_buffer == nullptr ||
        input_buffer == nullptr || left_output_buffer == nullptr || right_output_buffer == nullptr) {
        set_error("SQ FP8 matvec pair received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_rows == 0 || right_rows == 0 || cols == 0) {
        set_error("SQ FP8 matvec pair rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_scale_kind > 2u || right_scale_kind > 2u) {
        set_error("SQ FP8 matvec pair scale kind must be tensor(0), row(1), or row_block(2)");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((left_scale_kind == 2u && left_scale_block_cols == 0) ||
        (right_scale_kind == 2u && right_scale_block_cols == 0)) {
        set_error("SQ FP8 matvec pair row_block scale_block_cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(left_payload_buffer, left_scale_buffer) ||
        !buffers_share_backend(left_payload_buffer, right_payload_buffer) ||
        !buffers_share_backend(left_payload_buffer, right_scale_buffer) ||
        !buffers_share_backend(left_payload_buffer, input_buffer) ||
        !buffers_share_backend(left_payload_buffer, left_output_buffer) ||
        !buffers_share_backend(left_payload_buffer, right_output_buffer)) {
        set_error("SQ FP8 matvec pair buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(left_output_buffer, stream) ||
        !stream_matches_buffer(right_output_buffer, stream)) {
        set_error("SQ FP8 matvec pair stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    auto scale_values_for = [max_size, cols](size_t rows,
                                             uint32_t scale_kind,
                                             size_t scale_block_cols,
                                             const char *label,
                                             size_t *out) -> bool {
        if (out == nullptr) {
            return false;
        }
        if (scale_kind == 0u) {
            *out = 1;
            return true;
        }
        if (scale_kind == 1u) {
            *out = rows;
            return true;
        }
        const size_t blocks_per_row = (cols + scale_block_cols - 1) / scale_block_cols;
        if (blocks_per_row > max_size / rows) {
            set_error((std::string("SQ FP8 matvec pair ") + label +
                       " row_block scale count overflows")
                          .c_str());
            return false;
        }
        *out = rows * blocks_per_row;
        return true;
    };
    auto matrix_bytes_for = [max_size, cols](size_t rows, const char *label, size_t *out) -> bool {
        if (rows > max_size / cols) {
            set_error(
                (std::string("SQ FP8 matvec pair ") + label + " matrix element count overflows")
                    .c_str());
            return false;
        }
        *out = rows * cols;
        return true;
    };
    size_t left_payload_bytes = 0;
    size_t right_payload_bytes = 0;
    size_t left_scale_values = 0;
    size_t right_scale_values = 0;
    if (!matrix_bytes_for(left_rows, "left", &left_payload_bytes) ||
        !matrix_bytes_for(right_rows, "right", &right_payload_bytes) ||
        !scale_values_for(
            left_rows, left_scale_kind, left_scale_block_cols, "left", &left_scale_values) ||
        !scale_values_for(
            right_rows, right_scale_kind, right_scale_block_cols, "right", &right_scale_values)) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cols > max_size / sizeof(float) ||
        left_rows > max_size / sizeof(float) ||
        right_rows > max_size / sizeof(float) ||
        left_scale_values > max_size / sizeof(float) ||
        right_scale_values > max_size / sizeof(float)) {
        set_error("SQ FP8 matvec pair byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_bytes = cols * sizeof(float);
    const size_t left_output_bytes = left_rows * sizeof(float);
    const size_t right_output_bytes = right_rows * sizeof(float);
    const size_t left_scale_bytes = left_scale_values * sizeof(float);
    const size_t right_scale_bytes = right_scale_values * sizeof(float);
    if (left_payload_buffer->bytes < left_payload_bytes ||
        right_payload_buffer->bytes < right_payload_bytes ||
        left_scale_buffer->bytes < left_scale_bytes ||
        right_scale_buffer->bytes < right_scale_bytes ||
        input_buffer->bytes < input_bytes ||
        left_output_buffer->bytes < left_output_bytes ||
        right_output_buffer->bytes < right_output_bytes) {
        set_error("SQ FP8 matvec pair buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (left_payload_buffer->backend == BackendKind::Cpu) {
        if (!sq_fp8_matvec_pair_f32_host(
                static_cast<const uint8_t *>(left_payload_buffer->ptr),
                static_cast<const float *>(left_scale_buffer->ptr),
                left_scale_kind,
                left_scale_block_cols,
                static_cast<const uint8_t *>(right_payload_buffer->ptr),
                static_cast<const float *>(right_scale_buffer->ptr),
                right_scale_kind,
                right_scale_block_cols,
                static_cast<const float *>(input_buffer->ptr),
                left_rows,
                right_rows,
                cols,
                static_cast<float *>(left_output_buffer->ptr),
                static_cast<float *>(right_output_buffer->ptr))) {
            set_error("SQ FP8 matvec pair host path rejected invalid values");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (sq_fp8_matvec_pair_f32_hip_kernel(
            left_payload_buffer,
            left_scale_buffer,
            left_scale_kind,
            left_scale_block_cols,
            right_payload_buffer,
            right_scale_buffer,
            right_scale_kind,
            right_scale_block_cols,
            input_buffer,
            left_rows,
            right_rows,
            cols,
            left_output_buffer,
            right_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "SQ FP8 matvec pair HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    const ullm_status left_status = ullm_runtime_sq_fp8_matvec_f32(
        left_payload_buffer,
        left_scale_buffer,
        input_buffer,
        left_rows,
        cols,
        left_scale_kind,
        left_scale_block_cols,
        left_output_buffer,
        stream);
    if (left_status != ULLM_STATUS_OK) {
        return left_status;
    }
    return ullm_runtime_sq_fp8_matvec_f32(
        right_payload_buffer,
        right_scale_buffer,
        input_buffer,
        right_rows,
        cols,
        right_scale_kind,
        right_scale_block_cols,
        right_output_buffer,
        stream);
}

ullm_status ullm_runtime_sq_fp8_matvec_triple_f32(
    const ullm_runtime_buffer *first_payload_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    uint32_t first_scale_kind,
    size_t first_scale_block_cols,
    const ullm_runtime_buffer *second_payload_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    uint32_t second_scale_kind,
    size_t second_scale_block_cols,
    const ullm_runtime_buffer *third_payload_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    uint32_t third_scale_kind,
    size_t third_scale_block_cols,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream) {
    if (first_payload_buffer == nullptr || first_scale_buffer == nullptr ||
        second_payload_buffer == nullptr || second_scale_buffer == nullptr ||
        third_payload_buffer == nullptr || third_scale_buffer == nullptr ||
        input_buffer == nullptr || first_output_buffer == nullptr ||
        second_output_buffer == nullptr || third_output_buffer == nullptr) {
        set_error("SQ FP8 matvec triple received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (first_rows == 0 || second_rows == 0 || third_rows == 0 || cols == 0) {
        set_error("SQ FP8 matvec triple rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (first_scale_kind > 2u || second_scale_kind > 2u || third_scale_kind > 2u) {
        set_error("SQ FP8 matvec triple scale kind must be tensor(0), row(1), or row_block(2)");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((first_scale_kind == 2u && first_scale_block_cols == 0) ||
        (second_scale_kind == 2u && second_scale_block_cols == 0) ||
        (third_scale_kind == 2u && third_scale_block_cols == 0)) {
        set_error("SQ FP8 matvec triple row_block scale_block_cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(first_payload_buffer, first_scale_buffer) ||
        !buffers_share_backend(first_payload_buffer, second_payload_buffer) ||
        !buffers_share_backend(first_payload_buffer, second_scale_buffer) ||
        !buffers_share_backend(first_payload_buffer, third_payload_buffer) ||
        !buffers_share_backend(first_payload_buffer, third_scale_buffer) ||
        !buffers_share_backend(first_payload_buffer, input_buffer) ||
        !buffers_share_backend(first_payload_buffer, first_output_buffer) ||
        !buffers_share_backend(first_payload_buffer, second_output_buffer) ||
        !buffers_share_backend(first_payload_buffer, third_output_buffer)) {
        set_error("SQ FP8 matvec triple buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(first_output_buffer, stream) ||
        !stream_matches_buffer(second_output_buffer, stream) ||
        !stream_matches_buffer(third_output_buffer, stream)) {
        set_error("SQ FP8 matvec triple stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    auto scale_values_for = [max_size, cols](size_t rows,
                                             uint32_t scale_kind,
                                             size_t scale_block_cols,
                                             const char *label,
                                             size_t *out) -> bool {
        if (scale_kind == 0u) {
            *out = 1;
            return true;
        }
        if (scale_kind == 1u) {
            *out = rows;
            return true;
        }
        const size_t blocks_per_row = (cols + scale_block_cols - 1) / scale_block_cols;
        if (blocks_per_row > max_size / rows) {
            set_error((std::string("SQ FP8 matvec triple ") + label +
                       " row_block scale count overflows")
                          .c_str());
            return false;
        }
        *out = rows * blocks_per_row;
        return true;
    };
    auto matrix_bytes_for = [max_size, cols](size_t rows, const char *label, size_t *out) -> bool {
        if (rows > max_size / cols) {
            set_error(
                (std::string("SQ FP8 matvec triple ") + label + " matrix element count overflows")
                    .c_str());
            return false;
        }
        *out = rows * cols;
        return true;
    };
    size_t first_payload_bytes = 0;
    size_t second_payload_bytes = 0;
    size_t third_payload_bytes = 0;
    size_t first_scale_values = 0;
    size_t second_scale_values = 0;
    size_t third_scale_values = 0;
    if (!matrix_bytes_for(first_rows, "first", &first_payload_bytes) ||
        !matrix_bytes_for(second_rows, "second", &second_payload_bytes) ||
        !matrix_bytes_for(third_rows, "third", &third_payload_bytes) ||
        !scale_values_for(
            first_rows, first_scale_kind, first_scale_block_cols, "first", &first_scale_values) ||
        !scale_values_for(
            second_rows, second_scale_kind, second_scale_block_cols, "second", &second_scale_values) ||
        !scale_values_for(
            third_rows, third_scale_kind, third_scale_block_cols, "third", &third_scale_values)) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cols > max_size / sizeof(float) ||
        first_rows > max_size / sizeof(float) ||
        second_rows > max_size / sizeof(float) ||
        third_rows > max_size / sizeof(float) ||
        first_scale_values > max_size / sizeof(float) ||
        second_scale_values > max_size / sizeof(float) ||
        third_scale_values > max_size / sizeof(float)) {
        set_error("SQ FP8 matvec triple byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_bytes = cols * sizeof(float);
    const size_t first_output_bytes = first_rows * sizeof(float);
    const size_t second_output_bytes = second_rows * sizeof(float);
    const size_t third_output_bytes = third_rows * sizeof(float);
    const size_t first_scale_bytes = first_scale_values * sizeof(float);
    const size_t second_scale_bytes = second_scale_values * sizeof(float);
    const size_t third_scale_bytes = third_scale_values * sizeof(float);
    if (first_payload_buffer->bytes < first_payload_bytes ||
        second_payload_buffer->bytes < second_payload_bytes ||
        third_payload_buffer->bytes < third_payload_bytes ||
        first_scale_buffer->bytes < first_scale_bytes ||
        second_scale_buffer->bytes < second_scale_bytes ||
        third_scale_buffer->bytes < third_scale_bytes ||
        input_buffer->bytes < input_bytes ||
        first_output_buffer->bytes < first_output_bytes ||
        second_output_buffer->bytes < second_output_bytes ||
        third_output_buffer->bytes < third_output_bytes) {
        set_error("SQ FP8 matvec triple buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (first_payload_buffer->backend == BackendKind::Cpu) {
        if (!sq_fp8_matvec_triple_f32_host(
                static_cast<const uint8_t *>(first_payload_buffer->ptr),
                static_cast<const float *>(first_scale_buffer->ptr),
                first_scale_kind,
                first_scale_block_cols,
                static_cast<const uint8_t *>(second_payload_buffer->ptr),
                static_cast<const float *>(second_scale_buffer->ptr),
                second_scale_kind,
                second_scale_block_cols,
                static_cast<const uint8_t *>(third_payload_buffer->ptr),
                static_cast<const float *>(third_scale_buffer->ptr),
                third_scale_kind,
                third_scale_block_cols,
                static_cast<const float *>(input_buffer->ptr),
                first_rows,
                second_rows,
                third_rows,
                cols,
                static_cast<float *>(first_output_buffer->ptr),
                static_cast<float *>(second_output_buffer->ptr),
                static_cast<float *>(third_output_buffer->ptr))) {
            set_error("SQ FP8 matvec triple host path rejected invalid values");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (sq_fp8_matvec_triple_f32_hip_kernel(
            first_payload_buffer,
            first_scale_buffer,
            first_scale_kind,
            first_scale_block_cols,
            second_payload_buffer,
            second_scale_buffer,
            second_scale_kind,
            second_scale_block_cols,
            third_payload_buffer,
            third_scale_buffer,
            third_scale_kind,
            third_scale_block_cols,
            input_buffer,
            first_rows,
            second_rows,
            third_rows,
            cols,
            first_output_buffer,
            second_output_buffer,
            third_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "SQ FP8 matvec triple HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    const ullm_status first_status = ullm_runtime_sq_fp8_matvec_f32(
        first_payload_buffer,
        first_scale_buffer,
        input_buffer,
        first_rows,
        cols,
        first_scale_kind,
        first_scale_block_cols,
        first_output_buffer,
        stream);
    if (first_status != ULLM_STATUS_OK) {
        return first_status;
    }
    const ullm_status second_status = ullm_runtime_sq_fp8_matvec_f32(
        second_payload_buffer,
        second_scale_buffer,
        input_buffer,
        second_rows,
        cols,
        second_scale_kind,
        second_scale_block_cols,
        second_output_buffer,
        stream);
    if (second_status != ULLM_STATUS_OK) {
        return second_status;
    }
    return ullm_runtime_sq_fp8_matvec_f32(
        third_payload_buffer,
        third_scale_buffer,
        input_buffer,
        third_rows,
        cols,
        third_scale_kind,
        third_scale_block_cols,
        third_output_buffer,
        stream);
}
