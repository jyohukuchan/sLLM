ullm_status ullm_runtime_matvec_f32(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (matrix_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("f32 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(matrix_buffer, input_buffer) ||
        !buffers_share_backend(matrix_buffer, output_buffer)) {
        set_error("f32 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("f32 matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    if (matrix_elements > max_size / sizeof(float) ||
        cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float)) {
        set_error("f32 matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_matrix_bytes = matrix_elements * sizeof(float);
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    if (matrix_buffer->bytes < required_matrix_bytes) {
        set_error("f32 matvec matrix buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("f32 matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("f32 matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (matrix_buffer->backend == BackendKind::Cpu) {
        const auto *matrix = static_cast<const float *>(matrix_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        matvec_f32_host(matrix, input, rows, cols, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool disable_rocblas = std::getenv("ULLM_DISABLE_ROCBLAS_MATVEC") != nullptr;
    const bool require_rocblas = std::getenv("ULLM_REQUIRE_ROCBLAS_MATVEC") != nullptr;
    const bool enable_rocblas =
        require_rocblas || std::getenv("ULLM_ENABLE_ROCBLAS_MATVEC") != nullptr;
    if (disable_rocblas && require_rocblas) {
        set_error("ULLM_DISABLE_ROCBLAS_MATVEC and ULLM_REQUIRE_ROCBLAS_MATVEC are both set");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    if (enable_rocblas && !disable_rocblas) {
        std::string rocblas_error;
        if (rocblas_runtime().sgemv_row_major(
                matrix_buffer->hip_device_id,
                stream == nullptr ? nullptr : stream->stream,
                matrix_buffer->ptr,
                input_buffer->ptr,
                rows,
                cols,
                output_buffer->ptr,
                &rocblas_error)) {
            set_error("");
            return ULLM_STATUS_OK;
        }
        if (require_rocblas) {
            set_error(rocblas_error.empty() ? "rocBLAS SGEMV is unavailable" : rocblas_error.c_str());
            return ULLM_STATUS_RUNTIME_ERROR;
        }
    }

    std::string hip_kernel_error;
    if (matvec_f32_hip_kernel(
            matrix_buffer,
            input_buffer,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 matvec HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return matvec_f32_hip_staging(
        matrix_buffer,
        input_buffer,
        rows,
        cols,
        required_matrix_bytes,
        required_input_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

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

ullm_status ullm_runtime_matvec_bf16_f32(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (matrix_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("BF16 matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("BF16 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(matrix_buffer, input_buffer) ||
        !buffers_share_backend(matrix_buffer, output_buffer)) {
        set_error("BF16 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("BF16 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("BF16 matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    if (matrix_elements > max_size / sizeof(uint16_t) ||
        cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float)) {
        set_error("BF16 matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_matrix_bytes = matrix_elements * sizeof(uint16_t);
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    if (matrix_buffer->bytes < required_matrix_bytes) {
        set_error("BF16 matvec matrix buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("BF16 matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("BF16 matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (matrix_buffer->backend == BackendKind::Cpu) {
        const auto *matrix = static_cast<const uint16_t *>(matrix_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        matvec_bf16_f32_host(matrix, input, rows, cols, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (matvec_bf16_f32_hip_kernel(
            matrix_buffer,
            input_buffer,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "BF16 matvec HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return matvec_bf16_f32_hip_staging(
        matrix_buffer,
        input_buffer,
        rows,
        cols,
        required_matrix_bytes,
        required_input_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_bf16_row_f32(
    const ullm_runtime_buffer *matrix_buffer,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (matrix_buffer == nullptr || output_buffer == nullptr) {
        set_error("BF16 row received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("BF16 row rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_index >= rows) {
        set_error("BF16 row index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(matrix_buffer, output_buffer)) {
        set_error("BF16 row buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("BF16 row stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("BF16 row matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    if (matrix_elements > max_size / sizeof(uint16_t) || cols > max_size / sizeof(float)) {
        set_error("BF16 row byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_matrix_bytes = matrix_elements * sizeof(uint16_t);
    const size_t required_output_bytes = cols * sizeof(float);
    if (matrix_buffer->bytes < required_matrix_bytes) {
        set_error("BF16 row matrix buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("BF16 row output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (matrix_buffer->backend == BackendKind::Cpu) {
        const auto *matrix = static_cast<const uint16_t *>(matrix_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        bf16_row_f32_host(matrix, rows, cols, row_index, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (bf16_row_f32_hip_kernel(
            matrix_buffer,
            rows,
            cols,
            row_index,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_BF16_ROW_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "BF16 row HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return bf16_row_f32_hip_staging(
        matrix_buffer,
        rows,
        cols,
        row_index,
        required_matrix_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_top1_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || partial_values_buffer == nullptr || partial_indices_buffer == nullptr) {
        set_error("f32 top1 received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0 || partial_count == 0) {
        set_error("f32 top1 elements and partial_count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    constexpr size_t block_size = 256;
    const size_t expected_partials = (elements + block_size - 1) / block_size;
    if (partial_count != expected_partials) {
        set_error("f32 top1 partial_count does not match expected block count");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, partial_values_buffer) ||
        !buffers_share_backend(input_buffer, partial_indices_buffer)) {
        set_error("f32 top1 buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(partial_values_buffer, stream) ||
        !stream_matches_buffer(partial_indices_buffer, stream)) {
        set_error("f32 top1 stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(uint32_t)) {
        set_error("f32 top1 byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        set_error("f32 top1 elements exceed uint32 index range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_bytes = elements * sizeof(float);
    const size_t partial_values_bytes = partial_count * sizeof(float);
    const size_t partial_indices_bytes = partial_count * sizeof(uint32_t);
    if (input_buffer->bytes < input_bytes) {
        set_error("f32 top1 input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_values_buffer->bytes < partial_values_bytes) {
        set_error("f32 top1 partial values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_indices_buffer->bytes < partial_indices_bytes) {
        set_error("f32 top1 partial indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        top1_f32_partials_host(
            static_cast<const float *>(input_buffer->ptr),
            elements,
            partial_count,
            static_cast<float *>(partial_values_buffer->ptr),
            static_cast<uint32_t *>(partial_indices_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (top1_f32_hip_kernel(
            input_buffer,
            elements,
            partial_values_buffer,
            partial_indices_buffer,
            partial_count,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_TOP1_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 top1 HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return top1_f32_hip_staging(
        input_buffer,
        elements,
        input_bytes,
        partial_values_buffer,
        partial_indices_buffer,
        partial_count,
        partial_values_bytes,
        partial_indices_bytes,
        stream);
}

ullm_status ullm_runtime_top1_pairs_f32(
    const ullm_runtime_buffer *values_buffer,
    const ullm_runtime_buffer *indices_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream) {
    if (values_buffer == nullptr || indices_buffer == nullptr || partial_values_buffer == nullptr ||
        partial_indices_buffer == nullptr) {
        set_error("f32 top1 pairs received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0 || partial_count == 0) {
        set_error("f32 top1 pairs elements and partial_count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    constexpr size_t block_size = 256;
    const size_t expected_partials = (elements + block_size - 1) / block_size;
    if (partial_count != expected_partials) {
        set_error("f32 top1 pairs partial_count does not match expected block count");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(values_buffer, indices_buffer) ||
        !buffers_share_backend(values_buffer, partial_values_buffer) ||
        !buffers_share_backend(values_buffer, partial_indices_buffer)) {
        set_error("f32 top1 pairs buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(partial_values_buffer, stream) ||
        !stream_matches_buffer(partial_indices_buffer, stream)) {
        set_error("f32 top1 pairs stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float) ||
        elements > max_size / sizeof(uint32_t) ||
        partial_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(uint32_t)) {
        set_error("f32 top1 pairs byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t values_bytes = elements * sizeof(float);
    const size_t indices_bytes = elements * sizeof(uint32_t);
    const size_t partial_values_bytes = partial_count * sizeof(float);
    const size_t partial_indices_bytes = partial_count * sizeof(uint32_t);
    if (values_buffer->bytes < values_bytes) {
        set_error("f32 top1 pairs values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (indices_buffer->bytes < indices_bytes) {
        set_error("f32 top1 pairs indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_values_buffer->bytes < partial_values_bytes) {
        set_error("f32 top1 pairs partial values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_indices_buffer->bytes < partial_indices_bytes) {
        set_error("f32 top1 pairs partial indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (values_buffer->backend == BackendKind::Cpu) {
        top1_pairs_f32_partials_host(
            static_cast<const float *>(values_buffer->ptr),
            static_cast<const uint32_t *>(indices_buffer->ptr),
            elements,
            partial_count,
            static_cast<float *>(partial_values_buffer->ptr),
            static_cast<uint32_t *>(partial_indices_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (top1_pairs_f32_hip_kernel(
            values_buffer,
            indices_buffer,
            elements,
            partial_values_buffer,
            partial_indices_buffer,
            partial_count,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_TOP1_PAIRS_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_TOP1_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 top1 pairs HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return top1_pairs_f32_hip_staging(
        values_buffer,
        indices_buffer,
        elements,
        values_bytes,
        indices_bytes,
        partial_values_buffer,
        partial_indices_buffer,
        partial_count,
        partial_values_bytes,
        partial_indices_bytes,
        stream);
}

ullm_status ullm_runtime_rmsnorm_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 RMSNorm received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 RMSNorm elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("f32 RMSNorm epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 RMSNorm buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 RMSNorm stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 RMSNorm byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (input_buffer->bytes < required_bytes) {
        set_error("f32 RMSNorm input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < required_bytes) {
        set_error("f32 RMSNorm weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 RMSNorm output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *weight = static_cast<const float *>(weight_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        rmsnorm_f32_host(input, weight, elements, epsilon, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (rmsnorm_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            elements,
            epsilon,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_RMSNORM_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 RMSNorm HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return rmsnorm_f32_hip_staging(
        input_buffer,
        weight_buffer,
        elements,
        epsilon,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_segmented_rmsnorm_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 segmented RMSNorm received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (segments == 0 || segment_size == 0) {
        set_error("f32 segmented RMSNorm segments and segment size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("f32 segmented RMSNorm epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 segmented RMSNorm buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 segmented RMSNorm stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (segment_size > max_size / segments ||
        segment_size > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = segments * segment_size;
    if (elements > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm element byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_output_bytes = elements * sizeof(float);
    const size_t weight_bytes = segment_size * sizeof(float);
    if (input_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < weight_bytes) {
        set_error("f32 segmented RMSNorm weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        segmented_rmsnorm_f32_host(
            static_cast<const float *>(input_buffer->ptr),
            static_cast<const float *>(weight_buffer->ptr),
            segments,
            segment_size,
            epsilon,
            static_cast<float *>(output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (segmented_rmsnorm_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            segments,
            segment_size,
            epsilon,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_RMSNORM_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 segmented RMSNorm HIP kernel is unavailable"
                                           : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return segmented_rmsnorm_f32_hip_staging(
        input_buffer,
        weight_buffer,
        segments,
        segment_size,
        epsilon,
        input_output_bytes,
        weight_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_segmented_rmsnorm_silu_mul_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || gate_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 segmented RMSNorm SiLU-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (segments == 0 || segment_size == 0) {
        set_error("f32 segmented RMSNorm SiLU-mul segments and segment size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("f32 segmented RMSNorm SiLU-mul epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, gate_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 segmented RMSNorm SiLU-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 segmented RMSNorm SiLU-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (segment_size > max_size / segments ||
        segment_size > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm SiLU-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = segments * segment_size;
    if (elements > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm SiLU-mul element byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_output_bytes = elements * sizeof(float);
    const size_t weight_bytes = segment_size * sizeof(float);
    if (input_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < weight_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        segmented_rmsnorm_silu_mul_f32_host(
            static_cast<const float *>(input_buffer->ptr),
            static_cast<const float *>(weight_buffer->ptr),
            static_cast<const float *>(gate_buffer->ptr),
            segments,
            segment_size,
            epsilon,
            static_cast<float *>(output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (segmented_rmsnorm_silu_mul_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            gate_buffer,
            segments,
            segment_size,
            epsilon,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 segmented RMSNorm SiLU-mul HIP kernel is unavailable"
                                           : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return segmented_rmsnorm_silu_mul_f32_hip_staging(
        input_buffer,
        weight_buffer,
        gate_buffer,
        segments,
        segment_size,
        epsilon,
        input_output_bytes,
        weight_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_silu_mul_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr || up_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 SiLU-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 SiLU-mul elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(gate_buffer, up_buffer) ||
        !buffers_share_backend(gate_buffer, output_buffer)) {
        set_error("f32 SiLU-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 SiLU-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 SiLU-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (gate_buffer->bytes < required_bytes) {
        set_error("f32 SiLU-mul gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (up_buffer->bytes < required_bytes) {
        set_error("f32 SiLU-mul up buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 SiLU-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (gate_buffer->backend == BackendKind::Cpu) {
        const auto *gate = static_cast<const float *>(gate_buffer->ptr);
        const auto *up = static_cast<const float *>(up_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        silu_mul_f32_host(gate, up, elements, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (silu_mul_f32_hip_kernel(
            gate_buffer,
            up_buffer,
            elements,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_SILU_MUL_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 SiLU-mul HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return silu_mul_f32_hip_staging(
        gate_buffer,
        up_buffer,
        elements,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_sigmoid_mul_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 Sigmoid-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 Sigmoid-mul elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(gate_buffer, input_buffer) ||
        !buffers_share_backend(gate_buffer, output_buffer)) {
        set_error("f32 Sigmoid-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 Sigmoid-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 Sigmoid-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (gate_buffer->bytes < required_bytes) {
        set_error("f32 Sigmoid-mul gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_bytes) {
        set_error("f32 Sigmoid-mul input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 Sigmoid-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (gate_buffer->backend == BackendKind::Cpu) {
        const auto *gate = static_cast<const float *>(gate_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        sigmoid_mul_f32_host(gate, input, elements, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (sigmoid_mul_f32_hip_kernel(
            gate_buffer,
            input_buffer,
            elements,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 Sigmoid-mul HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return sigmoid_mul_f32_hip_staging(
        gate_buffer,
        input_buffer,
        elements,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_qwen35_split_q_gate_f32(
    const ullm_runtime_buffer *projected_buffer,
    size_t q_heads,
    size_t head_dim,
    ullm_runtime_buffer *query_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_stream *stream) {
    if (projected_buffer == nullptr || query_output_buffer == nullptr || gate_output_buffer == nullptr) {
        set_error("Qwen3.5 q/gate split received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_heads == 0 || head_dim == 0) {
        set_error("Qwen3.5 q/gate split q_heads and head_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(projected_buffer, query_output_buffer) ||
        !buffers_share_backend(projected_buffer, gate_output_buffer)) {
        set_error("Qwen3.5 q/gate split buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(query_output_buffer, stream) ||
        !stream_matches_buffer(gate_output_buffer, stream)) {
        set_error("Qwen3.5 q/gate split stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (head_dim > max_size / q_heads) {
        set_error("Qwen3.5 q/gate split element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_heads * head_dim;
    if (output_elements > max_size / 2 || output_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/gate split byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t projected_elements = output_elements * 2;
    if (projected_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/gate split projected byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t projected_bytes = projected_elements * sizeof(float);
    if (projected_buffer->bytes < projected_bytes) {
        set_error("Qwen3.5 q/gate split projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (query_output_buffer->bytes < output_bytes) {
        set_error("Qwen3.5 q/gate split query output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_output_buffer->bytes < output_bytes) {
        set_error("Qwen3.5 q/gate split gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (projected_buffer->backend == BackendKind::Cpu) {
        qwen35_split_q_gate_f32_host(
            static_cast<const float *>(projected_buffer->ptr),
            q_heads,
            head_dim,
            static_cast<float *>(query_output_buffer->ptr),
            static_cast<float *>(gate_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (qwen35_split_q_gate_f32_hip_kernel(
            projected_buffer,
            q_heads,
            head_dim,
            query_output_buffer,
            gate_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "Qwen3.5 q/gate split HIP kernel is unavailable"
                                           : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return qwen35_split_q_gate_f32_hip_staging(
        projected_buffer,
        q_heads,
        head_dim,
        projected_bytes,
        output_bytes,
        query_output_buffer,
        gate_output_buffer,
        stream);
}

ullm_status ullm_runtime_qwen35_qk_norm_rope_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream) {
    if (q_projected_buffer == nullptr || k_projected_buffer == nullptr ||
        q_weight_buffer == nullptr || k_weight_buffer == nullptr ||
        q_gate_output_buffer == nullptr || q_rope_output_buffer == nullptr ||
        k_rope_output_buffer == nullptr) {
        set_error("Qwen3.5 q/k norm RoPE received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_heads == 0 || kv_heads == 0 || head_dim == 0) {
        set_error("Qwen3.5 q/k norm RoPE heads and head_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        set_error("Qwen3.5 q/k norm RoPE rotary_dim must be even and no greater than head_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(rope_base) || rope_base <= 1.0f) {
        set_error("Qwen3.5 q/k norm RoPE base must be finite and greater than one");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("Qwen3.5 q/k norm RoPE epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_projected_buffer, k_projected_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_gate_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_rope_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_rope_output_buffer)) {
        set_error("Qwen3.5 q/k norm RoPE buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(q_gate_output_buffer, stream) ||
        !stream_matches_buffer(q_rope_output_buffer, stream) ||
        !stream_matches_buffer(k_rope_output_buffer, stream)) {
        set_error("Qwen3.5 q/k norm RoPE stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (head_dim > max_size / q_heads || head_dim > max_size / kv_heads) {
        set_error("Qwen3.5 q/k norm RoPE element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_output_elements = q_heads * head_dim;
    const size_t k_output_elements = kv_heads * head_dim;
    if (q_output_elements > max_size / 2 ||
        q_output_elements > max_size / sizeof(float) ||
        k_output_elements > max_size / sizeof(float) ||
        head_dim > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_elements = q_output_elements * 2;
    if (q_projected_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE q projected byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_bytes = q_projected_elements * sizeof(float);
    const size_t q_output_bytes = q_output_elements * sizeof(float);
    const size_t k_projected_bytes = k_output_elements * sizeof(float);
    const size_t k_output_bytes = k_output_elements * sizeof(float);
    const size_t weight_bytes = head_dim * sizeof(float);
    if (q_projected_buffer->bytes < q_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_projected_buffer->bytes < k_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE k projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE k weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_gate_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_rope_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_rope_output_buffer->bytes < k_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE k RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_projected_buffer->backend == BackendKind::Cpu) {
        qwen35_qk_norm_rope_f32_host(
            static_cast<const float *>(q_projected_buffer->ptr),
            static_cast<const float *>(k_projected_buffer->ptr),
            static_cast<const float *>(q_weight_buffer->ptr),
            static_cast<const float *>(k_weight_buffer->ptr),
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            static_cast<float *>(q_gate_output_buffer->ptr),
            static_cast<float *>(q_rope_output_buffer->ptr),
            static_cast<float *>(k_rope_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (qwen35_qk_norm_rope_f32_hip_kernel(
            q_projected_buffer,
            k_projected_buffer,
            q_weight_buffer,
            k_weight_buffer,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            q_gate_output_buffer,
            q_rope_output_buffer,
            k_rope_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty()
                      ? "Qwen3.5 q/k norm RoPE HIP kernel is unavailable"
                      : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return qwen35_qk_norm_rope_f32_hip_staging(
        q_projected_buffer,
        k_projected_buffer,
        q_weight_buffer,
        k_weight_buffer,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        q_projected_bytes,
        k_projected_bytes,
        weight_bytes,
        q_output_bytes,
        k_output_bytes,
        q_gate_output_buffer,
        q_rope_output_buffer,
        k_rope_output_buffer,
        stream);
}

ullm_status ullm_runtime_qwen35_qk_norm_rope_batch_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t sequence_len,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream) {
    if (q_projected_buffer == nullptr || k_projected_buffer == nullptr ||
        q_weight_buffer == nullptr || k_weight_buffer == nullptr ||
        q_gate_output_buffer == nullptr || q_rope_output_buffer == nullptr ||
        k_rope_output_buffer == nullptr) {
        set_error("Qwen3.5 q/k norm RoPE batch received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_heads == 0 || kv_heads == 0 || sequence_len == 0 || head_dim == 0) {
        set_error("Qwen3.5 q/k norm RoPE batch heads, sequence_len, and head_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        set_error(
            "Qwen3.5 q/k norm RoPE batch rotary_dim must be even and no greater than head_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(rope_base) || rope_base <= 1.0f) {
        set_error("Qwen3.5 q/k norm RoPE batch base must be finite and greater than one");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("Qwen3.5 q/k norm RoPE batch epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_projected_buffer, k_projected_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_gate_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_rope_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_rope_output_buffer)) {
        set_error("Qwen3.5 q/k norm RoPE batch buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(q_gate_output_buffer, stream) ||
        !stream_matches_buffer(q_rope_output_buffer, stream) ||
        !stream_matches_buffer(k_rope_output_buffer, stream)) {
        set_error("Qwen3.5 q/k norm RoPE batch stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (head_dim > max_size / q_heads || head_dim > max_size / kv_heads) {
        set_error("Qwen3.5 q/k norm RoPE batch per-token element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_output_elements_per_token = q_heads * head_dim;
    const size_t k_output_elements_per_token = kv_heads * head_dim;
    if (sequence_len > max_size / q_output_elements_per_token ||
        sequence_len > max_size / k_output_elements_per_token ||
        q_output_elements_per_token > max_size / 2 ||
        head_dim > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE batch element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_output_elements = sequence_len * q_output_elements_per_token;
    const size_t k_output_elements = sequence_len * k_output_elements_per_token;
    if (q_output_elements > max_size / 2) {
        set_error("Qwen3.5 q/k norm RoPE batch q projected element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_elements = q_output_elements * 2;
    if (q_projected_elements > max_size / sizeof(float) ||
        q_output_elements > max_size / sizeof(float) ||
        k_output_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_bytes = q_projected_elements * sizeof(float);
    const size_t q_output_bytes = q_output_elements * sizeof(float);
    const size_t k_projected_bytes = k_output_elements * sizeof(float);
    const size_t k_output_bytes = k_output_elements * sizeof(float);
    const size_t weight_bytes = head_dim * sizeof(float);
    if (q_projected_buffer->bytes < q_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE batch q projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_projected_buffer->bytes < k_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE batch k projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE batch q weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE batch k weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_gate_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE batch q gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_rope_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE batch q RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_rope_output_buffer->bytes < k_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE batch k RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_projected_buffer->backend == BackendKind::Cpu) {
        qwen35_qk_norm_rope_batch_f32_host(
            static_cast<const float *>(q_projected_buffer->ptr),
            static_cast<const float *>(k_projected_buffer->ptr),
            static_cast<const float *>(q_weight_buffer->ptr),
            static_cast<const float *>(k_weight_buffer->ptr),
            q_heads,
            kv_heads,
            sequence_len,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            static_cast<float *>(q_gate_output_buffer->ptr),
            static_cast<float *>(q_rope_output_buffer->ptr),
            static_cast<float *>(k_rope_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (qwen35_qk_norm_rope_batch_f32_hip_kernel(
            q_projected_buffer,
            k_projected_buffer,
            q_weight_buffer,
            k_weight_buffer,
            q_heads,
            kv_heads,
            sequence_len,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            q_gate_output_buffer,
            q_rope_output_buffer,
            k_rope_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty()
                      ? "Qwen3.5 q/k norm RoPE batch HIP kernel is unavailable"
                      : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return qwen35_qk_norm_rope_batch_f32_hip_staging(
        q_projected_buffer,
        k_projected_buffer,
        q_weight_buffer,
        k_weight_buffer,
        q_heads,
        kv_heads,
        sequence_len,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        q_projected_bytes,
        k_projected_bytes,
        weight_bytes,
        q_output_bytes,
        k_output_bytes,
        q_gate_output_buffer,
        q_rope_output_buffer,
        k_rope_output_buffer,
        stream);
}

ullm_status ullm_runtime_qwen35_qk_norm_rope_paged_kv_write_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *v_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    if (q_projected_buffer == nullptr || k_projected_buffer == nullptr ||
        v_projected_buffer == nullptr || q_weight_buffer == nullptr ||
        k_weight_buffer == nullptr || block_table_buffer == nullptr ||
        q_gate_output_buffer == nullptr || q_rope_output_buffer == nullptr ||
        k_cache_buffer == nullptr || v_cache_buffer == nullptr) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 ||
        block_size == 0 || cache_blocks == 0) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write rotary_dim must be even and no greater than head_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(rope_base) || rope_base <= 1.0f) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write base must be finite and greater than one");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_projected_buffer, k_projected_buffer) ||
        !buffers_share_backend(q_projected_buffer, v_projected_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, block_table_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_gate_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_rope_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_projected_buffer, v_cache_buffer)) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(q_gate_output_buffer, stream) ||
        !stream_matches_buffer(q_rope_output_buffer, stream) ||
        !stream_matches_buffer(k_cache_buffer, stream) ||
        !stream_matches_buffer(v_cache_buffer, stream)) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (cache_position >= physical_tokens) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write cache_position exceeds physical cache capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t block_index = cache_position / block_size;
    const size_t block_table_entries = block_index + 1;
    if (head_dim > max_size / q_heads || head_dim > max_size / kv_heads ||
        value_dim > max_size / kv_heads) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write token element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_output_elements = q_heads * head_dim;
    const size_t k_elements = kv_heads * head_dim;
    const size_t v_elements = kv_heads * value_dim;
    if (kv_heads > max_size / physical_tokens) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write kv head-cache count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = physical_tokens * kv_heads;
    if (head_dim > max_size / kv_head_cache || value_dim > max_size / kv_head_cache) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_cache_elements = kv_head_cache * head_dim;
    const size_t v_cache_elements = kv_head_cache * value_dim;
    if (q_output_elements > max_size / 2 ||
        q_output_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        head_dim > max_size / sizeof(float) ||
        block_table_entries > max_size / sizeof(std::uint32_t) ||
        k_cache_elements > max_size / sizeof(float) ||
        v_cache_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_elements = q_output_elements * 2;
    if (q_projected_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q projected byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_bytes = q_projected_elements * sizeof(float);
    const size_t k_projected_bytes = k_elements * sizeof(float);
    const size_t v_projected_bytes = v_elements * sizeof(float);
    const size_t weight_bytes = head_dim * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);
    const size_t q_output_bytes = q_output_elements * sizeof(float);
    const size_t k_cache_bytes = k_cache_elements * sizeof(float);
    const size_t v_cache_bytes = v_cache_elements * sizeof(float);
    if (q_projected_buffer->bytes < q_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_projected_buffer->bytes < k_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write k projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_projected_buffer->bytes < v_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write v projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write k weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_table_buffer->bytes < block_table_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write block table buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_gate_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_rope_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_cache_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_cache_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_projected_buffer->backend == BackendKind::Cpu) {
        const auto *block_table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        if (static_cast<size_t>(block_table[block_index]) >= cache_blocks) {
            set_error(
                "Qwen3.5 q/k norm RoPE paged KV write block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        qwen35_qk_norm_rope_paged_kv_write_f32_host(
            static_cast<const float *>(q_projected_buffer->ptr),
            static_cast<const float *>(k_projected_buffer->ptr),
            static_cast<const float *>(v_projected_buffer->ptr),
            static_cast<const float *>(q_weight_buffer->ptr),
            static_cast<const float *>(k_weight_buffer->ptr),
            block_table,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            cache_position,
            block_size,
            static_cast<float *>(q_gate_output_buffer->ptr),
            static_cast<float *>(q_rope_output_buffer->ptr),
            static_cast<float *>(k_cache_buffer->ptr),
            static_cast<float *>(v_cache_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (qwen35_qk_norm_rope_paged_kv_write_f32_hip_kernel(
            q_projected_buffer,
            k_projected_buffer,
            v_projected_buffer,
            q_weight_buffer,
            k_weight_buffer,
            block_table_buffer,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            cache_position,
            block_size,
            cache_blocks,
            q_gate_output_buffer,
            q_rope_output_buffer,
            k_cache_buffer,
            v_cache_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty()
                      ? "Qwen3.5 q/k norm RoPE paged KV write HIP kernel is unavailable"
                      : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return qwen35_qk_norm_rope_paged_kv_write_f32_hip_staging(
        q_projected_buffer,
        k_projected_buffer,
        v_projected_buffer,
        q_weight_buffer,
        k_weight_buffer,
        block_table_buffer,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        cache_position,
        block_size,
        q_projected_bytes,
        k_projected_bytes,
        v_projected_bytes,
        weight_bytes,
        block_table_bytes,
        q_output_bytes,
        k_cache_bytes,
        v_cache_bytes,
        q_gate_output_buffer,
        q_rope_output_buffer,
        k_cache_buffer,
        v_cache_buffer,
        stream);
}

ullm_status ullm_runtime_add_f32(
    const ullm_runtime_buffer *lhs_buffer,
    const ullm_runtime_buffer *rhs_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (lhs_buffer == nullptr || rhs_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 add received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 add elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(lhs_buffer, rhs_buffer) ||
        !buffers_share_backend(lhs_buffer, output_buffer)) {
        set_error("f32 add buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 add stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 add byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (lhs_buffer->bytes < required_bytes) {
        set_error("f32 add lhs buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rhs_buffer->bytes < required_bytes) {
        set_error("f32 add rhs buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 add output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (lhs_buffer->backend == BackendKind::Cpu) {
        const auto *lhs = static_cast<const float *>(lhs_buffer->ptr);
        const auto *rhs = static_cast<const float *>(rhs_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        add_f32_host(lhs, rhs, elements, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (add_f32_hip_kernel(
            lhs_buffer,
            rhs_buffer,
            elements,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_ADD_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 add HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return add_f32_hip_staging(
        lhs_buffer,
        rhs_buffer,
        elements,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_rope_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 RoPE received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (sequence_len == 0 || heads == 0 || head_dim == 0 || rotary_dim == 0) {
        set_error("f32 RoPE sequence_len, heads, head_dim, and rotary_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        set_error("f32 RoPE rotary_dim must be even and no greater than head_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(rope_base) || rope_base <= 1.0f) {
        set_error("f32 RoPE base must be finite and greater than one");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 RoPE buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 RoPE stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (heads > max_size / sequence_len) {
        set_error("f32 RoPE head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t head_sequence = heads * sequence_len;
    if (head_dim > max_size / head_sequence) {
        set_error("f32 RoPE element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = head_sequence * head_dim;
    if (elements > max_size / sizeof(float)) {
        set_error("f32 RoPE byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (input_buffer->bytes < required_bytes) {
        set_error("f32 RoPE input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        rope_f32_host(
            input,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (rope_f32_hip_kernel(
            input_buffer,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_ROPE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 RoPE HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return rope_f32_hip_staging(
        input_buffer,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_depthwise_conv1d_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 depthwise conv1d received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (channels == 0 || sequence_len == 0 || kernel_size == 0) {
        set_error("f32 depthwise conv1d channels, sequence length, and kernel size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 depthwise conv1d buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 depthwise conv1d stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (sequence_len > max_size / channels) {
        set_error("f32 depthwise conv1d input element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (kernel_size > max_size / channels) {
        set_error("f32 depthwise conv1d weight element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_elements = channels * sequence_len;
    const size_t weight_elements = channels * kernel_size;
    if (input_elements > max_size / sizeof(float) ||
        weight_elements > max_size / sizeof(float)) {
        set_error("f32 depthwise conv1d byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = input_elements * sizeof(float);
    const size_t required_weight_bytes = weight_elements * sizeof(float);
    const size_t required_output_bytes = required_input_bytes;
    if (input_buffer->bytes < required_input_bytes) {
        set_error("f32 depthwise conv1d input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < required_weight_bytes) {
        set_error("f32 depthwise conv1d weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("f32 depthwise conv1d output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *weight = static_cast<const float *>(weight_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        depthwise_conv1d_f32_host(
            input,
            weight,
            channels,
            sequence_len,
            kernel_size,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (depthwise_conv1d_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            channels,
            sequence_len,
            kernel_size,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_DEPTHWISE_CONV1D_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 depthwise conv1d HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return depthwise_conv1d_f32_hip_staging(
        input_buffer,
        weight_buffer,
        channels,
        sequence_len,
        kernel_size,
        required_input_bytes,
        required_weight_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}
