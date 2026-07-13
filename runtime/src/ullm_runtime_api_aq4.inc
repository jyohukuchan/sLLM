ullm_status ullm_runtime_aq4_dequant_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        output_buffer == nullptr || (scale_count > 0 && scale_values == nullptr)) {
        set_error("AQ4 dequant received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 dequant group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 dequant scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer)) {
        set_error("AQ4 dequant buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 dequant stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements > (static_cast<size_t>(-1) / sizeof(float))) {
        set_error("AQ4 dequant output byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    const size_t required_output_bytes = elements * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 dequant index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 dequant scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 dequant output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 dequant codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 dequant requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_dequant_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                elements,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    const HipAq4LaunchResult launch_result = aq4_dequant_hip_kernel(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values,
        scale_count,
        group_size,
        tensor_scale,
        elements,
        required_output_bytes,
        output_buffer,
        stream,
        &hip_kernel_error);
    if (launch_result == HipAq4LaunchResult::Ok) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (launch_result == HipAq4LaunchResult::InvalidArgument) {
        set_error(hip_kernel_error.c_str());
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "AQ4 HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_dequant_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values,
        scale_count,
        group_size,
        tensor_scale,
        elements,
        required_index_bytes,
        groups,
        required_output_bytes,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_row_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 row received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 row scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 row group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 row rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_index >= rows) {
        set_error("AQ4 row index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 row tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 row buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 row stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 row matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 row byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_output_bytes = cols * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 row index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 row scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 row scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 row scale override buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 row output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 row codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 row requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_row_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                row_scales,
                scale_count,
                group_size,
                tensor_scale,
                row_scale_count,
                rows,
                cols,
                row_index,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    const HipAq4LaunchResult launch_result = aq4_row_f32_hip_kernel(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        row_index,
        output_buffer,
        stream,
        &hip_kernel_error);
    if (launch_result == HipAq4LaunchResult::Ok) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (launch_result == HipAq4LaunchResult::InvalidArgument) {
        set_error(hip_kernel_error.c_str());
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_ROW_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "AQ4 row HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(cols);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = index_buffer->hip_device_id;
    if (!hip_runtime().copy_async(
            host_indices.data(),
            index_buffer->ptr,
            required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_indices.data(),
            scale_buffer->ptr,
            groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_codebook.data(),
            codebook_buffer->ptr,
            codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            required_scale_value_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && !hip_runtime().copy_async(
            host_row_scales.data(),
            row_scale_buffer->ptr,
            required_row_scale_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id))) {
        set_error("failed to copy AQ4 row HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 row HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_row_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            row_index,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 row output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 row HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_aq4_matvec_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_matvec_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input,
                row_scales,
                row_scale_count,
                rows,
                cols,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "AQ4 matvec HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_f32_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        input_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        required_index_bytes,
        groups,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_batch_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec batch received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec batch scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec batch group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0 || batch_count == 0) {
        set_error("AQ4 matvec batch rows, cols, and batch count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec batch tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec batch buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec batch stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec batch matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / batch_count ||
        rows > max_size / batch_count ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_elements = cols * batch_count;
    const size_t output_elements = rows * batch_count;
    if (input_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("AQ4 matvec batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = input_elements * sizeof(float);
    const size_t required_output_bytes = output_elements * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec batch index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec batch scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec batch scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec batch input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec batch row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec batch output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec batch codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec batch requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    // CPU buffers are directly inspectable, so validate every scale index before touching output.
    // HIP metadata validation remains a load-time/package responsibility: synchronizing a D2H
    // copy here would add a per-call barrier to every production projection.
    if (index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_scale_indices_host_are_valid(
                static_cast<const std::uint8_t *>(scale_buffer->ptr),
                groups,
                scale_count)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_matvec_batch_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input,
                row_scales,
                row_scale_count,
                rows,
                cols,
                batch_count,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    const Aq4MatvecBatchDispatchKind dispatch_kind =
        aq4_matvec_batch_dispatch_kind_for_device(
            index_buffer->hip_device_id,
            group_size,
            rows,
            cols,
            batch_count);
    const unsigned int register_batch_tile =
        dispatch_kind == Aq4MatvecBatchDispatchKind::RegisterBm4 ? kAq4RegisterBm4 :
        dispatch_kind == Aq4MatvecBatchDispatchKind::RegisterBm8 ? kAq4RegisterBm8 : 0u;
    if (register_batch_tile != 0u && aq4_matvec_batch_f32_register_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            batch_count,
            register_batch_tile,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (register_batch_tile != 0u && !hip_kernel_error.empty()) {
        hip_kernel_error.append("\n");
    }
    if (dispatch_kind == Aq4MatvecBatchDispatchKind::TiledLds &&
        aq4_matvec_batch_f32_tiled_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            batch_count,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (dispatch_kind == Aq4MatvecBatchDispatchKind::TiledLds && !hip_kernel_error.empty()) {
        hip_kernel_error.append("\n");
    }
    if (aq4_matvec_batch_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            batch_count,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec batch HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_batch_f32_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        input_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        batch_count,
        required_index_bytes,
        groups,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_batch_register_bm8_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 register BM8 batch received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0u) {
        set_error("AQ4 register BM8 batch scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size != 16u) {
        set_error("AQ4 register BM8 batch requires group size 16");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0u || cols == 0u || batch_count < kAq4RegisterBm8) {
        set_error("AQ4 register BM8 batch requires nonzero rows/cols and batch count at least 8");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows % kAq4RegisterRowsPerBlock != 0u || cols % kAq4TiledBk != 0u) {
        set_error("AQ4 register BM8 batch requires rows divisible by 32 and cols divisible by 128");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 register BM8 batch tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 register BM8 batch buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 register BM8 batch stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (index_buffer->backend != BackendKind::Hip) {
        set_error("AQ4 register BM8 batch requires a HIP gfx1201 backend");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!aq4_matvec_batch_register_shape_is_supported(
            hip_runtime().device_arch_name(index_buffer->hip_device_id),
            group_size,
            rows,
            cols,
            batch_count,
            kAq4RegisterBm8)) {
        set_error("AQ4 register BM8 batch requires exact gfx1201 and supported checked geometry");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t grid_x = (rows + kAq4RegisterRowsPerBlock - 1u) /
        kAq4RegisterRowsPerBlock;
    const size_t grid_y = (batch_count + kAq4RegisterBm8 - 1u) / kAq4RegisterBm8;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        set_error("AQ4 register BM8 batch dimensions exceed HIP grid limit");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 register BM8 batch matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2u;
    const size_t groups = elements / group_size;
    if (cols > max_size / batch_count || rows > max_size / batch_count ||
        scale_count > max_size / sizeof(float) || row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 register BM8 batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_elements = cols * batch_count;
    const size_t output_elements = rows * batch_count;
    if (input_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("AQ4 register BM8 batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = input_elements * sizeof(float);
    const size_t required_output_bytes = output_elements * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 register BM8 batch index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 register BM8 batch scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 register BM8 batch scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 register BM8 batch input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 register BM8 batch row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 register BM8 batch output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0u ||
        codebook_buffer->bytes / sizeof(float) < 16u) {
        set_error("AQ4 register BM8 batch requires at least 16 f32 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    std::string hip_kernel_error;
    if (!aq4_matvec_batch_f32_register_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            batch_count,
            kAq4RegisterBm8,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 register BM8 HIP kernel launch failed" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_aq4_matvec_batch_dispatch_kind ullm_runtime_aq4_matvec_batch_dispatch_kind_for_shape(
    uint32_t device_index,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    if (device_index == 0u) {
        return ULLM_AQ4_MATVEC_BATCH_DISPATCH_LEGACY;
    }
    const Aq4MatvecBatchDispatchKind kind = aq4_matvec_batch_dispatch_kind_for_device(
        static_cast<int>(device_index - 1u),
        group_size,
        rows,
        cols,
        batch_count);
    switch (kind) {
    case Aq4MatvecBatchDispatchKind::TiledLds:
        return ULLM_AQ4_MATVEC_BATCH_DISPATCH_TILED_LDS_BM8;
    case Aq4MatvecBatchDispatchKind::RegisterBm4:
        return ULLM_AQ4_MATVEC_BATCH_DISPATCH_REGISTER_BM4;
    case Aq4MatvecBatchDispatchKind::RegisterBm8:
        return ULLM_AQ4_MATVEC_BATCH_DISPATCH_REGISTER_BM8;
    case Aq4MatvecBatchDispatchKind::Legacy:
    default:
        return ULLM_AQ4_MATVEC_BATCH_DISPATCH_LEGACY;
    }
}

ullm_status ullm_runtime_aq4_matvec_top1_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr ||
        partial_values_buffer == nullptr || partial_indices_buffer == nullptr) {
        set_error("AQ4 matvec top1 received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec top1 scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec top1 group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0 || partial_count == 0) {
        set_error("AQ4 matvec top1 rows, cols, and partial_count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        set_error("AQ4 matvec top1 rows exceed uint32 index range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t expected_partials = aq4_matvec_top1_partial_count(rows);
    if (partial_count != expected_partials) {
        set_error("AQ4 matvec top1 partial_count does not match expected row block count");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec top1 tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, partial_values_buffer) ||
        !buffers_share_backend(index_buffer, partial_indices_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec top1 buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(partial_values_buffer, stream) ||
        !stream_matches_buffer(partial_indices_buffer, stream)) {
        set_error("AQ4 matvec top1 stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec top1 matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(uint32_t)) {
        set_error("AQ4 matvec top1 byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    const size_t partial_values_bytes = partial_count * sizeof(float);
    const size_t partial_indices_bytes = partial_count * sizeof(uint32_t);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec top1 index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec top1 scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec top1 scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec top1 input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec top1 row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_values_buffer->bytes < partial_values_bytes) {
        set_error("AQ4 matvec top1 partial values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_indices_buffer->bytes < partial_indices_bytes) {
        set_error("AQ4 matvec top1 partial indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec top1 codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec top1 requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_top1_f32_host(
                static_cast<const std::uint8_t *>(index_buffer->ptr),
                static_cast<const std::uint8_t *>(scale_buffer->ptr),
                static_cast<const float *>(codebook_buffer->ptr),
                static_cast<const float *>(scale_values_buffer->ptr),
                scale_count,
                group_size,
                tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(row_scale_buffer->ptr),
                row_scale_count,
                rows,
                cols,
                partial_count,
                static_cast<float *>(partial_values_buffer->ptr),
                static_cast<uint32_t *>(partial_indices_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_top1_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            partial_values_buffer,
            partial_indices_buffer,
            partial_count,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_TOP1_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec top1 HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_partial_values(partial_count);
    std::vector<uint32_t> host_partial_indices(partial_count);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_indices.data(),
            index_buffer->ptr,
            required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_indices.data(),
            scale_buffer->ptr,
            groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_codebook.data(),
            codebook_buffer->ptr,
            codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            required_scale_value_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             required_row_scale_bytes,
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec top1 HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec top1 HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_top1_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            partial_count,
            host_partial_values.data(),
            host_partial_indices.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            partial_values_buffer->ptr,
            host_partial_values.data(),
            partial_values_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            partial_indices_buffer->ptr,
            host_partial_indices.data(),
            partial_indices_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec top1 outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec top1 HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_aq4_matvec_add_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr ||
        residual_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec add received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec add scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec add group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 matvec add rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec add tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, residual_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec add buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec add stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec add matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec add byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec add index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec add scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec add scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec add input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (residual_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec add residual buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec add row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec add output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec add codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec add requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *residual = static_cast<const float *>(residual_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_matvec_add_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input,
                residual,
                row_scales,
                row_scale_count,
                rows,
                cols,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_add_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            residual_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec add HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_add_f32_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        input_buffer,
        residual_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        required_index_bytes,
        groups,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_pair_f32(
    const ullm_runtime_buffer *left_index_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    const ullm_runtime_buffer *left_codebook_buffer,
    const ullm_runtime_buffer *left_scale_values_buffer,
    const ullm_runtime_buffer *left_row_scale_buffer,
    size_t left_scale_count,
    size_t left_group_size,
    float left_tensor_scale,
    size_t left_row_scale_count,
    const ullm_runtime_buffer *right_index_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    const ullm_runtime_buffer *right_codebook_buffer,
    const ullm_runtime_buffer *right_scale_values_buffer,
    const ullm_runtime_buffer *right_row_scale_buffer,
    size_t right_scale_count,
    size_t right_group_size,
    float right_tensor_scale,
    size_t right_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream) {
    if (left_index_buffer == nullptr || left_scale_buffer == nullptr ||
        left_codebook_buffer == nullptr || left_scale_values_buffer == nullptr ||
        right_index_buffer == nullptr || right_scale_buffer == nullptr ||
        right_codebook_buffer == nullptr || right_scale_values_buffer == nullptr ||
        input_buffer == nullptr || left_output_buffer == nullptr || right_output_buffer == nullptr) {
        set_error("AQ4 matvec pair received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_scale_count == 0 || right_scale_count == 0) {
        set_error("AQ4 matvec pair scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_group_size == 0 || right_group_size == 0) {
        set_error("AQ4 matvec pair group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_rows == 0 || right_rows == 0 || cols == 0) {
        set_error("AQ4 matvec pair rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(left_tensor_scale) || left_tensor_scale <= 0.0f ||
        !std::isfinite(right_tensor_scale) || right_tensor_scale <= 0.0f) {
        set_error("AQ4 matvec pair tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(left_index_buffer, left_scale_buffer) ||
        !buffers_share_backend(left_index_buffer, left_codebook_buffer) ||
        !buffers_share_backend(left_index_buffer, left_scale_values_buffer) ||
        !buffers_share_backend(left_index_buffer, right_index_buffer) ||
        !buffers_share_backend(left_index_buffer, right_scale_buffer) ||
        !buffers_share_backend(left_index_buffer, right_codebook_buffer) ||
        !buffers_share_backend(left_index_buffer, right_scale_values_buffer) ||
        !buffers_share_backend(left_index_buffer, input_buffer) ||
        !buffers_share_backend(left_index_buffer, left_output_buffer) ||
        !buffers_share_backend(left_index_buffer, right_output_buffer) ||
        (left_row_scale_buffer != nullptr &&
         !buffers_share_backend(left_index_buffer, left_row_scale_buffer)) ||
        (right_row_scale_buffer != nullptr &&
         !buffers_share_backend(left_index_buffer, right_row_scale_buffer))) {
        set_error("AQ4 matvec pair buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(left_output_buffer, stream) ||
        !stream_matches_buffer(right_output_buffer, stream)) {
        set_error("AQ4 matvec pair stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / left_rows || cols > max_size / right_rows ||
        left_rows > max_size - right_rows) {
        set_error("AQ4 matvec pair matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t left_elements = left_rows * cols;
    const size_t right_elements = right_rows * cols;
    const size_t left_required_index_bytes = left_elements / 2 + (left_elements % 2);
    const size_t right_required_index_bytes = right_elements / 2 + (right_elements % 2);
    const size_t left_groups =
        left_elements / left_group_size + (left_elements % left_group_size == 0 ? 0 : 1);
    const size_t right_groups =
        right_elements / right_group_size + (right_elements % right_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        left_rows > max_size / sizeof(float) ||
        right_rows > max_size / sizeof(float) ||
        left_scale_count > max_size / sizeof(float) ||
        right_scale_count > max_size / sizeof(float) ||
        left_row_scale_count > max_size / sizeof(float) ||
        right_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec pair byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t left_required_output_bytes = left_rows * sizeof(float);
    const size_t right_required_output_bytes = right_rows * sizeof(float);
    const size_t left_required_scale_value_bytes = left_scale_count * sizeof(float);
    const size_t right_required_scale_value_bytes = right_scale_count * sizeof(float);
    const size_t left_required_row_scale_bytes = left_row_scale_count * sizeof(float);
    const size_t right_required_row_scale_bytes = right_row_scale_count * sizeof(float);
    if (left_index_buffer->bytes < left_required_index_bytes ||
        right_index_buffer->bytes < right_required_index_bytes) {
        set_error("AQ4 matvec pair index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_scale_buffer->bytes < left_groups || right_scale_buffer->bytes < right_groups) {
        set_error("AQ4 matvec pair scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_scale_values_buffer->bytes < left_required_scale_value_bytes ||
        right_scale_values_buffer->bytes < right_required_scale_value_bytes) {
        set_error("AQ4 matvec pair scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec pair input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((left_row_scale_buffer != nullptr &&
         left_row_scale_buffer->bytes < left_required_row_scale_bytes) ||
        (right_row_scale_buffer != nullptr &&
         right_row_scale_buffer->bytes < right_required_row_scale_bytes)) {
        set_error("AQ4 matvec pair row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_output_buffer->bytes < left_required_output_bytes ||
        right_output_buffer->bytes < right_required_output_bytes) {
        set_error("AQ4 matvec pair output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_codebook_buffer->bytes % sizeof(float) != 0 ||
        right_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec pair codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t left_codebook_entries = left_codebook_buffer->bytes / sizeof(float);
    const size_t right_codebook_entries = right_codebook_buffer->bytes / sizeof(float);
    if (left_codebook_entries < 16 || right_codebook_entries < 16) {
        set_error("AQ4 matvec pair requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (left_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(left_index_buffer->ptr),
                static_cast<const std::uint8_t *>(left_scale_buffer->ptr),
                static_cast<const float *>(left_codebook_buffer->ptr),
                static_cast<const float *>(left_scale_values_buffer->ptr),
                left_scale_count,
                left_group_size,
                left_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                left_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(left_row_scale_buffer->ptr),
                left_row_scale_count,
                left_rows,
                cols,
                static_cast<float *>(left_output_buffer->ptr)) ||
            !aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(right_index_buffer->ptr),
                static_cast<const std::uint8_t *>(right_scale_buffer->ptr),
                static_cast<const float *>(right_codebook_buffer->ptr),
                static_cast<const float *>(right_scale_values_buffer->ptr),
                right_scale_count,
                right_group_size,
                right_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                right_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(right_row_scale_buffer->ptr),
                right_row_scale_count,
                right_rows,
                cols,
                static_cast<float *>(right_output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_pair_f32_hip_kernel(
            left_index_buffer,
            left_scale_buffer,
            left_codebook_buffer,
            left_scale_values_buffer,
            left_row_scale_buffer,
            left_scale_count,
            left_group_size,
            left_tensor_scale,
            left_row_scale_count,
            right_index_buffer,
            right_scale_buffer,
            right_codebook_buffer,
            right_scale_values_buffer,
            right_row_scale_buffer,
            right_scale_count,
            right_group_size,
            right_tensor_scale,
            right_row_scale_count,
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

    set_error(
        hip_kernel_error.empty() ? "AQ4 matvec pair HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_aq4_matvec_triple_f32(
    const ullm_runtime_buffer *first_index_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    const ullm_runtime_buffer *first_codebook_buffer,
    const ullm_runtime_buffer *first_scale_values_buffer,
    const ullm_runtime_buffer *first_row_scale_buffer,
    size_t first_scale_count,
    size_t first_group_size,
    float first_tensor_scale,
    size_t first_row_scale_count,
    const ullm_runtime_buffer *second_index_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    const ullm_runtime_buffer *second_codebook_buffer,
    const ullm_runtime_buffer *second_scale_values_buffer,
    const ullm_runtime_buffer *second_row_scale_buffer,
    size_t second_scale_count,
    size_t second_group_size,
    float second_tensor_scale,
    size_t second_row_scale_count,
    const ullm_runtime_buffer *third_index_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    const ullm_runtime_buffer *third_codebook_buffer,
    const ullm_runtime_buffer *third_scale_values_buffer,
    const ullm_runtime_buffer *third_row_scale_buffer,
    size_t third_scale_count,
    size_t third_group_size,
    float third_tensor_scale,
    size_t third_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream) {
    struct Matrix {
        const ullm_runtime_buffer *index = nullptr;
        const ullm_runtime_buffer *scale = nullptr;
        const ullm_runtime_buffer *codebook = nullptr;
        const ullm_runtime_buffer *scale_values = nullptr;
        const ullm_runtime_buffer *row_scale = nullptr;
        size_t scale_count = 0;
        size_t group_size = 0;
        float tensor_scale = 0.0f;
        size_t row_scale_count = 0;
        size_t rows = 0;
        ullm_runtime_buffer *output = nullptr;
        const char *label = "";
    };
    std::array<Matrix, 3> matrices{{
        {first_index_buffer,
         first_scale_buffer,
         first_codebook_buffer,
         first_scale_values_buffer,
         first_row_scale_buffer,
         first_scale_count,
         first_group_size,
         first_tensor_scale,
         first_row_scale_count,
         first_rows,
         first_output_buffer,
         "first"},
        {second_index_buffer,
         second_scale_buffer,
         second_codebook_buffer,
         second_scale_values_buffer,
         second_row_scale_buffer,
         second_scale_count,
         second_group_size,
         second_tensor_scale,
         second_row_scale_count,
         second_rows,
         second_output_buffer,
         "second"},
        {third_index_buffer,
         third_scale_buffer,
         third_codebook_buffer,
         third_scale_values_buffer,
         third_row_scale_buffer,
         third_scale_count,
         third_group_size,
         third_tensor_scale,
         third_row_scale_count,
         third_rows,
         third_output_buffer,
         "third"},
    }};
    if (input_buffer == nullptr) {
        set_error("AQ4 matvec triple received a null input pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    for (const Matrix &matrix : matrices) {
        if (matrix.index == nullptr || matrix.scale == nullptr || matrix.codebook == nullptr ||
            matrix.scale_values == nullptr || matrix.output == nullptr) {
            set_error("AQ4 matvec triple received a null pointer");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.scale_count == 0) {
            set_error("AQ4 matvec triple scale table is empty");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.group_size == 0) {
            set_error("AQ4 matvec triple group size must be greater than zero");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.rows == 0 || cols == 0) {
            set_error("AQ4 matvec triple rows and cols must be greater than zero");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (!std::isfinite(matrix.tensor_scale) || matrix.tensor_scale <= 0.0f) {
            set_error("AQ4 matvec triple tensor scale must be finite and greater than zero");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }

    const ullm_runtime_buffer *base = first_index_buffer;
    for (const Matrix &matrix : matrices) {
        if (!buffers_share_backend(base, matrix.index) ||
            !buffers_share_backend(base, matrix.scale) ||
            !buffers_share_backend(base, matrix.codebook) ||
            !buffers_share_backend(base, matrix.scale_values) ||
            !buffers_share_backend(base, matrix.output) ||
            (matrix.row_scale != nullptr && !buffers_share_backend(base, matrix.row_scale))) {
            set_error("AQ4 matvec triple buffers belong to different backends or devices");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (!stream_matches_buffer(matrix.output, stream)) {
            set_error("AQ4 matvec triple stream belongs to a different backend or device");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!buffers_share_backend(base, input_buffer)) {
        set_error("AQ4 matvec triple buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (first_rows > max_size - second_rows ||
        first_rows + second_rows > max_size - third_rows) {
        set_error("AQ4 matvec triple matrix row count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    std::array<size_t, 3> elements{};
    std::array<size_t, 3> required_index_bytes{};
    std::array<size_t, 3> groups{};
    std::array<size_t, 3> required_output_bytes{};
    std::array<size_t, 3> required_scale_value_bytes{};
    std::array<size_t, 3> required_row_scale_bytes{};
    for (size_t index = 0; index < matrices.size(); ++index) {
        const Matrix &matrix = matrices[index];
        if (cols > max_size / matrix.rows ||
            matrix.rows > max_size / sizeof(float) ||
            matrix.scale_count > max_size / sizeof(float) ||
            matrix.row_scale_count > max_size / sizeof(float)) {
            set_error("AQ4 matvec triple byte size overflows");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        elements[index] = matrix.rows * cols;
        required_index_bytes[index] = elements[index] / 2 + (elements[index] % 2);
        groups[index] =
            elements[index] / matrix.group_size +
            (elements[index] % matrix.group_size == 0 ? 0 : 1);
        required_output_bytes[index] = matrix.rows * sizeof(float);
        required_scale_value_bytes[index] = matrix.scale_count * sizeof(float);
        required_row_scale_bytes[index] = matrix.row_scale_count * sizeof(float);
    }
    if (cols > max_size / sizeof(float)) {
        set_error("AQ4 matvec triple input byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec triple input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < matrices.size(); ++index) {
        const Matrix &matrix = matrices[index];
        if (matrix.index->bytes < required_index_bytes[index]) {
            set_error("AQ4 matvec triple index buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.scale->bytes < groups[index]) {
            set_error("AQ4 matvec triple scale index buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.scale_values->bytes < required_scale_value_bytes[index]) {
            set_error("AQ4 matvec triple scale value buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.row_scale != nullptr &&
            matrix.row_scale->bytes < required_row_scale_bytes[index]) {
            set_error("AQ4 matvec triple row scale buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.output->bytes < required_output_bytes[index]) {
            set_error("AQ4 matvec triple output buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.codebook->bytes % sizeof(float) != 0 ||
            matrix.codebook->bytes / sizeof(float) < 16) {
            set_error("AQ4 matvec triple requires at least 16 f32 codebook entries");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }

    if (base->backend == BackendKind::Cpu) {
        for (const Matrix &matrix : matrices) {
            if (!aq4_matvec_f32_host(
                    static_cast<const std::uint8_t *>(matrix.index->ptr),
                    static_cast<const std::uint8_t *>(matrix.scale->ptr),
                    static_cast<const float *>(matrix.codebook->ptr),
                    static_cast<const float *>(matrix.scale_values->ptr),
                    matrix.scale_count,
                    matrix.group_size,
                    matrix.tensor_scale,
                    static_cast<const float *>(input_buffer->ptr),
                    matrix.row_scale == nullptr
                        ? nullptr
                        : static_cast<const float *>(matrix.row_scale->ptr),
                    matrix.row_scale_count,
                    matrix.rows,
                    cols,
                    static_cast<float *>(matrix.output->ptr))) {
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_triple_f32_hip_kernel(
            first_index_buffer,
            first_scale_buffer,
            first_codebook_buffer,
            first_scale_values_buffer,
            first_row_scale_buffer,
            first_scale_count,
            first_group_size,
            first_tensor_scale,
            first_row_scale_count,
            second_index_buffer,
            second_scale_buffer,
            second_codebook_buffer,
            second_scale_values_buffer,
            second_row_scale_buffer,
            second_scale_count,
            second_group_size,
            second_tensor_scale,
            second_row_scale_count,
            third_index_buffer,
            third_scale_buffer,
            third_codebook_buffer,
            third_scale_values_buffer,
            third_row_scale_buffer,
            third_scale_count,
            third_group_size,
            third_tensor_scale,
            third_row_scale_count,
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

    set_error(
        hip_kernel_error.empty() ? "AQ4 matvec triple HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_aq4_matvec_qkv_z_gate_beta_f32(
    const ullm_runtime_buffer *qkv_index_buffer,
    const ullm_runtime_buffer *qkv_scale_buffer,
    const ullm_runtime_buffer *qkv_codebook_buffer,
    const ullm_runtime_buffer *qkv_scale_values_buffer,
    const ullm_runtime_buffer *qkv_row_scale_buffer,
    size_t qkv_scale_count,
    size_t qkv_group_size,
    float qkv_tensor_scale,
    size_t qkv_row_scale_count,
    const ullm_runtime_buffer *z_index_buffer,
    const ullm_runtime_buffer *z_scale_buffer,
    const ullm_runtime_buffer *z_codebook_buffer,
    const ullm_runtime_buffer *z_scale_values_buffer,
    const ullm_runtime_buffer *z_row_scale_buffer,
    size_t z_scale_count,
    size_t z_group_size,
    float z_tensor_scale,
    size_t z_row_scale_count,
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t qkv_rows,
    size_t z_rows,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *qkv_output_buffer,
    ullm_runtime_buffer *z_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    if (qkv_index_buffer == nullptr || qkv_scale_buffer == nullptr ||
        qkv_codebook_buffer == nullptr || qkv_scale_values_buffer == nullptr ||
        z_index_buffer == nullptr || z_scale_buffer == nullptr ||
        z_codebook_buffer == nullptr || z_scale_values_buffer == nullptr ||
        a_index_buffer == nullptr || a_scale_buffer == nullptr ||
        a_codebook_buffer == nullptr || a_scale_values_buffer == nullptr ||
        b_index_buffer == nullptr || b_scale_buffer == nullptr ||
        b_codebook_buffer == nullptr || b_scale_values_buffer == nullptr ||
        input_buffer == nullptr || a_log_buffer == nullptr || dt_bias_buffer == nullptr ||
        qkv_output_buffer == nullptr || z_output_buffer == nullptr ||
        gate_output_buffer == nullptr || beta_output_buffer == nullptr) {
        set_error("AQ4 qkv/z gate/beta received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_scale_count == 0 || z_scale_count == 0 ||
        a_scale_count == 0 || b_scale_count == 0) {
        set_error("AQ4 qkv/z gate/beta scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_group_size == 0 || z_group_size == 0 ||
        a_group_size == 0 || b_group_size == 0) {
        set_error("AQ4 qkv/z gate/beta group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_rows == 0 || z_rows == 0 || heads == 0 || cols == 0) {
        set_error("AQ4 qkv/z gate/beta rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(qkv_tensor_scale) || qkv_tensor_scale <= 0.0f ||
        !std::isfinite(z_tensor_scale) || z_tensor_scale <= 0.0f ||
        !std::isfinite(a_tensor_scale) || a_tensor_scale <= 0.0f ||
        !std::isfinite(b_tensor_scale) || b_tensor_scale <= 0.0f) {
        set_error("AQ4 qkv/z gate/beta tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(qkv_index_buffer, qkv_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, qkv_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, qkv_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_index_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_index_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_index_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, input_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_log_buffer) ||
        !buffers_share_backend(qkv_index_buffer, dt_bias_buffer) ||
        !buffers_share_backend(qkv_index_buffer, qkv_output_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_output_buffer) ||
        !buffers_share_backend(qkv_index_buffer, gate_output_buffer) ||
        !buffers_share_backend(qkv_index_buffer, beta_output_buffer) ||
        (qkv_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, qkv_row_scale_buffer)) ||
        (z_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, z_row_scale_buffer)) ||
        (a_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, a_row_scale_buffer)) ||
        (b_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, b_row_scale_buffer))) {
        set_error("AQ4 qkv/z gate/beta buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(qkv_output_buffer, stream) ||
        !stream_matches_buffer(z_output_buffer, stream) ||
        !stream_matches_buffer(gate_output_buffer, stream) ||
        !stream_matches_buffer(beta_output_buffer, stream)) {
        set_error("AQ4 qkv/z gate/beta stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / qkv_rows || cols > max_size / z_rows ||
        cols > max_size / heads) {
        set_error("AQ4 qkv/z gate/beta matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_elements = qkv_rows * cols;
    const size_t z_elements = z_rows * cols;
    const size_t gate_elements = heads * cols;
    const size_t qkv_required_index_bytes = qkv_elements / 2 + (qkv_elements % 2);
    const size_t z_required_index_bytes = z_elements / 2 + (z_elements % 2);
    const size_t gate_required_index_bytes = gate_elements / 2 + (gate_elements % 2);
    const size_t qkv_groups =
        qkv_elements / qkv_group_size + (qkv_elements % qkv_group_size == 0 ? 0 : 1);
    const size_t z_groups =
        z_elements / z_group_size + (z_elements % z_group_size == 0 ? 0 : 1);
    const size_t a_groups =
        gate_elements / a_group_size + (gate_elements % a_group_size == 0 ? 0 : 1);
    const size_t b_groups =
        gate_elements / b_group_size + (gate_elements % b_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        qkv_rows > max_size / sizeof(float) ||
        z_rows > max_size / sizeof(float) ||
        heads > max_size / sizeof(float) ||
        qkv_scale_count > max_size / sizeof(float) ||
        z_scale_count > max_size / sizeof(float) ||
        a_scale_count > max_size / sizeof(float) ||
        b_scale_count > max_size / sizeof(float) ||
        qkv_row_scale_count > max_size / sizeof(float) ||
        z_row_scale_count > max_size / sizeof(float) ||
        a_row_scale_count > max_size / sizeof(float) ||
        b_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 qkv/z gate/beta byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t qkv_required_output_bytes = qkv_rows * sizeof(float);
    const size_t z_required_output_bytes = z_rows * sizeof(float);
    const size_t gate_required_output_bytes = heads * sizeof(float);
    const size_t qkv_required_scale_value_bytes = qkv_scale_count * sizeof(float);
    const size_t z_required_scale_value_bytes = z_scale_count * sizeof(float);
    const size_t a_required_scale_value_bytes = a_scale_count * sizeof(float);
    const size_t b_required_scale_value_bytes = b_scale_count * sizeof(float);
    const size_t qkv_required_row_scale_bytes = qkv_row_scale_count * sizeof(float);
    const size_t z_required_row_scale_bytes = z_row_scale_count * sizeof(float);
    const size_t a_required_row_scale_bytes = a_row_scale_count * sizeof(float);
    const size_t b_required_row_scale_bytes = b_row_scale_count * sizeof(float);
    if (qkv_index_buffer->bytes < qkv_required_index_bytes ||
        z_index_buffer->bytes < z_required_index_bytes ||
        a_index_buffer->bytes < gate_required_index_bytes ||
        b_index_buffer->bytes < gate_required_index_bytes) {
        set_error("AQ4 qkv/z gate/beta index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_scale_buffer->bytes < qkv_groups ||
        z_scale_buffer->bytes < z_groups ||
        a_scale_buffer->bytes < a_groups ||
        b_scale_buffer->bytes < b_groups) {
        set_error("AQ4 qkv/z gate/beta scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_scale_values_buffer->bytes < qkv_required_scale_value_bytes ||
        z_scale_values_buffer->bytes < z_required_scale_value_bytes ||
        a_scale_values_buffer->bytes < a_required_scale_value_bytes ||
        b_scale_values_buffer->bytes < b_required_scale_value_bytes) {
        set_error("AQ4 qkv/z gate/beta scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 qkv/z gate/beta input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_log_buffer->bytes < gate_required_output_bytes ||
        dt_bias_buffer->bytes < gate_required_output_bytes) {
        set_error("AQ4 qkv/z gate/beta parameter buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((qkv_row_scale_buffer != nullptr &&
         qkv_row_scale_buffer->bytes < qkv_required_row_scale_bytes) ||
        (z_row_scale_buffer != nullptr &&
         z_row_scale_buffer->bytes < z_required_row_scale_bytes) ||
        (a_row_scale_buffer != nullptr &&
         a_row_scale_buffer->bytes < a_required_row_scale_bytes) ||
        (b_row_scale_buffer != nullptr &&
         b_row_scale_buffer->bytes < b_required_row_scale_bytes)) {
        set_error("AQ4 qkv/z gate/beta row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_output_buffer->bytes < qkv_required_output_bytes ||
        z_output_buffer->bytes < z_required_output_bytes ||
        gate_output_buffer->bytes < gate_required_output_bytes ||
        beta_output_buffer->bytes < gate_required_output_bytes) {
        set_error("AQ4 qkv/z gate/beta output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_codebook_buffer->bytes % sizeof(float) != 0 ||
        z_codebook_buffer->bytes % sizeof(float) != 0 ||
        a_codebook_buffer->bytes % sizeof(float) != 0 ||
        b_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 qkv/z gate/beta codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_codebook_entries = qkv_codebook_buffer->bytes / sizeof(float);
    const size_t z_codebook_entries = z_codebook_buffer->bytes / sizeof(float);
    const size_t a_codebook_entries = a_codebook_buffer->bytes / sizeof(float);
    const size_t b_codebook_entries = b_codebook_buffer->bytes / sizeof(float);
    if (qkv_codebook_entries < 16 || z_codebook_entries < 16 ||
        a_codebook_entries < 16 || b_codebook_entries < 16) {
        set_error("AQ4 qkv/z gate/beta requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (qkv_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(qkv_index_buffer->ptr),
                static_cast<const std::uint8_t *>(qkv_scale_buffer->ptr),
                static_cast<const float *>(qkv_codebook_buffer->ptr),
                static_cast<const float *>(qkv_scale_values_buffer->ptr),
                qkv_scale_count,
                qkv_group_size,
                qkv_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                qkv_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(qkv_row_scale_buffer->ptr),
                qkv_row_scale_count,
                qkv_rows,
                cols,
                static_cast<float *>(qkv_output_buffer->ptr)) ||
            !aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(z_index_buffer->ptr),
                static_cast<const std::uint8_t *>(z_scale_buffer->ptr),
                static_cast<const float *>(z_codebook_buffer->ptr),
                static_cast<const float *>(z_scale_values_buffer->ptr),
                z_scale_count,
                z_group_size,
                z_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                z_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(z_row_scale_buffer->ptr),
                z_row_scale_count,
                z_rows,
                cols,
                static_cast<float *>(z_output_buffer->ptr)) ||
            !aq4_matvec_gate_beta_f32_host(
                static_cast<const std::uint8_t *>(a_index_buffer->ptr),
                static_cast<const std::uint8_t *>(a_scale_buffer->ptr),
                static_cast<const float *>(a_codebook_buffer->ptr),
                static_cast<const float *>(a_scale_values_buffer->ptr),
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                a_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(a_row_scale_buffer->ptr),
                a_row_scale_count,
                static_cast<const std::uint8_t *>(b_index_buffer->ptr),
                static_cast<const std::uint8_t *>(b_scale_buffer->ptr),
                static_cast<const float *>(b_codebook_buffer->ptr),
                static_cast<const float *>(b_scale_values_buffer->ptr),
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                b_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(b_row_scale_buffer->ptr),
                b_row_scale_count,
                static_cast<const float *>(input_buffer->ptr),
                static_cast<const float *>(a_log_buffer->ptr),
                static_cast<const float *>(dt_bias_buffer->ptr),
                heads,
                cols,
                static_cast<float *>(gate_output_buffer->ptr),
                static_cast<float *>(beta_output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_qkv_z_gate_beta_f32_hip_kernel(
            qkv_index_buffer,
            qkv_scale_buffer,
            qkv_codebook_buffer,
            qkv_scale_values_buffer,
            qkv_row_scale_buffer,
            qkv_scale_count,
            qkv_group_size,
            qkv_tensor_scale,
            qkv_row_scale_count,
            z_index_buffer,
            z_scale_buffer,
            z_codebook_buffer,
            z_scale_values_buffer,
            z_row_scale_buffer,
            z_scale_count,
            z_group_size,
            z_tensor_scale,
            z_row_scale_count,
            a_index_buffer,
            a_scale_buffer,
            a_codebook_buffer,
            a_scale_values_buffer,
            a_row_scale_buffer,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_count,
            b_index_buffer,
            b_scale_buffer,
            b_codebook_buffer,
            b_scale_values_buffer,
            b_row_scale_buffer,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_count,
            input_buffer,
            a_log_buffer,
            dt_bias_buffer,
            qkv_rows,
            z_rows,
            heads,
            cols,
            qkv_output_buffer,
            z_output_buffer,
            gate_output_buffer,
            beta_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    set_error(
        hip_kernel_error.empty() ? "AQ4 qkv/z gate/beta HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_aq4_matvec_silu_mul_f32(
    const ullm_runtime_buffer *gate_index_buffer,
    const ullm_runtime_buffer *gate_scale_buffer,
    const ullm_runtime_buffer *gate_codebook_buffer,
    const ullm_runtime_buffer *gate_scale_values_buffer,
    const ullm_runtime_buffer *gate_row_scale_buffer,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    size_t gate_row_scale_count,
    const ullm_runtime_buffer *up_index_buffer,
    const ullm_runtime_buffer *up_scale_buffer,
    const ullm_runtime_buffer *up_codebook_buffer,
    const ullm_runtime_buffer *up_scale_values_buffer,
    const ullm_runtime_buffer *up_row_scale_buffer,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    size_t up_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_index_buffer == nullptr || gate_scale_buffer == nullptr ||
        gate_codebook_buffer == nullptr || gate_scale_values_buffer == nullptr ||
        up_index_buffer == nullptr || up_scale_buffer == nullptr ||
        up_codebook_buffer == nullptr || up_scale_values_buffer == nullptr ||
        input_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec SiLU-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_scale_count == 0 || up_scale_count == 0) {
        set_error("AQ4 matvec SiLU-mul scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_group_size == 0 || up_group_size == 0) {
        set_error("AQ4 matvec SiLU-mul group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 matvec SiLU-mul rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(gate_tensor_scale) || gate_tensor_scale <= 0.0f ||
        !std::isfinite(up_tensor_scale) || up_tensor_scale <= 0.0f) {
        set_error("AQ4 matvec SiLU-mul tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(gate_index_buffer, gate_scale_buffer) ||
        !buffers_share_backend(gate_index_buffer, gate_codebook_buffer) ||
        !buffers_share_backend(gate_index_buffer, gate_scale_values_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_index_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_scale_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_codebook_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_scale_values_buffer) ||
        !buffers_share_backend(gate_index_buffer, input_buffer) ||
        !buffers_share_backend(gate_index_buffer, output_buffer) ||
        (gate_row_scale_buffer != nullptr &&
         !buffers_share_backend(gate_index_buffer, gate_row_scale_buffer)) ||
        (up_row_scale_buffer != nullptr &&
         !buffers_share_backend(gate_index_buffer, up_row_scale_buffer))) {
        set_error("AQ4 matvec SiLU-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec SiLU-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec SiLU-mul matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t gate_required_index_bytes = elements / 2 + (elements % 2);
    const size_t up_required_index_bytes = gate_required_index_bytes;
    const size_t gate_groups =
        elements / gate_group_size + (elements % gate_group_size == 0 ? 0 : 1);
    const size_t up_groups = elements / up_group_size + (elements % up_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float) ||
        gate_scale_count > max_size / sizeof(float) ||
        up_scale_count > max_size / sizeof(float) ||
        gate_row_scale_count > max_size / sizeof(float) ||
        up_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec SiLU-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    const size_t gate_required_scale_value_bytes = gate_scale_count * sizeof(float);
    const size_t up_required_scale_value_bytes = up_scale_count * sizeof(float);
    const size_t gate_required_row_scale_bytes = gate_row_scale_count * sizeof(float);
    const size_t up_required_row_scale_bytes = up_row_scale_count * sizeof(float);
    if (gate_index_buffer->bytes < gate_required_index_bytes ||
        up_index_buffer->bytes < up_required_index_bytes) {
        set_error("AQ4 matvec SiLU-mul index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_scale_buffer->bytes < gate_groups || up_scale_buffer->bytes < up_groups) {
        set_error("AQ4 matvec SiLU-mul scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_scale_values_buffer->bytes < gate_required_scale_value_bytes ||
        up_scale_values_buffer->bytes < up_required_scale_value_bytes) {
        set_error("AQ4 matvec SiLU-mul scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec SiLU-mul input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_row_scale_buffer != nullptr &&
         gate_row_scale_buffer->bytes < gate_required_row_scale_bytes) ||
        (up_row_scale_buffer != nullptr &&
         up_row_scale_buffer->bytes < up_required_row_scale_bytes)) {
        set_error("AQ4 matvec SiLU-mul row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec SiLU-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_codebook_buffer->bytes % sizeof(float) != 0 ||
        up_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec SiLU-mul codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t gate_codebook_entries = gate_codebook_buffer->bytes / sizeof(float);
    const size_t up_codebook_entries = up_codebook_buffer->bytes / sizeof(float);
    if (gate_codebook_entries < 16 || up_codebook_entries < 16) {
        set_error("AQ4 matvec SiLU-mul requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (gate_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_silu_mul_f32_host(
                static_cast<const std::uint8_t *>(gate_index_buffer->ptr),
                static_cast<const std::uint8_t *>(gate_scale_buffer->ptr),
                static_cast<const float *>(gate_codebook_buffer->ptr),
                static_cast<const float *>(gate_scale_values_buffer->ptr),
                gate_scale_count,
                gate_group_size,
                gate_tensor_scale,
                gate_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(gate_row_scale_buffer->ptr),
                gate_row_scale_count,
                static_cast<const std::uint8_t *>(up_index_buffer->ptr),
                static_cast<const std::uint8_t *>(up_scale_buffer->ptr),
                static_cast<const float *>(up_codebook_buffer->ptr),
                static_cast<const float *>(up_scale_values_buffer->ptr),
                up_scale_count,
                up_group_size,
                up_tensor_scale,
                up_row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(up_row_scale_buffer->ptr),
                up_row_scale_count,
                static_cast<const float *>(input_buffer->ptr),
                rows,
                cols,
                static_cast<float *>(output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_silu_mul_f32_hip_kernel(
            gate_index_buffer,
            gate_scale_buffer,
            gate_codebook_buffer,
            gate_scale_values_buffer,
            gate_row_scale_buffer,
            gate_scale_count,
            gate_group_size,
            gate_tensor_scale,
            gate_row_scale_count,
            up_index_buffer,
            up_scale_buffer,
            up_codebook_buffer,
            up_scale_values_buffer,
            up_row_scale_buffer,
            up_scale_count,
            up_group_size,
            up_tensor_scale,
            up_row_scale_count,
            input_buffer,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec SiLU-mul HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_silu_mul_f32_hip_staging(
        gate_index_buffer,
        gate_scale_buffer,
        gate_codebook_buffer,
        gate_scale_values_buffer,
        gate_row_scale_buffer,
        gate_scale_count,
        gate_group_size,
        gate_tensor_scale,
        gate_row_scale_count,
        gate_required_index_bytes,
        gate_groups,
        gate_codebook_entries,
        up_index_buffer,
        up_scale_buffer,
        up_codebook_buffer,
        up_scale_values_buffer,
        up_row_scale_buffer,
        up_scale_count,
        up_group_size,
        up_tensor_scale,
        up_row_scale_count,
        up_required_index_bytes,
        up_groups,
        up_codebook_entries,
        input_buffer,
        rows,
        cols,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_gate_beta_f32(
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    if (a_index_buffer == nullptr || a_scale_buffer == nullptr ||
        a_codebook_buffer == nullptr || a_scale_values_buffer == nullptr ||
        b_index_buffer == nullptr || b_scale_buffer == nullptr ||
        b_codebook_buffer == nullptr || b_scale_values_buffer == nullptr ||
        input_buffer == nullptr || a_log_buffer == nullptr || dt_bias_buffer == nullptr ||
        gate_output_buffer == nullptr || beta_output_buffer == nullptr) {
        set_error("AQ4 matvec gate/beta received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_scale_count == 0 || b_scale_count == 0) {
        set_error("AQ4 matvec gate/beta scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_group_size == 0 || b_group_size == 0) {
        set_error("AQ4 matvec gate/beta group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (heads == 0 || cols == 0) {
        set_error("AQ4 matvec gate/beta heads and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(a_tensor_scale) || a_tensor_scale <= 0.0f ||
        !std::isfinite(b_tensor_scale) || b_tensor_scale <= 0.0f) {
        set_error("AQ4 matvec gate/beta tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(a_index_buffer, a_scale_buffer) ||
        !buffers_share_backend(a_index_buffer, a_codebook_buffer) ||
        !buffers_share_backend(a_index_buffer, a_scale_values_buffer) ||
        !buffers_share_backend(a_index_buffer, b_index_buffer) ||
        !buffers_share_backend(a_index_buffer, b_scale_buffer) ||
        !buffers_share_backend(a_index_buffer, b_codebook_buffer) ||
        !buffers_share_backend(a_index_buffer, b_scale_values_buffer) ||
        !buffers_share_backend(a_index_buffer, input_buffer) ||
        !buffers_share_backend(a_index_buffer, a_log_buffer) ||
        !buffers_share_backend(a_index_buffer, dt_bias_buffer) ||
        !buffers_share_backend(a_index_buffer, gate_output_buffer) ||
        !buffers_share_backend(a_index_buffer, beta_output_buffer) ||
        (a_row_scale_buffer != nullptr &&
         !buffers_share_backend(a_index_buffer, a_row_scale_buffer)) ||
        (b_row_scale_buffer != nullptr &&
         !buffers_share_backend(a_index_buffer, b_row_scale_buffer))) {
        set_error("AQ4 matvec gate/beta buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(gate_output_buffer, stream) ||
        !stream_matches_buffer(beta_output_buffer, stream)) {
        set_error("AQ4 matvec gate/beta stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / heads) {
        set_error("AQ4 matvec gate/beta matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = heads * cols;
    const size_t a_required_index_bytes = elements / 2 + (elements % 2);
    const size_t b_required_index_bytes = a_required_index_bytes;
    const size_t a_groups = elements / a_group_size + (elements % a_group_size == 0 ? 0 : 1);
    const size_t b_groups = elements / b_group_size + (elements % b_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        heads > max_size / sizeof(float) ||
        a_scale_count > max_size / sizeof(float) ||
        b_scale_count > max_size / sizeof(float) ||
        a_row_scale_count > max_size / sizeof(float) ||
        b_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec gate/beta byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = heads * sizeof(float);
    const size_t a_required_scale_value_bytes = a_scale_count * sizeof(float);
    const size_t b_required_scale_value_bytes = b_scale_count * sizeof(float);
    const size_t a_required_row_scale_bytes = a_row_scale_count * sizeof(float);
    const size_t b_required_row_scale_bytes = b_row_scale_count * sizeof(float);
    if (a_index_buffer->bytes < a_required_index_bytes ||
        b_index_buffer->bytes < b_required_index_bytes) {
        set_error("AQ4 matvec gate/beta index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_scale_buffer->bytes < a_groups || b_scale_buffer->bytes < b_groups) {
        set_error("AQ4 matvec gate/beta scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_scale_values_buffer->bytes < a_required_scale_value_bytes ||
        b_scale_values_buffer->bytes < b_required_scale_value_bytes) {
        set_error("AQ4 matvec gate/beta scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec gate/beta input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_log_buffer->bytes < required_output_bytes ||
        dt_bias_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec gate/beta parameter buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((a_row_scale_buffer != nullptr &&
         a_row_scale_buffer->bytes < a_required_row_scale_bytes) ||
        (b_row_scale_buffer != nullptr &&
         b_row_scale_buffer->bytes < b_required_row_scale_bytes)) {
        set_error("AQ4 matvec gate/beta row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_output_buffer->bytes < required_output_bytes ||
        beta_output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec gate/beta output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_codebook_buffer->bytes % sizeof(float) != 0 ||
        b_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec gate/beta codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t a_codebook_entries = a_codebook_buffer->bytes / sizeof(float);
    const size_t b_codebook_entries = b_codebook_buffer->bytes / sizeof(float);
    if (a_codebook_entries < 16 || b_codebook_entries < 16) {
        set_error("AQ4 matvec gate/beta requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (a_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_gate_beta_f32_host(
                static_cast<const std::uint8_t *>(a_index_buffer->ptr),
                static_cast<const std::uint8_t *>(a_scale_buffer->ptr),
                static_cast<const float *>(a_codebook_buffer->ptr),
                static_cast<const float *>(a_scale_values_buffer->ptr),
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                a_row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(a_row_scale_buffer->ptr),
                a_row_scale_count,
                static_cast<const std::uint8_t *>(b_index_buffer->ptr),
                static_cast<const std::uint8_t *>(b_scale_buffer->ptr),
                static_cast<const float *>(b_codebook_buffer->ptr),
                static_cast<const float *>(b_scale_values_buffer->ptr),
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                b_row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(b_row_scale_buffer->ptr),
                b_row_scale_count,
                static_cast<const float *>(input_buffer->ptr),
                static_cast<const float *>(a_log_buffer->ptr),
                static_cast<const float *>(dt_bias_buffer->ptr),
                heads,
                cols,
                static_cast<float *>(gate_output_buffer->ptr),
                static_cast<float *>(beta_output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_gate_beta_f32_hip_kernel(
            a_index_buffer,
            a_scale_buffer,
            a_codebook_buffer,
            a_scale_values_buffer,
            a_row_scale_buffer,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_count,
            b_index_buffer,
            b_scale_buffer,
            b_codebook_buffer,
            b_scale_values_buffer,
            b_row_scale_buffer,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_count,
            input_buffer,
            a_log_buffer,
            dt_bias_buffer,
            heads,
            cols,
            gate_output_buffer,
            beta_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec gate/beta HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_gate_beta_f32_hip_staging(
        a_index_buffer,
        a_scale_buffer,
        a_codebook_buffer,
        a_scale_values_buffer,
        a_row_scale_buffer,
        a_scale_count,
        a_group_size,
        a_tensor_scale,
        a_row_scale_count,
        a_required_index_bytes,
        a_groups,
        a_codebook_entries,
        b_index_buffer,
        b_scale_buffer,
        b_codebook_buffer,
        b_scale_values_buffer,
        b_row_scale_buffer,
        b_scale_count,
        b_group_size,
        b_tensor_scale,
        b_row_scale_count,
        b_required_index_bytes,
        b_groups,
        b_codebook_entries,
        input_buffer,
        a_log_buffer,
        dt_bias_buffer,
        heads,
        cols,
        gate_output_buffer,
        beta_output_buffer,
        stream);
}
