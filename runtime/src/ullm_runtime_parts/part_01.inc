
ullm_status sigmoid_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_gate(elements);
    std::vector<float> host_input(elements);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = gate_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 sigmoid-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 sigmoid-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    sigmoid_mul_f32_host(host_gate.data(), host_input.data(), elements, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 sigmoid-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 sigmoid-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

#include "../parts/qwen35_rope_kernels.inc"
#include "../parts/add_kernels.inc"
#include "../parts/rope_kernels.inc"
#include "../parts/causal_attention_kernels.inc"
#include "../parts/decode_attention_kernels.inc"
#include "../parts/cached_prefix_attention_kernels.inc"
#include "../parts/paged_decode_attention_kernels.inc"
#include "../parts/paged_kv_write_kernels.inc"
#include "../parts/paged_chunk_kernels.inc"
#include "../parts/paged_wmma_chunk_kernels.inc"
#include "../parts/depthwise_conv1d_kernels.inc"
#include "../parts/linear_attn_prepare_kernels.inc"
#include "../parts/linear_attn_recurrent_kernels.inc"
#include "../parts/runtime_epilogue.inc"
