// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "moe_gfx1201.h"

namespace {

constexpr uint32_t kMoeWeightF32 = ULLM_MOE_WEIGHT_DTYPE_F32;
constexpr uint32_t kMoeWeightBf16 = ULLM_MOE_WEIGHT_DTYPE_BF16;

bool moe_checked_mul(size_t lhs, size_t rhs, size_t *out) {
    if (out == nullptr || (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool moe_checked_bytes(size_t elements, size_t element_bytes, size_t *out) {
    return moe_checked_mul(elements, element_bytes, out);
}

bool moe_weight_dtype_is_valid(ullm_moe_weight_dtype dtype) {
    return dtype == ULLM_MOE_WEIGHT_DTYPE_F32 || dtype == ULLM_MOE_WEIGHT_DTYPE_BF16;
}

size_t moe_weight_element_bytes(ullm_moe_weight_dtype dtype) {
    return dtype == ULLM_MOE_WEIGHT_DTYPE_F32 ? sizeof(float) : sizeof(uint16_t);
}

float moe_bf16_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float moe_f32_to_bf16_roundtrip(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return value;
    }
    const uint32_t rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    const uint32_t bf16_bits = rounded & 0xffff0000u;
    float result = 0.0f;
    std::memcpy(&result, &bf16_bits, sizeof(result));
    return result;
}

float moe_host_weight(const void *weights, ullm_moe_weight_dtype dtype, size_t index) {
    if (dtype == ULLM_MOE_WEIGHT_DTYPE_F32) {
        return static_cast<const float *>(weights)[index];
    }
    return moe_bf16_to_f32(static_cast<const uint16_t *>(weights)[index]);
}

bool moe_buffers_and_stream_match(
    const ullm_runtime_buffer *first,
    const ullm_runtime_buffer *second,
    const ullm_runtime_buffer *third,
    const ullm_runtime_buffer *fourth,
    const ullm_runtime_buffer *fifth,
    const ullm_runtime_buffer *output,
    ullm_runtime_stream *stream) {
    const ullm_runtime_buffer *buffers[] = {first, second, third, fourth, fifth};
    for (const ullm_runtime_buffer *buffer : buffers) {
        if (buffer != nullptr && !buffers_share_backend(first, buffer)) {
            return false;
        }
        if (buffer != nullptr && !stream_matches_buffer(buffer, stream)) {
            return false;
        }
    }
    return output != nullptr && buffers_share_backend(first, output) &&
           stream_matches_buffer(output, stream);
}

bool moe_gpu_route(
    const ullm_runtime_buffer *hidden_buffer,
    const ullm_runtime_buffer *router_weight_buffer,
    ullm_moe_weight_dtype router_weight_dtype,
    size_t tokens,
    size_t hidden_size,
    size_t num_experts,
    size_t top_k,
    ullm_runtime_buffer *routing_scores_buffer,
    ullm_runtime_buffer *selected_expert_ids_buffer,
    ullm_runtime_buffer *boundary_tie_flags_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
#if defined(ULLM_RUNTIME_ROCM_MOE_GFX1201)
    char gpu_error[1024]{};
    const int launched = ullm_moe_gfx1201_route_f32(
        hidden_buffer->ptr,
        router_weight_buffer->ptr,
        static_cast<uint32_t>(router_weight_dtype),
        tokens,
        hidden_size,
        num_experts,
        top_k,
        routing_scores_buffer->ptr,
        selected_expert_ids_buffer->ptr,
        boundary_tie_flags_buffer->ptr,
        stream == nullptr ? nullptr : stream->stream,
        hidden_buffer->hip_device_id,
        gpu_error,
        sizeof(gpu_error));
    if (launched != 0) {
        return true;
    }
    if (error != nullptr) {
        *error = gpu_error[0] == '\0' ? "MoE route gfx1201 kernel launch failed" : gpu_error;
    }
    return false;
#else
    (void)hidden_buffer;
    (void)router_weight_buffer;
    (void)router_weight_dtype;
    (void)tokens;
    (void)hidden_size;
    (void)num_experts;
    (void)top_k;
    (void)routing_scores_buffer;
    (void)selected_expert_ids_buffer;
    (void)boundary_tie_flags_buffer;
    (void)stream;
    if (error != nullptr) {
        *error = "MoE GPU support requires the rocm-moe-gfx1201 runtime feature";
    }
    return false;
#endif
}

bool moe_gpu_gather(
    const ullm_runtime_buffer *hidden_buffer,
    size_t tokens,
    size_t hidden_size,
    size_t top_k,
    ullm_runtime_buffer *gathered_hidden_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
#if defined(ULLM_RUNTIME_ROCM_MOE_GFX1201)
    char gpu_error[1024]{};
    const int launched = ullm_moe_gfx1201_gather_f32(
        hidden_buffer->ptr,
        tokens,
        hidden_size,
        top_k,
        gathered_hidden_buffer->ptr,
        stream == nullptr ? nullptr : stream->stream,
        hidden_buffer->hip_device_id,
        gpu_error,
        sizeof(gpu_error));
    if (launched != 0) {
        return true;
    }
    if (error != nullptr) {
        *error = gpu_error[0] == '\0' ? "MoE gather gfx1201 kernel launch failed" : gpu_error;
    }
    return false;
#else
    (void)hidden_buffer;
    (void)tokens;
    (void)hidden_size;
    (void)top_k;
    (void)gathered_hidden_buffer;
    (void)stream;
    if (error != nullptr) {
        *error = "MoE GPU support requires the rocm-moe-gfx1201 runtime feature";
    }
    return false;
#endif
}

bool moe_gpu_grouped_gemm(
    const ullm_runtime_buffer *weight_buffer,
    ullm_moe_weight_dtype weight_dtype,
    const ullm_runtime_buffer *expert_ids_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t assignments,
    size_t num_experts,
    size_t rows_per_expert,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
#if defined(ULLM_RUNTIME_ROCM_MOE_GFX1201)
    char gpu_error[1024]{};
    const int launched = ullm_moe_gfx1201_grouped_gemm_f32(
        weight_buffer->ptr,
        static_cast<uint32_t>(weight_dtype),
        expert_ids_buffer->ptr,
        input_buffer->ptr,
        assignments,
        num_experts,
        rows_per_expert,
        cols,
        output_buffer->ptr,
        stream == nullptr ? nullptr : stream->stream,
        weight_buffer->hip_device_id,
        gpu_error,
        sizeof(gpu_error));
    if (launched != 0) {
        return true;
    }
    if (error != nullptr) {
        *error = gpu_error[0] == '\0' ? "MoE grouped GEMM gfx1201 kernel launch failed" : gpu_error;
    }
    return false;
#else
    (void)weight_buffer;
    (void)weight_dtype;
    (void)expert_ids_buffer;
    (void)input_buffer;
    (void)assignments;
    (void)num_experts;
    (void)rows_per_expert;
    (void)cols;
    (void)output_buffer;
    (void)stream;
    if (error != nullptr) {
        *error = "MoE GPU support requires the rocm-moe-gfx1201 runtime feature";
    }
    return false;
#endif
}

bool moe_gpu_gated_silu(
    const ullm_runtime_buffer *gate_up_buffer,
    size_t assignments,
    size_t intermediate_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
#if defined(ULLM_RUNTIME_ROCM_MOE_GFX1201)
    char gpu_error[1024]{};
    const int launched = ullm_moe_gfx1201_gated_silu_f32(
        gate_up_buffer->ptr,
        assignments,
        intermediate_size,
        output_buffer->ptr,
        stream == nullptr ? nullptr : stream->stream,
        gate_up_buffer->hip_device_id,
        gpu_error,
        sizeof(gpu_error));
    if (launched != 0) {
        return true;
    }
    if (error != nullptr) {
        *error = gpu_error[0] == '\0' ? "MoE gated SiLU gfx1201 kernel launch failed" : gpu_error;
    }
    return false;
#else
    (void)gate_up_buffer;
    (void)assignments;
    (void)intermediate_size;
    (void)output_buffer;
    (void)stream;
    if (error != nullptr) {
        *error = "MoE GPU support requires the rocm-moe-gfx1201 runtime feature";
    }
    return false;
#endif
}

bool moe_gpu_scatter_weighted(
    const ullm_runtime_buffer *expert_output_buffer,
    const ullm_runtime_buffer *routing_scores_buffer,
    size_t tokens,
    size_t top_k,
    size_t hidden_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
#if defined(ULLM_RUNTIME_ROCM_MOE_GFX1201)
    char gpu_error[1024]{};
    const int launched = ullm_moe_gfx1201_scatter_weighted_f32(
        expert_output_buffer->ptr,
        routing_scores_buffer->ptr,
        tokens,
        top_k,
        hidden_size,
        output_buffer->ptr,
        stream == nullptr ? nullptr : stream->stream,
        expert_output_buffer->hip_device_id,
        gpu_error,
        sizeof(gpu_error));
    if (launched != 0) {
        return true;
    }
    if (error != nullptr) {
        *error = gpu_error[0] == '\0' ? "MoE scatter gfx1201 kernel launch failed" : gpu_error;
    }
    return false;
#else
    (void)expert_output_buffer;
    (void)routing_scores_buffer;
    (void)tokens;
    (void)top_k;
    (void)hidden_size;
    (void)output_buffer;
    (void)stream;
    if (error != nullptr) {
        *error = "MoE GPU support requires the rocm-moe-gfx1201 runtime feature";
    }
    return false;
#endif
}

bool moe_gpu_sigmoid_gate(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t tokens,
    size_t hidden_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
#if defined(ULLM_RUNTIME_ROCM_MOE_GFX1201)
    char gpu_error[1024]{};
    const int launched = ullm_moe_gfx1201_sigmoid_gate_f32(
        gate_buffer->ptr,
        input_buffer->ptr,
        tokens,
        hidden_size,
        output_buffer->ptr,
        stream == nullptr ? nullptr : stream->stream,
        gate_buffer->hip_device_id,
        gpu_error,
        sizeof(gpu_error));
    if (launched != 0) {
        return true;
    }
    if (error != nullptr) {
        *error = gpu_error[0] == '\0' ? "MoE sigmoid gate gfx1201 kernel launch failed" : gpu_error;
    }
    return false;
#else
    (void)gate_buffer;
    (void)input_buffer;
    (void)tokens;
    (void)hidden_size;
    (void)output_buffer;
    (void)stream;
    if (error != nullptr) {
        *error = "MoE GPU support requires the rocm-moe-gfx1201 runtime feature";
    }
    return false;
#endif
}

} // namespace

ullm_status ullm_runtime_moe_route_f32(
    const ullm_runtime_buffer *hidden_buffer,
    const ullm_runtime_buffer *router_weight_buffer,
    ullm_moe_weight_dtype router_weight_dtype,
    size_t tokens,
    size_t hidden_size,
    size_t num_experts,
    size_t top_k,
    ullm_runtime_buffer *routing_scores_buffer,
    ullm_runtime_buffer *selected_expert_ids_buffer,
    ullm_runtime_buffer *boundary_tie_flags_buffer,
    ullm_runtime_stream *stream) {
    if (hidden_buffer == nullptr || router_weight_buffer == nullptr || routing_scores_buffer == nullptr ||
        selected_expert_ids_buffer == nullptr || boundary_tie_flags_buffer == nullptr) {
        set_error("MoE route received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!moe_weight_dtype_is_valid(router_weight_dtype) || tokens == 0u || hidden_size == 0u ||
        num_experts == 0u || top_k == 0u || top_k > num_experts) {
        set_error("MoE route received invalid dimensions or weight dtype");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!moe_buffers_and_stream_match(
            hidden_buffer,
            router_weight_buffer,
            routing_scores_buffer,
            selected_expert_ids_buffer,
            boundary_tie_flags_buffer,
            routing_scores_buffer,
            stream)) {
        set_error("MoE route buffers or stream do not share a backend and device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    size_t router_elements = 0u;
    size_t route_elements = 0u;
    size_t router_bytes = 0u;
    size_t hidden_bytes = 0u;
    size_t score_bytes = 0u;
    size_t id_bytes = 0u;
    size_t flag_bytes = 0u;
    if (!moe_checked_mul(num_experts, hidden_size, &router_elements) ||
        !moe_checked_mul(tokens, hidden_size, &route_elements) ||
        !moe_checked_bytes(router_elements, moe_weight_element_bytes(router_weight_dtype), &router_bytes) ||
        !moe_checked_bytes(route_elements, sizeof(float), &hidden_bytes) ||
        !moe_checked_mul(tokens, top_k, &route_elements) ||
        !moe_checked_bytes(route_elements, sizeof(float), &score_bytes) ||
        !moe_checked_bytes(route_elements, sizeof(int32_t), &id_bytes) ||
        !moe_checked_bytes(tokens, sizeof(uint32_t), &flag_bytes)) {
        set_error("MoE route byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (hidden_buffer->bytes < hidden_bytes || router_weight_buffer->bytes < router_bytes ||
        routing_scores_buffer->bytes < score_bytes || selected_expert_ids_buffer->bytes < id_bytes ||
        boundary_tie_flags_buffer->bytes < flag_bytes) {
        set_error("MoE route buffer is too small");
        return ULLM_STATUS_BUFFER_TOO_SMALL;
    }

    if (hidden_buffer->backend == BackendKind::Cpu) {
        const float *hidden = static_cast<const float *>(hidden_buffer->ptr);
        const void *router_weights = router_weight_buffer->ptr;
        float *scores = static_cast<float *>(routing_scores_buffer->ptr);
        int32_t *ids = static_cast<int32_t *>(selected_expert_ids_buffer->ptr);
        uint32_t *flags = static_cast<uint32_t *>(boundary_tie_flags_buffer->ptr);
        std::vector<float> probabilities(num_experts);
        std::vector<size_t> order(num_experts);
        for (size_t token = 0u; token < tokens; ++token) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t expert = 0u; expert < num_experts; ++expert) {
                float value = 0.0f;
                const size_t input_base = token * hidden_size;
                const size_t weight_base = expert * hidden_size;
                for (size_t column = 0u; column < hidden_size; ++column) {
                    const float activation = router_weight_dtype == ULLM_MOE_WEIGHT_DTYPE_BF16
                        ? moe_f32_to_bf16_roundtrip(hidden[input_base + column])
                        : hidden[input_base + column];
                    value += activation *
                             moe_host_weight(router_weights, router_weight_dtype, weight_base + column);
                }
                if (router_weight_dtype == ULLM_MOE_WEIGHT_DTYPE_BF16) {
                    // HF's F.linear returns BF16 logits for BF16 Qwen3.5
                    // router inputs/weights. Keep F32 storage but reproduce
                    // that boundary before the explicitly F32 softmax.
                    value = moe_f32_to_bf16_roundtrip(value);
                }
                probabilities[expert] = value;
                maximum = std::max(maximum, value);
                order[expert] = expert;
            }
            float denominator = 0.0f;
            for (float &value : probabilities) {
                value = std::exp(value - maximum);
                denominator += value;
            }
            for (float &value : probabilities) {
                value /= denominator;
            }
            std::sort(order.begin(), order.end(), [&probabilities](size_t lhs, size_t rhs) {
                if (probabilities[lhs] == probabilities[rhs]) {
                    return lhs > rhs;
                }
                return probabilities[lhs] > probabilities[rhs];
            });
            float selected_sum = 0.0f;
            for (size_t rank = 0u; rank < top_k; ++rank) {
                selected_sum += probabilities[order[rank]];
            }
            const float boundary = probabilities[order[top_k - 1u]];
            uint32_t tie = 0u;
            for (size_t rank = top_k; rank < num_experts; ++rank) {
                if (probabilities[order[rank]] == boundary) {
                    tie = 1u;
                    break;
                }
            }
            const size_t output_base = token * top_k;
            flags[token] = tie;
            for (size_t rank = 0u; rank < top_k; ++rank) {
                ids[output_base + rank] = static_cast<int32_t>(order[rank]);
                const float score = probabilities[order[rank]] / selected_sum;
                scores[output_base + rank] = router_weight_dtype == ULLM_MOE_WEIGHT_DTYPE_BF16
                    ? moe_f32_to_bf16_roundtrip(score)
                    : score;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string gpu_error;
    if (!moe_gpu_route(
            hidden_buffer,
            router_weight_buffer,
            router_weight_dtype,
            tokens,
            hidden_size,
            num_experts,
            top_k,
            routing_scores_buffer,
            selected_expert_ids_buffer,
            boundary_tie_flags_buffer,
            stream,
            &gpu_error)) {
        set_error(gpu_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_moe_gather_f32(
    const ullm_runtime_buffer *hidden_buffer,
    size_t tokens,
    size_t hidden_size,
    size_t top_k,
    ullm_runtime_buffer *gathered_hidden_buffer,
    ullm_runtime_stream *stream) {
    if (hidden_buffer == nullptr || gathered_hidden_buffer == nullptr) {
        set_error("MoE gather received a null buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (tokens == 0u || hidden_size == 0u || top_k == 0u ||
        !moe_buffers_and_stream_match(hidden_buffer, gathered_hidden_buffer, nullptr, nullptr, nullptr,
                                      gathered_hidden_buffer, stream)) {
        set_error("MoE gather received invalid dimensions, buffers, or stream");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    size_t input_elements = 0u;
    size_t assignments = 0u;
    size_t output_elements = 0u;
    size_t input_bytes = 0u;
    size_t output_bytes = 0u;
    if (!moe_checked_mul(tokens, hidden_size, &input_elements) ||
        !moe_checked_mul(tokens, top_k, &assignments) ||
        !moe_checked_mul(assignments, hidden_size, &output_elements) ||
        !moe_checked_bytes(input_elements, sizeof(float), &input_bytes) ||
        !moe_checked_bytes(output_elements, sizeof(float), &output_bytes)) {
        set_error("MoE gather byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (hidden_buffer->bytes < input_bytes || gathered_hidden_buffer->bytes < output_bytes) {
        set_error("MoE gather buffer is too small");
        return ULLM_STATUS_BUFFER_TOO_SMALL;
    }
    if (hidden_buffer->backend == BackendKind::Cpu) {
        const float *hidden = static_cast<const float *>(hidden_buffer->ptr);
        float *output = static_cast<float *>(gathered_hidden_buffer->ptr);
        for (size_t assignment = 0u; assignment < assignments; ++assignment) {
            const size_t token = assignment / top_k;
            std::memcpy(output + assignment * hidden_size,
                        hidden + token * hidden_size,
                        hidden_size * sizeof(float));
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string gpu_error;
    if (!moe_gpu_gather(hidden_buffer, tokens, hidden_size, top_k, gathered_hidden_buffer, stream, &gpu_error)) {
        set_error(gpu_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_moe_grouped_gemm_f32(
    const ullm_runtime_buffer *weight_buffer,
    ullm_moe_weight_dtype weight_dtype,
    const ullm_runtime_buffer *expert_ids_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t assignments,
    size_t num_experts,
    size_t rows_per_expert,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (weight_buffer == nullptr || expert_ids_buffer == nullptr || input_buffer == nullptr ||
        output_buffer == nullptr || !moe_weight_dtype_is_valid(weight_dtype) || assignments == 0u ||
        num_experts == 0u || rows_per_expert == 0u || cols == 0u ||
        !moe_buffers_and_stream_match(weight_buffer, expert_ids_buffer, input_buffer, nullptr, nullptr,
                                      output_buffer, stream)) {
        set_error("MoE grouped GEMM received invalid arguments, buffers, or stream");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    size_t weight_elements = 0u;
    size_t input_elements = 0u;
    size_t output_elements = 0u;
    size_t weight_bytes = 0u;
    size_t ids_bytes = 0u;
    size_t input_bytes = 0u;
    size_t output_bytes = 0u;
    if (!moe_checked_mul(num_experts, rows_per_expert, &weight_elements) ||
        !moe_checked_mul(weight_elements, cols, &weight_elements) ||
        !moe_checked_mul(assignments, cols, &input_elements) ||
        !moe_checked_mul(assignments, rows_per_expert, &output_elements) ||
        !moe_checked_bytes(weight_elements, moe_weight_element_bytes(weight_dtype), &weight_bytes) ||
        !moe_checked_bytes(assignments, sizeof(int32_t), &ids_bytes) ||
        !moe_checked_bytes(input_elements, sizeof(float), &input_bytes) ||
        !moe_checked_bytes(output_elements, sizeof(float), &output_bytes)) {
        set_error("MoE grouped GEMM byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < weight_bytes || expert_ids_buffer->bytes < ids_bytes ||
        input_buffer->bytes < input_bytes || output_buffer->bytes < output_bytes) {
        set_error("MoE grouped GEMM buffer is too small");
        return ULLM_STATUS_BUFFER_TOO_SMALL;
    }
    if (weight_buffer->backend == BackendKind::Cpu) {
        const void *weights = weight_buffer->ptr;
        const int32_t *ids = static_cast<const int32_t *>(expert_ids_buffer->ptr);
        const float *input = static_cast<const float *>(input_buffer->ptr);
        float *output = static_cast<float *>(output_buffer->ptr);
        for (size_t assignment = 0u; assignment < assignments; ++assignment) {
            const int32_t expert = ids[assignment];
            if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
                set_error("MoE grouped GEMM received an expert ID out of range");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
            for (size_t row = 0u; row < rows_per_expert; ++row) {
                const size_t weight_base =
                    (static_cast<size_t>(expert) * rows_per_expert + row) * cols;
                const size_t input_base = assignment * cols;
                float value = 0.0f;
                for (size_t column = 0u; column < cols; ++column) {
                    value += moe_host_weight(weights, weight_dtype, weight_base + column) *
                             input[input_base + column];
                }
                output[assignment * rows_per_expert + row] = value;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string gpu_error;
    if (!moe_gpu_grouped_gemm(weight_buffer, weight_dtype, expert_ids_buffer, input_buffer,
                              assignments, num_experts, rows_per_expert, cols, output_buffer,
                              stream, &gpu_error)) {
        set_error(gpu_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_moe_gated_silu_f32(
    const ullm_runtime_buffer *gate_up_buffer,
    size_t assignments,
    size_t intermediate_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_up_buffer == nullptr || output_buffer == nullptr || assignments == 0u ||
        intermediate_size == 0u ||
        !moe_buffers_and_stream_match(gate_up_buffer, output_buffer, nullptr, nullptr, nullptr,
                                      output_buffer, stream)) {
        set_error("MoE gated SiLU received invalid arguments, buffers, or stream");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    size_t gate_elements = 0u;
    size_t output_elements = 0u;
    size_t gate_bytes = 0u;
    size_t output_bytes = 0u;
    if (!moe_checked_mul(assignments, intermediate_size, &output_elements) ||
        !moe_checked_mul(output_elements, 2u, &gate_elements) ||
        !moe_checked_bytes(gate_elements, sizeof(float), &gate_bytes) ||
        !moe_checked_bytes(output_elements, sizeof(float), &output_bytes)) {
        set_error("MoE gated SiLU byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_up_buffer->bytes < gate_bytes || output_buffer->bytes < output_bytes) {
        set_error("MoE gated SiLU buffer is too small");
        return ULLM_STATUS_BUFFER_TOO_SMALL;
    }
    if (gate_up_buffer->backend == BackendKind::Cpu) {
        const float *gate_up = static_cast<const float *>(gate_up_buffer->ptr);
        float *output = static_cast<float *>(output_buffer->ptr);
        for (size_t assignment = 0u; assignment < assignments; ++assignment) {
            const size_t base = assignment * 2u * intermediate_size;
            for (size_t channel = 0u; channel < intermediate_size; ++channel) {
                const float gate = gate_up[base + channel];
                output[assignment * intermediate_size + channel] =
                    (gate / (1.0f + std::exp(-gate))) * gate_up[base + intermediate_size + channel];
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string gpu_error;
    if (!moe_gpu_gated_silu(gate_up_buffer, assignments, intermediate_size, output_buffer, stream,
                             &gpu_error)) {
        set_error(gpu_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_moe_scatter_weighted_f32(
    const ullm_runtime_buffer *expert_output_buffer,
    const ullm_runtime_buffer *routing_scores_buffer,
    size_t tokens,
    size_t top_k,
    size_t hidden_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (expert_output_buffer == nullptr || routing_scores_buffer == nullptr || output_buffer == nullptr ||
        tokens == 0u || top_k == 0u || hidden_size == 0u ||
        !moe_buffers_and_stream_match(expert_output_buffer, routing_scores_buffer, output_buffer, nullptr,
                                      nullptr, output_buffer, stream)) {
        set_error("MoE scatter received invalid arguments, buffers, or stream");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    size_t assignments = 0u;
    size_t expert_elements = 0u;
    size_t output_elements = 0u;
    size_t score_bytes = 0u;
    size_t expert_bytes = 0u;
    size_t output_bytes = 0u;
    if (!moe_checked_mul(tokens, top_k, &assignments) ||
        !moe_checked_mul(assignments, hidden_size, &expert_elements) ||
        !moe_checked_mul(tokens, hidden_size, &output_elements) ||
        !moe_checked_bytes(assignments, sizeof(float), &score_bytes) ||
        !moe_checked_bytes(expert_elements, sizeof(float), &expert_bytes) ||
        !moe_checked_bytes(output_elements, sizeof(float), &output_bytes)) {
        set_error("MoE scatter byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (expert_output_buffer->bytes < expert_bytes || routing_scores_buffer->bytes < score_bytes ||
        output_buffer->bytes < output_bytes) {
        set_error("MoE scatter buffer is too small");
        return ULLM_STATUS_BUFFER_TOO_SMALL;
    }
    if (expert_output_buffer->backend == BackendKind::Cpu) {
        const float *expert_output = static_cast<const float *>(expert_output_buffer->ptr);
        const float *scores = static_cast<const float *>(routing_scores_buffer->ptr);
        float *output = static_cast<float *>(output_buffer->ptr);
        for (size_t token = 0u; token < tokens; ++token) {
            const size_t assignment_base = token * top_k;
            for (size_t hidden = 0u; hidden < hidden_size; ++hidden) {
                float value = 0.0f;
                for (size_t rank = 0u; rank < top_k; ++rank) {
                    value += scores[assignment_base + rank] *
                             expert_output[(assignment_base + rank) * hidden_size + hidden];
                }
                output[token * hidden_size + hidden] = value;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string gpu_error;
    if (!moe_gpu_scatter_weighted(expert_output_buffer, routing_scores_buffer, tokens, top_k, hidden_size,
                                  output_buffer, stream, &gpu_error)) {
        set_error(gpu_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_moe_sigmoid_gate_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t tokens,
    size_t hidden_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr || tokens == 0u ||
        hidden_size == 0u ||
        !moe_buffers_and_stream_match(gate_buffer, input_buffer, output_buffer, nullptr, nullptr,
                                      output_buffer, stream)) {
        set_error("MoE sigmoid gate received invalid arguments, buffers, or stream");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    size_t elements = 0u;
    size_t gate_bytes = 0u;
    size_t data_bytes = 0u;
    if (!moe_checked_mul(tokens, hidden_size, &elements) ||
        !moe_checked_bytes(tokens, sizeof(float), &gate_bytes) ||
        !moe_checked_bytes(elements, sizeof(float), &data_bytes)) {
        set_error("MoE sigmoid gate byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer->bytes < gate_bytes || input_buffer->bytes < data_bytes ||
        output_buffer->bytes < data_bytes) {
        set_error("MoE sigmoid gate buffer is too small");
        return ULLM_STATUS_BUFFER_TOO_SMALL;
    }
    if (gate_buffer->backend == BackendKind::Cpu) {
        const float *gates = static_cast<const float *>(gate_buffer->ptr);
        const float *input = static_cast<const float *>(input_buffer->ptr);
        float *output = static_cast<float *>(output_buffer->ptr);
        for (size_t token = 0u; token < tokens; ++token) {
            const float scale = 1.0f / (1.0f + std::exp(-gates[token]));
            for (size_t hidden = 0u; hidden < hidden_size; ++hidden) {
                output[token * hidden_size + hidden] = scale * input[token * hidden_size + hidden];
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }
    std::string gpu_error;
    if (!moe_gpu_sigmoid_gate(gate_buffer, input_buffer, tokens, hidden_size, output_buffer, stream,
                              &gpu_error)) {
        set_error(gpu_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}
