// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "moe_gfx1201.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr unsigned int kThreads = 256u;
constexpr unsigned int kMaxExperts = 256u;
constexpr uint32_t kWeightF32 = 0u;
constexpr uint32_t kWeightBf16 = 1u;

void write_error(char* error, size_t capacity, std::string_view message) noexcept {
    if (error == nullptr || capacity == 0u) {
        return;
    }
    const size_t count = message.size() < capacity - 1u ? message.size() : capacity - 1u;
    std::memcpy(error, message.data(), count);
    error[count] = '\0';
}

void hip_check(hipError_t status, std::string_view operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + hipGetErrorString(status));
    }
}

void validate_device(int device_id) {
    hip_check(hipSetDevice(device_id), "hipSetDevice");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, device_id), "hipGetDeviceProperties");
    const bool is_gfx1201 =
        std::strncmp(properties.gcnArchName, "gfx1201", 7u) == 0 &&
        (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':');
    if (!is_gfx1201 || properties.major != 12 || properties.minor != 0) {
        throw std::runtime_error(
            std::string("MoE static GPU kernels require gfx1201 compute 12.0; selected ") +
            properties.gcnArchName);
    }
}

void validate_weight_dtype(uint32_t weight_dtype) {
    if (weight_dtype != kWeightF32 && weight_dtype != kWeightBf16) {
        throw std::runtime_error("MoE weight dtype must be F32 or raw IEEE BF16");
    }
}

unsigned int grid_for(size_t elements) {
    if (elements == 0u) {
        throw std::runtime_error("MoE GPU launch requires a nonzero element count");
    }
    const size_t grid = (elements + kThreads - 1u) / kThreads;
    if (grid > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::runtime_error("MoE GPU grid exceeds HIP unsigned-int limit");
    }
    return static_cast<unsigned int>(grid);
}

size_t checked_mul(size_t lhs, size_t rhs, std::string_view label) {
    if (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::runtime_error(std::string(label) + " overflows size_t");
    }
    return lhs * rhs;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ float f32_to_bf16_roundtrip(float value) {
    const unsigned int bits = __float_as_uint(value);
    const unsigned int exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return value;
    }
    const unsigned int rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(rounded & 0xffff0000u);
}

__device__ __forceinline__ float weight_at(const void* weights, uint32_t weight_dtype, size_t index) {
    if (weight_dtype == kWeightF32) {
        return static_cast<const float*>(weights)[index];
    }
    return bf16_to_f32(static_cast<const uint16_t*>(weights)[index]);
}

extern "C" __global__ void ullm_moe_route_f32_kernel(
    const float* hidden,
    const void* router_weights,
    uint32_t weight_dtype,
    unsigned long long tokens,
    unsigned long long hidden_size,
    unsigned int num_experts,
    unsigned int top_k,
    float* routing_scores,
    int32_t* selected_expert_ids,
    uint32_t* boundary_tie_flags) {
    const unsigned long long token = static_cast<unsigned long long>(blockIdx.x);
    if (token >= tokens) {
        return;
    }

    __shared__ float probabilities[kMaxExperts];
    __shared__ int32_t selected[kMaxExperts];

    const unsigned int expert = threadIdx.x;
    if (expert < num_experts) {
        float total = 0.0f;
        const unsigned long long input_base = token * hidden_size;
        const unsigned long long weight_base = static_cast<unsigned long long>(expert) * hidden_size;
        for (unsigned long long column = 0ull; column < hidden_size; ++column) {
            const float activation = weight_dtype == kWeightBf16
                ? f32_to_bf16_roundtrip(hidden[input_base + column])
                : hidden[input_base + column];
            total += activation *
                     weight_at(router_weights, weight_dtype, weight_base + column);
        }
        probabilities[expert] = weight_dtype == kWeightBf16 ? f32_to_bf16_roundtrip(total) : total;
    }
    __syncthreads();

    if (expert != 0u) {
        return;
    }

    float maximum = probabilities[0];
    for (unsigned int index = 1u; index < num_experts; ++index) {
        if (probabilities[index] > maximum) {
            maximum = probabilities[index];
        }
    }
    float denominator = 0.0f;
    for (unsigned int index = 0u; index < num_experts; ++index) {
        const float probability = expf(probabilities[index] - maximum);
        probabilities[index] = probability;
        denominator += probability;
    }
    for (unsigned int index = 0u; index < num_experts; ++index) {
        probabilities[index] /= denominator;
        selected[index] = -1;
    }

    float selected_sum = 0.0f;
    for (unsigned int rank = 0u; rank < top_k; ++rank) {
        float best = -3.402823466e+38F;
        int32_t best_index = -1;
        for (unsigned int index = 0u; index < num_experts; ++index) {
            bool already_selected = false;
            for (unsigned int prior = 0u; prior < rank; ++prior) {
                already_selected = selected[prior] == static_cast<int32_t>(index);
                if (already_selected) {
                    break;
                }
            }
            const float candidate = probabilities[index];
            if (!already_selected &&
                (candidate > best ||
                 (candidate == best && static_cast<int32_t>(index) > best_index))) {
                best = candidate;
                best_index = static_cast<int32_t>(index);
            }
        }
        selected[rank] = best_index;
        selected_sum += best;
    }

    const float boundary = probabilities[static_cast<unsigned int>(selected[top_k - 1u])];
    uint32_t boundary_tie = 0u;
    for (unsigned int index = 0u; index < num_experts; ++index) {
        bool is_selected = false;
        for (unsigned int rank = 0u; rank < top_k; ++rank) {
            if (selected[rank] == static_cast<int32_t>(index)) {
                is_selected = true;
                break;
            }
        }
        if (!is_selected && probabilities[index] == boundary) {
            boundary_tie = 1u;
            break;
        }
    }
    boundary_tie_flags[token] = boundary_tie;

    const unsigned long long output_base = token * static_cast<unsigned long long>(top_k);
    for (unsigned int rank = 0u; rank < top_k; ++rank) {
        const int32_t index = selected[rank];
        selected_expert_ids[output_base + rank] = index;
        const float score = probabilities[static_cast<unsigned int>(index)] / selected_sum;
        routing_scores[output_base + rank] =
            weight_dtype == kWeightBf16 ? f32_to_bf16_roundtrip(score) : score;
    }
}

extern "C" __global__ void ullm_moe_gather_f32_kernel(
    const float* hidden,
    unsigned long long assignments,
    unsigned long long hidden_size,
    unsigned int top_k,
    float* gathered_hidden) {
    const unsigned long long element =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = assignments * hidden_size;
    if (element >= total) {
        return;
    }
    const unsigned long long assignment = element / hidden_size;
    const unsigned long long column = element % hidden_size;
    const unsigned long long token = assignment / static_cast<unsigned long long>(top_k);
    gathered_hidden[element] = hidden[token * hidden_size + column];
}

extern "C" __global__ void ullm_moe_grouped_gemm_f32_kernel(
    const void* weights,
    uint32_t weight_dtype,
    const int32_t* expert_ids,
    const float* input,
    unsigned long long assignments,
    unsigned int num_experts,
    unsigned long long rows_per_expert,
    unsigned long long cols,
    float* output) {
    const unsigned long long element =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = assignments * rows_per_expert;
    if (element >= total) {
        return;
    }
    const unsigned long long assignment = element / rows_per_expert;
    const unsigned long long row = element % rows_per_expert;
    const int32_t expert = expert_ids[assignment];
    if (expert < 0 || static_cast<unsigned int>(expert) >= num_experts) {
        output[element] = nanf("");
        return;
    }
    const unsigned long long weight_base =
        (static_cast<unsigned long long>(expert) * rows_per_expert + row) * cols;
    const unsigned long long input_base = assignment * cols;
    float total_value = 0.0f;
    for (unsigned long long column = 0ull; column < cols; ++column) {
        total_value += weight_at(weights, weight_dtype, weight_base + column) *
                       input[input_base + column];
    }
    output[element] = total_value;
}

extern "C" __global__ void ullm_moe_gated_silu_f32_kernel(
    const float* gate_up,
    unsigned long long assignments,
    unsigned long long intermediate_size,
    float* output) {
    const unsigned long long element =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = assignments * intermediate_size;
    if (element >= total) {
        return;
    }
    const unsigned long long assignment = element / intermediate_size;
    const unsigned long long channel = element % intermediate_size;
    const unsigned long long base = assignment * 2ull * intermediate_size;
    const float gate = gate_up[base + channel];
    const float up = gate_up[base + intermediate_size + channel];
    output[element] = (gate / (1.0f + expf(-gate))) * up;
}

extern "C" __global__ void ullm_moe_scatter_weighted_f32_kernel(
    const float* expert_output,
    const float* routing_scores,
    unsigned long long tokens,
    unsigned int top_k,
    unsigned long long hidden_size,
    float* output) {
    const unsigned long long element =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = tokens * hidden_size;
    if (element >= total) {
        return;
    }
    const unsigned long long token = element / hidden_size;
    const unsigned long long column = element % hidden_size;
    float total_value = 0.0f;
    const unsigned long long assignment_base = token * static_cast<unsigned long long>(top_k);
    for (unsigned int rank = 0u; rank < top_k; ++rank) {
        total_value += routing_scores[assignment_base + rank] *
                       expert_output[(assignment_base + rank) * hidden_size + column];
    }
    output[element] = total_value;
}

extern "C" __global__ void ullm_moe_sigmoid_gate_f32_kernel(
    const float* gate,
    const float* input,
    unsigned long long tokens,
    unsigned long long hidden_size,
    float* output) {
    const unsigned long long element =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = tokens * hidden_size;
    if (element >= total) {
        return;
    }
    const unsigned long long token = element / hidden_size;
    const float scale = 1.0f / (1.0f + expf(-gate[token]));
    output[element] = scale * input[element];
}

template <typename Launch>
int launch_with_error(Launch&& launch, char* error, size_t error_capacity) {
    try {
        launch();
        hip_check(hipGetLastError(), "MoE GPU kernel launch");
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "MoE GPU kernel launch failed");
        return 0;
    }
}

} // namespace

extern "C" int ullm_moe_gfx1201_route_f32(
    const void* hidden_f32,
    const void* router_weights,
    uint32_t weight_dtype,
    size_t tokens,
    size_t hidden_size,
    size_t num_experts,
    size_t top_k,
    void* routing_scores_f32,
    void* selected_expert_ids_i32,
    void* boundary_tie_flags_u32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    return launch_with_error([&] {
        if (hidden_f32 == nullptr || router_weights == nullptr || routing_scores_f32 == nullptr ||
            selected_expert_ids_i32 == nullptr || boundary_tie_flags_u32 == nullptr) {
            throw std::runtime_error("MoE route received a null device pointer");
        }
        validate_weight_dtype(weight_dtype);
        if (tokens == 0u || hidden_size == 0u || num_experts == 0u ||
            num_experts > kMaxExperts || top_k == 0u || top_k > num_experts) {
            throw std::runtime_error("MoE route dimensions are unsupported by the gfx1201 baseline");
        }
        if (tokens > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::runtime_error("MoE route token count exceeds HIP grid limit");
        }
        validate_device(device_id);
        hipLaunchKernelGGL(ullm_moe_route_f32_kernel,
                           dim3(static_cast<unsigned int>(tokens)),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const float*>(hidden_f32),
                           router_weights,
                           weight_dtype,
                           static_cast<unsigned long long>(tokens),
                           static_cast<unsigned long long>(hidden_size),
                           static_cast<unsigned int>(num_experts),
                           static_cast<unsigned int>(top_k),
                           static_cast<float*>(routing_scores_f32),
                           static_cast<int32_t*>(selected_expert_ids_i32),
                           static_cast<uint32_t*>(boundary_tie_flags_u32));
    }, error, error_capacity);
}

extern "C" int ullm_moe_gfx1201_gather_f32(
    const void* hidden_f32,
    size_t tokens,
    size_t hidden_size,
    size_t top_k,
    void* gathered_hidden_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    return launch_with_error([&] {
        if (hidden_f32 == nullptr || gathered_hidden_f32 == nullptr) {
            throw std::runtime_error("MoE gather received a null device pointer");
        }
        if (tokens == 0u || hidden_size == 0u || top_k == 0u) {
            throw std::runtime_error("MoE gather dimensions must be nonzero");
        }
        const size_t assignments = checked_mul(tokens, top_k, "MoE gather assignments");
        const size_t elements = checked_mul(assignments, hidden_size, "MoE gather elements");
        validate_device(device_id);
        hipLaunchKernelGGL(ullm_moe_gather_f32_kernel,
                           dim3(grid_for(elements)),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const float*>(hidden_f32),
                           static_cast<unsigned long long>(assignments),
                           static_cast<unsigned long long>(hidden_size),
                           static_cast<unsigned int>(top_k),
                           static_cast<float*>(gathered_hidden_f32));
    }, error, error_capacity);
}

extern "C" int ullm_moe_gfx1201_grouped_gemm_f32(
    const void* weights,
    uint32_t weight_dtype,
    const void* expert_ids_i32,
    const void* input_f32,
    size_t assignments,
    size_t num_experts,
    size_t rows_per_expert,
    size_t cols,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    return launch_with_error([&] {
        if (weights == nullptr || expert_ids_i32 == nullptr || input_f32 == nullptr || output_f32 == nullptr) {
            throw std::runtime_error("MoE grouped GEMM received a null device pointer");
        }
        validate_weight_dtype(weight_dtype);
        if (assignments == 0u || num_experts == 0u || num_experts > kMaxExperts ||
            rows_per_expert == 0u || cols == 0u) {
            throw std::runtime_error("MoE grouped GEMM dimensions are unsupported by the gfx1201 baseline");
        }
        const size_t elements = checked_mul(assignments, rows_per_expert, "MoE grouped GEMM elements");
        validate_device(device_id);
        hipLaunchKernelGGL(ullm_moe_grouped_gemm_f32_kernel,
                           dim3(grid_for(elements)),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           weights,
                           weight_dtype,
                           static_cast<const int32_t*>(expert_ids_i32),
                           static_cast<const float*>(input_f32),
                           static_cast<unsigned long long>(assignments),
                           static_cast<unsigned int>(num_experts),
                           static_cast<unsigned long long>(rows_per_expert),
                           static_cast<unsigned long long>(cols),
                           static_cast<float*>(output_f32));
    }, error, error_capacity);
}

extern "C" int ullm_moe_gfx1201_gated_silu_f32(
    const void* gate_up_f32,
    size_t assignments,
    size_t intermediate_size,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    return launch_with_error([&] {
        if (gate_up_f32 == nullptr || output_f32 == nullptr) {
            throw std::runtime_error("MoE gated SiLU received a null device pointer");
        }
        if (assignments == 0u || intermediate_size == 0u) {
            throw std::runtime_error("MoE gated SiLU dimensions must be nonzero");
        }
        const size_t elements = checked_mul(assignments, intermediate_size, "MoE gated SiLU elements");
        validate_device(device_id);
        hipLaunchKernelGGL(ullm_moe_gated_silu_f32_kernel,
                           dim3(grid_for(elements)),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const float*>(gate_up_f32),
                           static_cast<unsigned long long>(assignments),
                           static_cast<unsigned long long>(intermediate_size),
                           static_cast<float*>(output_f32));
    }, error, error_capacity);
}

extern "C" int ullm_moe_gfx1201_scatter_weighted_f32(
    const void* expert_output_f32,
    const void* routing_scores_f32,
    size_t tokens,
    size_t top_k,
    size_t hidden_size,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    return launch_with_error([&] {
        if (expert_output_f32 == nullptr || routing_scores_f32 == nullptr || output_f32 == nullptr) {
            throw std::runtime_error("MoE scatter received a null device pointer");
        }
        if (tokens == 0u || top_k == 0u || hidden_size == 0u) {
            throw std::runtime_error("MoE scatter dimensions must be nonzero");
        }
        const size_t elements = checked_mul(tokens, hidden_size, "MoE scatter elements");
        validate_device(device_id);
        hipLaunchKernelGGL(ullm_moe_scatter_weighted_f32_kernel,
                           dim3(grid_for(elements)),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const float*>(expert_output_f32),
                           static_cast<const float*>(routing_scores_f32),
                           static_cast<unsigned long long>(tokens),
                           static_cast<unsigned int>(top_k),
                           static_cast<unsigned long long>(hidden_size),
                           static_cast<float*>(output_f32));
    }, error, error_capacity);
}

extern "C" int ullm_moe_gfx1201_sigmoid_gate_f32(
    const void* gate_f32,
    const void* input_f32,
    size_t tokens,
    size_t hidden_size,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    return launch_with_error([&] {
        if (gate_f32 == nullptr || input_f32 == nullptr || output_f32 == nullptr) {
            throw std::runtime_error("MoE sigmoid gate received a null device pointer");
        }
        if (tokens == 0u || hidden_size == 0u) {
            throw std::runtime_error("MoE sigmoid gate dimensions must be nonzero");
        }
        const size_t elements = checked_mul(tokens, hidden_size, "MoE sigmoid gate elements");
        validate_device(device_id);
        hipLaunchKernelGGL(ullm_moe_sigmoid_gate_f32_kernel,
                           dim3(grid_for(elements)),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const float*>(gate_f32),
                           static_cast<const float*>(input_f32),
                           static_cast<unsigned long long>(tokens),
                           static_cast<unsigned long long>(hidden_size),
                           static_cast<float*>(output_f32));
    }, error, error_capacity);
}
