// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "sq8_ck_gfx942_control.h"
#include "sq8_ck_gfx942_arch.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

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
        throw std::runtime_error(
            std::string(operation) + " failed: " + hipGetErrorString(status));
    }
}

void hipblas_check(hipblasStatus_t status, std::string_view operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with hipBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
}

void validate_gfx942_device(int device_id) {
    const char* visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (visible == nullptr || visible[0] == '\0' || std::strchr(visible, ',') != nullptr) {
        throw std::runtime_error(
            "SQ8_0 gfx942 B control requires HIP_VISIBLE_DEVICES to contain exactly one device token");
    }
    int device_count = 0;
    hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1 || device_id != 0) {
        throw std::runtime_error(
            "SQ8_0 gfx942 B control requires one visible HIP device at internal ordinal 0");
    }
    hip_check(hipSetDevice(device_id), "hipSetDevice");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, device_id), "hipGetDeviceProperties");
    properties.gcnArchName[sizeof(properties.gcnArchName) - 1u] = '\0';
    if (!ullm::sq8_ck_gfx942::is_exact_gfx942_gcn_arch_name(properties.gcnArchName)) {
        throw std::runtime_error(
            std::string("SQ8_0 gfx942 B control requires exact gcnArchName gfx942; selected ") +
            properties.gcnArchName);
    }
}

__device__ float ocp_e4m3fn_to_f32(unsigned char raw) {
    const unsigned int magnitude = raw & 0x7fu;
    if (magnitude == 0x7fu) {
        return __int_as_float(0x7fc00000u);
    }
    const float sign = (raw & 0x80u) == 0u ? 1.0f : -1.0f;
    const unsigned int exponent = (raw >> 3u) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0u) {
        return sign * static_cast<float>(mantissa) * 0.001953125f;
    }
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                         static_cast<int>(exponent) - 7);
}

__global__ void dequant_ocp_row_k128_to_bf16(
    const unsigned char* payload,
    const float* scales,
    hip_bfloat16* output,
    unsigned long long rows,
    unsigned long long cols) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = rows * cols;
    if (index >= elements) {
        return;
    }
    const unsigned long long row = index / cols;
    const unsigned long long col = index % cols;
    const unsigned long long scale_index = row * (cols / 128ull) + col / 128ull;
    output[index] = hip_bfloat16(ocp_e4m3fn_to_f32(payload[index]) * scales[scale_index]);
}

__global__ void dequant_ocp_block128x128_to_bf16(
    const unsigned char* payload,
    const float* scales,
    hip_bfloat16* output,
    unsigned long long rows,
    unsigned long long cols) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = rows * cols;
    if (index >= elements) {
        return;
    }
    const unsigned long long row = index / cols;
    const unsigned long long col = index % cols;
    const unsigned long long scale_index =
        (row / 128ull) * (cols / 128ull) + col / 128ull;
    output[index] = hip_bfloat16(ocp_e4m3fn_to_f32(payload[index]) * scales[scale_index]);
}

unsigned int dequant_grid(size_t elements, std::string_view label) {
    constexpr unsigned int kThreads = 256u;
    const size_t blocks = (elements + kThreads - 1u) / kThreads;
    if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::runtime_error(std::string(label) + " grid overflows");
    }
    return static_cast<unsigned int>(blocks);
}

} // namespace

extern "C" int ullm_sq8_ck_gfx942_control_dequant_ocp_bf16_projection(
    const void* activation_ocp_bytes,
    const void* activation_scale_f32,
    const void* weight_ocp_bytes,
    const void* weight_scale_f32,
    size_t m,
    size_t n,
    size_t k,
    void* activation_bf16,
    void* weight_bf16,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    try {
        validate_gfx942_device(device_id);
        if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            k > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("SQ8_0 gfx942 B control dimensions exceed hipBLAS int range");
        }

        constexpr unsigned int kThreads = 256u;
        hipLaunchKernelGGL(dequant_ocp_row_k128_to_bf16,
                           dim3(dequant_grid(m * k, "SQ8_0 gfx942 B activation dequant")),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const unsigned char*>(activation_ocp_bytes),
                           static_cast<const float*>(activation_scale_f32),
                           static_cast<hip_bfloat16*>(activation_bf16),
                           static_cast<unsigned long long>(m),
                           static_cast<unsigned long long>(k));
        hip_check(hipGetLastError(), "SQ8_0 gfx942 B OCP-to-BF16 activation launch");

        hipLaunchKernelGGL(dequant_ocp_block128x128_to_bf16,
                           dim3(dequant_grid(n * k, "SQ8_0 gfx942 B weight dequant")),
                           dim3(kThreads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const unsigned char*>(weight_ocp_bytes),
                           static_cast<const float*>(weight_scale_f32),
                           static_cast<hip_bfloat16*>(weight_bf16),
                           static_cast<unsigned long long>(n),
                           static_cast<unsigned long long>(k));
        hip_check(hipGetLastError(), "SQ8_0 gfx942 B OCP-to-BF16 weight launch");

        hipblasHandle_t handle = nullptr;
        hipblas_check(hipblasCreate(&handle), "hipblasCreate");
        try {
            hipblas_check(hipblasSetStream(handle, static_cast<hipStream_t>(stream)),
                          "hipblasSetStream");
            // Column-major hipBLAS sees row-major B[N,K] as KxN and row-major
            // A[M,K] as KxM.  B * A^T therefore lands in C[N,M], whose memory
            // is exactly the desired row-major C[M,N].
            const float alpha = 1.0f;
            const float beta = 0.0f;
            hipblas_check(hipblasGemmEx(handle,
                                        HIPBLAS_OP_N,
                                        HIPBLAS_OP_N,
                                        static_cast<int>(n),
                                        static_cast<int>(m),
                                        static_cast<int>(k),
                                        &alpha,
                                        weight_bf16,
                                        HIP_R_16BF,
                                        static_cast<int>(n),
                                        activation_bf16,
                                        HIP_R_16BF,
                                        static_cast<int>(k),
                                        &beta,
                                        output_f32,
                                        HIP_R_32F,
                                        static_cast<int>(n),
                                        HIPBLAS_COMPUTE_32F,
                                        HIPBLAS_GEMM_DEFAULT),
                          "hipblasGemmEx SQ8_0 gfx942 B BF16 GEMM");
            hipblas_check(hipblasDestroy(handle), "hipblasDestroy");
        } catch (...) {
            (void)hipblasDestroy(handle);
            throw;
        }
        hip_check(hipGetLastError(), "SQ8_0 gfx942 B BF16 GEMM launch");
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "SQ8_0 gfx942 B control projection failed");
        return 0;
    }
}
