// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "sq8_handwritten_gfx1201.h"

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr unsigned int kWmmaTile = 16u;
constexpr unsigned int kScaleBlock = 128u;
constexpr int kLaunchThreadsPerBlock = 32;

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

void validate_device(int device_id) {
    const char* visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (visible == nullptr || visible[0] == '\0' || std::strchr(visible, ',') != nullptr) {
        throw std::runtime_error(
            "SQ8 handwritten WMMA requires HIP_VISIBLE_DEVICES to contain exactly one device token");
    }
    int device_count = 0;
    hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1 || device_id != 0) {
        throw std::runtime_error(
            "SQ8 handwritten WMMA requires one visible HIP device at internal ordinal 0");
    }
    hip_check(hipSetDevice(device_id), "hipSetDevice");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, device_id), "hipGetDeviceProperties");
    const bool is_gfx1201 =
        std::strncmp(properties.gcnArchName, "gfx1201", 7u) == 0 &&
        (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':');
    if (!is_gfx1201 || properties.major != 12 || properties.minor != 0) {
        throw std::runtime_error(
            std::string("SQ8 handwritten WMMA requires gfx1201 compute 12.0; selected ") +
            properties.gcnArchName);
    }
}

bool is_exact_model_shape(size_t n, size_t k) {
    return (n == 5120u && k == 5120u) || (n == 1024u && k == 5120u) ||
           (n == 17408u && k == 5120u) || (n == 5120u && k == 17408u);
}

[[maybe_unused]] __device__ float bf16_roundtrip(float value) {
    // Match the existing CK route's BF16 workspace boundary before its F32
    // conversion.  Finite normal model values use IEEE round-to-nearest-even.
    const unsigned int bits = __float_as_uint(value);
    const unsigned int exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return value;
    }
    const unsigned int rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(rounded & 0xffff0000u);
}

/*
 * One wave computes a 16-output-N tile for one M=1 row.  The B operand is a
 * replicated 16-column view of the activation row: column zero is the scalar
 * result we retain, while the other columns only make the WMMA matrix shape
 * legal.  No K reduction is split across CTAs or waves.
 *
 * LDS: 256 B activation tile + 1024 B fragment store = 1280 B static.  The
 * fragment store is deliberately retained for this feasibility prototype so
 * that rocWMMA owns the opaque accumulator-lane layout.
 */
extern "C" __global__ void ullm_sq8_handwritten_gfx1201_m1_wmma_kernel(
    const unsigned char* activation,
    const float* activation_scales,
    const unsigned char* weight,
    const float* weight_scales,
    unsigned long long n,
    unsigned long long k,
    float* output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    using FragA = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float32_t, row_major>;

    const unsigned int lane = threadIdx.x;
    const unsigned long long n_tile = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long n_base = n_tile * kWmmaTile;
    if (lane >= 32u || n_base + kWmmaTile > n) {
        return;
    }

    __shared__ unsigned char activation_tile[kWmmaTile * kWmmaTile];
    __shared__ float fragment_tile[kWmmaTile * kWmmaTile];

    float scaled_total = 0.0f;
    const unsigned long long k_blocks = k / kScaleBlock;
    const unsigned long long n_scale_row = n_base / kScaleBlock;

    for (unsigned long long k_block = 0ull; k_block < k_blocks; ++k_block) {
        FragC raw_block;
        fill_fragment(raw_block, 0.0f);

        for (unsigned int sub_tile = 0u; sub_tile < kScaleBlock / kWmmaTile; ++sub_tile) {
            const unsigned long long k_base =
                k_block * kScaleBlock + static_cast<unsigned long long>(sub_tile) * kWmmaTile;
            const unsigned char value = activation[k_base + static_cast<unsigned long long>(lane & 15u)];
            // Column-major [K=16,N=16]: every column is the same activation vector.
            for (unsigned int column_pair = 0u; column_pair < 8u; ++column_pair) {
                activation_tile[lane + column_pair * 32u] = value;
            }
            __syncthreads();

            FragA a;
            FragB b;
            load_matrix_sync(
                a,
                reinterpret_cast<const float8_t*>(weight + n_base * k + k_base),
                static_cast<unsigned int>(k));
            load_matrix_sync(
                b,
                reinterpret_cast<const float8_t*>(activation_tile),
                kWmmaTile);
            mma_sync(raw_block, a, b, raw_block);
            // A second barrier prevents a lane from overwriting the shared B
            // tile while another lane still consumes it in the WMMA issue.
            __syncthreads();
        }

        store_matrix_sync(fragment_tile, raw_block, kWmmaTile);
        __syncthreads();
        if (lane < kWmmaTile) {
            const float scale = activation_scales[k_block] *
                                weight_scales[n_scale_row * k_blocks + k_block];
            scaled_total += fragment_tile[lane * kWmmaTile] * scale;
        }
        __syncthreads();
    }

    if (lane < kWmmaTile) {
        output[n_base + static_cast<unsigned long long>(lane)] = bf16_roundtrip(scaled_total);
    }
#else
    (void)activation;
    (void)activation_scales;
    (void)weight;
    (void)weight_scales;
    (void)n;
    (void)k;
    (void)output;
#endif
}

} // namespace

extern "C" int ullm_sq8_handwritten_gfx1201_m1_wmma_projection(
    const void* quantized_activation_ocp_e4m3,
    const void* activation_scale_f32,
    const void* weight_ocp_e4m3,
    const void* weight_scale_f32,
    size_t n,
    size_t k,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    try {
        if (quantized_activation_ocp_e4m3 == nullptr || activation_scale_f32 == nullptr ||
            weight_ocp_e4m3 == nullptr || weight_scale_f32 == nullptr || output_f32 == nullptr) {
            throw std::runtime_error("SQ8 handwritten WMMA projection received a null pointer");
        }
        if (!is_exact_model_shape(n, k) || n % kWmmaTile != 0u || k % kScaleBlock != 0u) {
            throw std::runtime_error(
                "SQ8 handwritten WMMA projection requires an exact Qwen3-14B SQ8_0 M=1 shape");
        }
        if (n / kWmmaTile > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::runtime_error("SQ8 handwritten WMMA projection grid overflows");
        }
        validate_device(device_id);
        hipLaunchKernelGGL(ullm_sq8_handwritten_gfx1201_m1_wmma_kernel,
                           dim3(static_cast<unsigned int>(n / kWmmaTile)),
                           dim3(static_cast<unsigned int>(kLaunchThreadsPerBlock)),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const unsigned char*>(quantized_activation_ocp_e4m3),
                           static_cast<const float*>(activation_scale_f32),
                           static_cast<const unsigned char*>(weight_ocp_e4m3),
                           static_cast<const float*>(weight_scale_f32),
                           static_cast<unsigned long long>(n),
                           static_cast<unsigned long long>(k),
                           static_cast<float*>(output_f32));
        hip_check(hipGetLastError(), "SQ8 handwritten WMMA projection launch");
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "SQ8 handwritten WMMA projection failed");
        return 0;
    }
}

extern "C" int ullm_sq8_handwritten_gfx1201_m1_wmma_resources(
    int device_id,
    uint32_t* vgpr_per_thread,
    size_t* static_lds_bytes,
    size_t* local_bytes_per_thread,
    int* threads_per_block,
    int* active_blocks_per_cu,
    char* error,
    size_t error_capacity) {
    try {
        if (vgpr_per_thread == nullptr || static_lds_bytes == nullptr ||
            local_bytes_per_thread == nullptr || threads_per_block == nullptr ||
            active_blocks_per_cu == nullptr) {
            throw std::runtime_error("SQ8 handwritten WMMA resource query received a null output");
        }
        validate_device(device_id);
        hipFuncAttributes attributes{};
        hip_check(hipFuncGetAttributes(
                      &attributes,
                      reinterpret_cast<const void*>(
                          ullm_sq8_handwritten_gfx1201_m1_wmma_kernel)),
                  "hipFuncGetAttributes");
        int blocks = 0;
        hip_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                      &blocks,
                      ullm_sq8_handwritten_gfx1201_m1_wmma_kernel,
                      kLaunchThreadsPerBlock,
                      0u),
                  "hipOccupancyMaxActiveBlocksPerMultiprocessor");
        *vgpr_per_thread = static_cast<uint32_t>(attributes.numRegs);
        *static_lds_bytes = attributes.sharedSizeBytes;
        *local_bytes_per_thread = attributes.localSizeBytes;
        // `maxThreadsPerBlock` is an architecture capability (1024 here),
        // not this private kernel's launch geometry. Report the latter.
        *threads_per_block = kLaunchThreadsPerBlock;
        *active_blocks_per_cu = blocks;
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "SQ8 handwritten WMMA resource query failed");
        return 0;
    }
}
