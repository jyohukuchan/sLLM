// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "sq8_ck_gfx1201.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace {

using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<
    RowMajor,
    ColumnMajor,
    ck::Tuple<>,
    RowMajor,
    ck::f8_t,
    float,
    ck::f8_t,
    float,
    ck::Tuple<>,
    ck::bhalf_t,
    1,
    128,
    128,
    PassThrough,
    PassThrough,
    PassThrough>;

static_assert(sizeof(ck::f8_t) == 1);
static_assert(sizeof(ck::bhalf_t) == 2);

constexpr uint32_t kDefault128 = 1u;
constexpr uint32_t kKPadding256 = 2u;
constexpr uint32_t kDefault256x128 = 3u;
constexpr uint32_t kDefault128x256 = 4u;

constexpr std::string_view kDefault128Type =
    "DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x128x128, "
    "WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 8x16, "
    "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, "
    "BlkGemmPipelinePrefetchStages: 2";
constexpr std::string_view kKPadding256Type =
    "DeviceGemmXdlUniversal<KPadding, RCR> BlkSize: 256, BlkTile: 16x128x256, "
    "WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 16x16, "
    "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, "
    "BlkGemmPipelinePrefetchStages: 2";
constexpr std::string_view kDefault256x128Type =
    "DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x256x128, "
    "WaveTile: 16x16, WaveMap: 1x4, VmemReadVec: 8x16, "
    "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, "
    "BlkGemmPipelinePrefetchStages: 2";
constexpr std::string_view kDefault128x256Type =
    "DeviceGemmXdlUniversal<Default, RCR> BlkSize: 256, BlkTile: 16x128x256, "
    "WaveTile: 16x16, WaveMap: 1x2, VmemReadVec: 16x16, "
    "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, "
    "BlkGemmPipelinePrefetchStages: 2";

void write_error(char *error, size_t capacity, std::string_view message) noexcept {
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
    const char *visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (visible == nullptr || visible[0] == '\0' || std::strchr(visible, ',') != nullptr) {
        throw std::runtime_error(
            "SQ8 CK requires HIP_VISIBLE_DEVICES to contain exactly one device token");
    }
    int device_count = 0;
    hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1 || device_id != 0) {
        throw std::runtime_error("SQ8 CK requires one visible HIP device at internal ordinal 0");
    }
    hip_check(hipSetDevice(device_id), "hipSetDevice");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, device_id), "hipGetDeviceProperties");
    if (std::strncmp(properties.gcnArchName, "gfx1201", 7u) != 0 ||
        properties.major != 12 || properties.minor != 0) {
        throw std::runtime_error(
            std::string("SQ8 CK requires gfx1201 compute 12.0; selected ") +
            properties.gcnArchName);
    }
}

__device__ float positive_ocp_e4m3_to_f32(unsigned int code) {
    const unsigned int exponent = (code >> 3u) & 15u;
    const unsigned int mantissa = code & 7u;
    if (exponent == 0u) {
        return static_cast<float>(mantissa) * 0.001953125f;
    }
    return ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                  static_cast<int>(exponent) - 7);
}

__device__ unsigned char f32_to_ocp_e4m3_rne(float value, float scale) {
    union {
        float as_float;
        unsigned int as_bits;
    } bits = {value};
    const unsigned int sign = (bits.as_bits >> 24u) & 0x80u;
    const float magnitude = fabsf(value / scale);
    if (magnitude == 0.0f) {
        return static_cast<unsigned char>(sign);
    }
    if (magnitude >= 448.0f) {
        return static_cast<unsigned char>(sign | 0x7eu);
    }

    unsigned int lower_bound = 0u;
    unsigned int upper_bound = 127u;
    while (lower_bound < upper_bound) {
        const unsigned int middle = lower_bound + (upper_bound - lower_bound) / 2u;
        if (positive_ocp_e4m3_to_f32(middle) < magnitude) {
            lower_bound = middle + 1u;
        } else {
            upper_bound = middle;
        }
    }
    const unsigned int upper = lower_bound;
    const float upper_value = positive_ocp_e4m3_to_f32(upper);
    if (upper_value == magnitude) {
        return static_cast<unsigned char>(sign | upper);
    }
    const unsigned int lower = upper - 1u;
    const double lower_distance =
        static_cast<double>(magnitude) - static_cast<double>(positive_ocp_e4m3_to_f32(lower));
    const double upper_distance = static_cast<double>(upper_value) - static_cast<double>(magnitude);
    const unsigned int encoded =
        lower_distance < upper_distance
            ? lower
            : (upper_distance < lower_distance ? upper : ((lower & 1u) == 0u ? lower : upper));
    return static_cast<unsigned char>(sign | encoded);
}

__global__ void quantize_activation_block128(
    const float *input,
    unsigned char *output,
    float *scales,
    unsigned long long k) {
    __shared__ float absolute_values[128];
    __shared__ float block_scale;

    const unsigned long long block = static_cast<unsigned long long>(blockIdx.x);
    const unsigned long long lane = static_cast<unsigned long long>(threadIdx.x);
    const unsigned long long k_blocks = k / 128ull;
    const unsigned long long row = block / k_blocks;
    const unsigned long long block_k = block % k_blocks;
    const unsigned long long element = row * k + block_k * 128ull + lane;

    absolute_values[lane] = fabsf(input[element]);
    __syncthreads();
    for (unsigned int stride = 64u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            absolute_values[lane] =
                fmaxf(absolute_values[lane], absolute_values[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0ull) {
        const float maximum = absolute_values[0];
        block_scale =
            maximum == 0.0f ? 1.0f : fmaxf(maximum / 448.0f, __uint_as_float(1u));
        scales[block] = block_scale;
    }
    __syncthreads();
    output[element] = f32_to_ocp_e4m3_rne(input[element], block_scale);
}

__global__ void bf16_to_f32(
    const std::uint16_t *input,
    float *output,
    unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __uint_as_float(static_cast<unsigned int>(input[index]) << 16u);
    }
}

struct Registry {
    std::vector<std::unique_ptr<DeviceOp>> default_instances;
    std::vector<std::unique_ptr<DeviceOp>> kpadding_instances;

    Registry() {
        using namespace ck::tensor_operation::device::instance;
        add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
            default_instances);
        add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
            kpadding_instances);
    }
};

Registry &registry() {
    static Registry value;
    return value;
}

std::pair<std::vector<std::unique_ptr<DeviceOp>> *, std::string_view> dispatch(uint32_t id) {
    switch (id) {
    case kDefault128:
        return {&registry().default_instances, kDefault128Type};
    case kKPadding256:
        return {&registry().kpadding_instances, kKPadding256Type};
    case kDefault256x128:
        return {&registry().default_instances, kDefault256x128Type};
    case kDefault128x256:
        return {&registry().default_instances, kDefault128x256Type};
    default:
        throw std::runtime_error("SQ8 CK received an unknown implementation id");
    }
}

DeviceOp &select_operation(uint32_t implementation) {
    auto [instances, expected_type] = dispatch(implementation);
    DeviceOp *selected = nullptr;
    size_t matches = 0u;
    for (const auto &instance : *instances) {
        if (instance->GetTypeString() == expected_type) {
            selected = instance.get();
            ++matches;
        }
    }
    if (matches != 1u || selected == nullptr) {
        throw std::runtime_error(
            "SQ8 CK measured GetTypeString did not resolve to exactly one instance: " +
            std::string(expected_type));
    }
    return *selected;
}

} // namespace

extern "C" int ullm_sq8_ck_gfx1201_quantize_activation(
    const void *input,
    void *quantized,
    void *scales,
    size_t m,
    size_t k,
    void *stream,
    int device_id,
    char *error,
    size_t error_capacity) {
    try {
        validate_device(device_id);
        const size_t blocks = m * (k / 128u);
        if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::runtime_error("SQ8 CK activation quantization grid overflows");
        }
        hipLaunchKernelGGL(quantize_activation_block128,
                           dim3(static_cast<unsigned int>(blocks)),
                           dim3(128u),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const float *>(input),
                           static_cast<unsigned char *>(quantized),
                           static_cast<float *>(scales),
                           static_cast<unsigned long long>(k));
        hip_check(hipGetLastError(), "SQ8 CK activation quantization launch");
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception &exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "SQ8 CK activation quantization failed");
        return 0;
    }
}

extern "C" int ullm_sq8_ck_gfx1201_projection(
    const void *quantized_activation,
    const void *activation_scales,
    const void *weight,
    const void *weight_scales,
    size_t m,
    size_t n,
    size_t k,
    void *workspace,
    void *output,
    void *stream,
    int device_id,
    uint32_t implementation,
    char *error,
    size_t error_capacity) {
    try {
        validate_device(device_id);
        DeviceOp &operation = select_operation(implementation);
        auto argument = operation.MakeArgumentPointer(
            quantized_activation,
            weight,
            std::array<const void *, 0>{},
            workspace,
            static_cast<ck::index_t>(m),
            static_cast<ck::index_t>(n),
            static_cast<ck::index_t>(k),
            static_cast<ck::index_t>(k),
            static_cast<ck::index_t>(k),
            std::array<ck::index_t, 0>{},
            static_cast<ck::index_t>(n),
            activation_scales,
            weight_scales,
            PassThrough{},
            PassThrough{},
            PassThrough{});
        if (!operation.IsSupportedArgument(argument.get())) {
            throw std::runtime_error(
                "SQ8 CK measured instance rejected the projection argument");
        }
        auto invoker = operation.MakeInvokerPointer();
        StreamConfig config;
        config.stream_id_ = static_cast<hipStream_t>(stream);
        config.time_kernel_ = false;
        config.log_level_ = 0;
        config.flush_cache = false;
        (void)invoker->Run(argument.get(), config);
        hip_check(hipGetLastError(), "SQ8 CK ABScale GEMM launch");

        const size_t elements = m * n;
        constexpr unsigned int threads = 256u;
        const size_t blocks = (elements + threads - 1u) / threads;
        if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::runtime_error("SQ8 CK BF16 conversion grid overflows");
        }
        hipLaunchKernelGGL(bf16_to_f32,
                           dim3(static_cast<unsigned int>(blocks)),
                           dim3(threads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const std::uint16_t *>(workspace),
                           static_cast<float *>(output),
                           static_cast<unsigned long long>(elements));
        hip_check(hipGetLastError(), "SQ8 CK BF16-to-F32 launch");
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception &exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "SQ8 CK projection failed");
        return 0;
    }
}
