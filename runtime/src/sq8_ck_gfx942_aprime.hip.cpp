// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "sq8_ck_gfx942_aprime.h"
#include "sq8_ck_gfx942_arch.h"

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

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
#include <type_traits>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/amd_ck_fp8.hpp"

namespace {

using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// This is intentionally the installed archive's OCP type identity.  The
// pointer values passed to it below are not OCP values: they are FNUZ-derived
// raw byte buffers, deliberately treated as opaque by this ABI bridge.
using OcpAbiOpaqueByte = ck::f8_ocp_t;
using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<
    RowMajor,
    ColumnMajor,
    ck::Tuple<>,
    RowMajor,
    OcpAbiOpaqueByte,
    float,
    OcpAbiOpaqueByte,
    float,
    ck::Tuple<>,
    ck::bhalf_t,
    1,
    128,
    128,
    PassThrough,
    PassThrough,
    PassThrough>;

static_assert(sizeof(OcpAbiOpaqueByte) == 1);
static_assert(sizeof(ck::bhalf_t) == 2);
static_assert(std::is_same_v<ck::f8_t, OcpAbiOpaqueByte>,
              "A′ must use the archive's f8_ocp_t ABI; build with CK_USE_OCP_FP8=1");

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

void validate_gfx942_device(int device_id) {
    const char* visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (visible == nullptr || visible[0] == '\0' || std::strchr(visible, ',') != nullptr) {
        throw std::runtime_error(
            "SQ8_0 gfx942 A′ requires HIP_VISIBLE_DEVICES to contain exactly one device token");
    }
    int device_count = 0;
    hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1 || device_id != 0) {
        throw std::runtime_error(
            "SQ8_0 gfx942 A′ requires one visible HIP device at internal ordinal 0");
    }
    hip_check(hipSetDevice(device_id), "hipSetDevice");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, device_id), "hipGetDeviceProperties");
    properties.gcnArchName[sizeof(properties.gcnArchName) - 1u] = '\0';
    if (!ullm::sq8_ck_gfx942::is_exact_gfx942_gcn_arch_name(properties.gcnArchName)) {
        throw std::runtime_error(
            std::string("SQ8_0 gfx942 A′ requires exact gcnArchName gfx942; selected ") +
            properties.gcnArchName);
    }
}

__global__ void bf16_to_f32(
    const std::uint16_t* input,
    float* output,
    unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __uint_as_float(static_cast<unsigned int>(input[index]) << 16u);
    }
}

__global__ void fnuz_fragment_probe_kernel(
    const unsigned char* a_fnuz_16x32_row_major,
    const unsigned char* b_fnuz_32x16_column_major,
    float* matrix_f32_16x16,
    float* fragment_f32_lane64x4) {
    const unsigned int lane = threadIdx.x;
    if (lane >= 64u) {
        return;
    }
    using namespace rocwmma;
    using FragA = fragment<matrix_a, 16, 16, 32, float8_fnuz_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 32, float8_fnuz_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 32, float32_t, row_major>;
    static_assert(FragC::num_elements == 4u,
                  "the physical fragment probe records four FP32 accumulator registers per lane");
    FragA a;
    FragB b;
    FragC c;
    fill_fragment(c, 0.0f);
    load_matrix_sync(a,
                     reinterpret_cast<const float8_fnuz_t*>(a_fnuz_16x32_row_major),
                     32u);
    load_matrix_sync(b,
                     reinterpret_cast<const float8_fnuz_t*>(b_fnuz_32x16_column_major),
                     32u);
    mma_sync(c, a, b, c);
    for (unsigned int index = 0u; index < FragC::num_elements; ++index) {
        fragment_f32_lane64x4[lane * FragC::num_elements + index] = c[index];
    }
    store_matrix_sync(matrix_f32_16x16, c, 16u, mem_row_major);
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

Registry& registry() {
    static Registry value;
    return value;
}

std::pair<std::vector<std::unique_ptr<DeviceOp>>*, std::string_view> dispatch(uint32_t id) {
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
        throw std::runtime_error("SQ8_0 gfx942 A′ received an unknown implementation id");
    }
}

DeviceOp& select_operation(uint32_t implementation) {
    auto [instances, expected_type] = dispatch(implementation);
    DeviceOp* selected = nullptr;
    size_t matches = 0u;
    for (const auto& instance : *instances) {
        if (instance->GetTypeString() == expected_type) {
            selected = instance.get();
            ++matches;
        }
    }
    if (matches != 1u || selected == nullptr) {
        throw std::runtime_error(
            "SQ8_0 gfx942 A′ measured GetTypeString did not resolve to exactly one instance: " +
            std::string(expected_type));
    }
    return *selected;
}

} // namespace

extern "C" int ullm_sq8_ck_gfx942_aprime_projection_fnuz_prepacked(
    const void* activation_fnuz_prepacked_bytes,
    const void* activation_scale_f32_x2,
    const void* weight_fnuz_prepacked_bytes,
    const void* weight_scale_f32_x2,
    size_t m,
    size_t n,
    size_t k,
    void* workspace_bf16,
    void* output_f32,
    void* stream,
    int device_id,
    uint32_t implementation,
    char* error,
    size_t error_capacity) {
    try {
        validate_gfx942_device(device_id);
        DeviceOp& operation = select_operation(implementation);

        // The two byte pointers are FNUZ-prepacked opaque storage.  The
        // f8_ocp_t type below exists solely to bind the installed CK archive;
        // it must not be read as a request to reinterpret canonical OCP bytes.
        auto argument = operation.MakeArgumentPointer(
            activation_fnuz_prepacked_bytes,
            weight_fnuz_prepacked_bytes,
            std::array<const void*, 0>{},
            workspace_bf16,
            static_cast<ck::index_t>(m),
            static_cast<ck::index_t>(n),
            static_cast<ck::index_t>(k),
            static_cast<ck::index_t>(k),
            static_cast<ck::index_t>(k),
            std::array<ck::index_t, 0>{},
            static_cast<ck::index_t>(n),
            activation_scale_f32_x2,
            weight_scale_f32_x2,
            PassThrough{},
            PassThrough{},
            PassThrough{});
        if (!operation.IsSupportedArgument(argument.get())) {
            throw std::runtime_error(
                "SQ8_0 gfx942 A′ measured instance rejected the FNUZ-prepacked projection argument");
        }
        auto invoker = operation.MakeInvokerPointer();
        StreamConfig config;
        config.stream_id_ = static_cast<hipStream_t>(stream);
        config.time_kernel_ = false;
        config.log_level_ = 0;
        config.flush_cache = false;
        (void)invoker->Run(argument.get(), config);
        hip_check(hipGetLastError(), "SQ8_0 gfx942 A′ CK ABScale GEMM launch");

        const size_t elements = m * n;
        constexpr unsigned int threads = 256u;
        const size_t blocks = (elements + threads - 1u) / threads;
        if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::runtime_error("SQ8_0 gfx942 A′ BF16 conversion grid overflows");
        }
        hipLaunchKernelGGL(bf16_to_f32,
                           dim3(static_cast<unsigned int>(blocks)),
                           dim3(threads),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const std::uint16_t*>(workspace_bf16),
                           static_cast<float*>(output_f32),
                           static_cast<unsigned long long>(elements));
        hip_check(hipGetLastError(), "SQ8_0 gfx942 A′ BF16-to-F32 launch");
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "SQ8_0 gfx942 A′ projection failed");
        return 0;
    }
}

extern "C" int ullm_sq8_ck_gfx942_aprime_fragment_probe_fnuz(
    const void* a_fnuz_16x32_row_major,
    const void* b_fnuz_32x16_column_major,
    void* matrix_f32_16x16,
    void* fragment_f32_lane64x4,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity) {
    try {
        validate_gfx942_device(device_id);
        hipLaunchKernelGGL(fnuz_fragment_probe_kernel,
                           dim3(1u),
                           dim3(64u),
                           0u,
                           static_cast<hipStream_t>(stream),
                           static_cast<const unsigned char*>(a_fnuz_16x32_row_major),
                           static_cast<const unsigned char*>(b_fnuz_32x16_column_major),
                           static_cast<float*>(matrix_f32_16x16),
                           static_cast<float*>(fragment_f32_lane64x4));
        hip_check(hipGetLastError(), "SQ8_0 gfx942 A′ FNUZ fragment probe launch");
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "SQ8_0 gfx942 A′ fragment probe failed");
        return 0;
    }
}
