// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void hip_check(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

const char* status_name(hipblasStatus_t status)
{
    switch(status)
    {
    case HIPBLAS_STATUS_SUCCESS: return "success";
    case HIPBLAS_STATUS_NOT_INITIALIZED: return "not_initialized";
    case HIPBLAS_STATUS_ALLOC_FAILED: return "alloc_failed";
    case HIPBLAS_STATUS_INVALID_VALUE: return "invalid_value";
    case HIPBLAS_STATUS_MAPPING_ERROR: return "mapping_error";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "execution_failed";
    case HIPBLAS_STATUS_INTERNAL_ERROR: return "internal_error";
    case HIPBLAS_STATUS_NOT_SUPPORTED: return "not_supported";
    case HIPBLAS_STATUS_ARCH_MISMATCH: return "arch_mismatch";
    default: return "other";
    }
}

std::string json_string(std::string_view value)
{
    std::string encoded;
    encoded.reserve(value.size() + 2);
    encoded.push_back('"');
    constexpr char hex[] = "0123456789abcdef";
    for(const unsigned char ch : value)
    {
        switch(ch)
        {
        case '"': encoded += "\\\""; break;
        case '\\': encoded += "\\\\"; break;
        case '\n': encoded += "\\n"; break;
        case '\r': encoded += "\\r"; break;
        case '\t': encoded += "\\t"; break;
        default:
            if(ch < 0x20)
            {
                encoded += "\\u00";
                encoded.push_back(hex[ch >> 4]);
                encoded.push_back(hex[ch & 0x0f]);
            }
            else
            {
                encoded.push_back(static_cast<char>(ch));
            }
        }
    }
    encoded.push_back('"');
    return encoded;
}

struct DeviceBuffer
{
    void* pointer = nullptr;

    explicit DeviceBuffer(std::size_t bytes)
    {
        hip_check(hipMalloc(&pointer, std::max<std::size_t>(bytes, 1)), "hipMalloc");
    }

    ~DeviceBuffer()
    {
        if(pointer != nullptr)
        {
            static_cast<void>(hipFree(pointer));
        }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct BlockProbe
{
    bool supported = false;
    std::string status;
    std::string detail;
};

struct ScalarProbe
{
    bool passed = false;
    std::string set_problem_status;
    std::string heuristic_status;
    std::size_t algorithm_count = 0;
    std::string initialize_status;
    std::string run_status;
    std::string solution;
    std::string kernel;
    float output = 0.0f;
    float expected = 0.0f;
};

BlockProbe probe_required_block_scale(hipblasLtHandle_t handle)
{
    constexpr std::int64_t n = 256;
    constexpr std::int64_t m = 8;
    constexpr std::int64_t k = 256;
    DeviceBuffer weight(static_cast<std::size_t>(n * k));
    DeviceBuffer activation(static_cast<std::size_t>(m * k));
    DeviceBuffer output(static_cast<std::size_t>(n * m) * sizeof(float));
    DeviceBuffer weight_scale(static_cast<std::size_t>((n / 128) * (k / 128)) * sizeof(float));
    DeviceBuffer activation_scale(static_cast<std::size_t>(m) * sizeof(float));

    float alpha = 1.0f;
    float beta  = 0.0f;
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(weight.pointer);
    inputs.setB(activation.pointer);
    inputs.setC(output.pointer);
    inputs.setD(output.pointer);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setScaleA(weight_scale.pointer);
    inputs.setScaleB(activation_scale.pointer);

    try
    {
        hipblaslt_ext::GemmEpilogue epilogue;
        epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F);
        epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_T,
                                 HIPBLAS_OP_N,
                                 HIP_R_8F_E4M3,
                                 HIP_R_8F_E4M3,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIPBLAS_COMPUTE_32F);
        const auto status = gemm.setProblem(n, m, k, 1, epilogue, inputs);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            return {false, status_name(status), "setProblem rejected required A block scale"};
        }
        hipblaslt_ext::GemmPreference preference;
        preference.setMaxWorkspaceBytes(64ULL << 20);
        std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
        const auto heuristic = gemm.algoGetHeuristic(8, preference, algorithms);
        return {heuristic == HIPBLAS_STATUS_SUCCESS && !algorithms.empty(),
                status_name(heuristic),
                algorithms.empty() ? "no algorithm for required A block scale"
                                   : "required A block scale unexpectedly has an algorithm"};
    }
    catch(const std::exception& error)
    {
        return {false, "exception", error.what()};
    }
}

ScalarProbe probe_scalar_fp8(hipblasLtHandle_t handle, std::int64_t m)
{
    constexpr std::int64_t n = 5120;
    constexpr std::int64_t k = 5120;
    DeviceBuffer weight(static_cast<std::size_t>(n * k));
    DeviceBuffer activation(static_cast<std::size_t>(m * k));
    DeviceBuffer output(static_cast<std::size_t>(n * m) * sizeof(float));
    DeviceBuffer weight_scale(sizeof(float));
    DeviceBuffer activation_scale(sizeof(float));
    hip_check(hipMemset(weight.pointer, 0x38, static_cast<std::size_t>(n * k)),
              "fill scalar sanity weight");
    hip_check(hipMemset(activation.pointer, 0x38, static_cast<std::size_t>(m * k)),
              "fill scalar sanity activation");
    const float host_weight_scale     = 2.0f;
    const float host_activation_scale = 3.0f;
    hip_check(hipMemcpy(weight_scale.pointer,
                        &host_weight_scale,
                        sizeof(float),
                        hipMemcpyHostToDevice),
              "copy scalar weight scale");
    hip_check(hipMemcpy(activation_scale.pointer,
                        &host_activation_scale,
                        sizeof(float),
                        hipMemcpyHostToDevice),
              "copy scalar activation scale");

    float alpha = 1.0f;
    float beta  = 0.0f;
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(weight.pointer);
    inputs.setB(activation.pointer);
    inputs.setC(output.pointer);
    inputs.setD(output.pointer);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setScaleA(weight_scale.pointer);
    inputs.setScaleB(activation_scale.pointer);

    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
    epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);

    ScalarProbe report;
    report.expected = static_cast<float>(k) * host_weight_scale * host_activation_scale;
    hipblaslt_ext::Gemm gemm(handle,
                             HIPBLAS_OP_T,
                             HIPBLAS_OP_N,
                             HIP_R_8F_E4M3,
                             HIP_R_8F_E4M3,
                             HIP_R_32F,
                             HIP_R_32F,
                             HIPBLAS_COMPUTE_32F);
    auto status               = gemm.setProblem(n, m, k, 1, epilogue, inputs);
    report.set_problem_status = status_name(status);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        return report;
    }

    hipblaslt_ext::GemmPreference preference;
    preference.setMaxWorkspaceBytes(64ULL << 20);
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
    status                    = gemm.algoGetHeuristic(8, preference, algorithms);
    report.heuristic_status   = status_name(status);
    report.algorithm_count    = algorithms.size();
    if(status != HIPBLAS_STATUS_SUCCESS || algorithms.empty())
    {
        return report;
    }

    DeviceBuffer workspace(algorithms.front().workspaceSize);
    gemm.setMaxWorkspaceBytes(algorithms.front().workspaceSize);
    status                   = gemm.initialize(algorithms.front().algo, workspace.pointer, false, nullptr);
    report.initialize_status = status_name(status);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        return report;
    }
    report.solution = gemm.getSolutionName();
    report.kernel   = gemm.getKernelName();
    status          = gemm.run(nullptr);
    report.run_status = status_name(status);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        return report;
    }
    hip_check(hipDeviceSynchronize(), "synchronize scalar sanity GEMM");
    hip_check(hipMemcpy(&report.output,
                        output.pointer,
                        sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy scalar sanity output");
    report.passed = std::isfinite(report.output) && report.output == report.expected;
    return report;
}

int parse_device(int argc, char** argv)
{
    int device = 1;
    for(int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if(argument == "--device" && index + 1 < argc)
        {
            device = std::stoi(argv[++index]);
        }
        else if(argument == "--help" || argument == "-h")
        {
            std::cout << "usage: probe-sq8-hipblaslt [--device HIP_DEVICE_ID]\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("unknown or incomplete argument: " + std::string(argument));
        }
    }
    return device;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const int device = parse_device(argc, argv);
        hip_check(hipSetDevice(device), "hipSetDevice");
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
        if(std::string_view(properties.gcnArchName).find("gfx1201") == std::string_view::npos)
        {
            throw std::runtime_error("SQ8 hipBLASLt route probe requires gfx1201");
        }

        hipblasLtHandle_t handle = nullptr;
        const auto create_status = hipblasLtCreate(&handle);
        if(create_status != HIPBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::string("hipblasLtCreate: ") + status_name(create_status));
        }
        const auto block  = probe_required_block_scale(handle);
        const auto scalar = probe_scalar_fp8(handle, 8);
        const auto destroy_status = hipblasLtDestroy(handle);
        if(destroy_status != HIPBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::string("hipblasLtDestroy: ") +
                                     status_name(destroy_status));
        }

        const bool passed = !block.supported && scalar.passed;
        std::cout << "{\n"
                  << "  \"schema_version\": \"sq8-hipblaslt-route-probe-v0.1\",\n"
                  << "  \"device\": {\"hip_device_id\": " << device
                  << ", \"name\": " << json_string(properties.name)
                  << ", \"arch\": " << json_string(properties.gcnArchName) << "},\n"
                  << "  \"required_operation\": {\"input\": \"f8_e4m3_ocp_x_f8_e4m3_ocp\", "
                     "\"weight_scale\": \"block_128x128_f32\", \"activation_scale\": "
                     "\"outer_vector_f32\", \"output\": \"f32\"},\n"
                  << "  \"block_scale_probe\": {\"supported\": "
                  << (block.supported ? "true" : "false") << ", \"status\": "
                  << json_string(block.status) << ", \"detail\": " << json_string(block.detail)
                  << "},\n"
                  << "  \"scalar_fp8_sanity\": {\"passed\": "
                  << (scalar.passed ? "true" : "false")
                  << ", \"set_problem_status\": " << json_string(scalar.set_problem_status)
                  << ", \"heuristic_status\": " << json_string(scalar.heuristic_status)
                  << ", \"algorithm_count\": " << scalar.algorithm_count
                  << ", \"initialize_status\": " << json_string(scalar.initialize_status)
                  << ", \"run_status\": " << json_string(scalar.run_status)
                  << ", \"output0\": " << scalar.output << ", \"expected0\": "
                  << scalar.expected << ", \"solution\": " << json_string(scalar.solution)
                  << ", \"kernel\": " << json_string(scalar.kernel) << "},\n"
                  << "  \"selection\": \"rejected\",\n"
                  << "  \"reason_code\": \"block_128x128_scale_unsupported\",\n"
                  << "  \"passed\": " << (passed ? "true" : "false") << "\n"
                  << "}\n";
        return passed ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "probe-sq8-hipblaslt: " << error.what() << '\n';
        return 2;
    }
}
