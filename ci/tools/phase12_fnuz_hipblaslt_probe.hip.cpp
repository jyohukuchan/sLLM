// Standalone Phase 12 gfx942 hipBLASLt E4M3FNUZ solution probe.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check_hip(const hipError_t status, const char *operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void check_lt(const hipblasStatus_t status, const char *operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with hipBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

} // namespace

int main() {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  float *weight_scales = nullptr;
  float *activation_scales = nullptr;
  try {
    constexpr std::uint64_t m = 3U;
    constexpr std::uint64_t k = 128U;
    constexpr std::uint64_t n = 256U;
    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1) {
      throw std::runtime_error("FNUZ probe requires exactly one visible GPU");
    }
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    if (std::string(properties.gcnArchName).rfind("gfx942", 0U) != 0U) {
      throw std::runtime_error("FNUZ probe requires exact gfx942");
    }

    check_hip(
        hipMalloc(reinterpret_cast<void **>(&weight_scales), n * sizeof(float)),
        "hipMalloc(weight scales)");
    check_hip(hipMalloc(reinterpret_cast<void **>(&activation_scales),
                        m * sizeof(float)),
              "hipMalloc(activation scales)");
    check_lt(hipblasLtCreate(&handle), "hipblasLtCreate");
    check_lt(
        hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "hipblasLtMatmulDescCreate");
    const hipblasOperation_t trans_a = HIPBLAS_OP_T;
    const hipblasOperation_t trans_b = HIPBLAS_OP_N;
    check_lt(hipblasLtMatmulDescSetAttribute(operation,
                                             HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &trans_a, sizeof(trans_a)),
             "set TRANSA");
    check_lt(hipblasLtMatmulDescSetAttribute(operation,
                                             HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &trans_b, sizeof(trans_b)),
             "set TRANSB");
    void *weight_scale_pointer = weight_scales;
    void *activation_scale_pointer = activation_scales;
    const hipblasLtMatmulMatrixScale_t scale_mode =
        HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    check_lt(hipblasLtMatmulDescSetAttribute(
                 operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                 &weight_scale_pointer, sizeof(weight_scale_pointer)),
             "set A scale pointer");
    check_lt(hipblasLtMatmulDescSetAttribute(
                 operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                 &activation_scale_pointer, sizeof(activation_scale_pointer)),
             "set B scale pointer");
    check_lt(hipblasLtMatmulDescSetAttribute(operation,
                                             HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                             &scale_mode, sizeof(scale_mode)),
             "set A scale mode");
    check_lt(hipblasLtMatmulDescSetAttribute(operation,
                                             HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                             &scale_mode, sizeof(scale_mode)),
             "set B scale mode");
    check_lt(hipblasLtMatrixLayoutCreate(&a, HIP_R_8F_E4M3_FNUZ, k, n,
                                         static_cast<std::int64_t>(k)),
             "create A layout");
    check_lt(hipblasLtMatrixLayoutCreate(&b, HIP_R_8F_E4M3_FNUZ, k, m,
                                         static_cast<std::int64_t>(k)),
             "create B layout");
    check_lt(hipblasLtMatrixLayoutCreate(&c, HIP_R_16BF, n, m,
                                         static_cast<std::int64_t>(n)),
             "create C layout");
    check_lt(hipblasLtMatrixLayoutCreate(&d, HIP_R_16BF, n, m,
                                         static_cast<std::int64_t>(n)),
             "create D layout");
    check_lt(hipblasLtMatmulPreferenceCreate(&preference),
             "hipblasLtMatmulPreferenceCreate");
    const std::uint64_t workspace_limit = 0U;
    check_lt(hipblasLtMatmulPreferenceSetAttribute(
                 preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                 &workspace_limit, sizeof(workspace_limit)),
             "set workspace preference");

    hipblasLtMatmulHeuristicResult_t results[8]{};
    int solution_count = 0;
    check_lt(hipblasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, d,
                                             preference, 8, results,
                                             &solution_count),
             "hipblasLtMatmulAlgoGetHeuristic");
    if (solution_count < 1 || results[0].state != HIPBLAS_STATUS_SUCCESS ||
        results[0].workspaceSize != 0U) {
      throw std::runtime_error(
          "hipBLASLt returned no zero-workspace FNUZ solution");
    }

    check_lt(hipblasLtMatmulPreferenceDestroy(preference),
             "destroy preference");
    preference = nullptr;
    check_lt(hipblasLtMatrixLayoutDestroy(d), "destroy D layout");
    d = nullptr;
    check_lt(hipblasLtMatrixLayoutDestroy(c), "destroy C layout");
    c = nullptr;
    check_lt(hipblasLtMatrixLayoutDestroy(b), "destroy B layout");
    b = nullptr;
    check_lt(hipblasLtMatrixLayoutDestroy(a), "destroy A layout");
    a = nullptr;
    check_lt(hipblasLtMatmulDescDestroy(operation), "destroy operation");
    operation = nullptr;
    check_lt(hipblasLtDestroy(handle), "hipblasLtDestroy");
    handle = nullptr;
    check_hip(hipFree(activation_scales), "hipFree(activation scales)");
    activation_scales = nullptr;
    check_hip(hipFree(weight_scales), "hipFree(weight scales)");
    weight_scales = nullptr;

    std::cout << "{\"schema_version\":"
                 "\"phase12-fnuz-hipblaslt-probe-v1\","
                 "\"state\":\"PASS\",\"target\":\"gfx942\","
                 "\"dtype\":\"e4m3fnuz\",\"m\":3,\"k\":128,"
                 "\"n\":256,\"requested_solutions\":8,"
                 "\"solution_count\":"
              << solution_count << ",\"workspace_bytes\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    if (preference != nullptr) {
      (void)hipblasLtMatmulPreferenceDestroy(preference);
    }
    if (d != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(d);
    }
    if (c != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(c);
    }
    if (b != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(b);
    }
    if (a != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(a);
    }
    if (operation != nullptr) {
      (void)hipblasLtMatmulDescDestroy(operation);
    }
    if (handle != nullptr) {
      (void)hipblasLtDestroy(handle);
    }
    if (activation_scales != nullptr) {
      (void)hipFree(activation_scales);
    }
    if (weight_scales != nullptr) {
      (void)hipFree(weight_scales);
    }
    std::cerr << "Phase 12 FNUZ hipBLASLt probe: " << error.what() << '\n';
    return 2;
  }
}
