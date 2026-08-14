// Bounded Phase 8 library-candidate probe. This is not linked into sLLM.
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    return static_cast<uint16_t>(bits >> 16U);
  }
  const uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  return static_cast<uint16_t>(
      upper + (lower > UINT32_C(0x8000) ||
               (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)));
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool hip_ok(const hipError_t status, const char *const label) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", label, hipGetErrorString(status));
  return false;
}

bool blas_ok(const hipblasStatus_t status, const char *const label) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s: hipBLAS status %d\n", label,
               static_cast<int>(status));
  return false;
}

} // namespace

int main(int argc, char **argv) {
  if ((argc != 2 && argc != 5) || (std::strcmp(argv[1], "gfx1030") != 0 &&
                                   std::strcmp(argv[1], "gfx1201") != 0)) {
    std::fprintf(stderr, "usage: phase8_hipblas_poc gfx1030|gfx1201 [M K N]\n");
    return 2;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "device properties") ||
      std::strcmp(properties.gcnArchName, argv[1]) != 0) {
    std::fprintf(stderr, "visible device is not the requested exact target\n");
    return 1;
  }
  hipblasHandle_t handle = nullptr;
  if (!blas_ok(hipblasCreate(&handle), "hipblasCreate")) {
    return 1;
  }
  const int m = argc == 5 ? std::atoi(argv[2]) : 17;
  const int k = argc == 5 ? std::atoi(argv[3]) : 257;
  const int n = argc == 5 ? std::atoi(argv[4]) : 65;
  if (m <= 0 || k <= 0 || n <= 0 || m > 1024 || k > 16384 || n > 262144) {
    std::fprintf(stderr, "shape is outside the bounded probe contract\n");
    return 2;
  }
  std::vector<uint16_t> activation(static_cast<size_t>(m) * k);
  std::vector<uint16_t> weight(static_cast<size_t>(n) * k);
  std::vector<uint16_t> expected(static_cast<size_t>(m) * n);
  for (size_t index = 0; index != activation.size(); ++index) {
    activation[index] = f32_to_bf16(
        static_cast<float>(static_cast<int>(index % 13) - 6) / 16.0F);
  }
  for (size_t index = 0; index != weight.size(); ++index) {
    weight[index] = f32_to_bf16(
        static_cast<float>(static_cast<int>(index % 17) - 8) / 32.0F);
  }
  for (int row = 0; row != m; ++row) {
    for (int column = 0; column != n; ++column) {
      float sum = 0.0F;
      for (int reduction = 0; reduction != k; ++reduction) {
        sum +=
            bf16_to_f32(activation[static_cast<size_t>(row) * k + reduction]) *
            bf16_to_f32(weight[static_cast<size_t>(column) * k + reduction]);
      }
      expected[static_cast<size_t>(row) * n + column] = f32_to_bf16(sum);
    }
  }

  uint16_t *device_activation = nullptr;
  uint16_t *device_weight = nullptr;
  uint16_t *device_output = nullptr;
  const size_t activation_bytes = activation.size() * sizeof(uint16_t);
  const size_t weight_bytes = weight.size() * sizeof(uint16_t);
  const size_t output_bytes = expected.size() * sizeof(uint16_t);
  if (!hip_ok(hipMalloc(&device_activation, activation_bytes),
              "activation allocation") ||
      !hip_ok(hipMalloc(&device_weight, weight_bytes), "weight allocation") ||
      !hip_ok(hipMalloc(&device_output, output_bytes), "output allocation") ||
      !hip_ok(hipMemcpy(device_activation, activation.data(), activation_bytes,
                        hipMemcpyHostToDevice),
              "activation upload") ||
      !hip_ok(hipMemcpy(device_weight, weight.data(), weight_bytes,
                        hipMemcpyHostToDevice),
              "weight upload")) {
    return 1;
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto gemm = [&]() {
    // Row-major output [M,N] aliases column-major [N,M]. Weight [N,K]
    // aliases column-major [K,N], hence op(A)=transpose(weight).
    return hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k, &alpha,
                         device_weight, HIPBLAS_R_16B, k, device_activation,
                         HIPBLAS_R_16B, k, &beta, device_output, HIPBLAS_R_16B,
                         n, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  };
  if (!blas_ok(gemm(), "hipblasGemmEx warmup") ||
      !hip_ok(hipDeviceSynchronize(), "warmup synchronize")) {
    return 1;
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!hip_ok(hipEventCreate(&start), "start event") ||
      !hip_ok(hipEventCreate(&stop), "stop event") ||
      !hip_ok(hipEventRecord(start), "start record")) {
    return 1;
  }
  constexpr int iterations = 10;
  for (int iteration = 0; iteration != iterations; ++iteration) {
    if (!blas_ok(gemm(), "hipblasGemmEx measured")) {
      return 1;
    }
  }
  if (!hip_ok(hipEventRecord(stop), "stop record") ||
      !hip_ok(hipEventSynchronize(stop), "stop synchronize")) {
    return 1;
  }
  float elapsed_ms = 0.0F;
  if (!hip_ok(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed time")) {
    return 1;
  }
  std::vector<uint16_t> actual(expected.size());
  if (!hip_ok(hipMemcpy(actual.data(), device_output, output_bytes,
                        hipMemcpyDeviceToHost),
              "output readback")) {
    return 1;
  }
  float max_abs_error = 0.0F;
  for (size_t index = 0; index != actual.size(); ++index) {
    max_abs_error =
        std::fmax(max_abs_error, std::fabs(bf16_to_f32(actual[index]) -
                                           bf16_to_f32(expected[index])));
  }
  const bool pass = std::isfinite(max_abs_error) && max_abs_error <= 0.016F;
  std::printf("{\"protocol\":\"phase8-hipblas-poc-v1\",\"state\":\"%s\","
              "\"target\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,"
              "\"weight_copy\":false,\"workspace_bytes\":0,"
              "\"average_kernel_ns\":%.0f,\"max_abs_error\":%.9g}\n",
              pass ? "PASS" : "FAIL", argv[1], m, k, n,
              static_cast<double>(elapsed_ms) * 1000000.0 / iterations,
              static_cast<double>(max_abs_error));
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  (void)hipFree(device_activation);
  (void)hipFree(device_weight);
  (void)hipFree(device_output);
  hipblasDestroy(handle);
  return pass ? 0 : 1;
}
