// Model-free Phase 10 hipBLASLt OCP E4M3 outer-vector scaling proof.
//
// This intentionally uses row-major Qwen linear semantics: D[M,N] =
// A[M,K] * transpose(B[N,K]).  A and B are E4M3FN bytes, their FP32 scales
// have M and N entries, accumulation is FP32, and D is BF16.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_hip(hipError_t status, const char *what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

void check_blas(hipblasStatus_t status, const char *what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": status=" +
                             std::to_string(static_cast<int>(status)));
  }
}

float decode_e4m3fn(uint8_t bits) {
  const float sign = (bits & 0x80U) == 0 ? 1.0F : -1.0F;
  const uint8_t exponent = (bits >> 3U) & 0x0fU;
  const uint8_t mantissa = bits & 0x07U;
  if (exponent == 0) {
    return mantissa == 0 ? std::copysign(0.0F, sign)
                         : sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  }
  if (exponent == 0x0fU && mantissa == 0x07U) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         std::ldexp(1.0F, static_cast<int>(exponent) - 7);
}

uint8_t encode_e4m3fn(float value) {
  if (std::isnan(value)) return 0x7fU;
  const uint8_t sign = std::signbit(value) ? 0x80U : 0U;
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F) return sign;
  if (!std::isfinite(magnitude) || magnitude >= 448.0F) return sign | 0x7eU;
  uint8_t best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (uint16_t candidate = 0; candidate <= 0x7eU; ++candidate) {
    const float error = std::fabs(decode_e4m3fn(static_cast<uint8_t>(candidate)) - magnitude);
    if (error < best_error ||
        (error == best_error && (candidate & 1U) == 0 && (best & 1U) != 0)) {
      best = static_cast<uint8_t>(candidate);
      best_error = error;
    }
  }
  return sign | best;
}

template <typename T> struct DeviceBuffer {
  T *pointer = nullptr;
  explicit DeviceBuffer(size_t count) {
    check_hip(hipMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)), "hipMalloc");
  }
  ~DeviceBuffer() { if (pointer != nullptr) hipFree(pointer); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct LtObjects {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  ~LtObjects() {
    if (preference) hipblasLtMatmulPreferenceDestroy(preference);
    if (d) hipblasLtMatrixLayoutDestroy(d);
    if (c) hipblasLtMatrixLayoutDestroy(c);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (a) hipblasLtMatrixLayoutDestroy(a);
    if (operation) hipblasLtMatmulDescDestroy(operation);
    if (handle) hipblasLtDestroy(handle);
  }
};

}  // namespace

int main(int argc, char **argv) {
  try {
    const int64_t m = argc > 1 ? std::stoll(argv[1]) : 3;
    const int64_t k = argc > 2 ? std::stoll(argv[2]) : 129;
    const int64_t n = argc > 3 ? std::stoll(argv[3]) : 257;
    if (m <= 0 || k <= 0 || n <= 0) throw std::runtime_error("dimensions must be positive");

    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    if (std::string(properties.gcnArchName).rfind("gfx1201", 0) != 0) {
      throw std::runtime_error("native Phase 10 PoC requires exact gfx1201");
    }

    std::vector<float> a_source(static_cast<size_t>(m * k));
    std::vector<float> b_source(static_cast<size_t>(n * k));
    for (size_t index = 0; index < a_source.size(); ++index)
      a_source[index] = std::sin(static_cast<float>(index) * 0.013F) * 2.25F;
    for (size_t index = 0; index < b_source.size(); ++index)
      b_source[index] = std::cos(static_cast<float>(index) * 0.007F) * 1.75F;

    std::vector<float> a_scale(static_cast<size_t>(m));
    std::vector<float> b_scale(static_cast<size_t>(n));
    std::vector<uint8_t> a_fp8(a_source.size());
    std::vector<uint8_t> b_fp8(b_source.size());
    for (int64_t row = 0; row < m; ++row) {
      float amax = 0.0F;
      for (int64_t column = 0; column < k; ++column)
        amax = std::max(amax, std::fabs(a_source[static_cast<size_t>(row * k + column)]));
      a_scale[static_cast<size_t>(row)] = amax == 0.0F ? 1.0F : amax / 448.0F;
      for (int64_t column = 0; column < k; ++column) {
        const size_t index = static_cast<size_t>(row * k + column);
        a_fp8[index] = encode_e4m3fn(a_source[index] / a_scale[static_cast<size_t>(row)]);
      }
    }
    for (int64_t row = 0; row < n; ++row) {
      float amax = 0.0F;
      for (int64_t column = 0; column < k; ++column)
        amax = std::max(amax, std::fabs(b_source[static_cast<size_t>(row * k + column)]));
      b_scale[static_cast<size_t>(row)] = amax == 0.0F ? 1.0F : amax / 448.0F;
      for (int64_t column = 0; column < k; ++column) {
        const size_t index = static_cast<size_t>(row * k + column);
        b_fp8[index] = encode_e4m3fn(b_source[index] / b_scale[static_cast<size_t>(row)]);
      }
    }

    DeviceBuffer<uint8_t> da(a_fp8.size()), db(b_fp8.size());
    DeviceBuffer<float> das(a_scale.size()), dbs(b_scale.size());
    DeviceBuffer<hip_bfloat16> dc(static_cast<size_t>(m * n)), dd(static_cast<size_t>(m * n));
    check_hip(hipMemcpy(da.pointer, a_fp8.data(), a_fp8.size(), hipMemcpyHostToDevice), "copy A");
    check_hip(hipMemcpy(db.pointer, b_fp8.data(), b_fp8.size(), hipMemcpyHostToDevice), "copy B");
    check_hip(hipMemcpy(das.pointer, a_scale.data(), a_scale.size() * sizeof(float), hipMemcpyHostToDevice), "copy A scale");
    check_hip(hipMemcpy(dbs.pointer, b_scale.data(), b_scale.size() * sizeof(float), hipMemcpyHostToDevice), "copy B scale");
    check_hip(hipMemset(dc.pointer, 0, static_cast<size_t>(m * n) * sizeof(hip_bfloat16)), "clear C");

    LtObjects lt;
    check_blas(hipblasLtCreate(&lt.handle), "hipblasLtCreate");
    check_blas(hipblasLtMatmulDescCreate(&lt.operation, HIPBLAS_COMPUTE_32F, HIP_R_32F), "create operation");
    // Row-major [M,K] and [N,K] buffers are presented as column-major [K,M]
    // and [K,N].  The resulting column-major [M,N] buffer is read with the
    // corresponding column-major index below.
    hipblasOperation_t trans_a = HIPBLAS_OP_T;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    check_blas(hipblasLtMatmulDescSetAttribute(lt.operation, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)), "set trans A");
    check_blas(hipblasLtMatmulDescSetAttribute(lt.operation, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)), "set trans B");
    void *a_scale_pointer = das.pointer;
    void *b_scale_pointer = dbs.pointer;
    check_blas(hipblasLtMatmulDescSetAttribute(lt.operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_pointer, sizeof(a_scale_pointer)), "set A scale");
    check_blas(hipblasLtMatmulDescSetAttribute(lt.operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_pointer, sizeof(b_scale_pointer)), "set B scale");
    hipblasLtMatmulMatrixScale_t scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    check_blas(hipblasLtMatmulDescSetAttribute(lt.operation, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)), "set A scale mode");
    check_blas(hipblasLtMatmulDescSetAttribute(lt.operation, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)), "set B scale mode");

    check_blas(hipblasLtMatrixLayoutCreate(&lt.a, HIP_R_8F_E4M3, k, m, k), "create A layout");
    check_blas(hipblasLtMatrixLayoutCreate(&lt.b, HIP_R_8F_E4M3, k, n, k), "create B layout");
    check_blas(hipblasLtMatrixLayoutCreate(&lt.c, HIP_R_16BF, m, n, m), "create C layout");
    check_blas(hipblasLtMatrixLayoutCreate(&lt.d, HIP_R_16BF, m, n, m), "create D layout");
    check_blas(hipblasLtMatmulPreferenceCreate(&lt.preference), "create preference");
    uint64_t workspace_limit = 64U * 1024U * 1024U;
    check_blas(hipblasLtMatmulPreferenceSetAttribute(lt.preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)), "set workspace preference");
    hipblasLtMatmulHeuristicResult_t heuristic{};
    int solution_count = 0;
    const auto query = hipblasLtMatmulAlgoGetHeuristic(lt.handle, lt.operation, lt.a, lt.b, lt.c, lt.d, lt.preference, 1, &heuristic, &solution_count);
    if (query != HIPBLAS_STATUS_SUCCESS || solution_count != 1 || heuristic.state != HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("hipBLASLt returned no supported FP8 solution");

    DeviceBuffer<uint8_t> workspace(std::max<size_t>(1, heuristic.workspaceSize));
    const float alpha = 1.0F, beta = 0.0F;
    check_blas(hipblasLtMatmul(lt.handle, lt.operation, &alpha, da.pointer, lt.a, db.pointer, lt.b,
                               &beta, dc.pointer, lt.c, dd.pointer, lt.d, &heuristic.algo,
                               workspace.pointer, heuristic.workspaceSize, nullptr), "hipblasLtMatmul");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    std::vector<hip_bfloat16> output(static_cast<size_t>(m * n));
    check_hip(hipMemcpy(output.data(), dd.pointer, output.size() * sizeof(hip_bfloat16), hipMemcpyDeviceToHost), "copy D");
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t column = 0; column < n; ++column) {
        double expected = 0.0;
        for (int64_t inner = 0; inner < k; ++inner) {
          expected += static_cast<double>(decode_e4m3fn(a_fp8[static_cast<size_t>(row * k + inner)]) * a_scale[static_cast<size_t>(row)]) *
                      static_cast<double>(decode_e4m3fn(b_fp8[static_cast<size_t>(column * k + inner)]) * b_scale[static_cast<size_t>(column)]);
        }
        const double actual = static_cast<float>(output[static_cast<size_t>(row + column * m)]);
        const double absolute = std::fabs(actual - expected);
        max_abs_error = std::max(max_abs_error, absolute);
        max_rel_error = std::max(max_rel_error, absolute / std::max(1.0, std::fabs(expected)));
      }
    }
    if (max_rel_error > 0.02 || !std::isfinite(max_abs_error))
      throw std::runtime_error("numerical oracle failed");
    std::printf("phase10_fp8_hipblaslt: PASS target=%s m=%lld k=%lld n=%lld solutions=%d workspace=%zu max_abs=%.9g max_rel=%.9g scale=outer-vector-f32 output=bf16 accumulation=fp32 fallback=false\n",
                properties.gcnArchName, static_cast<long long>(m), static_cast<long long>(k),
                static_cast<long long>(n), solution_count, heuristic.workspaceSize,
                max_abs_error, max_rel_error);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase10_fp8_hipblaslt: FAIL: %s\n", error.what());
    return 1;
  }
}
