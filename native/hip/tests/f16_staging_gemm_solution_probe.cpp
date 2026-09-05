// Focused exact-target probe for the transient-FP16 GEMM shared by packed
// staging pipelines.  This is a developer evidence tool, not part of the
// public runtime build.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

extern "C" rocblas_status rocblas_gemm_ex_get_solutions(
    rocblas_handle handle, rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k, const void *alpha,
    const void *a, rocblas_datatype a_type, rocblas_int lda, const void *b,
    rocblas_datatype b_type, rocblas_int ldb, const void *beta, const void *c,
    rocblas_datatype c_type, rocblas_int ldc, void *d, rocblas_datatype d_type,
    rocblas_int ldd, rocblas_datatype compute_type, rocblas_gemm_algo algorithm,
    uint32_t flags, rocblas_int *solutions, rocblas_int *solution_count);

namespace {

constexpr int kWarmups = 5;
constexpr int kMeasured = 21;

bool parse_positive_int(const char *const text, int *const value) {
  if (text == nullptr || value == nullptr) {
    return false;
  }
  const std::string_view input(text);
  int parsed = 0;
  const auto result =
      std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
      parsed <= 0) {
    return false;
  }
  *value = parsed;
  return true;
}

bool parse_device(const char *const text, int *const value) {
  if (text == nullptr || value == nullptr) {
    return false;
  }
  const std::string_view input(text);
  int parsed = -1;
  const auto result =
      std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
      parsed < 0) {
    return false;
  }
  *value = parsed;
  return true;
}

bool parse_solution(const char *const text, int *const value) {
  if (text == nullptr || value == nullptr) {
    return false;
  }
  const std::string_view input(text);
  int parsed = 0;
  const auto result =
      std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
      parsed == 0) {
    return false;
  }
  *value = parsed;
  return true;
}

bool checked_bytes(const int rows, const int columns, const uint64_t element,
                   std::size_t *const bytes) {
  if (rows <= 0 || columns <= 0 || bytes == nullptr) {
    return false;
  }
  const uint64_t row_count = static_cast<uint64_t>(rows);
  const uint64_t column_count = static_cast<uint64_t>(columns);
  if (row_count > UINT64_MAX / column_count ||
      row_count * column_count > UINT64_MAX / element ||
      row_count * column_count * element > SIZE_MAX) {
    return false;
  }
  *bytes = static_cast<std::size_t>(row_count * column_count * element);
  return true;
}

bool exact_target(const char *const name, const std::string_view target) {
  if (name == nullptr) {
    return false;
  }
  const std::string_view actual(name);
  return actual == target ||
         (actual.size() > target.size() && actual.starts_with(target) &&
          actual[target.size()] == ':');
}

uint64_t hash_bytes(const std::span<const uint8_t> bytes) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (const uint8_t byte : bytes) {
    hash = (hash ^ byte) * UINT64_C(1099511628211);
  }
  return hash;
}

bool measure_us(const hipStream_t stream, const hipEvent_t start,
                const hipEvent_t stop, const std::function<bool()> &launch,
                float *const median_us) {
  if (median_us == nullptr) {
    return false;
  }
  for (int iteration = 0; iteration < kWarmups; ++iteration) {
    if (!launch() || hipStreamSynchronize(stream) != hipSuccess) {
      return false;
    }
  }
  std::vector<float> samples;
  samples.reserve(kMeasured);
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (hipEventRecord(start, stream) != hipSuccess || !launch() ||
        hipEventRecord(stop, stream) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) {
      return false;
    }
    float elapsed_ms = 0.0F;
    if (hipEventElapsedTime(&elapsed_ms, start, stop) != hipSuccess) {
      return false;
    }
    samples.push_back(elapsed_ms * 1000.0F);
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[samples.size() / 2U];
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  int m = 0;
  int k = 0;
  int n = 0;
  int requested_solution = 0;
  if ((argc != 5 && argc != 6) || !parse_device(argv[1], &device) ||
      !parse_positive_int(argv[2], &m) || !parse_positive_int(argv[3], &k) ||
      !parse_positive_int(argv[4], &n) ||
      (argc == 6 && !parse_solution(argv[5], &requested_solution))) {
    std::fprintf(stderr, "usage: f16_staging_gemm_solution_probe DEVICE M K N "
                         "[NONZERO_SOLUTION]\n");
    return EXIT_FAILURE;
  }
  if (hipSetDevice(device) != hipSuccess) {
    std::fprintf(stderr, "hipSetDevice failed\n");
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
    std::fprintf(stderr, "hipGetDeviceProperties failed\n");
    return EXIT_FAILURE;
  }
  const std::string_view target =
      exact_target(properties.gcnArchName, "gfx1030")   ? "gfx1030"
      : exact_target(properties.gcnArchName, "gfx1201") ? "gfx1201"
                                                        : "";
  if (target.empty()) {
    std::fprintf(stderr, "exact gfx1030 or gfx1201 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::fprintf(stderr, "device=%d arch=%s pci=%04x:%02x:%02x\n", device,
               properties.gcnArchName, properties.pciDomainID,
               properties.pciBusID, properties.pciDeviceID);

  std::size_t activation_bytes = 0U;
  std::size_t weight_bytes = 0U;
  std::size_t output_bytes = 0U;
  if (!checked_bytes(m, k, 2U, &activation_bytes) ||
      !checked_bytes(n, k, 2U, &weight_bytes) ||
      !checked_bytes(m, n, 4U, &output_bytes)) {
    std::fprintf(stderr, "shape byte size overflow\n");
    return EXIT_FAILURE;
  }

  void *activation = nullptr;
  void *weight = nullptr;
  void *output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  rocblas_handle rocblas = nullptr;
  hipblasHandle_t hipblas = nullptr;
  const bool allow_atomics =
      std::getenv("SLLM_F16_STAGING_PROBE_ALLOW_ATOMICS") != nullptr &&
      std::string_view(std::getenv("SLLM_F16_STAGING_PROBE_ALLOW_ATOMICS")) ==
          "1";
  const rocblas_atomics_mode requested_atomics_mode =
      allow_atomics ? rocblas_atomics_allowed : rocblas_atomics_not_allowed;
  const auto cleanup = [&]() {
    if (hipblas != nullptr) {
      (void)hipblasDestroy(hipblas);
    }
    if (rocblas != nullptr) {
      (void)rocblas_destroy_handle(rocblas);
    }
    if (stop != nullptr) {
      (void)hipEventDestroy(stop);
    }
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
    if (output != nullptr) {
      (void)hipFree(output);
    }
    if (weight != nullptr) {
      (void)hipFree(weight);
    }
    if (activation != nullptr) {
      (void)hipFree(activation);
    }
  };

  if (hipMalloc(&activation, activation_bytes) != hipSuccess ||
      hipMalloc(&weight, weight_bytes) != hipSuccess ||
      hipMalloc(&output, output_bytes) != hipSuccess ||
      hipMemset(activation, 0x38, activation_bytes) != hipSuccess ||
      hipMemset(weight, 0x3c, weight_bytes) != hipSuccess ||
      hipMemset(output, 0, output_bytes) != hipSuccess ||
      hipStreamCreate(&stream) != hipSuccess ||
      hipEventCreate(&start) != hipSuccess ||
      hipEventCreate(&stop) != hipSuccess ||
      rocblas_create_handle(&rocblas) != rocblas_status_success ||
      rocblas_set_pointer_mode(rocblas, rocblas_pointer_mode_host) !=
          rocblas_status_success ||
      rocblas_set_atomics_mode(rocblas, requested_atomics_mode) !=
          rocblas_status_success ||
      rocblas_set_stream(rocblas, stream) != rocblas_status_success ||
      hipblasCreate(&hipblas) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(hipblas, stream) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "GPU allocation or handle initialization failed\n");
    cleanup();
    return EXIT_FAILURE;
  }
  rocblas_atomics_mode atomics_mode =
      allow_atomics ? rocblas_atomics_not_allowed : rocblas_atomics_allowed;
  if (rocblas_get_atomics_mode(rocblas, &atomics_mode) !=
          rocblas_status_success ||
      atomics_mode != requested_atomics_mode) {
    std::fprintf(stderr, "rocBLAS atomics mode contract failed\n");
    cleanup();
    return EXIT_FAILURE;
  }
  const char *const atomics_name = allow_atomics ? "allowed" : "not_allowed";
  std::fprintf(stderr, "rocblas_atomics=%s\n", atomics_name);

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto run_rocblas = [&](const rocblas_gemm_algo algorithm,
                               const int32_t solution) {
    return (rocblas_gemm_ex)(rocblas, rocblas_operation_transpose,
                             rocblas_operation_none, n, m, k, &alpha, weight,
                             rocblas_datatype_f16_r, k, activation,
                             rocblas_datatype_f16_r, k, &beta, output,
                             rocblas_datatype_f32_r, n, output,
                             rocblas_datatype_f32_r, n, rocblas_datatype_f32_r,
                             algorithm, solution, 0U) == rocblas_status_success;
  };
  const auto run_hipblas = [&]() {
    return hipblasGemmEx(hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k, &alpha,
                         weight, HIPBLAS_R_16F, k, activation, HIPBLAS_R_16F, k,
                         &beta, output, HIPBLAS_R_32F, n, HIPBLAS_COMPUTE_32F,
                         HIPBLAS_GEMM_DEFAULT) == HIPBLAS_STATUS_SUCCESS;
  };

  std::printf("target\tm\tk\tn\tprovider\tsolution\tmedian_us\tstatus\n");
  float median_us = 0.0F;
  const bool hipblas_ok =
      measure_us(stream, start, stop, run_hipblas, &median_us);
  std::printf("%.*s\t%d\t%d\t%d\thipblas-default\t0\t%.3f\t%s\n",
              static_cast<int>(target.size()), target.data(), m, k, n,
              hipblas_ok ? median_us : 0.0F,
              hipblas_ok ? "PASS" : "UNSUPPORTED");
  const bool standard_ok = measure_us(
      stream, start, stop,
      [&]() { return run_rocblas(rocblas_gemm_algo_standard, 0); }, &median_us);
  std::printf("%.*s\t%d\t%d\t%d\trocblas-standard\t0\t%.3f\t%s\n",
              static_cast<int>(target.size()), target.data(), m, k, n,
              standard_ok ? median_us : 0.0F,
              standard_ok ? "PASS" : "UNSUPPORTED");

  rocblas_int solution_count = 0;
  rocblas_status query_status = rocblas_gemm_ex_get_solutions(
      rocblas, rocblas_operation_transpose, rocblas_operation_none, n, m, k,
      &alpha, weight, rocblas_datatype_f16_r, k, activation,
      rocblas_datatype_f16_r, k, &beta, output, rocblas_datatype_f32_r, n,
      output, rocblas_datatype_f32_r, n, rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0U, nullptr, &solution_count);
  if (query_status != rocblas_status_success || solution_count <= 0) {
    std::fprintf(
        stderr, "rocblas solution count query failed: status=%d count=%d\n",
        static_cast<int>(query_status), static_cast<int>(solution_count));
    cleanup();
    return EXIT_FAILURE;
  }
  std::vector<rocblas_int> solutions(static_cast<std::size_t>(solution_count));
  query_status = rocblas_gemm_ex_get_solutions(
      rocblas, rocblas_operation_transpose, rocblas_operation_none, n, m, k,
      &alpha, weight, rocblas_datatype_f16_r, k, activation,
      rocblas_datatype_f16_r, k, &beta, output, rocblas_datatype_f32_r, n,
      output, rocblas_datatype_f32_r, n, rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0U, solutions.data(), &solution_count);
  if (query_status != rocblas_status_success) {
    std::fprintf(stderr, "rocblas solution list query failed: status=%d\n",
                 static_cast<int>(query_status));
    cleanup();
    return EXIT_FAILURE;
  }
  solutions.resize(static_cast<std::size_t>(solution_count));
  bool requested_solution_found = requested_solution == 0;
  for (const rocblas_int solution : solutions) {
    if (requested_solution != 0 && solution != requested_solution) {
      continue;
    }
    requested_solution_found = true;
    const bool ok = measure_us(
        stream, start, stop,
        [&]() {
          return run_rocblas(rocblas_gemm_algo_solution_index, solution);
        },
        &median_us);
    bool deterministic = ok;
    uint64_t reference_hash = 0U;
    std::vector<uint8_t> reference(output_bytes);
    std::vector<uint8_t> repeated(output_bytes);
    if (deterministic && hipMemcpy(reference.data(), output, output_bytes,
                                   hipMemcpyDeviceToHost) == hipSuccess) {
      reference_hash = hash_bytes(reference);
      for (int repeat = 0; repeat < 8 && deterministic; ++repeat) {
        deterministic =
            run_rocblas(rocblas_gemm_algo_solution_index, solution) &&
            hipStreamSynchronize(stream) == hipSuccess &&
            hipMemcpy(repeated.data(), output, output_bytes,
                      hipMemcpyDeviceToHost) == hipSuccess &&
            repeated == reference;
      }
    } else {
      deterministic = false;
    }
    std::printf("%.*s\t%d\t%d\t%d\trocblas-solution\t%d\t%.3f\t%s "
                "atomics=%s repeat_bitwise=%s hash=%016llx\n",
                static_cast<int>(target.size()), target.data(), m, k, n,
                static_cast<int>(solution), ok ? median_us : 0.0F,
                ok ? "PASS" : "UNSUPPORTED", atomics_name,
                deterministic ? "PASS" : "FAIL",
                static_cast<unsigned long long>(reference_hash));
    if (!deterministic) {
      cleanup();
      return EXIT_FAILURE;
    }
  }
  if (!requested_solution_found) {
    std::fprintf(stderr, "requested solution %d is absent from %d entries\n",
                 requested_solution, static_cast<int>(solution_count));
    cleanup();
    return EXIT_FAILURE;
  }
  cleanup();
  return EXIT_SUCCESS;
}
