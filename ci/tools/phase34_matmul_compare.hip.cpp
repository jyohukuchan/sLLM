// Bounded Phase 34 comparison for the existing gfx1030 BF16 providers.
// This tool is not linked into sLLM. It compares the production tiled16
// kernel with hipblasGemmEx using the same buffers, stream, dtype contract,
// and deterministic operands.

#include "matmul_kernel_internal.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Timing {
  double first_ns = 0.0;
  double median_ns = 0.0;
  double mad_ns = 0.0;
  double min_ns = 0.0;
  double max_ns = 0.0;
};

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U) {
      return static_cast<uint16_t>((bits >> 16U) | UINT32_C(0x0040));
    }
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

uint64_t parse_u64(const char *const text, const char *const label) {
  if (text == nullptr || text[0] == '\0' || text[0] == '-') {
    std::fprintf(stderr, "%s must be a positive integer\n", label);
    std::exit(2);
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == nullptr || *end != '\0' || value == 0ULL) {
    std::fprintf(stderr, "%s must be a positive integer\n", label);
    std::exit(2);
  }
  return static_cast<uint64_t>(value);
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2U;
  return values.size() % 2U == 0U ? (values[middle - 1U] + values[middle]) / 2.0
                                  : values[middle];
}

template <typename Launch>
bool measure(Launch &&launch, const int warmups, const int measured,
             const hipStream_t stream, Timing *const timing) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!hip_ok(hipEventCreate(&start), "start event create") ||
      !hip_ok(hipEventCreate(&stop), "stop event create")) {
    return false;
  }
  auto one = [&](double *const elapsed_ns) {
    if (!hip_ok(hipEventRecord(start, stream), "start event record") ||
        !launch() ||
        !hip_ok(hipEventRecord(stop, stream), "stop event record") ||
        !hip_ok(hipEventSynchronize(stop), "stop event synchronize")) {
      return false;
    }
    float elapsed_ms = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&elapsed_ms, start, stop),
                "event elapsed time")) {
      return false;
    }
    *elapsed_ns = static_cast<double>(elapsed_ms) * 1000000.0;
    return true;
  };
  if (!one(&timing->first_ns)) {
    return false;
  }
  double ignored = 0.0;
  for (int iteration = 0; iteration < warmups; ++iteration) {
    if (!one(&ignored)) {
      return false;
    }
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(measured));
  for (int iteration = 0; iteration < measured; ++iteration) {
    double elapsed_ns = 0.0;
    if (!one(&elapsed_ns)) {
      return false;
    }
    samples.push_back(elapsed_ns);
  }
  timing->median_ns = median(samples);
  std::vector<double> deviations;
  deviations.reserve(samples.size());
  for (const double sample : samples) {
    deviations.push_back(std::fabs(sample - timing->median_ns));
  }
  timing->mad_ns = median(deviations);
  timing->min_ns = *std::min_element(samples.begin(), samples.end());
  timing->max_ns = *std::max_element(samples.begin(), samples.end());
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return true;
}

uint64_t digest_words(const std::vector<uint16_t> &words) {
  uint64_t digest = UINT64_C(1469598103934665603);
  for (const uint16_t word : words) {
    digest ^= static_cast<uint8_t>(word & UINT16_C(0xff));
    digest *= UINT64_C(1099511628211);
    digest ^= static_cast<uint8_t>(word >> 8U);
    digest *= UINT64_C(1099511628211);
  }
  return digest;
}

double bf16_half_ulp(const double reference) {
  if (reference == 0.0 || std::fabs(reference) < std::ldexp(1.0, -126)) {
    return std::ldexp(1.0, -134);
  }
  int exponent = 0;
  (void)std::frexp(std::fabs(reference), &exponent);
  return std::ldexp(1.0, exponent - 9);
}

struct OracleResult {
  double max_tiled_error = 0.0;
  double max_blas_error = 0.0;
  uint32_t tiled_bound_violations = 0U;
  uint32_t blas_bound_violations = 0U;
  uint32_t samples = 0U;
};

OracleResult sampled_oracle(const uint64_t m, const uint64_t k,
                            const uint64_t n,
                            const std::vector<uint16_t> &activation,
                            const std::vector<uint16_t> &weight,
                            const std::vector<uint16_t> &tiled,
                            const std::vector<uint16_t> &blas) {
  const uint64_t rows[] = {0U, m / 3U, m / 2U, m - 1U};
  const uint64_t columns[] = {0U, n / 7U, n / 3U, n / 2U, n - 1U};
  const double unit_roundoff = std::ldexp(1.0, -24);
  const double gamma = static_cast<double>(k) * unit_roundoff /
                       (1.0 - static_cast<double>(k) * unit_roundoff);
  OracleResult result{};
  for (const uint64_t row : rows) {
    for (const uint64_t column : columns) {
      double reference = 0.0;
      double sum_abs = 0.0;
      for (uint64_t reduction = 0U; reduction < k; ++reduction) {
        const double left = static_cast<double>(
            bf16_to_f32(activation[static_cast<size_t>(row * k + reduction)]));
        const double right = static_cast<double>(
            bf16_to_f32(weight[static_cast<size_t>(column * k + reduction)]));
        reference += left * right;
        sum_abs += std::fabs(left * right);
      }
      const size_t index = static_cast<size_t>(row * n + column);
      const double tiled_error =
          std::fabs(static_cast<double>(bf16_to_f32(tiled[index])) - reference);
      const double blas_error =
          std::fabs(static_cast<double>(bf16_to_f32(blas[index])) - reference);
      const double bound = gamma * sum_abs + bf16_half_ulp(reference);
      result.max_tiled_error = std::max(result.max_tiled_error, tiled_error);
      result.max_blas_error = std::max(result.max_blas_error, blas_error);
      result.tiled_bound_violations += tiled_error > bound ? 1U : 0U;
      result.blas_bound_violations += blas_error > bound ? 1U : 0U;
      ++result.samples;
    }
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 7 || std::strcmp(argv[1], "gfx1030") != 0) {
    std::fprintf(stderr, "usage: phase34_matmul_compare gfx1030 M K N WARMUPS "
                         "MEASURED\n");
    return 2;
  }
  const uint64_t m = parse_u64(argv[2], "M");
  const uint64_t k = parse_u64(argv[3], "K");
  const uint64_t n = parse_u64(argv[4], "N");
  const uint64_t warmups_u64 = parse_u64(argv[5], "WARMUPS");
  const uint64_t measured_u64 = parse_u64(argv[6], "MEASURED");
  if (m > 16385U || k > 9216U || n > 9216U || warmups_u64 > 100U ||
      measured_u64 > 100U || m > static_cast<uint64_t>(INT32_MAX) ||
      k > static_cast<uint64_t>(INT32_MAX) ||
      n > static_cast<uint64_t>(INT32_MAX)) {
    std::fprintf(stderr, "shape or sample count is outside Phase 34 bounds\n");
    return 2;
  }
  const int warmups = static_cast<int>(warmups_u64);
  const int measured = static_cast<int>(measured_u64);
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "device properties") ||
      std::strcmp(properties.gcnArchName, "gfx1030") != 0) {
    std::fprintf(stderr, "visible device is not exact gfx1030\n");
    return 1;
  }
  if (m > std::numeric_limits<size_t>::max() / k ||
      n > std::numeric_limits<size_t>::max() / k ||
      m > std::numeric_limits<size_t>::max() / n) {
    std::fprintf(stderr, "shape element count overflow\n");
    return 2;
  }
  const size_t activation_count = static_cast<size_t>(m * k);
  const size_t weight_count = static_cast<size_t>(n * k);
  const size_t output_count = static_cast<size_t>(m * n);
  std::vector<uint16_t> activation(activation_count);
  std::vector<uint16_t> weight(weight_count);
  // Mix signs and exponent ranges so a provider-order difference is not
  // hidden by an exactly representable fixed-point reduction.
  constexpr uint16_t finite_pattern[] = {
      UINT16_C(0x3f80), UINT16_C(0xbf80), UINT16_C(0x3c00), UINT16_C(0xbc00),
      UINT16_C(0x4000), UINT16_C(0xc000), UINT16_C(0x3eab), UINT16_C(0xbe2b),
      UINT16_C(0x3880), UINT16_C(0x4180), UINT16_C(0xc180), UINT16_C(0x3d4d),
      UINT16_C(0xbdcd)};
  constexpr size_t pattern_size =
      sizeof(finite_pattern) / sizeof(finite_pattern[0]);
  for (size_t index = 0; index < activation.size(); ++index) {
    activation[index] =
        finite_pattern[(index * 17U + index / 97U + index / 4099U) %
                       pattern_size];
  }
  for (size_t index = 0; index < weight.size(); ++index) {
    weight[index] =
        finite_pattern[(index * 29U + index / 53U + index / 2053U + 3U) %
                       pattern_size];
  }

  uint16_t *device_activation = nullptr;
  uint16_t *device_weight = nullptr;
  uint16_t *device_tiled = nullptr;
  uint16_t *device_blas = nullptr;
  hipStream_t stream = nullptr;
  hipblasHandle_t handle = nullptr;
  const size_t activation_bytes = activation_count * sizeof(uint16_t);
  const size_t weight_bytes = weight_count * sizeof(uint16_t);
  const size_t output_bytes = output_count * sizeof(uint16_t);
  if (!hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
              "stream create") ||
      !hip_ok(hipMalloc(&device_activation, activation_bytes),
              "activation allocation") ||
      !hip_ok(hipMalloc(&device_weight, weight_bytes), "weight allocation") ||
      !hip_ok(hipMalloc(&device_tiled, output_bytes), "tiled allocation") ||
      !hip_ok(hipMalloc(&device_blas, output_bytes), "blas allocation") ||
      !hip_ok(hipMemcpyAsync(device_activation, activation.data(),
                             activation_bytes, hipMemcpyHostToDevice, stream),
              "activation upload") ||
      !hip_ok(hipMemcpyAsync(device_weight, weight.data(), weight_bytes,
                             hipMemcpyHostToDevice, stream),
              "weight upload") ||
      !hip_ok(hipStreamSynchronize(stream), "upload synchronize")) {
    return 1;
  }
  const auto handle_start = std::chrono::steady_clock::now();
  if (!blas_ok(hipblasCreate(&handle), "hipblasCreate") ||
      !blas_ok(hipblasSetStream(handle, stream), "hipblasSetStream")) {
    return 1;
  }
  const auto handle_stop = std::chrono::steady_clock::now();
  const uint64_t handle_create_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(handle_stop -
                                                           handle_start)
          .count());
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto tiled_launch = [&]() {
    return hip_ok(sllm_matmul_kernel::launch(
                      device_activation, device_weight, device_tiled, m, k, n,
                      sllm_matmul_kernel::KernelVariant::PrefillTiled16,
                      stream),
                  "tiled16 launch");
  };
  auto blas_launch = [&]() {
    return blas_ok(
        hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(n),
                      static_cast<int>(m), static_cast<int>(k), &alpha,
                      device_weight, HIPBLAS_R_16B, static_cast<int>(k),
                      device_activation, HIPBLAS_R_16B, static_cast<int>(k),
                      &beta, device_blas, HIPBLAS_R_16B, static_cast<int>(n),
                      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
        "hipblasGemmEx");
  };
  Timing tiled_timing{};
  Timing blas_timing{};
  if (!measure(tiled_launch, warmups, measured, stream, &tiled_timing) ||
      !measure(blas_launch, warmups, measured, stream, &blas_timing)) {
    return 1;
  }
  if (!tiled_launch() || !blas_launch() ||
      !hip_ok(hipStreamSynchronize(stream), "final synchronize")) {
    return 1;
  }
  std::vector<uint16_t> tiled(output_count);
  std::vector<uint16_t> blas(output_count);
  if (!hip_ok(hipMemcpy(tiled.data(), device_tiled, output_bytes,
                        hipMemcpyDeviceToHost),
              "tiled readback") ||
      !hip_ok(hipMemcpy(blas.data(), device_blas, output_bytes,
                        hipMemcpyDeviceToHost),
              "blas readback")) {
    return 1;
  }
  const uint64_t tiled_digest = digest_words(tiled);
  const uint64_t blas_digest = digest_words(blas);
  uint64_t mismatch_count = 0U;
  double max_provider_abs_difference = 0.0;
  for (size_t index = 0; index < output_count; ++index) {
    mismatch_count += tiled[index] != blas[index] ? 1U : 0U;
    max_provider_abs_difference =
        std::max(max_provider_abs_difference,
                 std::fabs(static_cast<double>(bf16_to_f32(tiled[index])) -
                           static_cast<double>(bf16_to_f32(blas[index]))));
  }
  const OracleResult oracle =
      sampled_oracle(m, k, n, activation, weight, tiled, blas);
  if (!tiled_launch() || !blas_launch() ||
      !hip_ok(hipStreamSynchronize(stream), "repeat synchronize")) {
    return 1;
  }
  std::vector<uint16_t> tiled_repeat(output_count);
  std::vector<uint16_t> blas_repeat(output_count);
  if (!hip_ok(hipMemcpy(tiled_repeat.data(), device_tiled, output_bytes,
                        hipMemcpyDeviceToHost),
              "tiled repeat readback") ||
      !hip_ok(hipMemcpy(blas_repeat.data(), device_blas, output_bytes,
                        hipMemcpyDeviceToHost),
              "blas repeat readback")) {
    return 1;
  }
  const bool tiled_repeat_equal = tiled == tiled_repeat;
  const bool blas_repeat_equal = blas == blas_repeat;
  const bool pass = oracle.tiled_bound_violations == 0U &&
                    oracle.blas_bound_violations == 0U && tiled_repeat_equal &&
                    blas_repeat_equal;
  std::printf("{\"schema_version\":\"phase34-matmul-compare-v1\","
              "\"state\":\"%s\",\"target\":\"gfx1030\",\"m\":%llu,"
              "\"k\":%llu,\"n\":%llu,\"warmups\":%d,\"measured\":%d,"
              "\"handle_create_ns\":%llu,"
              "\"tiled\":{\"first_ns\":%.0f,\"median_ns\":%.0f,"
              "\"mad_ns\":%.0f,\"min_ns\":%.0f,\"max_ns\":%.0f,"
              "\"digest\":\"%016llx\",\"repeat_equal\":%s},"
              "\"hipblas\":{\"first_ns\":%.0f,\"median_ns\":%.0f,"
              "\"mad_ns\":%.0f,\"min_ns\":%.0f,\"max_ns\":%.0f,"
              "\"digest\":\"%016llx\",\"repeat_equal\":%s},"
              "\"hipblas_over_tiled\":%.9f,\"mismatch_count\":%llu,"
              "\"max_provider_abs_difference\":%.9g,"
              "\"oracle_samples\":%u,\"tiled_max_abs_error\":%.9g,"
              "\"hipblas_max_abs_error\":%.9g,"
              "\"tiled_bound_violations\":%u,"
              "\"hipblas_bound_violations\":%u,\"workspace_bytes\":0}\n",
              pass ? "PASS" : "FAIL", static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(n), warmups, measured,
              static_cast<unsigned long long>(handle_create_ns),
              tiled_timing.first_ns, tiled_timing.median_ns,
              tiled_timing.mad_ns, tiled_timing.min_ns, tiled_timing.max_ns,
              static_cast<unsigned long long>(tiled_digest),
              tiled_repeat_equal ? "true" : "false", blas_timing.first_ns,
              blas_timing.median_ns, blas_timing.mad_ns, blas_timing.min_ns,
              blas_timing.max_ns, static_cast<unsigned long long>(blas_digest),
              blas_repeat_equal ? "true" : "false",
              blas_timing.median_ns / tiled_timing.median_ns,
              static_cast<unsigned long long>(mismatch_count),
              max_provider_abs_difference, oracle.samples,
              oracle.max_tiled_error, oracle.max_blas_error,
              oracle.tiled_bound_violations, oracle.blas_bound_violations);
  (void)hipblasDestroy(handle);
  (void)hipFree(device_activation);
  (void)hipFree(device_weight);
  (void)hipFree(device_tiled);
  (void)hipFree(device_blas);
  (void)hipStreamDestroy(stream);
  return pass ? 0 : 1;
}
