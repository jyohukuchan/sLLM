// Exact-gfx1030 Phase 10 FP8 byte-decode emulation proof.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

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

void check(hipError_t status, const char *what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

__host__ __device__ float decode_e4m3fn(uint8_t bits) {
  const float sign = (bits & 0x80U) == 0 ? 1.0F : -1.0F;
  const uint8_t exponent = (bits >> 3U) & 0x0fU;
  const uint8_t mantissa = bits & 0x07U;
  if (exponent == 0)
    return mantissa == 0 ? copysignf(0.0F, sign)
                         : sign * static_cast<float>(mantissa) * ldexpf(1.0F, -9);
  if (exponent == 0x0fU && mantissa == 0x07U) return NAN;
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 7);
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

__global__ void fp8_outer_emulation(const uint8_t *__restrict__ activation,
                                    const float *__restrict__ activation_scale,
                                    const uint8_t *__restrict__ weight,
                                    const float *__restrict__ weight_scale,
                                    hip_bfloat16 *__restrict__ output,
                                    uint64_t m, uint64_t k, uint64_t n) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= m * n) return;
  const uint64_t row = index / n;
  const uint64_t column = index - row * n;
  float sum = 0.0F;
  for (uint64_t inner = 0; inner < k; ++inner)
    sum = fmaf(decode_e4m3fn(activation[row * k + inner]),
               decode_e4m3fn(weight[column * k + inner]), sum);
  output[index] = hip_bfloat16(sum * activation_scale[row] * weight_scale[column]);
}

template <typename T> struct DeviceBuffer {
  T *pointer = nullptr;
  explicit DeviceBuffer(size_t count) { check(hipMalloc(&pointer, count * sizeof(T)), "hipMalloc"); }
  ~DeviceBuffer() { if (pointer != nullptr) (void)hipFree(pointer); }
};

}  // namespace

int main(int argc, char **argv) {
  try {
    const uint64_t m = argc > 1 ? std::stoull(argv[1]) : 3;
    const uint64_t k = argc > 2 ? std::stoull(argv[2]) : 129;
    const uint64_t n = argc > 3 ? std::stoull(argv[3]) : 257;
    if (m == 0 || k == 0 || n == 0) throw std::runtime_error("dimensions must be positive");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    if (std::string(properties.gcnArchName).rfind("gfx1030", 0) != 0)
      throw std::runtime_error("emulation PoC requires exact gfx1030");

    std::vector<float> a_source(static_cast<size_t>(m * k));
    std::vector<float> b_source(static_cast<size_t>(n * k));
    for (size_t index = 0; index < a_source.size(); ++index)
      a_source[index] = std::sin(static_cast<float>(index) * 0.031F) * 2.0F;
    for (size_t index = 0; index < b_source.size(); ++index)
      b_source[index] = std::cos(static_cast<float>(index) * 0.019F) * 1.5F;
    std::vector<float> a_scale(m), b_scale(n);
    std::vector<uint8_t> a_fp8(a_source.size()), b_fp8(b_source.size());
    for (uint64_t row = 0; row < m; ++row) {
      float amax = 0.0F;
      for (uint64_t column = 0; column < k; ++column)
        amax = std::max(amax, std::fabs(a_source[row * k + column]));
      a_scale[row] = amax == 0.0F ? 1.0F : amax / 448.0F;
      for (uint64_t column = 0; column < k; ++column)
        a_fp8[row * k + column] = encode_e4m3fn(a_source[row * k + column] / a_scale[row]);
    }
    for (uint64_t row = 0; row < n; ++row) {
      float amax = 0.0F;
      for (uint64_t column = 0; column < k; ++column)
        amax = std::max(amax, std::fabs(b_source[row * k + column]));
      b_scale[row] = amax == 0.0F ? 1.0F : amax / 448.0F;
      for (uint64_t column = 0; column < k; ++column)
        b_fp8[row * k + column] = encode_e4m3fn(b_source[row * k + column] / b_scale[row]);
    }

    DeviceBuffer<uint8_t> da(a_fp8.size()), db(b_fp8.size());
    DeviceBuffer<float> das(a_scale.size()), dbs(b_scale.size());
    DeviceBuffer<hip_bfloat16> dd(static_cast<size_t>(m * n));
    check(hipMemcpy(da.pointer, a_fp8.data(), a_fp8.size(), hipMemcpyHostToDevice), "copy A");
    check(hipMemcpy(db.pointer, b_fp8.data(), b_fp8.size(), hipMemcpyHostToDevice), "copy B");
    check(hipMemcpy(das.pointer, a_scale.data(), a_scale.size() * sizeof(float), hipMemcpyHostToDevice), "copy A scales");
    check(hipMemcpy(dbs.pointer, b_scale.data(), b_scale.size() * sizeof(float), hipMemcpyHostToDevice), "copy B scales");
    const uint32_t threads = 256;
    const uint32_t blocks = static_cast<uint32_t>((m * n + threads - 1) / threads);
    hipLaunchKernelGGL(fp8_outer_emulation, dim3(blocks), dim3(threads), 0, nullptr,
                       da.pointer, das.pointer, db.pointer, dbs.pointer, dd.pointer, m, k, n);
    check(hipGetLastError(), "fp8_outer_emulation launch");
    check(hipDeviceSynchronize(), "fp8_outer_emulation synchronize");
    std::vector<hip_bfloat16> output(static_cast<size_t>(m * n));
    check(hipMemcpy(output.data(), dd.pointer, output.size() * sizeof(hip_bfloat16), hipMemcpyDeviceToHost), "copy output");
    double max_abs_error = 0.0, max_rel_error = 0.0;
    for (uint64_t row = 0; row < m; ++row) {
      for (uint64_t column = 0; column < n; ++column) {
        double expected = 0.0;
        for (uint64_t inner = 0; inner < k; ++inner)
          expected += static_cast<double>(decode_e4m3fn(a_fp8[row * k + inner]) * a_scale[row]) *
                      static_cast<double>(decode_e4m3fn(b_fp8[column * k + inner]) * b_scale[column]);
        const double actual = static_cast<float>(output[row * n + column]);
        const double absolute = std::fabs(actual - expected);
        max_abs_error = std::max(max_abs_error, absolute);
        max_rel_error = std::max(max_rel_error, absolute / std::max(1.0, std::fabs(expected)));
      }
    }
    if (!std::isfinite(max_abs_error) || max_rel_error > 0.02)
      throw std::runtime_error("numerical oracle failed");
    std::printf("phase10_fp8_gfx1030: PASS target=%s provider=byte-decode-emulation native_fp8=false m=%llu k=%llu n=%llu max_abs=%.9g max_rel=%.9g fallback=false\n",
                properties.gcnArchName, static_cast<unsigned long long>(m),
                static_cast<unsigned long long>(k), static_cast<unsigned long long>(n),
                max_abs_error, max_rel_error);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase10_fp8_gfx1030: FAIL: %s\n", error.what());
    return 1;
  }
}
