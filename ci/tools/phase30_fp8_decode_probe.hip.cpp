#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

__global__ void decode_all_e4m3fn(float *output) {
  const unsigned int code = blockIdx.x * blockDim.x + threadIdx.x;
  if (code < 256U) {
#if defined(__gfx1201__)
    output[code] = __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(code), 0);
#else
    output[code] = 0.0F;
#endif
  }
}

float decode_reference(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits >> 7U);
  const uint32_t exponent = static_cast<uint32_t>((bits >> 3U) & 0x0fU);
  const uint32_t mantissa = static_cast<uint32_t>(bits & 0x07U);
  if (exponent == 0x0fU && mantissa == 0x07U) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float magnitude = 0.0F;
  if (exponent == 0U) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -9);
  } else {
    magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
  }
  return sign == 0U ? magnitude : -magnitude;
}

uint32_t float_bits(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool check_hip(const hipError_t status, const char *operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}

} // namespace

int main() {
  int device = 0;
  hipDeviceProp_t properties{};
  if (!check_hip(hipGetDevice(&device), "hipGetDevice") ||
      !check_hip(hipGetDeviceProperties(&properties, device),
                 "hipGetDeviceProperties")) {
    return 2;
  }
  const std::string architecture(properties.gcnArchName);
  if (architecture.rfind("gfx1201", 0) != 0) {
    std::fprintf(stderr,
                 "phase30_fp8_decode: FAIL target=%s reason=wrong-target\n",
                 architecture.c_str());
    return 2;
  }

  float *device_output = nullptr;
  if (!check_hip(hipMalloc(&device_output, 256U * sizeof(float)),
                 "hipMalloc")) {
    return 2;
  }
  decode_all_e4m3fn<<<1, 256>>>(device_output);
  if (!check_hip(hipGetLastError(), "decode_all_e4m3fn launch") ||
      !check_hip(hipDeviceSynchronize(), "decode_all_e4m3fn synchronize")) {
    static_cast<void>(hipFree(device_output));
    return 2;
  }

  std::vector<float> actual(256U);
  if (!check_hip(hipMemcpy(actual.data(), device_output,
                           actual.size() * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "hipMemcpy")) {
    static_cast<void>(hipFree(device_output));
    return 2;
  }
  static_cast<void>(hipFree(device_output));

  unsigned int mismatches = 0U;
  unsigned int nan_codes = 0U;
  for (unsigned int code = 0U; code < 256U; ++code) {
    const float expected = decode_reference(static_cast<uint8_t>(code));
    if (std::isnan(expected)) {
      ++nan_codes;
      if (!std::isnan(actual[code])) {
        ++mismatches;
      }
    } else if (float_bits(actual[code]) != float_bits(expected)) {
      ++mismatches;
    }
  }

  std::printf("phase30_fp8_decode: %s target=%s provider=native-scalar-e4m3fn "
              "codes=256 nan_codes=%u mismatches=%u fallback=false\n",
              mismatches == 0U ? "PASS" : "FAIL", architecture.c_str(),
              nan_codes, mismatches);
  return mismatches == 0U ? 0 : 1;
}
