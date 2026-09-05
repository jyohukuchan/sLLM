// Exhaustive FP16 storage conversion oracle for Phase 78 attention.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void phase78_convert_all_fp16(uint32_t *output) {
  const uint32_t raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= 65536U)
    return;
  // Keep infinity and NaN payload/sign bits identical to the storage decoder.
  if ((raw & 0x7c00U) == 0x7c00U) {
    output[raw] =
        ((raw & 0x8000U) << 16U) | 0x7f800000U | ((raw & 0x3ffU) << 13U);
  } else {
    output[raw] = __float_as_uint(
        __half2float(__ushort_as_half(static_cast<uint16_t>(raw))));
  }
}

int main(int argc, char **argv) {
  if (argc != 2 || (std::strcmp(argv[1], "gfx1030") != 0 &&
                    std::strcmp(argv[1], "gfx1201") != 0))
    return 2;
  int count = 0;
  hipDeviceProp_t properties{};
  if (hipGetDeviceCount(&count) != hipSuccess || count != 1 ||
      hipSetDevice(0) != hipSuccess ||
      hipGetDeviceProperties(&properties, 0) != hipSuccess ||
      std::strncmp(properties.gcnArchName, argv[1], 7U) != 0)
    return 3;
  std::printf("target=%s pci=%04x:%02x:%02x count=%d\n", properties.gcnArchName,
              properties.pciDomainID, properties.pciBusID,
              properties.pciDeviceID, count);
  uint32_t *device_output = nullptr;
  std::vector<uint32_t> output(65536U);
  if (hipMalloc(&device_output, output.size() * sizeof(uint32_t)) != hipSuccess)
    return 4;
  hipLaunchKernelGGL(phase78_convert_all_fp16, dim3(256), dim3(256), 0, 0,
                     device_output);
  bool ok =
      hipGetLastError() == hipSuccess &&
      hipMemcpy(output.data(), device_output, output.size() * sizeof(uint32_t),
                hipMemcpyDeviceToHost) == hipSuccess;
  uint32_t mismatches = 0;
  for (uint32_t raw = 0; ok && raw < 65536U; ++raw) {
    const uint32_t exponent = (raw >> 10U) & 31U;
    const uint32_t fraction = raw & 1023U;
    uint32_t expected;
    if (exponent == 31U) {
      expected = ((raw & 0x8000U) << 16U) | 0x7f800000U | (fraction << 13U);
    } else {
      const double magnitude =
          exponent == 0U
              ? std::ldexp(static_cast<double>(fraction), -24)
              : std::ldexp(1.0 + static_cast<double>(fraction) / 1024.0,
                           static_cast<int>(exponent) - 15);
      const float value =
          static_cast<float>((raw & 0x8000U) ? -magnitude : magnitude);
      std::memcpy(&expected, &value, sizeof(expected));
    }
    if (output[raw] != expected) {
      if (mismatches < 8U)
        std::printf("raw=%04x actual=%08x expected=%08x\n", raw, output[raw],
                    expected);
      ++mismatches;
    }
  }
  ok = hipFree(device_output) == hipSuccess && ok;
  std::printf("encodings=65536 bit_mismatches=%u cleanup=%s status=%s\n",
              mismatches, ok ? "PASS" : "FAIL",
              ok && mismatches == 0 ? "PASS" : "FAIL");
  return ok && mismatches == 0 ? 0 : 1;
}
