#ifndef SLLM_LOWP_TEST_COMMON_HPP
#define SLLM_LOWP_TEST_COMMON_HPP

#include <lowp/lowp.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "unknown"
#endif

namespace lowp_test {

inline bool &hip_cleanup_ok() {
  static bool ok = true;
  return ok;
}

inline bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << ": " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

struct DeviceBuffer final {
  void *data = nullptr;
  std::size_t bytes = 0U;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (data != nullptr) {
      const hipError_t status = hipFree(data);
      if (status != hipSuccess) {
        hip_cleanup_ok() = false;
        std::cerr << "hipFree: " << hipGetErrorName(status) << " ("
                  << hipGetErrorString(status) << ")\n";
      }
    }
  }

  bool allocate(const std::size_t size) {
    bytes = size;
    return hip_ok(hipMalloc(&data, size), "hipMalloc");
  }
};

inline bool copy_to_device(const DeviceBuffer &destination, const void *source,
                           const std::size_t bytes) {
  return bytes <= destination.bytes &&
         hip_ok(
             hipMemcpy(destination.data, source, bytes, hipMemcpyHostToDevice),
             "hipMemcpyHostToDevice");
}

inline bool copy_from_device(void *destination, const DeviceBuffer &source,
                             const std::size_t bytes) {
  return bytes <= source.bytes &&
         hip_ok(
             hipMemcpy(destination, source.data, bytes, hipMemcpyDeviceToHost),
             "hipMemcpyDeviceToHost");
}

inline uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

inline float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline float decode_e4m3fn(const uint8_t raw) {
  const uint32_t sign = static_cast<uint32_t>(raw & 0x80U) << 24U;
  const uint32_t magnitude = raw & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    float result = static_cast<float>(mantissa) * 0x1p-9F;
    uint32_t bits = 0U;
    std::memcpy(&bits, &result, sizeof(bits));
    bits |= sign;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }
  if (magnitude == 0x7fU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const uint32_t bits = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline float decode_e8m0(const uint8_t raw) {
  if (raw == 0xffU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const uint32_t bits =
      raw == 0U ? UINT32_C(0x00400000) : static_cast<uint32_t>(raw) << 23U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline bool make_mxfp8_plan(const uint64_t m, const uint64_t n,
                            const uint64_t k, lowp_matmul_plan_t *const plan) {
  lowp_matmul_request_t request{};
  request.struct_size = sizeof(request);
  request.version = LOWP_ABI_VERSION;
  request.format = LOWP_MXFP8_E4M3_W8A8;
  request.target = lowp_target_from_name(SLLM_TEST_EXPECTED_TARGET);
  request.weight_layout = LOWP_ROW_MAJOR_BLOCK_SCALED;
  request.activation_layout = LOWP_ROW_MAJOR_BLOCK_SCALED;
  request.m = m;
  request.n = n;
  request.k = k;
  plan->struct_size = sizeof(*plan);
  plan->version = LOWP_ABI_VERSION;
  const lowp_status_t status = lowp_matmul_plan(&request, plan);
  if (status != LOWP_SUCCESS || plan->supported == 0U) {
    std::cerr << "lowp_matmul_plan failed status=" << status << " reason="
              << (plan->reason == nullptr ? "<none>" : plan->reason) << '\n';
    return false;
  }
  return true;
}

inline bool finite_bf16(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

} // namespace lowp_test

#endif
