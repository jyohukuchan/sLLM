#ifndef SLLM_RMSNORM_KERNEL_INTERNAL_HPP
#define SLLM_RMSNORM_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_rmsnorm_kernel {

constexpr const char *kLogicalKernelId = "rmsnorm.baseline.wave32.v1";
constexpr const char *kDeviceSymbol = "sllm_rmsnorm_baseline_wave32_v1";

#if defined(__HIPCC__) || defined(__CUDACC__)
#define SLLM_RMSNORM_HOST_DEVICE __host__ __device__
#else
#define SLLM_RMSNORM_HOST_DEVICE
#endif

/*
 * Convert one IEEE-754 binary32 value to BF16 using round-to-nearest-even.
 * NaNs are canonicalized to quiet BF16 NaNs while retaining the sign and the
 * high payload bits that BF16 can represent.  This is deliberately header
 * local so the device kernel and host contract test use the same conversion.
 */
SLLM_RMSNORM_HOST_DEVICE static inline uint32_t
float_bits(const float value) noexcept {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return __float_as_uint(value);
#else
  uint32_t bits = 0U;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

SLLM_RMSNORM_HOST_DEVICE static inline uint16_t
float_to_bf16_rne_bits(const float value) noexcept {
  const uint32_t bits = float_bits(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);

  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      constexpr uint16_t quiet_nan = UINT16_C(0x7fc0);
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | quiet_nan | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }

  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

#undef SLLM_RMSNORM_HOST_DEVICE

hipError_t launch(const uint16_t *activation, const uint16_t *raw_scale,
                  uint16_t *output, uint32_t normalized_size,
                  uint32_t row_count, float epsilon,
                  hipStream_t stream) noexcept;

} // namespace sllm_rmsnorm_kernel

#endif // SLLM_RMSNORM_KERNEL_INTERNAL_HPP
