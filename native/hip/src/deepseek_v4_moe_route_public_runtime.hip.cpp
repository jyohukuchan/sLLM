#include "deepseek_v4_moe_route_api.hpp"
#include "deepseek_v4_moe_route_kernel_internal.hpp"

/* Keep the established public runtime implementation byte-for-byte intact.
 * This additive translation unit appends the DeepSeek V4-specific lifecycle
 * while sharing its opaque handle registry and completion accounting. */
#include "public_runtime.hip.cpp"

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
namespace sllm_deepseek_v4_moe_route_kernel {
namespace {
std::atomic<int32_t> test_device_status{SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK};
}

hipError_t launch(const uint16_t *, const float *, const int32_t *, int32_t *,
                  float *, int32_t *, int32_t *, int32_t *, int32_t *,
                  int32_t *const status, uint64_t, uint32_t, uint32_t, float,
                  hipStream_t) noexcept {
  if (status == nullptr) {
    return hipErrorInvalidValue;
  }
  *status = test_device_status.load(std::memory_order_relaxed);
  return hipSuccess;
}
} // namespace sllm_deepseek_v4_moe_route_kernel

extern "C" void
sllm_test_deepseek_v4_moe_route_device_status(const int32_t status) noexcept {
  sllm_deepseek_v4_moe_route_kernel::test_device_status.store(
      status, std::memory_order_relaxed);
}
#endif

#include "deepseek_v4_moe_route_runtime.inc"
