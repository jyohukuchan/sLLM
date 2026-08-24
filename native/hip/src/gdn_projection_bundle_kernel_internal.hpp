#ifndef SLLM_GDN_PROJECTION_BUNDLE_KERNEL_INTERNAL_HPP
#define SLLM_GDN_PROJECTION_BUNDLE_KERNEL_INTERNAL_HPP

#include <cstdint>
#include <hip/hip_runtime.h>

namespace sllm_gdn_projection_bundle_kernel {
hipError_t launch(const uint16_t *, const uint16_t *, const uint16_t *,
                  const uint16_t *, const uint16_t *, uint16_t *, uint16_t *,
                  uint16_t *, uint16_t *, uint64_t, hipStream_t) noexcept;
}

#endif
