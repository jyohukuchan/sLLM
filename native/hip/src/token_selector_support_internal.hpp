#ifndef SLLM_TOKEN_SELECTOR_SUPPORT_INTERNAL_HPP
#define SLLM_TOKEN_SELECTOR_SUPPORT_INTERNAL_HPP

#include "sllm/hip.h"

#include <cstddef>
#include <cstdint>

namespace sllm_token_selector_support {

inline constexpr uint32_t kVersionV1 = UINT32_C(1);
inline constexpr uint32_t kMaxCountV1 = UINT32_C(20);
inline constexpr uint32_t kBytesV1 = UINT32_C(256);
// A one-tile fixed-K20 selector reserves two 20-record regions (320 bytes).
// The 256-byte record is written at workspace base only after those reads.
inline constexpr uint32_t kMinimumWorkspaceBytesV1 = UINT32_C(320);

/* Private fixed-K20 p/q side-output layout.  It is written byte-wise by the
 * kernel because the workspace is exposed as a U8 view and may be offset by a
 * caller before it is retained. */
struct RecordV1 final {
  uint32_t version;
  uint32_t status;
  uint32_t count;
  uint32_t reserved;
  uint32_t ids[kMaxCountV1];
  double probabilities[kMaxCountV1];
};

static_assert(sizeof(RecordV1) == kBytesV1,
              "fixed-K20 support record ABI must be 256 bytes");
static_assert(offsetof(RecordV1, ids) == 16U,
              "fixed-K20 support ids offset changed");
static_assert(offsetof(RecordV1, probabilities) == 96U,
              "fixed-K20 support probability offset changed");

} // namespace sllm_token_selector_support

#endif // SLLM_TOKEN_SELECTOR_SUPPORT_INTERNAL_HPP
