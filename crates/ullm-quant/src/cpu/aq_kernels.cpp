#include "aq_kernels.h"

#include <algorithm>

extern "C" {

ullm_aq_kernel_version ullm_aq_get_kernel_version() {
    return ullm_aq_kernel_version{0, 1, 0};
}

std::size_t ullm_aq_pack_nibbles(
    const std::uint8_t * low,
    const std::uint8_t * high,
    std::uint8_t * output,
    std::size_t len) {
    if (low == nullptr || high == nullptr || output == nullptr) {
        return 0;
    }

    for (std::size_t i = 0; i < len; ++i) {
        const std::uint8_t lo = low[i] & 0x0f;
        const std::uint8_t hi = (high[i] & 0x0f) << 4;
        output[i] = static_cast<std::uint8_t>(lo | hi);
    }

    return len;
}

}

