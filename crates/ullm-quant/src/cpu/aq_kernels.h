#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct ullm_aq_kernel_version {
    std::uint32_t major;
    std::uint32_t minor;
    std::uint32_t patch;
};

ullm_aq_kernel_version ullm_aq_get_kernel_version();

std::size_t ullm_aq_pack_nibbles(
    const std::uint8_t * low,
    const std::uint8_t * high,
    std::uint8_t * output,
    std::size_t len);

}

