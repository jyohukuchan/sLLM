// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "ullm_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#if defined(ULLM_HAVE_HIP_RUNTIME_API)
#include <hip/hip_runtime_api.h>
#include <hip/hip_deprecated.h>
#endif

namespace {

constexpr unsigned int kAq4MatvecTop1RowsPerBlock = 8u;
constexpr unsigned int kRocwmmaCachedPrefixValueGroupWidth = 64u;
// The tiled AQ4 GEMM is intentionally conservative in its first production slice.
// Keep these values in one place so the host classifier, launcher, and HIPRTC source
// cannot silently drift apart.
constexpr unsigned int kAq4TiledBm = 8u;
constexpr unsigned int kAq4TiledBn = 32u;
constexpr unsigned int kAq4TiledBk = 128u;
constexpr unsigned int kAq4TiledThreads = 256u;
static_assert(kAq4TiledBm * kAq4TiledBn == kAq4TiledThreads, "AQ4 tiled block must have one thread per output");
static_assert(kAq4TiledBk % 16u == 0u, "AQ4 tiled K tile must preserve group16 geometry");
constexpr unsigned int kAq4RegisterBm4 = 4u;
constexpr unsigned int kAq4RegisterBm8 = 8u;
constexpr unsigned int kAq4RegisterGroup8 = 8u;
constexpr unsigned int kAq4RegisterGroup16 = 16u;
constexpr unsigned int kAq4RegisterRowsPerBlock = 32u;
constexpr unsigned int kAq4RegisterLanesPerRow = 8u;
constexpr unsigned int kAq4RegisterThreads = 256u;
static_assert(
    kAq4RegisterRowsPerBlock * kAq4RegisterLanesPerRow == kAq4RegisterThreads,
    "AQ4 register tile must map eight lanes to each row");
// One WMMA workgroup contains eight wave32s, each producing a 16x16 column slice of a
// group16/M=128 output tile with 16 rows and 128 batch columns.
constexpr unsigned int kAq4WmmaPrototypeBm = 128u;
constexpr unsigned int kAq4WmmaPrototypeBn = 16u;
constexpr unsigned int kAq4WmmaPrototypeBk = 32u;
constexpr unsigned int kAq4WmmaPrototypeV3Bk = 64u;
constexpr unsigned int kAq4WmmaPrototypeThreads = 256u;
static_assert(
    (kAq4WmmaPrototypeBm / 16u) * 32u == kAq4WmmaPrototypeThreads,
    "AQ4 WMMA prototype requires eight wave32s");
static_assert(
    kAq4WmmaPrototypeBk == 32u,
    "AQ4 WMMA prototype Wide-K tile must contain two K=16 WMMA operations");
static_assert(
    kAq4WmmaPrototypeV3Bk == 2u * kAq4WmmaPrototypeBk,
    "AQ4 WMMA prototype v3 stage must contain two Wide-K microtiles");

thread_local std::string last_error;

void set_error(const char *message) {
    last_error = message == nullptr ? "" : message;
}

void copy_cstr(char *dst, size_t dst_len, const std::string &src) {
    if (dst == nullptr || dst_len == 0) {
        return;
    }
    const size_t copy_len = std::min(dst_len - 1, src.size());
    std::memcpy(dst, src.data(), copy_len);
    dst[copy_len] = '\0';
}

bool aq4_rows_per_block_is_valid(unsigned long value) {
    return value >= 1ul && value <= 32ul && 256ul % value == 0ul;
}

unsigned int aq4_rows_per_block_from_env(
    const char *primary_env_name,
    const char *fallback_env_name,
    unsigned int fallback) {
    auto parse = [](const char *env_name, unsigned int *out) {
        if (env_name == nullptr || out == nullptr) {
            return false;
        }
        const char *raw = std::getenv(env_name);
        if (raw == nullptr || raw[0] == '\0') {
            return false;
        }
        char *end = nullptr;
        const unsigned long value = std::strtoul(raw, &end, 10);
        if (end == raw || *end != '\0' || !aq4_rows_per_block_is_valid(value)) {
            return false;
        }
        *out = static_cast<unsigned int>(value);
        return true;
    };

    unsigned int value = fallback;
    if (parse(primary_env_name, &value)) {
        return value;
    }
    if (parse(fallback_env_name, &value)) {
        return value;
    }
    return fallback;
}

unsigned int aq4_matvec_top1_rows_per_block_from_env() {
    return aq4_rows_per_block_from_env(
        "ULLM_AQ4_MATVEC_TOP1_RPB",
        nullptr,
        kAq4MatvecTop1RowsPerBlock);
}

unsigned int block_size_from_env(const char *env_name, unsigned int fallback) {
    const char *raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 32ul || value > 256ul ||
        (value & (value - 1ul)) != 0ul) {
        return fallback;
    }
    return static_cast<unsigned int>(value);
}

bool rocwmma_cached_prefix_value_group_width_is_valid(unsigned long value) {
    return value == 16ul || value == 32ul || value == 64ul || value == 128ul || value == 256ul;
}

unsigned int rocwmma_cached_prefix_default_value_group_width(size_t new_tokens) {
    return new_tokens < 64u ? 16u : kRocwmmaCachedPrefixValueGroupWidth;
}

unsigned int rocwmma_cached_prefix_value_group_width(size_t new_tokens) {
    const char *raw = std::getenv("ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH");
    if (raw == nullptr || raw[0] == '\0') {
        return rocwmma_cached_prefix_default_value_group_width(new_tokens);
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || !rocwmma_cached_prefix_value_group_width_is_valid(value)) {
        return rocwmma_cached_prefix_default_value_group_width(new_tokens);
    }
    return static_cast<unsigned int>(value);
}

#include "../infra/hip_runtime.inc"
#include "../infra/hiprtc_runtime_part_a.inc"
#include "../ullm_runtime_hiprtc_sources.inc"
#include "../infra/hiprtc_runtime_part_b.inc"
#include "../infra/rocblas_runtime.inc"

#include "../parts/runtime_structs.inc"
#include "../parts/probe_kernels.inc"
#include "../parts/aq4_kernels.inc"
#include "../parts/primitives_kernels_part_a.inc"
