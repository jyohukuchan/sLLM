// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_SQ8_CK_GFX942_ARCH_H
#define ULLM_SQ8_CK_GFX942_ARCH_H

#include <string_view>

namespace ullm::sq8_ck_gfx942 {

// HIP appends target modifiers such as ":sramecc+:xnack-" to gcnArchName.
// Accept only the exact gfx942 architecture token and those known modifiers.
// In particular, never derive this decision from compute-major/minor values,
// never accept a prefix such as gfx9420, and never accept an empty or unknown
// modifier suffix.
inline bool is_exact_gfx942_gcn_arch_name(const char* gcn_arch_name) {
    if (gcn_arch_name == nullptr) {
        return false;
    }
    constexpr std::string_view kArch = "gfx942";
    const std::string_view actual(gcn_arch_name);
    if (actual == kArch) {
        return true;
    }
    if (actual.size() <= kArch.size() || actual.substr(0, kArch.size()) != kArch) {
        return false;
    }

    bool saw_xnack = false;
    bool saw_sramecc = false;
    std::string_view modifiers = actual.substr(kArch.size());
    while (!modifiers.empty()) {
        if (modifiers.front() != ':') {
            return false;
        }
        modifiers.remove_prefix(1u);
        if (modifiers.starts_with("xnack+") || modifiers.starts_with("xnack-")) {
            if (saw_xnack) {
                return false;
            }
            saw_xnack = true;
            modifiers.remove_prefix(6u);
        } else if (modifiers.starts_with("sramecc+") || modifiers.starts_with("sramecc-")) {
            if (saw_sramecc) {
                return false;
            }
            saw_sramecc = true;
            modifiers.remove_prefix(8u);
        } else {
            return false;
        }
    }
    return saw_xnack || saw_sramecc;
}

} // namespace ullm::sq8_ck_gfx942

#endif
