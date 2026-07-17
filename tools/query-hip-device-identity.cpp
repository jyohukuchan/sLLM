// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Read-only HIP identity guard for a process whose visibility is already pinned.
//
// This executable neither allocates device memory nor creates a stream nor launches a kernel.
// It only queries HIP runtime properties for filtered ordinal 0 and writes one JSON record to
// stdout.  Callers must set HIP_VISIBLE_DEVICES and ULLM_HIP_VISIBLE_DEVICES before invoking it.

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kExpectedVisibility = "1";
constexpr std::string_view kExpectedArchitecture = "gfx1201";

std::string json_escape(std::string_view value) {
    std::ostringstream escaped;
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            escaped << "\\\"";
            break;
        case '\\':
            escaped << "\\\\";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (character < 0x20U) {
                constexpr char kHex[] = "0123456789abcdef";
                escaped << "\\u00" << kHex[(character >> 4U) & 0x0fU]
                        << kHex[character & 0x0fU];
            } else {
                escaped << static_cast<char>(character);
            }
            break;
        }
    }
    return escaped.str();
}

int report_invalid(const std::string_view reason) {
    std::cout << "{\"schema_version\":\"ullm.r9700_hip_device_guard.v1\","
              << "\"status\":\"invalid\",\"reason\":\"" << json_escape(reason)
              << "\"}" << std::endl;
    return 1;
}

const char *environment_value(const char *name) {
    const char *value = std::getenv(name);
    return value == nullptr ? "" : value;
}

std::string architecture_name(const char *raw_name) {
    const std::string raw = raw_name == nullptr ? "" : raw_name;
    return raw.substr(0, raw.find(':'));
}

} // namespace

int main(int argc, char **) {
    if (argc != 1) {
        return report_invalid("this guard accepts no arguments");
    }

    const std::string hip_visible = environment_value("HIP_VISIBLE_DEVICES");
    const std::string ullm_hip_visible = environment_value("ULLM_HIP_VISIBLE_DEVICES");
    if (hip_visible != kExpectedVisibility || ullm_hip_visible != kExpectedVisibility) {
        return report_invalid("HIP_VISIBLE_DEVICES and ULLM_HIP_VISIBLE_DEVICES must both equal 1");
    }

    int visible_device_count = 0;
    const hipError_t count_status = hipGetDeviceCount(&visible_device_count);
    if (count_status != hipSuccess) {
        return report_invalid(std::string("hipGetDeviceCount failed: ") + hipGetErrorString(count_status));
    }
    if (visible_device_count != 1) {
        return report_invalid("HIP visibility must expose exactly one GPU");
    }

    hipDeviceProp_t properties{};
    const hipError_t properties_status = hipGetDeviceProperties(&properties, 0);
    if (properties_status != hipSuccess) {
        return report_invalid(
            std::string("hipGetDeviceProperties(0) failed: ") + hipGetErrorString(properties_status));
    }
    const std::string raw_architecture = properties.gcnArchName;
    const std::string architecture = architecture_name(properties.gcnArchName);
    const std::string name = properties.name;
    if (architecture != kExpectedArchitecture) {
        return report_invalid("filtered HIP ordinal 0 is not gfx1201");
    }
    if (name.empty()) {
        return report_invalid("filtered HIP ordinal 0 has an empty device name");
    }

    char pci_bdf[32] = {};
    const hipError_t bdf_status = hipDeviceGetPCIBusId(pci_bdf, sizeof(pci_bdf), 0);
    if (bdf_status != hipSuccess || pci_bdf[0] == '\0') {
        return report_invalid(
            std::string("hipDeviceGetPCIBusId(0) failed: ") + hipGetErrorString(bdf_status));
    }

    int runtime_version = 0;
    int driver_version = 0;
    const hipError_t runtime_version_status = hipRuntimeGetVersion(&runtime_version);
    const hipError_t driver_version_status = hipDriverGetVersion(&driver_version);

    std::cout << "{\"schema_version\":\"ullm.r9700_hip_device_guard.v1\","
              << "\"status\":\"valid\","
              << "\"required\":{\"hip_visible_devices\":\"1\","
              << "\"ullm_hip_visible_devices\":\"1\","
              << "\"visible_hip_device_count\":1,\"architecture\":\"gfx1201\"},"
              << "\"actual\":{\"hip_visible_devices\":\"" << json_escape(hip_visible)
              << "\",\"ullm_hip_visible_devices\":\"" << json_escape(ullm_hip_visible)
              << "\",\"visible_hip_device_count\":" << visible_device_count
              << ",\"filtered_hip_ordinal\":0,\"architecture\":\""
              << json_escape(architecture) << "\",\"raw_architecture\":\""
              << json_escape(raw_architecture) << "\",\"name\":\"" << json_escape(name)
              << "\",\"pci_bdf\":\"" << json_escape(pci_bdf) << "\",\"hip_runtime_version\":";
    if (runtime_version_status == hipSuccess) {
        std::cout << runtime_version;
    } else {
        std::cout << "null";
    }
    std::cout << ",\"hip_driver_version\":";
    if (driver_version_status == hipSuccess) {
        std::cout << driver_version;
    } else {
        std::cout << "null";
    }
    std::cout << "}}" << std::endl;
    return 0;
}
