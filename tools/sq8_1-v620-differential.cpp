// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Thermal-guarded SQ8_1 runtime differential for the passive V620.  It never
// accepts a HIP ordinal alone: the selected HIP device must report the caller's
// exact PCI BDF through hipDeviceGetPCIBusId, and the junction sensor is found
// through that same BDF's DRM/sysfs path.

#include "ullm_runtime.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

#define HIP_CHECK(call)                                                                      \
    do {                                                                                     \
        const hipError_t status__ = (call);                                                  \
        if (status__ != hipSuccess) {                                                        \
            throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(status__)); \
        }                                                                                    \
    } while (false)

constexpr double kJunctionLimitC = 85.0;
constexpr std::size_t kRows = 7u;
constexpr std::size_t kCols = 65u;
constexpr std::size_t kGroup = 32u;
constexpr std::size_t kStride = 80u;
constexpr int kRepetitions = 8;
constexpr float kMaxAbsoluteTolerance = 0.01f;
constexpr double kRelativeL2Tolerance = 1e-6;

struct Options {
    std::string pci_bus_id;
    std::filesystem::path jsonl_output;
};

struct Device {
    int ordinal = -1;
    std::string name;
    std::string arch;
    std::string bdf;
};

struct JunctionSensor {
    std::string card;
    std::filesystem::path hwmon;
    std::filesystem::path temp2_input;
    std::string label;
};

std::ofstream g_output;

[[noreturn]] void usage(const char* program) {
    std::cerr << "usage: " << program
              << " --pci-bus-id 0000:03:00.0 --jsonl-output /absolute/path\n";
    std::exit(2);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto need = [&]() -> std::string {
            if (++index >= argc) {
                usage(argv[0]);
            }
            return argv[index];
        };
        if (argument == "--pci-bus-id") {
            options.pci_bus_id = need();
        } else if (argument == "--jsonl-output") {
            options.jsonl_output = need();
        } else {
            usage(argv[0]);
        }
    }
    if (options.pci_bus_id.empty() || options.jsonl_output.empty() ||
        !options.jsonl_output.is_absolute()) {
        usage(argv[0]);
    }
    return options;
}

void open_output(const std::filesystem::path& output) {
    std::error_code error;
    if (std::filesystem::exists(output, error)) {
        throw std::runtime_error("refusing to overwrite differential output: " + output.string());
    }
    if (error) {
        throw std::runtime_error("failed to stat output: " + error.message());
    }
    std::filesystem::create_directories(output.parent_path(), error);
    if (error) {
        throw std::runtime_error("failed to create output directory: " + error.message());
    }
    g_output.open(output);
    if (!g_output) {
        throw std::runtime_error("failed to open output: " + output.string());
    }
}

void emit(const std::string& line) {
    std::cout << line << '\n';
    std::cout.flush();
    g_output << line << '\n';
    g_output.flush();
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return value;
}

bool is_gfx1030(std::string_view arch) {
    return arch == "gfx1030" || (arch.size() > 8u && arch.substr(0u, 8u) == "gfx1030:");
}

std::vector<Device> enumerate_devices() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    std::vector<Device> devices;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t property{};
        char bdf[32]{};
        HIP_CHECK(hipGetDeviceProperties(&property, ordinal));
        HIP_CHECK(hipDeviceGetPCIBusId(bdf, static_cast<int>(sizeof(bdf)), ordinal));
        devices.push_back(Device{ordinal, property.name, property.gcnArchName, bdf});
    }
    return devices;
}

Device select_v620(const std::string& requested_bdf) {
    const std::vector<Device> devices = enumerate_devices();
    std::ostringstream inventory;
    inventory << "{\"record\":\"device_inventory\",\"devices\":[";
    for (std::size_t index = 0; index < devices.size(); ++index) {
        if (index != 0u) {
            inventory << ',';
        }
        inventory << "{\"hip_ordinal\":" << devices[index].ordinal
                  << ",\"name\":\"" << devices[index].name
                  << "\",\"gcn_arch_name\":\"" << devices[index].arch
                  << "\",\"pci_bus_id\":\"" << devices[index].bdf << "\"}";
    }
    inventory << "]}";
    emit(inventory.str());
    std::vector<Device> matches;
    for (const Device& device : devices) {
        if (is_gfx1030(device.arch) && lower(device.bdf) == lower(requested_bdf)) {
            matches.push_back(device);
        }
    }
    if (matches.size() != 1u) {
        throw std::runtime_error("requested PCI BDF did not identify exactly one gfx1030 HIP device");
    }
    HIP_CHECK(hipSetDevice(matches.front().ordinal));
    return matches.front();
}

std::string read_line(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string line;
    if (!input || !std::getline(input, line)) {
        throw std::runtime_error("failed to read " + path.string());
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

bool is_card(std::string_view name) {
    return name.size() > 4u && name.substr(0u, 4u) == "card" &&
        std::all_of(name.begin() + 4, name.end(), [](unsigned char value) {
            return std::isdigit(value) != 0;
        });
}

JunctionSensor sensor_for_bdf(const std::string& bdf) {
    const std::filesystem::path drm_root("/sys/class/drm");
    std::error_code error;
    std::vector<JunctionSensor> matches;
    for (const auto& entry : std::filesystem::directory_iterator(drm_root, error)) {
        if (error) {
            throw std::runtime_error("failed to enumerate DRM devices: " + error.message());
        }
        const std::string card = entry.path().filename().string();
        if (!is_card(card)) {
            continue;
        }
        std::ifstream uevent(entry.path() / "device" / "uevent");
        std::string line;
        std::string card_bdf;
        while (uevent && std::getline(uevent, line)) {
            constexpr std::string_view prefix = "PCI_SLOT_NAME=";
            if (line.rfind(prefix.data(), 0u) == 0u) {
                card_bdf = line.substr(prefix.size());
                break;
            }
        }
        if (lower(card_bdf) != lower(bdf)) {
            continue;
        }
        const auto hwmon_root = entry.path() / "device" / "hwmon";
        for (const auto& hwmon : std::filesystem::directory_iterator(hwmon_root, error)) {
            if (error) {
                throw std::runtime_error("failed to enumerate matching hwmon: " + error.message());
            }
            const auto input = hwmon.path() / "temp2_input";
            const auto label = hwmon.path() / "temp2_label";
            if (!std::filesystem::exists(input, error) || error || !std::filesystem::exists(label, error) || error) {
                if (error) {
                    throw std::runtime_error("failed to inspect matching hwmon: " + error.message());
                }
                continue;
            }
            const std::string label_value = lower(read_line(label));
            if (label_value == "junction") {
                matches.push_back(JunctionSensor{card, hwmon.path(), input, label_value});
            }
        }
    }
    if (matches.size() != 1u) {
        throw std::runtime_error("requested BDF did not identify exactly one junction temp2_input sensor");
    }
    return matches.front();
}

class ThermalGuard {
public:
    ThermalGuard(Device device, JunctionSensor sensor) : device_(std::move(device)), sensor_(std::move(sensor)) {
        emit("{\"record\":\"thermal_sensor\",\"hip_ordinal\":" +
             std::to_string(device_.ordinal) + ",\"pci_bus_id\":\"" + device_.bdf +
             "\",\"drm_card\":\"" + sensor_.card + "\",\"temp2_input\":\"" +
             sensor_.temp2_input.string() + "\",\"temp2_label\":\"" + sensor_.label +
             "\",\"junction_limit_c\":85.0}");
    }

    void check(std::string_view phase) const {
        const std::string raw = read_line(sensor_.temp2_input);
        char* end = nullptr;
        const long long milli_c = std::strtoll(raw.c_str(), &end, 10);
        if (end == raw.c_str() || *end != '\0' || milli_c <= 0 || milli_c > 200'000) {
            throw std::runtime_error("invalid junction temperature reading");
        }
        const double celsius = static_cast<double>(milli_c) / 1000.0;
        std::ostringstream record;
        record << std::setprecision(8) << "{\"record\":\"thermal\",\"phase\":\"" << phase
               << "\",\"pci_bus_id\":\"" << device_.bdf << "\",\"drm_card\":\""
               << sensor_.card << "\",\"temperature_c\":" << celsius
               << ",\"junction_limit_c\":85.0}";
        emit(record.str());
        if (celsius >= kJunctionLimitC) {
            throw std::runtime_error("junction temperature reached the 85 C guard");
        }
    }

private:
    Device device_;
    JunctionSensor sensor_;
};

float f16_to_f32(std::uint16_t bits) {
    const float sign = (bits & 0x8000u) == 0u ? 1.0f : -1.0f;
    const unsigned exponent = (bits >> 10u) & 0x1fu;
    const unsigned mantissa = bits & 0x03ffu;
    if (exponent == 0u) {
        return mantissa == 0u ? sign * 0.0f : sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 0x1fu) {
        return mantissa == 0u ? sign * std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, static_cast<int>(exponent) - 15);
}

std::uint16_t f32_to_f16_rne(float value) {
    const _Float16 narrowed = static_cast<_Float16>(value);
    std::uint16_t bits = 0u;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    return bits;
}

float ceil_f16(float value) {
    std::uint16_t bits = f32_to_f16_rne(value);
    if (bits == 0u) {
        return f16_to_f32(1u);
    }
    if (bits >= 0x7c00u) {
        throw std::runtime_error("FP16 scale overflow");
    }
    float stored = f16_to_f32(bits);
    if (stored < value) {
        ++bits;
        if (bits >= 0x7c00u) {
            throw std::runtime_error("FP16 scale overflow");
        }
        stored = f16_to_f32(bits);
    }
    return stored;
}

int round_ties_even(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    int result = static_cast<int>(lower);
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1) != 0)) {
        ++result;
    }
    return result;
}

struct Packed {
    std::vector<std::uint8_t> payload;
    std::vector<std::uint16_t> scales;
};

Packed pack_weights(const std::vector<float>& weights) {
    Packed packed{std::vector<std::uint8_t>(kRows * kStride, 0u), std::vector<std::uint16_t>(kRows * 3u)};
    for (std::size_t row = 0u; row < kRows; ++row) {
        for (std::size_t block = 0u; block < 3u; ++block) {
            const std::size_t start = block * kGroup;
            const std::size_t stop = std::min(start + kGroup, kCols);
            float maximum = 0.0f;
            for (std::size_t col = start; col < stop; ++col) {
                maximum = std::max(maximum, std::fabs(weights[row * kCols + col]));
            }
            const float scale = maximum == 0.0f ? 1.0f : ceil_f16(maximum / 127.0f);
            packed.scales[row * 3u + block] = f32_to_f16_rne(scale);
            for (std::size_t col = start; col < stop; ++col) {
                const int code = std::clamp(round_ties_even(weights[row * kCols + col] / scale), -127, 127);
                packed.payload[row * kStride + col] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
            }
        }
    }
    return packed;
}

std::vector<float> reference_w8a16(const Packed& packed, const std::vector<float>& input) {
    std::vector<float> output(kRows, 0.0f);
    for (std::size_t row = 0u; row < kRows; ++row) {
        for (std::size_t block = 0u; block < 3u; ++block) {
            const auto start = block * kGroup;
            const auto stop = std::min(start + kGroup, kCols);
            float partial = 0.0f;
            for (std::size_t col = start; col < stop; ++col) {
                partial += static_cast<float>(static_cast<std::int8_t>(packed.payload[row * kStride + col])) * input[col];
            }
            output[row] += partial * f16_to_f32(packed.scales[row * 3u + block]);
        }
    }
    return output;
}

std::vector<float> reference_w8a8(const Packed& packed, const std::vector<float>& input) {
    std::array<std::int8_t, kCols> activation{};
    std::array<float, 3u> activation_scales{};
    for (std::size_t block = 0u; block < 3u; ++block) {
        const auto start = block * kGroup;
        const auto stop = std::min(start + kGroup, kCols);
        float maximum = 0.0f;
        for (std::size_t col = start; col < stop; ++col) {
            maximum = std::max(maximum, std::fabs(input[col]));
        }
        activation_scales[block] = maximum == 0.0f ? 1.0f : ceil_f16(maximum / 127.0f);
        for (std::size_t col = start; col < stop; ++col) {
            activation[col] = static_cast<std::int8_t>(std::clamp(
                round_ties_even(input[col] / activation_scales[block]), -127, 127));
        }
    }
    std::vector<float> output(kRows, 0.0f);
    for (std::size_t row = 0u; row < kRows; ++row) {
        for (std::size_t block = 0u; block < 3u; ++block) {
            const auto start = block * kGroup;
            const auto stop = std::min(start + kGroup, kCols);
            std::int32_t dot = 0;
            for (std::size_t col = start; col < stop; ++col) {
                dot += static_cast<std::int32_t>(static_cast<std::int8_t>(packed.payload[row * kStride + col])) *
                    static_cast<std::int32_t>(activation[col]);
            }
            output[row] += static_cast<float>(dot) * f16_to_f32(packed.scales[row * 3u + block]) * activation_scales[block];
        }
    }
    return output;
}

std::vector<std::uint8_t> f32_bytes(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<float> read_f32s(ullm_runtime_buffer* buffer, ullm_runtime_stream* stream) {
    std::vector<std::uint8_t> bytes(kRows * sizeof(float));
    if (ullm_runtime_buffer_copy_to_host(buffer, 0u, bytes.data(), bytes.size(), stream) != ULLM_STATUS_OK ||
        ullm_runtime_stream_synchronize(stream) != ULLM_STATUS_OK) {
        throw std::runtime_error("failed to read runtime output");
    }
    std::vector<float> values(kRows);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

struct ErrorMetric {
    float max_abs = 0.0f;
    double relative_l2 = 0.0;
};

ErrorMetric compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    double error_sumsq = 0.0;
    double reference_sumsq = 0.0;
    float max_abs = 0.0f;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        const float delta = actual[index] - expected[index];
        max_abs = std::max(max_abs, std::fabs(delta));
        error_sumsq += static_cast<double>(delta) * delta;
        reference_sumsq += static_cast<double>(expected[index]) * expected[index];
    }
    return ErrorMetric{max_abs, std::sqrt(error_sumsq / reference_sumsq)};
}

void check_runtime(ullm_status status, const char* label) {
    if (status == ULLM_STATUS_OK) {
        return;
    }
    std::array<char, 1024> error{};
    std::size_t length = error.size();
    ullm_runtime_get_last_error(error.data(), &length);
    throw std::runtime_error(std::string(label) + ": " + error.data());
}

int run(const Options& options) {
    const Device selected = select_v620(options.pci_bus_id);
    ThermalGuard thermal(selected, sensor_for_bdf(selected.bdf));
    thermal.check("before_runtime_context");

    ullm_runtime_context* context = nullptr;
    ullm_runtime_stream* stream = nullptr;
    ullm_runtime_buffer* payload_buffer = nullptr;
    ullm_runtime_buffer* scale_buffer = nullptr;
    ullm_runtime_buffer* input_buffer = nullptr;
    ullm_runtime_buffer* w8a16_output = nullptr;
    ullm_runtime_buffer* w8a8_output = nullptr;
    try {
        check_runtime(ullm_runtime_context_create(static_cast<std::uint32_t>(selected.ordinal + 1), &context), "context_create");
        int current = -1;
        char current_bdf[32]{};
        HIP_CHECK(hipGetDevice(&current));
        HIP_CHECK(hipDeviceGetPCIBusId(current_bdf, static_cast<int>(sizeof(current_bdf)), current));
        if (current != selected.ordinal || lower(current_bdf) != lower(selected.bdf)) {
            throw std::runtime_error("runtime context did not preserve the selected HIP BDF");
        }
        check_runtime(ullm_runtime_stream_create(context, &stream), "stream_create");

        std::vector<float> weights(kRows * kCols);
        std::vector<float> input(kCols);
        for (std::size_t index = 0u; index < weights.size(); ++index) {
            weights[index] = static_cast<float>((index * 37u + 11u) % 251u) - 125.0f;
        }
        for (std::size_t index = 0u; index < input.size(); ++index) {
            input[index] = static_cast<float>((index * 19u + 7u) % 253u) - 126.0f;
        }
        const Packed packed = pack_weights(weights);
        const std::vector<float> expected_w8a16 = reference_w8a16(packed, input);
        const std::vector<float> expected_w8a8 = reference_w8a8(packed, input);
        const std::vector<std::uint8_t> input_bytes = f32_bytes(input);

        check_runtime(ullm_runtime_buffer_alloc(context, packed.payload.size(), &payload_buffer), "payload_alloc");
        check_runtime(ullm_runtime_buffer_alloc(context, packed.scales.size() * sizeof(std::uint16_t), &scale_buffer), "scale_alloc");
        check_runtime(ullm_runtime_buffer_alloc(context, input_bytes.size(), &input_buffer), "input_alloc");
        check_runtime(ullm_runtime_buffer_alloc(context, kRows * sizeof(float), &w8a16_output), "w8a16_alloc");
        check_runtime(ullm_runtime_buffer_alloc(context, kRows * sizeof(float), &w8a8_output), "w8a8_alloc");
        check_runtime(ullm_runtime_buffer_copy_from_host(payload_buffer, 0u, packed.payload.data(), packed.payload.size(), stream), "payload_upload");
        check_runtime(ullm_runtime_buffer_copy_from_host(scale_buffer, 0u, packed.scales.data(), packed.scales.size() * sizeof(std::uint16_t), stream), "scale_upload");
        check_runtime(ullm_runtime_buffer_copy_from_host(input_buffer, 0u, input_bytes.data(), input_bytes.size(), stream), "input_upload");
        check_runtime(ullm_runtime_stream_synchronize(stream), "input_upload_sync");
        thermal.check("after_upload");

        for (int repetition = 0; repetition < kRepetitions; ++repetition) {
            thermal.check("before_w8a16_launch_" + std::to_string(repetition));
            check_runtime(ullm_runtime_sq8_1_matvec_w8a16_f32(
                payload_buffer, scale_buffer, input_buffer, kRows, kCols, kStride, w8a16_output, stream), "w8a16_launch");
            check_runtime(ullm_runtime_stream_synchronize(stream), "w8a16_sync");
            thermal.check("after_w8a16_launch_" + std::to_string(repetition));
        }
        for (int repetition = 0; repetition < kRepetitions; ++repetition) {
            thermal.check("before_w8a8_launch_" + std::to_string(repetition));
            check_runtime(ullm_runtime_sq8_1_matvec_w8a8_explicit_f32(
                payload_buffer, scale_buffer, input_buffer, kRows, kCols, kStride, w8a8_output, stream), "w8a8_launch");
            check_runtime(ullm_runtime_stream_synchronize(stream), "w8a8_sync");
            thermal.check("after_w8a8_launch_" + std::to_string(repetition));
        }
        const ErrorMetric w8a16 = compare(read_f32s(w8a16_output, stream), expected_w8a16);
        const ErrorMetric w8a8 = compare(read_f32s(w8a8_output, stream), expected_w8a8);
        if (w8a16.max_abs > kMaxAbsoluteTolerance || w8a16.relative_l2 > kRelativeL2Tolerance ||
            w8a8.max_abs > kMaxAbsoluteTolerance || w8a8.relative_l2 > kRelativeL2Tolerance) {
            throw std::runtime_error("GPU differential exceeded the fixed numerical tolerance");
        }
        std::ostringstream record;
        record << std::setprecision(10)
               << "{\"record\":\"differential\",\"status\":\"passed\",\"rows\":" << kRows
               << ",\"cols\":" << kCols << ",\"payload_row_stride\":" << kStride
               << ",\"w8a16_max_abs\":" << w8a16.max_abs
               << ",\"w8a16_relative_l2\":" << w8a16.relative_l2
               << ",\"w8a8_max_abs\":" << w8a8.max_abs
               << ",\"w8a8_relative_l2\":" << w8a8.relative_l2
               << ",\"max_absolute_tolerance\":" << kMaxAbsoluteTolerance
               << ",\"relative_l2_tolerance\":" << kRelativeL2Tolerance << "}";
        emit(record.str());
    } catch (...) {
        ullm_runtime_buffer_destroy(w8a8_output);
        ullm_runtime_buffer_destroy(w8a16_output);
        ullm_runtime_buffer_destroy(input_buffer);
        ullm_runtime_buffer_destroy(scale_buffer);
        ullm_runtime_buffer_destroy(payload_buffer);
        ullm_runtime_stream_destroy(stream);
        ullm_runtime_context_destroy(context);
        throw;
    }
    ullm_runtime_buffer_destroy(w8a8_output);
    ullm_runtime_buffer_destroy(w8a16_output);
    ullm_runtime_buffer_destroy(input_buffer);
    ullm_runtime_buffer_destroy(scale_buffer);
    ullm_runtime_buffer_destroy(payload_buffer);
    ullm_runtime_stream_destroy(stream);
    ullm_runtime_context_destroy(context);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        open_output(options.jsonl_output);
        return run(options);
    } catch (const std::exception& error) {
        if (g_output.is_open()) {
            emit(std::string("{\"record\":\"failure\",\"message\":\"") + error.what() + "\"}");
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
