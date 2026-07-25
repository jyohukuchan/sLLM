// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Isolated SQ9_0 versus canonical SQ8_0 V620 viability benchmark.
//
// This program deliberately does not use a HIP ordinal supplied by the caller.
// It enumerates every HIP device, requires an exact gfx1030 architecture match,
// and then requires the requested PCI BDF to match one of those gfx1030 devices
// before it can create a stream or allocate a device buffer.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace {

#define HIP_CHECK(call)                                                                      \
    do {                                                                                     \
        const hipError_t status__ = (call);                                                  \
        if (status__ != hipSuccess) {                                                        \
            throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(status__)); \
        }                                                                                    \
    } while (false)

constexpr int kThreads = 256;
constexpr int kBlockElements = 128;
constexpr int kThreadsPerBlockTile = 8;
constexpr int kTileGroups = kThreads / kThreadsPerBlockTile;
constexpr int kReplicaCount = 6;
constexpr double kV620PeakBandwidthGBps = 512.0;
constexpr double kJunctionLimitC = 85.0;
// card0 idles at 41--42 C in the verified chassis airflow path.  Holding every
// timing point to this narrow ceiling makes both the thermal starting condition
// and the performance state substantially more comparable than a broad 45 C
// ceiling, while leaving a wide margin below the 85 C hard guard.
constexpr double kDefaultCooldownC = 42.0;
constexpr std::size_t kDefaultCooldownPollMs = 5'000;
constexpr std::size_t kDefaultCooldownTimeoutS = 900;

enum Format : int {
    kSq8 = 0,
    kSq9Lane = 1,
    kSq9Lds = 2,
    kFp16 = 3,
};

struct Shape {
    const char* name;
    const char* model;
    const char* projection;
    std::size_t rows;
    std::size_t cols;
};

// The shapes come from the Qwen3-14B-FP8 canonical SQ8_0 artifact and the
// Qwen3.5-9B package/configuration retained in this repository's evidence.
constexpr std::array<Shape, 6> kShapes = {{
    {"qwen3_14b_q_proj", "Qwen3-14B-FP8", "self_attn.q_proj", 5'120, 5'120},
    {"qwen3_14b_mlp_gate", "Qwen3-14B-FP8", "mlp.gate_proj", 17'408, 5'120},
    {"qwen3_14b_mlp_down", "Qwen3-14B-FP8", "mlp.down_proj", 5'120, 17'408},
    {"qwen35_9b_linear_qkv", "Qwen3.5-9B", "linear_attn.in_proj_qkv", 8'192, 4'096},
    {"qwen35_9b_mlp_up", "Qwen3.5-9B", "mlp.up_proj", 12'288, 4'096},
    {"qwen35_9b_mlp_down", "Qwen3.5-9B", "mlp.down_proj", 4'096, 12'288},
}};

struct Options {
    std::string pci_bus_id;
    std::string suite = "full";
    std::string shape = "all";
    bool shape_explicit = false;
    std::vector<std::size_t> m_values;
    bool m_values_explicit = false;
    std::string dequant = "auto";
    int warmups = 4;
    int trials = 9;
    int launches_per_trial = kReplicaCount;
    double cooldown_c = kDefaultCooldownC;
    std::size_t cooldown_poll_ms = kDefaultCooldownPollMs;
    std::size_t cooldown_timeout_s = kDefaultCooldownTimeoutS;
    int thermal_max_retries = 1;
    bool preflight_only = false;
    std::string jsonl_output;
    std::string thermal_history_output;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --pci-bus-id 0000:03:00.0 [--suite full|smoke]"
              << " [--shape all|shape-name]"
              << " [--m-values 1,8,32,128,512] [--dequant auto|on|off]"
              << " [--warmups N] [--trials N] [--launches-per-trial N]"
              << " [--cooldown-c C] [--cooldown-poll-ms N]"
              << " [--cooldown-timeout-s N] [--thermal-max-retries N]"
              << " [--preflight-only]"
              << " [--jsonl-output PATH] [--thermal-history-output PATH]\n";
    std::exit(2);
}

std::size_t parse_positive(std::string_view text, const char* label) {
    char* end = nullptr;
    const auto value = std::strtoull(std::string(text).c_str(), &end, 10);
    if (text.empty() || text.front() == '-' || end == nullptr || *end != '\0' || value == 0) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<std::size_t>(value);
}

std::size_t parse_nonnegative(std::string_view text, const char* label) {
    char* end = nullptr;
    const auto value = std::strtoull(std::string(text).c_str(), &end, 10);
    if (text.empty() || text.front() == '-' || end == nullptr || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<std::size_t>(value);
}

double parse_positive_double(std::string_view text, const char* label) {
    char* end = nullptr;
    const double value = std::strtod(std::string(text).c_str(), &end);
    if (end == nullptr || *end != '\0' || !std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return value;
}

std::vector<std::size_t> parse_m_values(std::string_view text) {
    std::vector<std::size_t> values;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        if (end == begin) {
            throw std::runtime_error("invalid m-values");
        }
        values.push_back(parse_positive(text.substr(begin, end - begin), "m-values"));
        if (comma == std::string_view::npos) {
            break;
        }
        if (end + 1u == text.size()) {
            throw std::runtime_error("invalid m-values");
        }
        begin = end + 1u;
    }
    if (values.empty() || !std::is_sorted(values.begin(), values.end()) ||
        std::adjacent_find(values.begin(), values.end()) != values.end()) {
        throw std::runtime_error("m-values must be a strictly increasing comma-separated list");
    }
    return values;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        auto need = [&]() -> std::string_view {
            if (++index >= argc) {
                usage(argv[0]);
            }
            return argv[index];
        };
        if (arg == "--pci-bus-id") {
            options.pci_bus_id = need();
        } else if (arg == "--suite") {
            options.suite = need();
        } else if (arg == "--shape") {
            options.shape = need();
            options.shape_explicit = true;
        } else if (arg == "--m-values") {
            options.m_values = parse_m_values(need());
            options.m_values_explicit = true;
        } else if (arg == "--dequant") {
            options.dequant = need();
        } else if (arg == "--warmups") {
            options.warmups = static_cast<int>(parse_positive(need(), "warmups"));
        } else if (arg == "--trials") {
            options.trials = static_cast<int>(parse_positive(need(), "trials"));
        } else if (arg == "--launches-per-trial") {
            options.launches_per_trial =
                static_cast<int>(parse_positive(need(), "launches-per-trial"));
        } else if (arg == "--cooldown-c") {
            options.cooldown_c = parse_positive_double(need(), "cooldown-c");
        } else if (arg == "--cooldown-poll-ms") {
            options.cooldown_poll_ms = parse_positive(need(), "cooldown-poll-ms");
        } else if (arg == "--cooldown-timeout-s") {
            options.cooldown_timeout_s = parse_positive(need(), "cooldown-timeout-s");
        } else if (arg == "--thermal-max-retries") {
            options.thermal_max_retries =
                static_cast<int>(parse_nonnegative(need(), "thermal-max-retries"));
        } else if (arg == "--preflight-only") {
            options.preflight_only = true;
        } else if (arg == "--jsonl-output") {
            options.jsonl_output = need();
        } else if (arg == "--thermal-history-output") {
            options.thermal_history_output = need();
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            usage(argv[0]);
        }
    }
    if (options.pci_bus_id.empty() || (options.suite != "full" && options.suite != "smoke") ||
        (options.dequant != "auto" && options.dequant != "on" && options.dequant != "off") ||
        options.cooldown_c >= kJunctionLimitC ||
        (!options.jsonl_output.empty() &&
         options.jsonl_output == options.thermal_history_output)) {
        usage(argv[0]);
    }
    return options;
}

std::string json_escape(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += ch; break;
        }
    }
    return result;
}

std::ofstream g_jsonl_output;
std::ofstream g_thermal_history_output;

void open_new_output_file(std::ofstream& output, const std::string& raw_path, const char* label) {
    if (raw_path.empty()) {
        return;
    }
    const std::filesystem::path path(raw_path);
    std::error_code error;
    if (std::filesystem::exists(path, error)) {
        throw std::runtime_error(std::string(label) + " already exists: " + path.string());
    }
    if (error) {
        throw std::runtime_error(std::string("failed to stat ") + label + ": " + error.message());
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            throw std::runtime_error(
                std::string("failed to create ") + label + " parent: " + error.message());
        }
    }
    output.open(path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(std::string("failed to open ") + label + ": " + path.string());
    }
}

void configure_outputs(const Options& options) {
    open_new_output_file(g_jsonl_output, options.jsonl_output, "jsonl output");
    open_new_output_file(
        g_thermal_history_output, options.thermal_history_output, "thermal history output");
}

void emit_json_line(const std::string& json, bool thermal_record = false) {
    std::cout << json << '\n';
    std::cout.flush();
    if (g_jsonl_output.is_open()) {
        g_jsonl_output << json << '\n';
        g_jsonl_output.flush();
    }
    if (thermal_record && g_thermal_history_output.is_open()) {
        g_thermal_history_output << json << '\n';
        g_thermal_history_output.flush();
    }
}

bool is_gfx1030(std::string_view arch) {
    return arch == "gfx1030" ||
           (arch.size() > 7 && arch.substr(0, 8) == "gfx1030:");
}

struct DeviceIdentity {
    int ordinal = -1;
    std::string name;
    std::string gcn_arch_name;
    std::string pci_bus_id;
};

std::vector<DeviceIdentity> enumerate_devices() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    std::vector<DeviceIdentity> devices;
    devices.reserve(static_cast<std::size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        char pci_bus_id[32]{};
        HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), ordinal));
        devices.push_back(DeviceIdentity{
            ordinal,
            properties.name,
            properties.gcnArchName,
            pci_bus_id,
        });
    }
    return devices;
}

void emit_device_inventory(const char* phase, const std::vector<DeviceIdentity>& devices) {
    std::ostringstream out;
    out << "{\"record\":\"device_inventory\",\"phase\":\"" << phase
        << "\",\"devices\":[";
    for (std::size_t index = 0; index < devices.size(); ++index) {
        const auto& device = devices[index];
        if (index != 0) {
            out << ',';
        }
        out << "{\"hip_ordinal\":" << device.ordinal
            << ",\"name\":\"" << json_escape(device.name)
            << "\",\"gcnArchName\":\"" << json_escape(device.gcn_arch_name)
            << "\",\"pci_bus_id\":\"" << json_escape(device.pci_bus_id) << "\"}";
    }
    out << "]}";
    emit_json_line(out.str());
}

DeviceIdentity select_v620_by_arch_and_bdf(const std::string& requested_bdf) {
    const auto devices = enumerate_devices();
    emit_device_inventory("preflight", devices);
    std::vector<DeviceIdentity> matches;
    for (const auto& device : devices) {
        if (is_gfx1030(device.gcn_arch_name) && device.pci_bus_id == requested_bdf) {
            matches.push_back(device);
        }
    }
    if (matches.size() != 1) {
        throw std::runtime_error(
            "refusing GPU execution: requested PCI BDF did not select exactly one gfx1030 device");
    }
    const DeviceIdentity selected = matches.front();
    HIP_CHECK(hipSetDevice(selected.ordinal));
    return selected;
}

void assert_selected_v620(const DeviceIdentity& expected, const char* phase) {
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, expected.ordinal));
    char pci_bus_id[32]{};
    HIP_CHECK(hipDeviceGetPCIBusId(
        pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), expected.ordinal));
    if (!is_gfx1030(properties.gcnArchName) || expected.pci_bus_id != pci_bus_id) {
        throw std::runtime_error("refusing GPU execution: selected device identity changed or is not gfx1030");
    }
    std::ostringstream out;
    out << "{\"record\":\"device_identity\",\"phase\":\"" << phase
        << "\",\"hip_ordinal\":" << expected.ordinal
        << ",\"name\":\"" << json_escape(properties.name)
        << "\",\"gcnArchName\":\"" << json_escape(properties.gcnArchName)
        << "\",\"pci_bus_id\":\"" << json_escape(pci_bus_id) << "\"}";
    emit_json_line(out.str());
}

std::string ascii_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool is_drm_card_name(std::string_view name) {
    if (name.size() <= 4u || name.substr(0, 4) != "card") {
        return false;
    }
    return std::all_of(name.begin() + 4, name.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
}

std::string read_single_line(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string value;
    if (!input || !std::getline(input, value)) {
        throw std::runtime_error("failed to read " + path.string());
    }
    if (!value.empty() && value.back() == '\r') {
        value.pop_back();
    }
    return value;
}

struct JunctionSensor {
    std::string drm_card;
    std::string pci_bus_id;
    std::filesystem::path hwmon_path;
    std::filesystem::path temp2_input_path;
    std::string temp2_label;
};

JunctionSensor find_junction_sensor_for_bdf(const std::string& requested_bdf) {
    const std::string normalized_bdf = ascii_lower(requested_bdf);
    const std::filesystem::path drm_root("/sys/class/drm");
    std::error_code error;
    std::filesystem::directory_iterator entries(drm_root, error);
    if (error) {
        throw std::runtime_error("failed to enumerate /sys/class/drm: " + error.message());
    }

    std::vector<JunctionSensor> matches;
    for (const std::filesystem::directory_entry& entry : entries) {
        const std::string card_name = entry.path().filename().string();
        if (!is_drm_card_name(card_name)) {
            continue;
        }
        const std::filesystem::path uevent_path = entry.path() / "device" / "uevent";
        std::ifstream uevent(uevent_path);
        if (!uevent) {
            continue;
        }
        std::string line;
        std::string card_bdf;
        while (std::getline(uevent, line)) {
            constexpr std::string_view kPrefix = "PCI_SLOT_NAME=";
            if (line.rfind(kPrefix.data(), 0) == 0) {
                card_bdf = line.substr(kPrefix.size());
                break;
            }
        }
        if (ascii_lower(card_bdf) != normalized_bdf) {
            continue;
        }

        const std::filesystem::path hwmon_root = entry.path() / "device" / "hwmon";
        std::filesystem::directory_iterator hwmons(hwmon_root, error);
        if (error) {
            throw std::runtime_error(
                "failed to enumerate hwmon for " + card_name + ": " + error.message());
        }
        for (const std::filesystem::directory_entry& hwmon : hwmons) {
            const std::filesystem::path temp2_input = hwmon.path() / "temp2_input";
            if (!std::filesystem::exists(temp2_input, error)) {
                if (error) {
                    throw std::runtime_error(
                        "failed to stat " + temp2_input.string() + ": " + error.message());
                }
                continue;
            }
            std::string label;
            const std::filesystem::path label_path = hwmon.path() / "temp2_label";
            if (std::filesystem::exists(label_path, error)) {
                if (error) {
                    throw std::runtime_error(
                        "failed to stat " + label_path.string() + ": " + error.message());
                }
                label = read_single_line(label_path);
            }
            // Do not silently reinterpret an arbitrary temp2 channel as the
            // junction sensor.  This exact hwmon implementation labels it
            // `junction`; a changed driver/sysfs layout must fail closed.
            if (ascii_lower(label) != "junction") {
                continue;
            }
            matches.push_back(JunctionSensor{
                card_name,
                card_bdf,
                hwmon.path(),
                temp2_input,
                label,
            });
        }
    }
    if (matches.size() != 1u) {
        throw std::runtime_error(
            "refusing GPU execution: expected exactly one DRM hwmon temp2_input labeled junction for PCI BDF " +
            requested_bdf + ", found " + std::to_string(matches.size()));
    }
    return matches.front();
}

std::int64_t unix_epoch_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

class ThermalLimitExceeded final : public std::runtime_error {
public:
    ThermalLimitExceeded(double temperature_c, std::string phase)
        : std::runtime_error("junction temperature reached the 85 C guard at " + phase),
          temperature_c_(temperature_c),
          phase_(std::move(phase)) {}

    double temperature_c() const { return temperature_c_; }
    const std::string& phase() const { return phase_; }

private:
    double temperature_c_;
    std::string phase_;
};

class ThermalCooldownTimeout final : public std::runtime_error {
public:
    ThermalCooldownTimeout(double temperature_c, std::string phase)
        : std::runtime_error("junction cooldown did not reach target at " + phase),
          temperature_c_(temperature_c),
          phase_(std::move(phase)) {}

    double temperature_c() const { return temperature_c_; }
    const std::string& phase() const { return phase_; }

private:
    double temperature_c_;
    std::string phase_;
};

class ThermalGuard {
public:
    ThermalGuard(const DeviceIdentity& device, const Options& options)
        : device_(device),
          sensor_(find_junction_sensor_for_bdf(device.pci_bus_id)),
          cooldown_c_(options.cooldown_c),
          cooldown_poll_ms_(options.cooldown_poll_ms),
          cooldown_timeout_s_(options.cooldown_timeout_s),
          max_retries_(options.thermal_max_retries) {}

    void emit_sensor_mapping() const {
        std::ostringstream out;
        out << "{\"record\":\"thermal_sensor\""
            << ",\"hip_ordinal\":" << device_.ordinal
            << ",\"gcnArchName\":\"" << json_escape(device_.gcn_arch_name)
            << "\",\"pci_bus_id\":\"" << json_escape(device_.pci_bus_id)
            << "\",\"drm_card\":\"" << json_escape(sensor_.drm_card)
            << "\",\"hwmon\":\"" << json_escape(sensor_.hwmon_path.filename().string())
            << "\",\"temp2_input\":\"" << json_escape(sensor_.temp2_input_path.string())
            << "\",\"temp2_label\":\"" << json_escape(sensor_.temp2_label)
            << "\",\"junction_limit_c\":" << kJunctionLimitC
            << ",\"cooldown_target_c\":" << cooldown_c_
            << "}";
        emit_json_line(out.str(), true);
    }

    int max_retries() const { return max_retries_; }

    double check_before_launch(std::string_view phase) {
        return sample(phase, true);
    }

    double check_after_launch(std::string_view phase) {
        return sample(phase, true);
    }

    void wait_for_cooldown(std::string_view phase) {
        emit_cooldown_event("cooldown_wait_begin", phase, std::nullopt);
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(cooldown_timeout_s_);
        for (;;) {
            const double temperature_c = sample(std::string(phase) + ":poll", false);
            if (temperature_c <= cooldown_c_) {
                emit_cooldown_event("cooldown_complete", phase, temperature_c);
                return;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                emit_cooldown_event("cooldown_timeout", phase, temperature_c);
                throw ThermalCooldownTimeout(temperature_c, std::string(phase));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_poll_ms_));
        }
    }

private:
    double sample(std::string_view phase, bool enforce_limit) const {
        const std::string raw = read_single_line(sensor_.temp2_input_path);
        char* end = nullptr;
        const long long millidegrees = std::strtoll(raw.c_str(), &end, 10);
        if (end == raw.c_str() || *end != '\0' || millidegrees <= 0 || millidegrees > 200'000) {
            throw std::runtime_error(
                "invalid junction temperature from " + sensor_.temp2_input_path.string());
        }
        const double temperature_c = static_cast<double>(millidegrees) / 1'000.0;
        std::ostringstream out;
        out << std::setprecision(8)
            << "{\"record\":\"thermal\",\"event\":\"sample\""
            << ",\"unix_epoch_ms\":" << unix_epoch_ms()
            << ",\"phase\":\"" << json_escape(phase)
            << "\",\"pci_bus_id\":\"" << json_escape(device_.pci_bus_id)
            << "\",\"drm_card\":\"" << json_escape(sensor_.drm_card)
            << "\",\"temperature_c\":" << temperature_c
            << ",\"junction_limit_c\":" << kJunctionLimitC
            << ",\"cooldown_target_c\":" << cooldown_c_
            << ",\"enforce_limit\":" << (enforce_limit ? "true" : "false")
            << "}";
        emit_json_line(out.str(), true);
        if (enforce_limit && temperature_c >= kJunctionLimitC) {
            emit_cooldown_event("guard_trip", phase, temperature_c);
            throw ThermalLimitExceeded(temperature_c, std::string(phase));
        }
        return temperature_c;
    }

    void emit_cooldown_event(const char* event, std::string_view phase,
                             std::optional<double> temperature_c) const {
        std::ostringstream out;
        out << std::setprecision(8)
            << "{\"record\":\"thermal\",\"event\":\"" << event
            << "\",\"unix_epoch_ms\":" << unix_epoch_ms()
            << ",\"phase\":\"" << json_escape(phase)
            << "\",\"pci_bus_id\":\"" << json_escape(device_.pci_bus_id)
            << "\",\"drm_card\":\"" << json_escape(sensor_.drm_card)
            << "\",\"junction_limit_c\":" << kJunctionLimitC
            << ",\"cooldown_target_c\":" << cooldown_c_;
        if (temperature_c.has_value()) {
            out << ",\"temperature_c\":" << *temperature_c;
        }
        out << "}";
        emit_json_line(out.str(), true);
    }

    DeviceIdentity device_;
    JunctionSensor sensor_;
    double cooldown_c_;
    std::size_t cooldown_poll_ms_;
    std::size_t cooldown_timeout_s_;
    int max_retries_;
};

std::uint32_t lcg_next(std::uint32_t& state) {
    state = state * 1'664'525u + 1'013'904'223u;
    return state;
}

std::uint16_t f32_to_bf16_rne(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16u);
}

float bf16_bits_to_f32_host(std::uint16_t bits) {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint16_t f32_to_half_bits(float value) {
    const _Float16 half_value = static_cast<_Float16>(value);
    std::uint16_t bits = 0;
    static_assert(sizeof(half_value) == sizeof(bits));
    std::memcpy(&bits, &half_value, sizeof(bits));
    return bits;
}

float half_bits_to_f32_host(std::uint16_t bits) {
    _Float16 half_value{};
    std::memcpy(&half_value, &bits, sizeof(bits));
    return static_cast<float>(half_value);
}

float sq8_e4m3fn_to_f32_host(std::uint8_t value) {
    const std::uint32_t raw = value;
    const std::uint32_t sign = raw >> 7u;
    const std::uint32_t exponent = (raw >> 3u) & 0x0fu;
    const std::uint32_t mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const std::uint32_t fp32_bits =
        (sign << 31u) | ((exponent + 120u) << 23u) | (mantissa << 20u);
    float result = 0.0f;
    std::memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

std::size_t round_up_128(std::size_t value) {
    return (value + kBlockElements - 1u) / kBlockElements * kBlockElements;
}

struct HostMatrix {
    Shape shape{};
    std::size_t stored_cols = 0;
    std::size_t scale_rows = 0;
    std::size_t scale_cols = 0;
    std::vector<std::uint8_t> sq8_weight;
    // Canonical SQ8_0 stores BF16 scales.  The V620 fallback runtime decodes
    // those bytes on the host and uploads the resulting F32 scale grid.
    std::vector<float> sq8_scale_f32;
    // One contiguous SQ9_0 payload: [low-plane bytes][high-plane bytes].
    std::vector<std::uint8_t> sq9_payload;
    std::vector<std::uint16_t> fp16_weight;
};

HostMatrix make_host_matrix(const Shape& shape, int replica) {
    HostMatrix matrix;
    matrix.shape = shape;
    matrix.stored_cols = round_up_128(shape.cols);
    matrix.scale_rows = (shape.rows + kBlockElements - 1u) / kBlockElements;
    matrix.scale_cols = (shape.cols + kBlockElements - 1u) / kBlockElements;
    const std::size_t logical_elements = shape.rows * shape.cols;
    const std::size_t stored_elements = shape.rows * matrix.stored_cols;
    const std::size_t low_plane_bytes = stored_elements;
    const std::size_t high_plane_bytes = stored_elements / 8u;
    matrix.sq8_weight.resize(logical_elements);
    matrix.sq8_scale_f32.resize(matrix.scale_rows * matrix.scale_cols);
    matrix.sq9_payload.assign(low_plane_bytes + high_plane_bytes, 0u);
    matrix.fp16_weight.resize(logical_elements);

    std::uint32_t state = 0x6d2b79f5u ^ static_cast<std::uint32_t>(shape.rows) ^
                          (static_cast<std::uint32_t>(shape.cols) << 1u) ^
                          (static_cast<std::uint32_t>(replica) * 0x9e3779b9u);
    for (std::size_t block_row = 0; block_row < matrix.scale_rows; ++block_row) {
        for (std::size_t block_col = 0; block_col < matrix.scale_cols; ++block_col) {
            const float scale = 0.0078125f * (1.0f + 0.125f * static_cast<float>(lcg_next(state) & 7u));
            matrix.sq8_scale_f32[block_row * matrix.scale_cols + block_col] =
                bf16_bits_to_f32_host(f32_to_bf16_rne(scale));
        }
    }
    auto* sq9_low = matrix.sq9_payload.data();
    auto* sq9_high = matrix.sq9_payload.data() + low_plane_bytes;
    const std::size_t high_stride = matrix.stored_cols / 8u;
    for (std::size_t row = 0; row < shape.rows; ++row) {
        for (std::size_t col = 0; col < shape.cols; ++col) {
            const std::size_t logical = row * shape.cols + col;
            // Finite E4M3FN codes: exp 3..10.  The branch for 0x7f/0xff remains
            // present in the kernel because the payload format permits that sentinel.
            const std::uint8_t sq8_code = static_cast<std::uint8_t>(
                ((lcg_next(state) & 1u) << 7u) |
                ((3u + (lcg_next(state) & 7u)) << 3u) |
                (lcg_next(state) & 7u));
            matrix.sq8_weight[logical] = sq8_code;
            // Finite signed E5M3 code.  exp 8..13 avoids nonfinite/subnormal test
            // data while every decode still follows the normative q << 7 path.
            const std::uint16_t sq9_code = static_cast<std::uint16_t>(
                ((lcg_next(state) & 1u) << 8u) |
                ((8u + (lcg_next(state) & 7u)) << 3u) |
                (lcg_next(state) & 7u));
            const std::size_t physical = row * matrix.stored_cols + col;
            sq9_low[physical] = static_cast<std::uint8_t>(sq9_code & 0xffu);
            sq9_high[row * high_stride + (col >> 3u)] |= static_cast<std::uint8_t>(
                ((sq9_code >> 8u) & 1u) << (col & 7u));
            const float fp16_value =
                (static_cast<int>(lcg_next(state) & 0x7ffu) - 1024) * (1.0f / 2048.0f);
            matrix.fp16_weight[logical] = f32_to_half_bits(fp16_value);
        }
    }
    return matrix;
}

std::vector<float> make_input(std::size_t m, std::size_t cols) {
    std::vector<float> values(m * cols);
    std::uint32_t state = 0x12345678u;
    for (float& value : values) {
        value = (static_cast<int>(lcg_next(state) & 0x7ffu) - 1024) * (1.0f / 2048.0f);
    }
    return values;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { reset(); }

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        HIP_CHECK(hipMalloc(&ptr_, count * sizeof(T)));
    }

    void upload(const std::vector<T>& values, hipStream_t stream) {
        if (count_ != values.size()) {
            throw std::runtime_error("device upload size mismatch");
        }
        HIP_CHECK(hipMemcpyAsync(
            ptr_, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice, stream));
    }

    T* get() const { return ptr_; }
    std::size_t count() const { return count_; }

private:
    void reset() noexcept {
        if (ptr_ != nullptr) {
            (void)hipFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

struct DeviceReplica {
    DeviceBuffer<std::uint8_t> sq8_weight;
    DeviceBuffer<float> sq8_scale_f32;
    DeviceBuffer<std::uint8_t> sq9_payload;
    DeviceBuffer<std::uint16_t> fp16_weight;
};

struct DeviceMatrix {
    Shape shape{};
    std::size_t stored_cols = 0;
    std::size_t scale_rows = 0;
    std::size_t scale_cols = 0;
    std::vector<DeviceReplica> replicas;
    DeviceBuffer<float> input;
    DeviceBuffer<float> output;
};

DeviceMatrix upload_matrix(const Shape& shape, std::size_t max_m, ThermalGuard& thermal,
                           hipStream_t stream) {
    DeviceMatrix matrix;
    matrix.shape = shape;
    matrix.stored_cols = round_up_128(shape.cols);
    matrix.scale_rows = (shape.rows + kBlockElements - 1u) / kBlockElements;
    matrix.scale_cols = (shape.cols + kBlockElements - 1u) / kBlockElements;
    matrix.replicas.resize(kReplicaCount);
    for (int replica = 0; replica < kReplicaCount; ++replica) {
        const std::string phase = std::string("upload:") + shape.name + ":replica:" +
            std::to_string(replica);
        thermal.check_before_launch(phase + ":before");
        HostMatrix host = make_host_matrix(shape, replica);
        auto& device = matrix.replicas[static_cast<std::size_t>(replica)];
        device.sq8_weight.allocate(host.sq8_weight.size());
        device.sq8_scale_f32.allocate(host.sq8_scale_f32.size());
        device.sq9_payload.allocate(host.sq9_payload.size());
        device.fp16_weight.allocate(host.fp16_weight.size());
        device.sq8_weight.upload(host.sq8_weight, stream);
        device.sq8_scale_f32.upload(host.sq8_scale_f32, stream);
        device.sq9_payload.upload(host.sq9_payload, stream);
        device.fp16_weight.upload(host.fp16_weight, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        thermal.check_after_launch(phase + ":after");
    }
    const std::vector<float> input = make_input(max_m, shape.cols);
    thermal.check_before_launch(std::string("upload:") + shape.name + ":input_before");
    matrix.input.allocate(input.size());
    matrix.input.upload(input, stream);
    matrix.output.allocate(max_m * shape.rows);
    HIP_CHECK(hipStreamSynchronize(stream));
    thermal.check_after_launch(std::string("upload:") + shape.name + ":input_after");
    return matrix;
}

__device__ __forceinline__ float sq8_e4m3fn_to_f32_device(unsigned char value) {
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7u;
    const unsigned int exponent = (raw >> 3u) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return __uint_as_float(0x7fc00000u);
    }
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const unsigned int fp32_bits =
        (sign << 31u) | ((exponent + 120u) << 23u) | (mantissa << 20u);
    return __uint_as_float(fp32_bits);
}

__global__ void sq9_v620_sanity_kernel(float* output) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 256u) {
        output[index] = static_cast<float>(index) + 0.25f;
    }
}

__device__ __forceinline__ float sq9_e5m3_to_f32_device(std::uint16_t code) {
    // q is already a validated nine-bit field assembled from lo8 and hi1.
    // No mask occurs between q assembly and this normative one-shot shift.
    return __half2float(__ushort_as_half(static_cast<unsigned short>(code << 7u)));
}

__device__ __forceinline__ std::uint8_t packed_byte(const uint4& packed, int index) {
    const unsigned int word = index < 4 ? packed.x :
                              (index < 8 ? packed.y : (index < 12 ? packed.z : packed.w));
    return static_cast<std::uint8_t>((word >> ((index & 3) * 8)) & 0xffu);
}

template <int F, int MTILE, bool UseLdsHighPlane>
__global__ void gemm_kernel(
    const std::uint8_t* sq8_weight,
    const float* sq8_scale_f32,
    const std::uint8_t* sq9_payload,
    const std::uint16_t* fp16_weight,
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    std::uint64_t stored_cols,
    std::uint64_t scale_cols,
    std::uint64_t m) {
    const unsigned int tid = threadIdx.x;
    const std::uint64_t row = blockIdx.x;
    const std::uint64_t m_base = static_cast<std::uint64_t>(blockIdx.y) * MTILE;
    const unsigned int group = tid / kThreadsPerBlockTile;
    const unsigned int lane = tid % kThreadsPerBlockTile;
    __shared__ float reduction[MTILE][kThreads];
    __shared__ float scale_shared[kTileGroups];
    __shared__ uint4 high_shared[kTileGroups];
    float sums[MTILE] = {};

    if (row < rows && m_base < m) {
        const std::uint64_t tiles = (cols + kBlockElements - 1u) / kBlockElements;
        const std::uint64_t high_stride = stored_cols / 8u;
        const std::uint8_t* const sq9_low = sq9_payload;
        const std::uint8_t* const sq9_high = sq9_payload + rows * stored_cols;
        for (std::uint64_t tile = group; tile < tiles; tile += kTileGroups) {
            if constexpr (F == kSq8) {
                if (lane == 0u) {
                    const std::uint64_t scale_index = (row / kBlockElements) * scale_cols + tile;
                    scale_shared[group] = sq8_scale_f32[scale_index];
                }
                __syncthreads();
            } else if constexpr (F == kSq9Lane && UseLdsHighPlane) {
                if (lane == 0u) {
                    const auto* high_words = reinterpret_cast<const uint4*>(
                        sq9_high + row * high_stride + tile * (kBlockElements / 8));
                    high_shared[group] = *high_words;
                }
                __syncthreads();
            }

            const std::uint64_t column_base = tile * kBlockElements + lane * 16u;
            uint4 packed_a{};
            uint4 packed_b{};
            if constexpr (F == kSq8) {
                const auto* words = reinterpret_cast<const uint4*>(
                    sq8_weight + row * cols + tile * kBlockElements + lane * 16u);
                packed_a = *words;
            } else if constexpr (F == kSq9Lane) {
                const auto* words = reinterpret_cast<const uint4*>(
                    sq9_low + row * stored_cols + tile * kBlockElements + lane * 16u);
                packed_a = *words;
            } else {
                const auto* words = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const std::uint8_t*>(fp16_weight + row * cols) +
                    (tile * kBlockElements + lane * 16u) * sizeof(std::uint16_t));
                packed_a = words[0];
                packed_b = words[1];
            }

#pragma unroll
            for (int element = 0; element < 16; ++element) {
                const std::uint64_t col = column_base + static_cast<std::uint64_t>(element);
                if (col < cols) {
                    float weight = 0.0f;
                    if constexpr (F == kSq8) {
                        weight = sq8_e4m3fn_to_f32_device(packed_byte(packed_a, element)) *
                                 scale_shared[group];
                    } else if constexpr (F == kSq9Lane) {
                        const unsigned int high_byte_index =
                            (lane * 16u + static_cast<unsigned int>(element)) >> 3u;
                        const std::uint8_t high_byte = UseLdsHighPlane
                            ? packed_byte(high_shared[group], static_cast<int>(high_byte_index))
                            : sq9_high[row * high_stride + tile * (kBlockElements / 8) +
                                       high_byte_index];
                        const unsigned int bit =
                            (lane * 16u + static_cast<unsigned int>(element)) & 7u;
                        const std::uint16_t code = static_cast<std::uint16_t>(
                            static_cast<std::uint16_t>(packed_byte(packed_a, element)) |
                            (static_cast<std::uint16_t>((high_byte >> bit) & 1u) << 8u));
                        weight = sq9_e5m3_to_f32_device(code);
                    } else {
                        const uint4& source = element < 8 ? packed_a : packed_b;
                        const int half_index = (element < 8 ? element : element - 8) * 2;
                        const std::uint16_t bits = static_cast<std::uint16_t>(
                            static_cast<std::uint16_t>(packed_byte(source, half_index)) |
                            (static_cast<std::uint16_t>(packed_byte(source, half_index + 1)) << 8u));
                        weight = __half2float(__ushort_as_half(bits));
                    }
#pragma unroll
                    for (int local_m = 0; local_m < MTILE; ++local_m) {
                        if (m_base + static_cast<std::uint64_t>(local_m) < m) {
                            sums[local_m] = fmaf(
                                weight,
                                input[(m_base + static_cast<std::uint64_t>(local_m)) * cols + col],
                                sums[local_m]);
                        }
                    }
                }
            }
            if constexpr (F == kSq8 || (F == kSq9Lane && UseLdsHighPlane)) {
                __syncthreads();
            }
        }
    }

#pragma unroll
    for (int local_m = 0; local_m < MTILE; ++local_m) {
        reduction[local_m][tid] = sums[local_m];
    }
    __syncthreads();
    for (unsigned int offset = kThreads / 2u; offset != 0u; offset >>= 1u) {
        if (tid < offset) {
#pragma unroll
            for (int local_m = 0; local_m < MTILE; ++local_m) {
                reduction[local_m][tid] += reduction[local_m][tid + offset];
            }
        }
        __syncthreads();
    }
    if (tid == 0u && row < rows) {
#pragma unroll
        for (int local_m = 0; local_m < MTILE; ++local_m) {
            if (m_base + static_cast<std::uint64_t>(local_m) < m) {
                output[(m_base + static_cast<std::uint64_t>(local_m)) * rows + row] =
                    reduction[local_m][0];
            }
        }
    }
}

template <int F, bool UseLdsHighPlane>
__global__ void dequant_sum_kernel(
    const std::uint8_t* sq8_weight,
    const float* sq8_scale_f32,
    const std::uint8_t* sq9_payload,
    const std::uint16_t* fp16_weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    std::uint64_t stored_cols,
    std::uint64_t scale_cols) {
    const unsigned int tid = threadIdx.x;
    const std::uint64_t row = blockIdx.x;
    const unsigned int group = tid / kThreadsPerBlockTile;
    const unsigned int lane = tid % kThreadsPerBlockTile;
    __shared__ float reduction[kThreads];
    __shared__ float scale_shared[kTileGroups];
    __shared__ uint4 high_shared[kTileGroups];
    float sum = 0.0f;
    if (row < rows) {
        const std::uint64_t tiles = (cols + kBlockElements - 1u) / kBlockElements;
        const std::uint64_t high_stride = stored_cols / 8u;
        const std::uint8_t* const sq9_low = sq9_payload;
        const std::uint8_t* const sq9_high = sq9_payload + rows * stored_cols;
        for (std::uint64_t tile = group; tile < tiles; tile += kTileGroups) {
            if constexpr (F == kSq8) {
                if (lane == 0u) {
                    scale_shared[group] =
                        sq8_scale_f32[(row / kBlockElements) * scale_cols + tile];
                }
                __syncthreads();
            } else if constexpr (F == kSq9Lane && UseLdsHighPlane) {
                if (lane == 0u) {
                    const auto* high_words = reinterpret_cast<const uint4*>(
                        sq9_high + row * high_stride + tile * (kBlockElements / 8));
                    high_shared[group] = *high_words;
                }
                __syncthreads();
            }
            const std::uint64_t column_base = tile * kBlockElements + lane * 16u;
            uint4 packed_a{};
            uint4 packed_b{};
            if constexpr (F == kSq8) {
                packed_a = *reinterpret_cast<const uint4*>(
                    sq8_weight + row * cols + tile * kBlockElements + lane * 16u);
            } else if constexpr (F == kSq9Lane) {
                packed_a = *reinterpret_cast<const uint4*>(
                    sq9_low + row * stored_cols + tile * kBlockElements + lane * 16u);
            } else {
                const auto* words = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const std::uint8_t*>(fp16_weight + row * cols) +
                    (tile * kBlockElements + lane * 16u) * sizeof(std::uint16_t));
                packed_a = words[0];
                packed_b = words[1];
            }
#pragma unroll
            for (int element = 0; element < 16; ++element) {
                if (column_base + static_cast<std::uint64_t>(element) < cols) {
                    float value = 0.0f;
                    if constexpr (F == kSq8) {
                        value = sq8_e4m3fn_to_f32_device(packed_byte(packed_a, element)) *
                                scale_shared[group];
                    } else if constexpr (F == kSq9Lane) {
                        const unsigned int high_byte_index =
                            (lane * 16u + static_cast<unsigned int>(element)) >> 3u;
                        const std::uint8_t high_byte = UseLdsHighPlane
                            ? packed_byte(high_shared[group], static_cast<int>(high_byte_index))
                            : sq9_high[row * high_stride + tile * (kBlockElements / 8) +
                                       high_byte_index];
                        const unsigned int bit =
                            (lane * 16u + static_cast<unsigned int>(element)) & 7u;
                        const std::uint16_t code = static_cast<std::uint16_t>(
                            static_cast<std::uint16_t>(packed_byte(packed_a, element)) |
                            (static_cast<std::uint16_t>((high_byte >> bit) & 1u) << 8u));
                        value = sq9_e5m3_to_f32_device(code);
                    } else {
                        const uint4& source = element < 8 ? packed_a : packed_b;
                        const int half_index = (element < 8 ? element : element - 8) * 2;
                        const std::uint16_t bits = static_cast<std::uint16_t>(
                            static_cast<std::uint16_t>(packed_byte(source, half_index)) |
                            (static_cast<std::uint16_t>(packed_byte(source, half_index + 1)) << 8u));
                        value = __half2float(__ushort_as_half(bits));
                    }
                    sum += value;
                }
            }
            if constexpr (F == kSq8 || (F == kSq9Lane && UseLdsHighPlane)) {
                __syncthreads();
            }
        }
    }
    reduction[tid] = sum;
    __syncthreads();
    for (unsigned int offset = kThreads / 2u; offset != 0u; offset >>= 1u) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0u && row < rows) {
        output[row] = reduction[0];
    }
}

// This control streams the same physical payload layout but does not perform
// E4M3/E5M3-to-float conversion or a reconstruction multiply.  It provides a
// direct dequant-versus-raw-stream comparison for the ALU-bound question.
template <int F, bool UseLdsHighPlane>
__global__ void raw_stream_kernel(
    const std::uint8_t* sq8_weight,
    const float* sq8_scale_f32,
    const std::uint8_t* sq9_payload,
    const std::uint16_t* fp16_weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    std::uint64_t stored_cols,
    std::uint64_t scale_cols) {
    const unsigned int tid = threadIdx.x;
    const std::uint64_t row = blockIdx.x;
    const unsigned int group = tid / kThreadsPerBlockTile;
    const unsigned int lane = tid % kThreadsPerBlockTile;
    __shared__ float reduction[kThreads];
    __shared__ float scale_shared[kTileGroups];
    __shared__ uint4 high_shared[kTileGroups];
    float sum = 0.0f;
    if (row < rows) {
        const std::uint64_t tiles = (cols + kBlockElements - 1u) / kBlockElements;
        const std::uint64_t high_stride = stored_cols / 8u;
        const std::uint8_t* const sq9_low = sq9_payload;
        const std::uint8_t* const sq9_high = sq9_payload + rows * stored_cols;
        for (std::uint64_t tile = group; tile < tiles; tile += kTileGroups) {
            if constexpr (F == kSq8) {
                if (lane == 0u) {
                    scale_shared[group] =
                        sq8_scale_f32[(row / kBlockElements) * scale_cols + tile];
                }
                __syncthreads();
            } else if constexpr (F == kSq9Lane && UseLdsHighPlane) {
                if (lane == 0u) {
                    high_shared[group] = *reinterpret_cast<const uint4*>(
                        sq9_high + row * high_stride + tile * (kBlockElements / 8));
                }
                __syncthreads();
            }
            const std::uint64_t column_base = tile * kBlockElements + lane * 16u;
            uint4 packed_a{};
            uint4 packed_b{};
            if constexpr (F == kSq8) {
                packed_a = *reinterpret_cast<const uint4*>(
                    sq8_weight + row * cols + tile * kBlockElements + lane * 16u);
            } else if constexpr (F == kSq9Lane) {
                packed_a = *reinterpret_cast<const uint4*>(
                    sq9_low + row * stored_cols + tile * kBlockElements + lane * 16u);
            } else {
                const auto* words = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const std::uint8_t*>(fp16_weight + row * cols) +
                    (tile * kBlockElements + lane * 16u) * sizeof(std::uint16_t));
                packed_a = words[0];
                packed_b = words[1];
            }
#pragma unroll
            for (int element = 0; element < 16; ++element) {
                if (column_base + static_cast<std::uint64_t>(element) < cols) {
                    if constexpr (F == kSq8) {
                        sum += static_cast<float>(packed_byte(packed_a, element)) * 0.000001f +
                               scale_shared[group] * 0.000000001f;
                    } else if constexpr (F == kSq9Lane) {
                        const unsigned int high_byte_index =
                            (lane * 16u + static_cast<unsigned int>(element)) >> 3u;
                        const std::uint8_t high_byte = UseLdsHighPlane
                            ? packed_byte(high_shared[group], static_cast<int>(high_byte_index))
                            : sq9_high[row * high_stride + tile * (kBlockElements / 8) +
                                       high_byte_index];
                        sum += static_cast<float>(packed_byte(packed_a, element)) * 0.000001f +
                               static_cast<float>(high_byte) * 0.000000001f;
                    } else {
                        const uint4& source = element < 8 ? packed_a : packed_b;
                        const int half_index = (element < 8 ? element : element - 8) * 2;
                        sum += static_cast<float>(packed_byte(source, half_index)) * 0.000001f +
                               static_cast<float>(packed_byte(source, half_index + 1)) * 0.000000001f;
                    }
                }
            }
            if constexpr (F == kSq8 || (F == kSq9Lane && UseLdsHighPlane)) {
                __syncthreads();
            }
        }
    }
    reduction[tid] = sum;
    __syncthreads();
    for (unsigned int offset = kThreads / 2u; offset != 0u; offset >>= 1u) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0u && row < rows) {
        output[row] = reduction[0];
    }
}

template <int F, int MTILE, bool UseLdsHighPlane>
void launch_gemm_templated(const DeviceMatrix& matrix, const DeviceReplica& replica,
                           std::size_t m, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(matrix.shape.rows),
                    static_cast<unsigned int>((m + MTILE - 1u) / MTILE));
    hipLaunchKernelGGL((gemm_kernel<F, MTILE, UseLdsHighPlane>), grid, dim3(kThreads), 0, stream,
                       replica.sq8_weight.get(), replica.sq8_scale_f32.get(),
                       replica.sq9_payload.get(), replica.fp16_weight.get(), matrix.input.get(),
                       matrix.output.get(), static_cast<std::uint64_t>(matrix.shape.rows),
                       static_cast<std::uint64_t>(matrix.shape.cols),
                       static_cast<std::uint64_t>(matrix.stored_cols),
                       static_cast<std::uint64_t>(matrix.scale_cols),
                       static_cast<std::uint64_t>(m));
    HIP_CHECK(hipGetLastError());
}

template <int F, bool UseLdsHighPlane>
void launch_dequant_templated(const DeviceMatrix& matrix, const DeviceReplica& replica,
                              hipStream_t stream) {
    hipLaunchKernelGGL((dequant_sum_kernel<F, UseLdsHighPlane>),
                       dim3(static_cast<unsigned int>(matrix.shape.rows)), dim3(kThreads), 0,
                       stream, replica.sq8_weight.get(), replica.sq8_scale_f32.get(),
                       replica.sq9_payload.get(), replica.fp16_weight.get(), matrix.output.get(),
                       static_cast<std::uint64_t>(matrix.shape.rows),
                       static_cast<std::uint64_t>(matrix.shape.cols),
                       static_cast<std::uint64_t>(matrix.stored_cols),
                       static_cast<std::uint64_t>(matrix.scale_cols));
    HIP_CHECK(hipGetLastError());
}

template <int F, bool UseLdsHighPlane>
void launch_raw_templated(const DeviceMatrix& matrix, const DeviceReplica& replica,
                          hipStream_t stream) {
    hipLaunchKernelGGL((raw_stream_kernel<F, UseLdsHighPlane>),
                       dim3(static_cast<unsigned int>(matrix.shape.rows)), dim3(kThreads), 0,
                       stream, replica.sq8_weight.get(), replica.sq8_scale_f32.get(),
                       replica.sq9_payload.get(), replica.fp16_weight.get(), matrix.output.get(),
                       static_cast<std::uint64_t>(matrix.shape.rows),
                       static_cast<std::uint64_t>(matrix.shape.cols),
                       static_cast<std::uint64_t>(matrix.stored_cols),
                       static_cast<std::uint64_t>(matrix.scale_cols));
    HIP_CHECK(hipGetLastError());
}

int tile_for_m(std::size_t m) {
    if (m == 1) {
        return 1;
    }
    if (m <= 8) {
        return 8;
    }
    // A 32-row tile would require 32 KiB for the reduction alone, before
    // the high-plane/scale staging.  Keep the V620 workgroup below the
    // conservative 32 KiB LDS ceiling and reuse weights across 16 rows.
    return 16;
}

void launch_gemm(Format format, const DeviceMatrix& matrix, const DeviceReplica& replica,
                 std::size_t m, hipStream_t stream) {
    const int m_tile = tile_for_m(m);
    switch (format) {
    case kSq8:
        if (m_tile == 1) launch_gemm_templated<kSq8, 1, false>(matrix, replica, m, stream);
        else if (m_tile == 8) launch_gemm_templated<kSq8, 8, false>(matrix, replica, m, stream);
        else launch_gemm_templated<kSq8, 16, false>(matrix, replica, m, stream);
        break;
    case kSq9Lane:
        if (m_tile == 1) launch_gemm_templated<kSq9Lane, 1, false>(matrix, replica, m, stream);
        else if (m_tile == 8) launch_gemm_templated<kSq9Lane, 8, false>(matrix, replica, m, stream);
        else launch_gemm_templated<kSq9Lane, 16, false>(matrix, replica, m, stream);
        break;
    case kSq9Lds:
        if (m_tile == 1) launch_gemm_templated<kSq9Lane, 1, true>(matrix, replica, m, stream);
        else if (m_tile == 8) launch_gemm_templated<kSq9Lane, 8, true>(matrix, replica, m, stream);
        else launch_gemm_templated<kSq9Lane, 16, true>(matrix, replica, m, stream);
        break;
    case kFp16:
        if (m_tile == 1) launch_gemm_templated<kFp16, 1, false>(matrix, replica, m, stream);
        else if (m_tile == 8) launch_gemm_templated<kFp16, 8, false>(matrix, replica, m, stream);
        else launch_gemm_templated<kFp16, 16, false>(matrix, replica, m, stream);
        break;
    }
}

void launch_dequant(Format format, const DeviceMatrix& matrix, const DeviceReplica& replica,
                    hipStream_t stream) {
    switch (format) {
    case kSq8: launch_dequant_templated<kSq8, false>(matrix, replica, stream); break;
    case kSq9Lane: launch_dequant_templated<kSq9Lane, false>(matrix, replica, stream); break;
    case kSq9Lds: launch_dequant_templated<kSq9Lane, true>(matrix, replica, stream); break;
    case kFp16: launch_dequant_templated<kFp16, false>(matrix, replica, stream); break;
    }
}

void launch_raw(Format format, const DeviceMatrix& matrix, const DeviceReplica& replica,
                hipStream_t stream) {
    switch (format) {
    case kSq8: launch_raw_templated<kSq8, false>(matrix, replica, stream); break;
    case kSq9Lane: launch_raw_templated<kSq9Lane, false>(matrix, replica, stream); break;
    case kSq9Lds: launch_raw_templated<kSq9Lane, true>(matrix, replica, stream); break;
    case kFp16: launch_raw_templated<kFp16, false>(matrix, replica, stream); break;
    }
}

const char* format_name(Format format) {
    switch (format) {
    case kSq8: return "SQ8_0";
    case kSq9Lane: return "SQ9_0";
    case kSq9Lds: return "SQ9_0";
    case kFp16: return "FP16";
    }
    return "unknown";
}

const char* variant_name(Format format) {
    switch (format) {
    case kSq8: return "v620_fallback_f8_e4m3_f32_block_128x128";
    case kSq9Lane: return "lo8_hi1_per_lane_high_byte";
    case kSq9Lds: return "lo8_hi1_cooperative_uint4_high_plane_lds";
    case kFp16: return "raw_fp16";
    }
    return "unknown";
}

std::size_t physical_weight_bytes(Format format, const DeviceMatrix& matrix) {
    const std::size_t elements = matrix.shape.rows * matrix.shape.cols;
    switch (format) {
    case kSq8:
        return elements + matrix.scale_rows * matrix.scale_cols * sizeof(float);
    case kSq9Lane:
    case kSq9Lds:
        return matrix.shape.rows * matrix.stored_cols * 9u / 8u;
    case kFp16:
        return elements * sizeof(std::uint16_t);
    }
    return 0;
}

float reference_weight(const HostMatrix& matrix, Format format, std::size_t row, std::size_t col) {
    const std::size_t logical = row * matrix.shape.cols + col;
    if (format == kSq8) {
        const float scale = matrix.sq8_scale_f32[(row / kBlockElements) * matrix.scale_cols +
                                                 (col / kBlockElements)];
        return sq8_e4m3fn_to_f32_host(matrix.sq8_weight[logical]) * scale;
    }
    if (format == kSq9Lane || format == kSq9Lds) {
        const std::size_t low_bytes = matrix.shape.rows * matrix.stored_cols;
        const std::uint8_t low = matrix.sq9_payload[row * matrix.stored_cols + col];
        const std::uint8_t high = matrix.sq9_payload[
            low_bytes + row * (matrix.stored_cols / 8u) + (col >> 3u)];
        const std::uint16_t code = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(low) |
            (static_cast<std::uint16_t>((high >> (col & 7u)) & 1u) << 8u));
        return half_bits_to_f32_host(static_cast<std::uint16_t>(code << 7u));
    }
    return half_bits_to_f32_host(matrix.fp16_weight[logical]);
}

std::vector<float> reference_gemv(const HostMatrix& matrix, Format format,
                                  const std::vector<float>& input) {
    std::vector<float> output(matrix.shape.rows, 0.0f);
    for (std::size_t row = 0; row < matrix.shape.rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < matrix.shape.cols; ++col) {
            sum = std::fmaf(reference_weight(matrix, format, row, col), input[col], sum);
        }
        output[row] = sum;
    }
    return output;
}

std::vector<float> reference_dequant_sums(const HostMatrix& matrix, Format format) {
    std::vector<float> output(matrix.shape.rows, 0.0f);
    for (std::size_t row = 0; row < matrix.shape.rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < matrix.shape.cols; ++col) {
            sum += reference_weight(matrix, format, row, col);
        }
        output[row] = sum;
    }
    return output;
}

struct Difference {
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t max_index = 0;
};

Difference compare_vectors(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("correctness vector size mismatch");
    }
    Difference result;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double diff = std::abs(static_cast<double>(expected[index]) - actual[index]);
        const double relative = diff / std::max(1.0, std::abs(static_cast<double>(expected[index])));
        if (diff > result.max_abs) {
            result.max_abs = diff;
            result.max_index = index;
        }
        result.max_rel = std::max(result.max_rel, relative);
    }
    return result;
}

void emit_correctness(const Shape& shape, Format format, const char* kind,
                      const Difference& diff, bool passed) {
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"record\":\"correctness\",\"shape\":\"" << shape.name
        << "\",\"model\":\"" << shape.model
        << "\",\"format\":\"" << format_name(format)
        << "\",\"variant\":\"" << variant_name(format)
        << "\",\"kind\":\"" << kind
        << "\",\"max_abs\":" << diff.max_abs
        << ",\"max_rel\":" << diff.max_rel
        << ",\"max_index\":" << diff.max_index
        << ",\"passed\":" << (passed ? "true" : "false") << "}";
    emit_json_line(out.str());
}

void verify_matrix(const DeviceMatrix& device, hipStream_t stream, ThermalGuard& thermal) {
    const HostMatrix host = make_host_matrix(device.shape, 0);
    const std::vector<float> input = make_input(1, device.shape.cols);
    std::vector<float> gpu(device.shape.rows);
    for (const Format format : {kSq8, kSq9Lane, kSq9Lds, kFp16}) {
        const std::string base = std::string("correctness:") + device.shape.name + ":" +
            format_name(format);
        thermal.check_before_launch(base + ":gemv_before");
        launch_gemm(format, device, device.replicas.front(), 1, stream);
        HIP_CHECK(hipMemcpyAsync(gpu.data(), device.output.get(), gpu.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        thermal.check_after_launch(base + ":gemv_after");
        const Difference dot_diff = compare_vectors(reference_gemv(host, format, input), gpu);
        const bool dot_passed = dot_diff.max_abs <= 0.01 && dot_diff.max_rel <= 0.002;
        emit_correctness(device.shape, format, "dequant_plus_gemv_m1", dot_diff, dot_passed);
        if (!dot_passed) {
            throw std::runtime_error("GPU GEMV output did not match CPU reference");
        }

        thermal.check_before_launch(base + ":dequant_before");
        launch_dequant(format, device, device.replicas.front(), stream);
        HIP_CHECK(hipMemcpyAsync(gpu.data(), device.output.get(), gpu.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        thermal.check_after_launch(base + ":dequant_after");
        const Difference dequant_diff = compare_vectors(reference_dequant_sums(host, format), gpu);
        const bool dequant_passed = dequant_diff.max_abs <= 0.01 && dequant_diff.max_rel <= 0.002;
        emit_correctness(device.shape, format, "dequant_only", dequant_diff, dequant_passed);
        if (!dequant_passed) {
            throw std::runtime_error("GPU dequant output did not match CPU reference");
        }
    }
}

struct Timing {
    std::vector<float> samples_ms;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double temperature_before_warmup_c = 0.0;
    double temperature_after_warmup_c = 0.0;
    double temperature_start_c = 0.0;
    double temperature_end_c = 0.0;
    double temperature_peak_c = 0.0;
    std::size_t thermal_sample_count = 0;
    int thermal_retry_count = 0;
};

void observe_timing_temperature(Timing& timing, double temperature_c) {
    if (timing.thermal_sample_count == 0u || temperature_c > timing.temperature_peak_c) {
        timing.temperature_peak_c = temperature_c;
    }
    ++timing.thermal_sample_count;
}

template <typename Launch>
Timing time_launch(const Options& options, ThermalGuard& thermal, std::string_view point_id,
                   hipStream_t stream, Launch&& launch) {
    const std::string point(point_id);
    Timing timing;
    thermal.wait_for_cooldown(point + ":before_warmup");
    timing.temperature_before_warmup_c =
        thermal.check_before_launch(point + ":warmup_start");
    observe_timing_temperature(timing, timing.temperature_before_warmup_c);
    for (int warmup = 0; warmup < options.warmups; ++warmup) {
        for (int iteration = 0; iteration < options.launches_per_trial; ++iteration) {
            observe_timing_temperature(
                timing, thermal.check_before_launch(
                            point + ":warmup:" + std::to_string(warmup) + ":" +
                            std::to_string(iteration) + ":before"));
            launch(warmup * options.launches_per_trial + iteration);
            HIP_CHECK(hipStreamSynchronize(stream));
            observe_timing_temperature(
                timing, thermal.check_after_launch(
                            point + ":warmup:" + std::to_string(warmup) + ":" +
                            std::to_string(iteration) + ":after"));
        }
    }
    timing.temperature_after_warmup_c =
        thermal.check_after_launch(point + ":warmup_complete");
    observe_timing_temperature(timing, timing.temperature_after_warmup_c);
    // Warmup is outside the timing sample.  Cool again so every point begins
    // from the same explicit temperature ceiling, rather than accumulated heat.
    thermal.wait_for_cooldown(point + ":before_timing");
    timing.temperature_start_c = thermal.check_before_launch(point + ":timing_start");
    observe_timing_temperature(timing, timing.temperature_start_c);

    hipEvent_t start{};
    hipEvent_t stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    timing.samples_ms.reserve(options.trials);
    try {
        for (int trial = 0; trial < options.trials; ++trial) {
            float trial_elapsed = 0.0f;
            for (int iteration = 0; iteration < options.launches_per_trial; ++iteration) {
                observe_timing_temperature(
                    timing, thermal.check_before_launch(
                                point + ":trial:" + std::to_string(trial) + ":" +
                                std::to_string(iteration) + ":before"));
                HIP_CHECK(hipEventRecord(start, stream));
                launch(trial * options.launches_per_trial + iteration);
                HIP_CHECK(hipEventRecord(stop, stream));
                HIP_CHECK(hipEventSynchronize(stop));
                float elapsed = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
                trial_elapsed += elapsed;
                observe_timing_temperature(
                    timing, thermal.check_after_launch(
                                point + ":trial:" + std::to_string(trial) + ":" +
                                std::to_string(iteration) + ":after"));
            }
            timing.samples_ms.push_back(
                trial_elapsed / static_cast<float>(options.launches_per_trial));
        }
    } catch (...) {
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        throw;
    }
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    timing.temperature_end_c = thermal.check_after_launch(point + ":timing_end");
    observe_timing_temperature(timing, timing.temperature_end_c);
    std::vector<float> sorted = timing.samples_ms;
    std::sort(sorted.begin(), sorted.end());
    timing.min_ms = sorted.front();
    timing.max_ms = sorted.back();
    timing.median_ms = sorted[sorted.size() / 2u];
    timing.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double variance = 0.0;
    for (const float sample : sorted) {
        const double delta = static_cast<double>(sample) - timing.mean_ms;
        variance += delta * delta;
    }
    timing.stddev_ms = std::sqrt(variance / sorted.size());
    return timing;
}

void emit_measurement_event(const char* event, std::string_view point_id, int attempt,
                            std::optional<double> temperature_c = std::nullopt,
                            std::string_view phase = {}) {
    std::ostringstream out;
    out << std::setprecision(8)
        << "{\"record\":\"measurement\",\"event\":\"" << event
        << "\",\"unix_epoch_ms\":" << unix_epoch_ms()
        << ",\"point_id\":\"" << json_escape(point_id)
        << "\",\"attempt\":" << attempt;
    if (temperature_c.has_value()) {
        out << ",\"temperature_c\":" << *temperature_c;
    }
    if (!phase.empty()) {
        out << ",\"phase\":\"" << json_escape(phase) << "\"";
    }
    out << "}";
    emit_json_line(out.str(), true);
}

template <typename Work>
Timing run_timing_point_with_recovery(ThermalGuard& thermal, std::string point_id, Work&& work) {
    for (int attempt = 0;; ++attempt) {
        emit_measurement_event("begin", point_id, attempt);
        try {
            Timing timing = work();
            timing.thermal_retry_count = attempt;
            emit_measurement_event("complete", point_id, attempt, timing.temperature_end_c);
            return timing;
        } catch (const ThermalLimitExceeded& error) {
            emit_measurement_event(
                "thermal_interrupt", point_id, attempt, error.temperature_c(), error.phase());
            thermal.wait_for_cooldown(point_id + ":post_trip");
            if (attempt >= thermal.max_retries()) {
                emit_measurement_event(
                    "thermal_retry_exhausted", point_id, attempt, error.temperature_c(), error.phase());
                throw;
            }
            emit_measurement_event("thermal_retry", point_id, attempt + 1);
        }
    }
}

void emit_timing(const char* kind, const DeviceMatrix& matrix, Format format, std::size_t m,
                 const Timing& timing) {
    const std::size_t m_tile = static_cast<std::size_t>(tile_for_m(m));
    const std::size_t weight_bytes = physical_weight_bytes(format, matrix);
    const std::size_t weighted_tile_reads =
        weight_bytes * ((m + m_tile - 1u) / m_tile);
    const double seconds = timing.median_ms / 1'000.0;
    const double element_ops = static_cast<double>(m) * static_cast<double>(matrix.shape.rows) *
                               static_cast<double>(matrix.shape.cols);
    const double weight_gbps = static_cast<double>(weighted_tile_reads) / seconds / 1.0e9;
    const double gflops = 2.0 * element_ops / seconds / 1.0e9;
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"record\":\"timing\",\"kind\":\"" << kind
        << "\",\"shape\":\"" << matrix.shape.name
        << "\",\"model\":\"" << matrix.shape.model
        << "\",\"projection\":\"" << matrix.shape.projection
        << "\",\"format\":\"" << format_name(format)
        << "\",\"variant\":\"" << variant_name(format)
        << "\",\"rows\":" << matrix.shape.rows
        << ",\"cols\":" << matrix.shape.cols
        << ",\"m\":" << m
        << ",\"m_tile\":" << m_tile
        << ",\"weight_bytes_resident\":" << weight_bytes
        << ",\"modeled_weight_bytes_per_launch\":" << weighted_tile_reads
        << ",\"median_ms\":" << timing.median_ms
        << ",\"mean_ms\":" << timing.mean_ms
        << ",\"stddev_ms\":" << timing.stddev_ms
        << ",\"min_ms\":" << timing.min_ms
        << ",\"max_ms\":" << timing.max_ms
        << ",\"samples_ms\":[";
    for (std::size_t index = 0; index < timing.samples_ms.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        out << timing.samples_ms[index];
    }
    out << "]"
        << ",\"temperature_before_warmup_c\":" << timing.temperature_before_warmup_c
        << ",\"temperature_after_warmup_c\":" << timing.temperature_after_warmup_c
        << ",\"temperature_start_c\":" << timing.temperature_start_c
        << ",\"temperature_end_c\":" << timing.temperature_end_c
        << ",\"temperature_peak_c\":" << timing.temperature_peak_c
        << ",\"thermal_sample_count\":" << timing.thermal_sample_count
        << ",\"thermal_retry_count\":" << timing.thermal_retry_count
        << ",\"modeled_weight_stream_GBps\":" << weight_gbps
        << ",\"modeled_weight_stream_pct_of_512GBps\":" <<
            (weight_gbps / kV620PeakBandwidthGBps * 100.0)
        << ",\"gflops\":" << gflops
        << ",\"ns_per_logical_fma\":" << (timing.median_ms * 1.0e6 / element_ops)
        << "}";
    emit_json_line(out.str());
}

void benchmark_gemm(const Options& options, const DeviceMatrix& matrix, Format format,
                    std::size_t m, ThermalGuard& thermal, hipStream_t stream) {
    const std::string point_id = std::string("dequant_plus_gemm:") + matrix.shape.name + ":" +
        format_name(format) + ":m" + std::to_string(m) + ":" + variant_name(format);
    const Timing timing = run_timing_point_with_recovery(thermal, point_id, [&]() {
        return time_launch(options, thermal, point_id, stream, [&](int sequence) {
            const auto& replica = matrix.replicas[
                static_cast<std::size_t>(sequence) % matrix.replicas.size()];
            launch_gemm(format, matrix, replica, m, stream);
        });
    });
    emit_timing("dequant_plus_gemm", matrix, format, m, timing);
    // Complete the per-point cooldown even for the final point in a process.
    // Earlier points would otherwise cool only incidentally when the following
    // point begins, while the last one could return with residual heat.
    thermal.wait_for_cooldown(point_id + ":after_timing");
}

void benchmark_dequant(const Options& options, const DeviceMatrix& matrix, Format format,
                       bool raw, ThermalGuard& thermal, hipStream_t stream) {
    const char* kind = raw ? "raw_payload_stream_control" : "dequant_only";
    const std::string point_id = std::string(kind) + ":" + matrix.shape.name + ":" +
        format_name(format) + ":" + variant_name(format);
    const Timing timing = run_timing_point_with_recovery(thermal, point_id, [&]() {
        return time_launch(options, thermal, point_id, stream, [&](int sequence) {
            const auto& replica = matrix.replicas[
                static_cast<std::size_t>(sequence) % matrix.replicas.size()];
            if (raw) {
                launch_raw(format, matrix, replica, stream);
            } else {
                launch_dequant(format, matrix, replica, stream);
            }
        });
    });
    emit_timing(raw ? "raw_payload_stream_control" : "dequant_only", matrix, format, 1, timing);
    thermal.wait_for_cooldown(point_id + ":after_timing");
}

void emit_layout_record(const DeviceMatrix& matrix) {
    const std::size_t elements = matrix.shape.rows * matrix.shape.cols;
    const std::size_t sq8_weight = elements;
    const std::size_t scale_count = matrix.scale_rows * matrix.scale_cols;
    const std::size_t sq8_artifact_scale = scale_count * sizeof(std::uint16_t);
    const std::size_t sq8_runtime_scale = scale_count * sizeof(float);
    const std::size_t sq8_runtime_total = sq8_weight + sq8_runtime_scale;
    const std::size_t sq9 = matrix.shape.rows * matrix.stored_cols * 9u / 8u;
    const double ratio = static_cast<double>(sq9) / static_cast<double>(sq8_runtime_total);
    std::ostringstream out;
    out << std::setprecision(12)
        << "{\"record\":\"layout\",\"shape\":\"" << matrix.shape.name
        << "\",\"rows\":" << matrix.shape.rows
        << ",\"cols\":" << matrix.shape.cols
        << ",\"sq8_weight_dtype\":\"F8_E4M3\""
        << ",\"sq8_artifact_scale_dtype\":\"BF16\""
        << ",\"sq8_runtime_scale_dtype\":\"F32\""
        << ",\"sq8_scale_layout\":\"block_2d_row_major\""
        << ",\"sq8_block_shape\":[128,128]"
        << ",\"sq8_weight_bytes\":" << sq8_weight
        << ",\"sq8_scale_count\":" << scale_count
        << ",\"sq8_artifact_scale_bytes\":" << sq8_artifact_scale
        << ",\"sq8_artifact_total_bytes\":" << (sq8_weight + sq8_artifact_scale)
        << ",\"sq8_runtime_scale_bytes\":" << sq8_runtime_scale
        << ",\"sq8_total_resident_bytes\":" << sq8_runtime_total
        << ",\"sq8_artifact_bytes_per_full_128x128_block\":16386"
        << ",\"sq8_runtime_bytes_per_full_128x128_block\":16388"
        << ",\"sq9_layout\":\"lo8_then_hi1\""
        << ",\"sq9_stored_cols\":" << matrix.stored_cols
        << ",\"sq9_low_plane_bytes\":" << matrix.shape.rows * matrix.stored_cols
        << ",\"sq9_high_plane_bytes\":" << matrix.shape.rows * matrix.stored_cols / 8u
        << ",\"sq9_total_resident_bytes\":" << sq9
        << ",\"sq9_bytes_per_full_128_element_tile\":144"
        << ",\"sq9_padding_elements\":" <<
            (matrix.shape.rows * matrix.stored_cols - elements)
        << ",\"sq9_over_sq8_total_byte_ratio\":" << ratio
        << ",\"sq9_over_sq8_total_byte_increase_pct\":" << (ratio - 1.0) * 100.0
        << "}";
    emit_json_line(out.str());
}

void run_shape(const Options& options, const Shape& shape,
               const std::vector<std::size_t>& m_values, bool include_dequant,
               ThermalGuard& thermal, hipStream_t stream) {
    if (m_values.empty()) {
        throw std::runtime_error("run_shape requires at least one M value");
    }
    const std::size_t max_m = m_values.back();
    thermal.wait_for_cooldown(std::string("shape:") + shape.name + ":before_upload");
    DeviceMatrix matrix = upload_matrix(shape, max_m, thermal, stream);
    emit_layout_record(matrix);
    thermal.check_before_launch(std::string("sanity:") + shape.name + ":before");
    hipLaunchKernelGGL(sq9_v620_sanity_kernel, dim3(1u), dim3(256u), 0u, stream,
                       matrix.output.get());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    thermal.check_after_launch(std::string("sanity:") + shape.name + ":after");
    emit_json_line("{\"record\":\"sanity_kernel\",\"passed\":true}");
    verify_matrix(matrix, stream, thermal);
    for (const std::size_t m : m_values) {
        for (const Format format : {kSq8, kSq9Lane, kSq9Lds, kFp16}) {
            benchmark_gemm(options, matrix, format, m, thermal, stream);
        }
    }
    if (include_dequant) {
        for (const Format format : {kSq8, kSq9Lane, kSq9Lds, kFp16}) {
            benchmark_dequant(options, matrix, format, false, thermal, stream);
            benchmark_dequant(options, matrix, format, true, thermal, stream);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        if (!options.preflight_only && options.suite == "full" &&
            (!options.m_values_explicit || !options.shape_explicit)) {
            throw std::runtime_error(
                "full suite requires explicit --shape and --m-values so shapes or M stages cannot be queued accidentally");
        }
        configure_outputs(options);
        const DeviceIdentity selected = select_v620_by_arch_and_bdf(options.pci_bus_id);
        ThermalGuard thermal(selected, options);
        thermal.emit_sensor_mapping();
        try {
            thermal.wait_for_cooldown("startup");
            assert_selected_v620(selected, "before_measurement");
            if (options.preflight_only) {
                emit_json_line("{\"record\":\"preflight\",\"passed\":true}");
                return 0;
            }
            hipStream_t stream{};
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            try {
            std::ostringstream metadata;
            metadata << "{\"record\":\"benchmark_metadata\",\"suite\":\""
                     << options.suite << "\",\"shape_selector\":\"" << options.shape
                     << "\",\"warmups\":" << options.warmups
                     << ",\"trials\":" << options.trials
                     << ",\"launches_per_trial\":" << options.launches_per_trial
                     << ",\"replica_count\":" << kReplicaCount
                     << ",\"v620_peak_bandwidth_GBps\":" << kV620PeakBandwidthGBps
                     << ",\"junction_limit_c\":" << kJunctionLimitC
                     << ",\"cooldown_target_c\":" << options.cooldown_c
                     << ",\"cooldown_poll_ms\":" << options.cooldown_poll_ms
                     << ",\"cooldown_timeout_s\":" << options.cooldown_timeout_s
                     << ",\"thermal_max_retries\":" << options.thermal_max_retries
                     << ",\"HIP_VISIBLE_DEVICES\":\""
                     << json_escape(std::getenv("HIP_VISIBLE_DEVICES") != nullptr
                                        ? std::getenv("HIP_VISIBLE_DEVICES")
                                        : "")
                     << "\""
                     << ",\"sq8_canonical_artifact_layout\":\"F8_E4M3 + BF16 block_2d row_major 128x128\""
                     << ",\"sq8_v620_fallback_resident_layout\":\"F8_E4M3 + F32 block_2d row_major 128x128\""
                     << ",\"sq9_layout\":\"SQ9_0 lo8_then_hi1 128-element tile, q<<7\""
                     << ",\"raw_payload_stream_control_is_load_only\":false"
                     << "}";
            emit_json_line(metadata.str());
            if (options.suite == "smoke") {
                if (options.shape != "all" && options.shape != kShapes.front().name) {
                    throw std::runtime_error("smoke suite supports only qwen3_14b_q_proj or all");
                }
                const std::vector<std::size_t> m_values = options.m_values_explicit
                    ? options.m_values : std::vector<std::size_t>{1u};
                if (m_values.size() != 1u || m_values.front() != 1u) {
                    throw std::runtime_error("smoke suite supports only M=1");
                }
                run_shape(
                    options, kShapes.front(), m_values, options.dequant == "on", thermal, stream);
            } else {
                bool matched_shape = false;
                for (const Shape& shape : kShapes) {
                    if (options.shape != "all" && options.shape != shape.name) {
                        continue;
                    }
                    matched_shape = true;
                    run_shape(
                        options, shape, options.m_values, options.dequant == "on", thermal, stream);
                }
                if (!matched_shape) {
                    throw std::runtime_error("unknown shape selector");
                }
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            assert_selected_v620(selected, "after_measurement");
            HIP_CHECK(hipStreamDestroy(stream));
            } catch (...) {
                (void)hipStreamDestroy(stream);
                throw;
            }
        } catch (const ThermalLimitExceeded& error) {
            emit_measurement_event(
                "benchmark_thermal_abort", "benchmark", 0, error.temperature_c(), error.phase());
            try {
                thermal.wait_for_cooldown("benchmark:after_thermal_abort");
            } catch (const ThermalCooldownTimeout& cooldown_error) {
                emit_measurement_event("benchmark_cooldown_timeout", "benchmark", 0,
                                       cooldown_error.temperature_c(), cooldown_error.phase());
                std::cerr << "bench-sq9-v620-viability-hip: " << cooldown_error.what() << '\n';
                return 4;
            }
            std::cerr << "bench-sq9-v620-viability-hip: " << error.what() << '\n';
            return 3;
        } catch (const ThermalCooldownTimeout& error) {
            emit_measurement_event(
                "benchmark_cooldown_timeout", "benchmark", 0, error.temperature_c(), error.phase());
            std::cerr << "bench-sq9-v620-viability-hip: " << error.what() << '\n';
            return 4;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench-sq9-v620-viability-hip: " << error.what() << '\n';
        return 1;
    }
}
