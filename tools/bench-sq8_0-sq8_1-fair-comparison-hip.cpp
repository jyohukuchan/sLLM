// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Thermal-guarded, same-process SQ8_0/SQ8_1 comparison for V620/gfx1030.
// The modules are compiled from the exact in-tree HIPRTC strings.  This tool
// is deliberately isolated: it changes no runtime dispatch, artifact, service,
// candidate, campaign, authorization, or active-model state.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <algorithm>
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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// Host-only factories for the exact runtime HIPRTC strings.
#include "../runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc"
#include "../runtime/src/kernels/sq8_1/sq8_1_matvec_hiprtc.inc"

namespace {

#define HIP_CHECK(call) do { \
    const hipError_t status__ = (call); \
    if (status__ != hipSuccess) throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(status__)); \
} while (false)

#define HIPRTC_CHECK(call) do { \
    const hiprtcResult status__ = (call); \
    if (status__ != HIPRTC_SUCCESS) throw std::runtime_error(std::string(#call) + ": " + hiprtcGetErrorString(status__)); \
} while (false)

constexpr std::size_t kRows = 5'120u;
constexpr std::size_t kCols = 5'120u;
constexpr std::size_t kSq8ScaleBlock = 128u;
constexpr std::size_t kSq8ScaleRows = kRows / kSq8ScaleBlock;
constexpr std::size_t kSq8ScaleCols = kCols / kSq8ScaleBlock;
constexpr std::size_t kSq81Groups = kCols / 32u;
constexpr std::size_t kSq81Stride = kCols;
constexpr std::size_t kSq81RowsPerBlock = 8u;
constexpr unsigned int kThreads = 256u;
constexpr int kReplicaCount = 6;
constexpr double kJunctionLimitC = 85.0;
constexpr double kCooldownC = 42.0;
constexpr double kPeakBandwidthGBps = 512.0;

enum class Variant { Sq8_0, Sq8_1W8A16, Sq8_1W8A8 };

struct Options {
    std::string bdf;
    std::filesystem::path jsonl;
    std::filesystem::path thermal;
    int m1_runs = 3;
    int m1_warmups = 32;
    int m1_trials = 31;
    int sweep_runs = 3;
    int sweep_warmups = 5;
    int sweep_trials = 9;
    std::vector<std::size_t> m_values{1u, 8u, 32u, 128u};
};

struct Device {
    int ordinal = -1;
    std::string bdf;
    std::string arch;
    std::string name;
};

struct Sensor {
    std::string card;
    std::filesystem::path input;
};

struct Timing {
    std::vector<float> samples;
    double start_c = 0.0;
    double end_c = 0.0;
    double peak_c = 0.0;
};

struct HostWeights {
    std::vector<std::uint8_t> sq8_payload;
    std::vector<float> sq8_scales;
    std::vector<std::uint8_t> sq81_payload;
    std::vector<std::uint16_t> sq81_scales;
};

class DeviceAllocation {
public:
    DeviceAllocation() = default;
    explicit DeviceAllocation(std::size_t bytes) { HIP_CHECK(hipMalloc(&ptr_, bytes)); }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation(DeviceAllocation &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    ~DeviceAllocation() { reset(); }
    void *get() const { return ptr_; }

private:
    void reset() noexcept {
        if (ptr_ != nullptr) (void)hipFree(ptr_);
        ptr_ = nullptr;
    }
    void *ptr_ = nullptr;
};

struct Replica {
    DeviceAllocation sq8_payload;
    DeviceAllocation sq8_scales;
    DeviceAllocation sq81_payload;
    DeviceAllocation sq81_scales;
};

struct Matrices {
    std::vector<Replica> replicas;
    HostWeights reference;
    std::vector<float> host_inputs;
    DeviceAllocation input;
    DeviceAllocation sq8_output;
    DeviceAllocation sq81_output;
    DeviceAllocation sq81_activation_codes;
    DeviceAllocation sq81_activation_scales;
};

struct Modules {
    hipModule_t sq8_module{};
    hipFunction_t sq8{};
    hipFunction_t sq8_batch{};
    hipModule_t sq81_module{};
    hipFunction_t w8a16{};
    hipFunction_t w8a8{};
    hipModule_t sq81_batch_prototype_module{};
    hipFunction_t prequant{};
    hipFunction_t w8a16_batch{};
    hipFunction_t w8a8_prequant_batch{};
    Modules() = default;
    Modules(const Modules &) = delete;
    Modules &operator=(const Modules &) = delete;
    Modules(Modules &&other) noexcept
        : sq8_module(other.sq8_module), sq8(other.sq8), sq8_batch(other.sq8_batch), sq81_module(other.sq81_module),
          w8a16(other.w8a16), w8a8(other.w8a8), sq81_batch_prototype_module(other.sq81_batch_prototype_module),
          prequant(other.prequant), w8a16_batch(other.w8a16_batch), w8a8_prequant_batch(other.w8a8_prequant_batch) {
        other.sq8_module = nullptr;
        other.sq8 = nullptr;
        other.sq8_batch = nullptr;
        other.sq81_module = nullptr;
        other.w8a16 = nullptr;
        other.w8a8 = nullptr;
        other.sq81_batch_prototype_module = nullptr;
        other.prequant = nullptr;
        other.w8a16_batch = nullptr;
        other.w8a8_prequant_batch = nullptr;
    }
    Modules &operator=(Modules &&other) noexcept {
        if (this != &other) {
            if (sq8_module != nullptr) (void)hipModuleUnload(sq8_module);
            if (sq81_module != nullptr) (void)hipModuleUnload(sq81_module);
            if (sq81_batch_prototype_module != nullptr) (void)hipModuleUnload(sq81_batch_prototype_module);
            sq8_module = other.sq8_module;
            sq8 = other.sq8;
            sq8_batch = other.sq8_batch;
            sq81_module = other.sq81_module;
            w8a16 = other.w8a16;
            w8a8 = other.w8a8;
            sq81_batch_prototype_module = other.sq81_batch_prototype_module;
            prequant = other.prequant;
            w8a16_batch = other.w8a16_batch;
            w8a8_prequant_batch = other.w8a8_prequant_batch;
            other.sq8_module = nullptr;
            other.sq8 = nullptr;
            other.sq8_batch = nullptr;
            other.sq81_module = nullptr;
            other.w8a16 = nullptr;
            other.w8a8 = nullptr;
            other.sq81_batch_prototype_module = nullptr;
            other.prequant = nullptr;
            other.w8a16_batch = nullptr;
            other.w8a8_prequant_batch = nullptr;
        }
        return *this;
    }
    ~Modules() {
        if (sq8_module != nullptr) (void)hipModuleUnload(sq8_module);
        if (sq81_module != nullptr) (void)hipModuleUnload(sq81_module);
        if (sq81_batch_prototype_module != nullptr) (void)hipModuleUnload(sq81_batch_prototype_module);
    }
};

std::ofstream g_jsonl;
std::ofstream g_thermal;

[[noreturn]] void usage(const char *argv0) {
    std::cerr << "usage: " << argv0
              << " --pci-bus-id 0000:03:00.0 --jsonl-output /absolute/path"
              << " --thermal-output /absolute/path [--m1-runs N] [--m1-warmups N]"
              << " [--m1-trials N] [--sweep-runs N] [--sweep-warmups N] [--sweep-trials N]"
              << " [--m-values 1,8,32,128]\n";
    std::exit(2);
}

int positive_int(std::string_view text, const char *label) {
    char *end = nullptr;
    const long value = std::strtol(std::string(text).c_str(), &end, 10);
    if (text.empty() || end == nullptr || *end != '\0' || value <= 0 || value > 1000000) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<int>(value);
}

std::vector<std::size_t> parse_m_values(std::string_view text) {
    std::vector<std::size_t> values;
    std::size_t begin = 0u;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        const std::string token(text.substr(begin, end - begin));
        char *parsed_end = nullptr;
        const unsigned long long value = std::strtoull(token.c_str(), &parsed_end, 10);
        if (token.empty() || parsed_end == nullptr || *parsed_end != '\0' || value == 0u || value > 128u) {
            throw std::runtime_error("m-values must be positive values no greater than 128");
        }
        values.push_back(static_cast<std::size_t>(value));
        if (comma == std::string_view::npos) break;
        begin = comma + 1u;
    }
    if (values.empty() || !std::is_sorted(values.begin(), values.end()) ||
        std::adjacent_find(values.begin(), values.end()) != values.end()) {
        throw std::runtime_error("m-values must be strictly increasing");
    }
    return values;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        auto need = [&]() -> std::string {
            if (++index >= argc) usage(argv[0]);
            return argv[index];
        };
        if (arg == "--pci-bus-id") options.bdf = need();
        else if (arg == "--jsonl-output") options.jsonl = need();
        else if (arg == "--thermal-output") options.thermal = need();
        else if (arg == "--m1-runs") options.m1_runs = positive_int(need(), "m1-runs");
        else if (arg == "--m1-warmups") options.m1_warmups = positive_int(need(), "m1-warmups");
        else if (arg == "--m1-trials") options.m1_trials = positive_int(need(), "m1-trials");
        else if (arg == "--sweep-runs") options.sweep_runs = positive_int(need(), "sweep-runs");
        else if (arg == "--sweep-warmups") options.sweep_warmups = positive_int(need(), "sweep-warmups");
        else if (arg == "--sweep-trials") options.sweep_trials = positive_int(need(), "sweep-trials");
        else if (arg == "--m-values") options.m_values = parse_m_values(need());
        else usage(argv[0]);
    }
    if (options.bdf.empty() || !options.jsonl.is_absolute() || !options.thermal.is_absolute() ||
        options.jsonl == options.thermal) usage(argv[0]);
    return options;
}

void open_new(std::ofstream &output, const std::filesystem::path &path) {
    std::error_code error;
    if (std::filesystem::exists(path, error) || error) {
        throw std::runtime_error("refusing to overwrite output: " + path.string());
    }
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) throw std::runtime_error("failed to create output directory: " + error.message());
    output.open(path);
    if (!output) throw std::runtime_error("failed to open output: " + path.string());
}

void emit(const std::string &record, bool thermal = false) {
    std::cout << record << '\n';
    std::cout.flush();
    g_jsonl << record << '\n';
    g_jsonl.flush();
    if (thermal) {
        g_thermal << record << '\n';
        g_thermal.flush();
    }
}

std::string lower(std::string value) {
    for (char &ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

bool is_gfx1030(std::string_view arch) {
    return arch == "gfx1030" || (arch.size() > 8u && arch.substr(0u, 8u) == "gfx1030:");
}

Device select_v620(const std::string &requested_bdf) {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    std::vector<Device> devices;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t property{};
        char bdf[32]{};
        HIP_CHECK(hipGetDeviceProperties(&property, ordinal));
        HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), ordinal));
        devices.push_back(Device{ordinal, bdf, property.gcnArchName, property.name});
    }
    std::ostringstream inventory;
    inventory << "{\"record\":\"device_inventory\",\"devices\":[";
    for (std::size_t index = 0; index < devices.size(); ++index) {
        if (index) inventory << ',';
        inventory << "{\"hip_ordinal\":" << devices[index].ordinal
                  << ",\"name\":\"" << devices[index].name
                  << "\",\"gcn_arch_name\":\"" << devices[index].arch
                  << "\",\"pci_bus_id\":\"" << devices[index].bdf << "\"}";
    }
    inventory << "]}";
    emit(inventory.str());
    std::vector<Device> matched;
    for (const Device &device : devices) {
        if (is_gfx1030(device.arch) && lower(device.bdf) == lower(requested_bdf)) {
            matched.push_back(device);
        }
    }
    if (matched.size() != 1u) throw std::runtime_error("requested BDF did not select exactly one gfx1030");
    HIP_CHECK(hipSetDevice(matched.front().ordinal));
    return matched.front();
}

std::string read_line(const std::filesystem::path &path) {
    std::ifstream input(path);
    std::string value;
    if (!input || !std::getline(input, value)) throw std::runtime_error("failed to read " + path.string());
    return value;
}

Sensor sensor_for_bdf(const std::string &bdf) {
    std::vector<Sensor> matched;
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator("/sys/class/drm", error)) {
        const std::string card = entry.path().filename().string();
        if (card.rfind("card", 0u) != 0u || card.size() == 4u) continue;
        const auto device = entry.path() / "device";
        const auto canonical = std::filesystem::weakly_canonical(device, error);
        if (error || lower(canonical.filename().string()) != lower(bdf)) {
            error.clear();
            continue;
        }
        for (const auto &hwmon : std::filesystem::directory_iterator(device / "hwmon", error)) {
            if (error) break;
            const auto temp = hwmon.path() / "temp2_input";
            const auto label = hwmon.path() / "temp2_label";
            if (std::filesystem::is_regular_file(temp, error) && !error &&
                std::filesystem::is_regular_file(label, error) && !error &&
                lower(read_line(label)) == "junction") {
                matched.push_back(Sensor{card, temp});
            }
            error.clear();
        }
        error.clear();
    }
    if (matched.size() != 1u) throw std::runtime_error("failed to map HIP BDF to exactly one own junction sensor");
    return matched.front();
}

class ThermalGuard {
public:
    ThermalGuard(Device device, Sensor sensor) : device_(std::move(device)), sensor_(std::move(sensor)) {
        std::ostringstream record;
        record << "{\"record\":\"thermal_sensor\",\"hip_ordinal\":" << device_.ordinal
               << ",\"pci_bus_id\":\"" << device_.bdf << "\",\"drm_card\":\"" << sensor_.card
               << "\",\"temp2_input\":\"" << sensor_.input.string()
               << "\",\"junction_limit_c\":85.0,\"cooldown_target_c\":42.0}";
        emit(record.str(), true);
    }

    double check(std::string_view phase) {
        const std::string raw = read_line(sensor_.input);
        char *end = nullptr;
        const long milli = std::strtol(raw.c_str(), &end, 10);
        if (end == nullptr || *end != '\0' || milli <= 0 || milli > 200000) {
            throw std::runtime_error("invalid junction reading");
        }
        const double celsius = static_cast<double>(milli) / 1000.0;
        std::ostringstream record;
        record << std::setprecision(8) << "{\"record\":\"thermal\",\"phase\":\"" << phase
               << "\",\"pci_bus_id\":\"" << device_.bdf << "\",\"drm_card\":\"" << sensor_.card
               << "\",\"temperature_c\":" << celsius << ",\"junction_limit_c\":85.0}";
        emit(record.str(), true);
        if (celsius >= kJunctionLimitC) throw std::runtime_error("thermal guard reached 85 C");
        return celsius;
    }

    void cooldown(std::string_view phase) {
        for (int attempt = 0; attempt < 180; ++attempt) {
            if (check(std::string(phase) + ":poll") <= kCooldownC) return;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        throw std::runtime_error("cooldown timed out");
    }

private:
    Device device_;
    Sensor sensor_;
};

std::uint32_t next(std::uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

float e4m3fn_to_f32(std::uint8_t value) {
    const std::uint32_t raw = value;
    const std::uint32_t sign = raw >> 7u;
    const std::uint32_t exponent = (raw >> 3u) & 0x0fu;
    const std::uint32_t mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) return std::numeric_limits<float>::quiet_NaN();
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    const std::uint32_t bits = (sign << 31u) | ((exponent + 120u) << 23u) | (mantissa << 20u);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float f16_to_f32(std::uint16_t bits) {
    _Float16 value{};
    std::memcpy(&value, &bits, sizeof(bits));
    return static_cast<float>(value);
}

std::uint16_t f32_to_f16_bits(float value) {
    const _Float16 narrowed = static_cast<_Float16>(value);
    std::uint16_t bits = 0u;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    return bits;
}

float ceil_f16(float value) {
    std::uint16_t bits = f32_to_f16_bits(value);
    if (bits == 0u) return f16_to_f32(1u);
    float stored = f16_to_f32(bits);
    if (stored < value) stored = f16_to_f32(++bits);
    return stored;
}

HostWeights make_weights(int replica) {
    HostWeights result;
    result.sq8_payload.resize(kRows * kCols);
    result.sq8_scales.resize(kSq8ScaleRows * kSq8ScaleCols);
    result.sq81_payload.resize(kRows * kSq81Stride);
    result.sq81_scales.resize(kRows * kSq81Groups);
    std::uint32_t state = 0x9146a3d5u ^ static_cast<std::uint32_t>(replica) * 0x9e3779b9u;
    for (float &scale : result.sq8_scales) {
        scale = 0.0078125f * (1.0f + 0.125f * static_cast<float>(next(state) & 7u));
    }
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t col = 0; col < kCols; ++col) {
            result.sq8_payload[row * kCols + col] = static_cast<std::uint8_t>(
                ((next(state) & 1u) << 7u) | ((3u + (next(state) & 7u)) << 3u) | (next(state) & 7u));
        }
        for (std::size_t group = 0; group < kSq81Groups; ++group) {
            const std::size_t start = group * 32u;
            float maximum = 0.0f;
            for (std::size_t index = 0; index < 32u; ++index) {
                const std::size_t col = start + index;
                const float source = e4m3fn_to_f32(result.sq8_payload[row * kCols + col]) *
                    result.sq8_scales[(row / kSq8ScaleBlock) * kSq8ScaleCols + col / kSq8ScaleBlock];
                maximum = std::max(maximum, std::fabs(source));
            }
            const float scale = maximum == 0.0f ? 1.0f : ceil_f16(maximum / 127.0f);
            result.sq81_scales[row * kSq81Groups + group] = f32_to_f16_bits(scale);
            for (std::size_t index = 0; index < 32u; ++index) {
                const std::size_t col = start + index;
                const float source = e4m3fn_to_f32(result.sq8_payload[row * kCols + col]) *
                    result.sq8_scales[(row / kSq8ScaleBlock) * kSq8ScaleCols + col / kSq8ScaleBlock];
                const int rounded = static_cast<int>(std::rint(source / scale));
                const int clamped = std::max(-127, std::min(127, rounded));
                result.sq81_payload[row * kSq81Stride + col] = static_cast<std::uint8_t>(
                    static_cast<std::int8_t>(clamped));
            }
        }
    }
    return result;
}

std::vector<float> make_inputs(std::size_t count) {
    std::vector<float> result(count * kCols);
    std::uint32_t state = 0x41d0c8e3u;
    for (float &value : result) {
        value = static_cast<float>(static_cast<int>(next(state) % 255u) - 127) * (1.0f / 127.0f);
    }
    return result;
}

Matrices upload_matrices(std::size_t max_m, hipStream_t stream, ThermalGuard &thermal) {
    Matrices result;
    result.replicas.reserve(kReplicaCount);
    for (int replica = 0; replica < kReplicaCount; ++replica) {
        thermal.check("upload:replica:" + std::to_string(replica) + ":before");
        HostWeights host = make_weights(replica);
        Replica device{
            DeviceAllocation(host.sq8_payload.size()),
            DeviceAllocation(host.sq8_scales.size() * sizeof(float)),
            DeviceAllocation(host.sq81_payload.size()),
            DeviceAllocation(host.sq81_scales.size() * sizeof(std::uint16_t)),
        };
        HIP_CHECK(hipMemcpyAsync(device.sq8_payload.get(), host.sq8_payload.data(), host.sq8_payload.size(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(device.sq8_scales.get(), host.sq8_scales.data(), host.sq8_scales.size() * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(device.sq81_payload.get(), host.sq81_payload.data(), host.sq81_payload.size(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(device.sq81_scales.get(), host.sq81_scales.data(), host.sq81_scales.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        if (replica == 0) result.reference = std::move(host);
        result.replicas.push_back(std::move(device));
        thermal.check("upload:replica:" + std::to_string(replica) + ":after");
    }
    result.host_inputs = make_inputs(max_m);
    result.input = DeviceAllocation(result.host_inputs.size() * sizeof(float));
    result.sq8_output = DeviceAllocation(max_m * kRows * sizeof(float));
    result.sq81_output = DeviceAllocation(max_m * kRows * sizeof(float));
    const std::size_t activation_stride = ((kCols + 15u) / 16u) * 16u;
    result.sq81_activation_codes = DeviceAllocation(max_m * activation_stride);
    result.sq81_activation_scales = DeviceAllocation(max_m * kSq81Groups * sizeof(float));
    thermal.check("upload:input:before");
    HIP_CHECK(hipMemcpyAsync(result.input.get(), result.host_inputs.data(), result.host_inputs.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    thermal.check("upload:input:after");
    return result;
}

void compile_source(const char *source, const char *name, hipModule_t *module) {
    hiprtcProgram program{};
    HIPRTC_CHECK(hiprtcCreateProgram(&program, source, name, 0, nullptr, nullptr));
    const char *options[] = {"--offload-arch=gfx1030", "--std=c++17", "-O3"};
    const hiprtcResult result = hiprtcCompileProgram(program, 3, options);
    if (result != HIPRTC_SUCCESS) {
        std::size_t size = 0u;
        (void)hiprtcGetProgramLogSize(program, &size);
        std::string log(size, '\0');
        if (size) (void)hiprtcGetProgramLog(program, log.data());
        (void)hiprtcDestroyProgram(&program);
        throw std::runtime_error(std::string("HIPRTC compile failed for ") + name + ": " + log);
    }
    std::size_t size = 0u;
    HIPRTC_CHECK(hiprtcGetCodeSize(program, &size));
    std::vector<char> code(size);
    HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    HIP_CHECK(hipModuleLoadData(module, code.data()));
}

// This source is deliberately benchmark-only.  It models a two-stage prefill
// implementation (one exact activation quantization per M row, followed by a
// 2-D output grid), but does not alter the runtime's C ABI or dispatch.
std::string sq8_1_batch_prototype_source() {
    return std::string(sq8_1_matvec_kernel_source()) + R"SQ81_BATCH(

extern "C" __global__ __launch_bounds__(256) void ullm_sq8_1_prequant_activation_batch_f32_kernel(
    const float *input,
    unsigned long long batch_count,
    unsigned long long cols,
    unsigned char *activation_codes,
    float *activation_scales) {
    const unsigned long long batch = static_cast<unsigned long long>(blockIdx.x);
    if (batch >= batch_count) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kUllmSq8_1WaveSize - 1u);
    const unsigned int logical_wave = tid >> 5u;
    const unsigned long long groups = (cols + 31ull) / 32ull;
    const unsigned long long activation_code_bytes = ((cols + 15ull) / 16ull) * 16ull;
    const float *batch_input = input + batch * cols;
    unsigned char *batch_codes = activation_codes + batch * activation_code_bytes;
    float *batch_scales = activation_scales + batch * groups;

    // This is the exact group-local quantization sequence used by the tiled
    // runtime W8A8 kernel, emitted once per input row instead of once per
    // eight-output-row CTA.
    for (unsigned long long group = logical_wave; group < groups;
         group += kUllmSq8_1RowsPerBlock) {
        const unsigned long long start = group * 32ull;
        const unsigned int count = static_cast<unsigned int>(
            (cols - start) < 32ull ? (cols - start) : 32ull);
        const float value = lane < count ? batch_input[start + lane] : 0.0f;
        float maximum = fabsf(value);
#pragma unroll
        for (unsigned int offset = kUllmSq8_1WaveSize >> 1u; offset > 0u; offset >>= 1u) {
            maximum = fmaxf(maximum, __shfl_down(maximum, offset, kUllmSq8_1WaveSize));
        }
        float activation_scale = lane == 0u
            ? (maximum == 0.0f ? 1.0f : ullm_sq8_1_ceil_f16(maximum / 127.0f))
            : 0.0f;
        activation_scale = ullm_sq8_1_wave32_broadcast_lane0(activation_scale);
        if (lane == 0u) {
            batch_scales[group] = activation_scale;
        }
        if (lane < count) {
            const int rounded = static_cast<int>(rintf(value / activation_scale));
            const int clamped = rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
            batch_codes[start + lane] = static_cast<unsigned char>(static_cast<signed char>(clamped));
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void ullm_sq8_1_matvec_w8a16_batch_prototype_f32_kernel(
    const unsigned char *payload,
    const unsigned short *weight_scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long payload_row_stride,
    unsigned long long batch_count,
    float *output) {
    const unsigned long long batch = static_cast<unsigned long long>(blockIdx.y);
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kUllmSq8_1WaveSize - 1u);
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) *
        static_cast<unsigned long long>(kUllmSq8_1RowsPerBlock) + (tid >> 5u);
    if (row >= rows || batch >= batch_count) {
        return;
    }
    float sum = 0.0f;
    const unsigned long long groups = (cols + 31ull) / 32ull;
    const unsigned char *row_payload = payload + row * payload_row_stride;
    const float *batch_input = input + batch * cols;
    for (unsigned long long group = lane; group < groups; group += kUllmSq8_1WaveSize) {
        const unsigned long long start = group * 32ull;
        const unsigned int count = static_cast<unsigned int>(
            (cols - start) < 32ull ? (cols - start) : 32ull);
        float dot = 0.0f;
        if (count == 32u) {
            const ullm_sq8_1_uint4 first =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start);
            const ullm_sq8_1_uint4 second =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start + 16ull);
            dot = ullm_sq8_1_dot32_w8a16(
                first, second, reinterpret_cast<const float4 *>(batch_input + start));
        } else {
            for (unsigned int index = 0u; index < count; ++index) {
                dot = fmaf(
                    static_cast<float>(ullm_sq8_1_signed_byte(row_payload[start + index])),
                    batch_input[start + index],
                    dot);
            }
        }
        sum += dot * ullm_sq8_1_f16_to_f32(weight_scales[row * groups + group]);
    }
    const float reduced = ullm_sq8_1_wave32_sum(sum);
    if (lane == 0u) {
        output[batch * rows + row] = reduced;
    }
}

extern "C" __global__ __launch_bounds__(256) void ullm_sq8_1_matvec_w8a8_prequant_batch_prototype_f32_kernel(
    const unsigned char *payload,
    const unsigned short *weight_scales,
    const unsigned char *activation_codes,
    const float *activation_scales,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long payload_row_stride,
    unsigned long long batch_count,
    float *output) {
    const unsigned long long batch = static_cast<unsigned long long>(blockIdx.y);
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kUllmSq8_1WaveSize - 1u);
    const unsigned int logical_wave = tid >> 5u;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) *
        static_cast<unsigned long long>(kUllmSq8_1RowsPerBlock) + logical_wave;
    if (row >= rows || batch >= batch_count) {
        return;
    }
    const unsigned long long groups = (cols + 31ull) / 32ull;
    const unsigned long long activation_code_bytes = ((cols + 15ull) / 16ull) * 16ull;
    const unsigned char *batch_codes = activation_codes + batch * activation_code_bytes;
    const float *batch_scales = activation_scales + batch * groups;
    const unsigned char *row_payload = payload + row * payload_row_stride;
    float sum = 0.0f;
    for (unsigned long long group = lane; group < groups; group += kUllmSq8_1WaveSize) {
        const unsigned long long start = group * 32ull;
        const unsigned int count = static_cast<unsigned int>(
            (cols - start) < 32ull ? (cols - start) : 32ull);
        int dot = 0;
        if (count == 32u) {
            const ullm_sq8_1_uint4 weights_first =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start);
            const ullm_sq8_1_uint4 weights_second =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(row_payload + start + 16ull);
            const ullm_sq8_1_uint4 activation_first =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(batch_codes + start);
            const ullm_sq8_1_uint4 activation_second =
                *reinterpret_cast<const ullm_sq8_1_uint4 *>(batch_codes + start + 16ull);
            dot = ullm_sq8_1_dot32_w8a8(
                weights_first, weights_second, activation_first, activation_second);
        } else {
            for (unsigned int index = 0u; index < count; ++index) {
                dot += ullm_sq8_1_signed_byte(row_payload[start + index]) *
                    static_cast<int>(static_cast<signed char>(batch_codes[start + index]));
            }
        }
        const float weight_scale = ullm_sq8_1_f16_to_f32(weight_scales[row * groups + group]);
        sum += static_cast<float>(dot) * weight_scale * batch_scales[group];
    }
    const float reduced = ullm_sq8_1_wave32_sum(sum);
    if (lane == 0u) {
        output[batch * rows + row] = reduced;
    }
}
)SQ81_BATCH";
}

Modules compile_modules() {
    Modules result;
    compile_source(sq_fp8_matvec_kernel_source(), "sq8_0_fair_runtime.hip", &result.sq8_module);
    try {
        HIP_CHECK(hipModuleGetFunction(&result.sq8, result.sq8_module, "ullm_sq_fp8_matvec_f32_kernel"));
        HIP_CHECK(hipModuleGetFunction(
            &result.sq8_batch, result.sq8_module, "ullm_sq_fp8_matvec_batch_f32_kernel"));
        compile_source(sq8_1_matvec_kernel_source(), "sq8_1_fair_runtime.hip", &result.sq81_module);
        HIP_CHECK(hipModuleGetFunction(&result.w8a16, result.sq81_module, "ullm_sq8_1_matvec_w8a16_f32_kernel"));
        HIP_CHECK(hipModuleGetFunction(&result.w8a8, result.sq81_module, "ullm_sq8_1_matvec_w8a8_explicit_f32_kernel"));
        const std::string batch_prototype = sq8_1_batch_prototype_source();
        compile_source(batch_prototype.c_str(), "sq8_1_batch_prototype.hip", &result.sq81_batch_prototype_module);
        HIP_CHECK(hipModuleGetFunction(
            &result.prequant,
            result.sq81_batch_prototype_module,
            "ullm_sq8_1_prequant_activation_batch_f32_kernel"));
        HIP_CHECK(hipModuleGetFunction(
            &result.w8a16_batch,
            result.sq81_batch_prototype_module,
            "ullm_sq8_1_matvec_w8a16_batch_prototype_f32_kernel"));
        HIP_CHECK(hipModuleGetFunction(
            &result.w8a8_prequant_batch,
            result.sq81_batch_prototype_module,
            "ullm_sq8_1_matvec_w8a8_prequant_batch_prototype_f32_kernel"));
    } catch (...) {
        if (result.sq81_batch_prototype_module != nullptr) (void)hipModuleUnload(result.sq81_batch_prototype_module);
        result.sq81_batch_prototype_module = nullptr;
        if (result.sq81_module != nullptr) (void)hipModuleUnload(result.sq81_module);
        result.sq81_module = nullptr;
        if (result.sq8_module != nullptr) (void)hipModuleUnload(result.sq8_module);
        result.sq8_module = nullptr;
        throw;
    }
    return result;
}

void launch(
    const Modules &modules,
    Variant variant,
    const Replica &replica,
    const Matrices &matrices,
    std::size_t input_index,
    hipStream_t stream) {
    void *input = static_cast<char *>(matrices.input.get()) + input_index * kCols * sizeof(float);
    if (variant == Variant::Sq8_0) {
        void *payload = replica.sq8_payload.get();
        void *scales = replica.sq8_scales.get();
        void *output = static_cast<char *>(matrices.sq8_output.get()) + input_index * kRows * sizeof(float);
        unsigned long long rows = kRows, cols = kCols, block_rows = kSq8ScaleBlock, block_cols = kSq8ScaleBlock;
        unsigned int scale_kind = 2u;
        void *params[] = {&payload, &scales, &input, &rows, &cols, &scale_kind, &block_rows, &block_cols, &output};
        HIP_CHECK(hipModuleLaunchKernel(
            modules.sq8,
            static_cast<unsigned int>(kRows),
            1u,
            1u,
            kThreads,
            1u,
            1u,
            0u,
            stream,
            params,
            nullptr));
        return;
    }
    void *payload = replica.sq81_payload.get();
    void *scales = replica.sq81_scales.get();
    void *output = static_cast<char *>(matrices.sq81_output.get()) + input_index * kRows * sizeof(float);
    unsigned long long rows = kRows, cols = kCols, stride = kSq81Stride;
    void *params[] = {&payload, &scales, &input, &rows, &cols, &stride, &output};
    const unsigned int dynamic_shared = variant == Variant::Sq8_1W8A8
        ? static_cast<unsigned int>(kSq81Stride + kSq81Groups * sizeof(float))
        : 0u;
    HIP_CHECK(hipModuleLaunchKernel(
        variant == Variant::Sq8_1W8A8 ? modules.w8a8 : modules.w8a16,
        static_cast<unsigned int>((kRows + kSq81RowsPerBlock - 1u) / kSq81RowsPerBlock), 1u, 1u,
        kThreads, 1u, 1u, dynamic_shared, stream, params, nullptr));
}

void launch_sq8_batch_for_numerical_gate(
    const Modules &modules,
    const Replica &replica,
    const Matrices &matrices,
    std::size_t batch_count,
    hipStream_t stream) {
    void *payload = replica.sq8_payload.get();
    void *scales = replica.sq8_scales.get();
    void *input = matrices.input.get();
    void *output = matrices.sq8_output.get();
    unsigned long long rows = kRows, cols = kCols, block_rows = kSq8ScaleBlock, block_cols = kSq8ScaleBlock;
    unsigned long long kernel_batch_count = batch_count;
    unsigned int scale_kind = 2u;
    void *params[] = {
        &payload,
        &scales,
        &input,
        &rows,
        &cols,
        &scale_kind,
        &block_rows,
        &block_cols,
        &kernel_batch_count,
        &output,
    };
    HIP_CHECK(hipModuleLaunchKernel(
        modules.sq8_batch,
        static_cast<unsigned int>(kRows),
        static_cast<unsigned int>(batch_count),
        1u,
        kThreads,
        1u,
        1u,
        0u,
        stream,
        params,
        nullptr));
}

// Benchmark-only two-stage M-batch path.  This is intentionally separate
// from `launch`: no runtime symbol, ABI, launcher, or dispatch rule uses it.
void launch_sq81_batch_prototype(
    const Modules &modules,
    Variant variant,
    const Replica &replica,
    const Matrices &matrices,
    std::size_t batch_count,
    hipStream_t stream) {
    if (variant == Variant::Sq8_0) {
        throw std::runtime_error("SQ8_0 is not a SQ8_1 batch prototype variant");
    }
    void *payload = replica.sq81_payload.get();
    void *weight_scales = replica.sq81_scales.get();
    void *input = matrices.input.get();
    void *output = matrices.sq81_output.get();
    unsigned long long rows = kRows, cols = kCols, stride = kSq81Stride;
    unsigned long long kernel_batch_count = batch_count;
    const unsigned int output_grid = static_cast<unsigned int>(
        (kRows + kSq81RowsPerBlock - 1u) / kSq81RowsPerBlock);
    if (variant == Variant::Sq8_1W8A16) {
        void *params[] = {
            &payload, &weight_scales, &input, &rows, &cols, &stride, &kernel_batch_count, &output};
        HIP_CHECK(hipModuleLaunchKernel(
            modules.w8a16_batch,
            output_grid,
            static_cast<unsigned int>(batch_count),
            1u,
            kThreads,
            1u,
            1u,
            0u,
            stream,
            params,
            nullptr));
        return;
    }

    void *activation_codes = matrices.sq81_activation_codes.get();
    void *activation_scales = matrices.sq81_activation_scales.get();
    void *prequant_params[] = {&input, &kernel_batch_count, &cols, &activation_codes, &activation_scales};
    HIP_CHECK(hipModuleLaunchKernel(
        modules.prequant,
        static_cast<unsigned int>(batch_count),
        1u,
        1u,
        kThreads,
        1u,
        1u,
        0u,
        stream,
        prequant_params,
        nullptr));
    void *params[] = {
        &payload,
        &weight_scales,
        &activation_codes,
        &activation_scales,
        &rows,
        &cols,
        &stride,
        &kernel_batch_count,
        &output,
    };
    HIP_CHECK(hipModuleLaunchKernel(
        modules.w8a8_prequant_batch,
        output_grid,
        static_cast<unsigned int>(batch_count),
        1u,
        kThreads,
        1u,
        1u,
        0u,
        stream,
        params,
        nullptr));
}

std::vector<float> cpu_sq8_reference(
    const Matrices &matrices,
    std::size_t rows,
    std::size_t input_index = 0u) {
    std::vector<float> result(rows, 0.0f);
    const HostWeights &weights = matrices.reference;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < kCols; ++col) {
            const float scale = weights.sq8_scales[(row / kSq8ScaleBlock) * kSq8ScaleCols + col / kSq8ScaleBlock];
            result[row] += e4m3fn_to_f32(weights.sq8_payload[row * kCols + col]) * scale *
                matrices.host_inputs[input_index * kCols + col];
        }
    }
    return result;
}

std::vector<float> cpu_sq81_reference(
    const Matrices &matrices,
    bool w8a8,
    std::size_t rows,
    std::size_t input_index = 0u) {
    std::vector<float> result(rows, 0.0f);
    const HostWeights &weights = matrices.reference;
    const float *input = matrices.host_inputs.data() + input_index * kCols;
    std::vector<std::int8_t> activation(kCols);
    std::vector<float> activation_scales(kSq81Groups, 1.0f);
    if (w8a8) {
        for (std::size_t group = 0; group < kSq81Groups; ++group) {
            const std::size_t start = group * 32u;
            float maximum = 0.0f;
            for (std::size_t index = 0; index < 32u; ++index) maximum = std::max(maximum, std::fabs(input[start + index]));
            const float scale = maximum == 0.0f ? 1.0f : ceil_f16(maximum / 127.0f);
            activation_scales[group] = scale;
            for (std::size_t index = 0; index < 32u; ++index) {
                const int rounded = static_cast<int>(std::rint(input[start + index] / scale));
                activation[start + index] = static_cast<std::int8_t>(std::max(-127, std::min(127, rounded)));
            }
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t group = 0; group < kSq81Groups; ++group) {
            const std::size_t start = group * 32u;
            const float weight_scale = f16_to_f32(weights.sq81_scales[row * kSq81Groups + group]);
            if (w8a8) {
                std::int32_t dot = 0;
                for (std::size_t index = 0; index < 32u; ++index) {
                    dot += static_cast<std::int32_t>(static_cast<std::int8_t>(weights.sq81_payload[row * kSq81Stride + start + index])) *
                        static_cast<std::int32_t>(activation[start + index]);
                }
                result[row] += static_cast<float>(dot) * weight_scale * activation_scales[group];
            } else {
                for (std::size_t index = 0; index < 32u; ++index) {
                    result[row] += static_cast<float>(static_cast<std::int8_t>(weights.sq81_payload[row * kSq81Stride + start + index])) *
                        weight_scale * input[start + index];
                }
            }
        }
    }
    return result;
}

struct Error { float max_abs = 0.0f; double relative_l2 = 0.0; };

Error error_against(const std::vector<float> &actual, const std::vector<float> &expected) {
    Error result;
    double error = 0.0, reference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float delta = actual[index] - expected[index];
        result.max_abs = std::max(result.max_abs, std::fabs(delta));
        error += static_cast<double>(delta) * delta;
        reference += static_cast<double>(expected[index]) * expected[index];
    }
    result.relative_l2 = std::sqrt(error / std::max(reference, 1.0e-30));
    return result;
}

void numerical_gate(const Modules &modules, const Matrices &matrices, hipStream_t stream, ThermalGuard &thermal) {
    constexpr std::size_t rows_to_check = 8u;
    const std::vector<std::pair<Variant, std::vector<float>>> references = {
        {Variant::Sq8_0, cpu_sq8_reference(matrices, rows_to_check)},
        {Variant::Sq8_1W8A16, cpu_sq81_reference(matrices, false, rows_to_check)},
        {Variant::Sq8_1W8A8, cpu_sq81_reference(matrices, true, rows_to_check)},
    };
    for (const auto &entry : references) {
        const char *name = entry.first == Variant::Sq8_0 ? "SQ8_0" :
            (entry.first == Variant::Sq8_1W8A16 ? "SQ8_1_W8A16" : "SQ8_1_W8A8");
        thermal.check(std::string("numerical:") + name + ":before");
        launch(modules, entry.first, matrices.replicas.front(), matrices, 0u, stream);
        const DeviceAllocation &device_output = entry.first == Variant::Sq8_0
            ? matrices.sq8_output
            : matrices.sq81_output;
        std::vector<float> actual(rows_to_check);
        HIP_CHECK(hipMemcpyAsync(actual.data(), device_output.get(), actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        thermal.check(std::string("numerical:") + name + ":after");
        const Error error = error_against(actual, entry.second);
        const bool passed = error.max_abs <= 0.05f && error.relative_l2 <= 1.0e-5;
        std::ostringstream record;
        record << std::setprecision(10) << "{\"record\":\"numerical_gate\",\"path\":\"" << name
               << "\",\"rows_checked\":8,\"cols\":5120,\"max_abs\":" << error.max_abs
               << ",\"relative_l2\":" << error.relative_l2 << ",\"passed\":"
               << (passed ? "true" : "false") << "}";
        emit(record.str());
        if (!passed) throw std::runtime_error(std::string("numerical gate failed for ") + name);
    }
    thermal.check("numerical:SQ8_0_batch:before");
    launch_sq8_batch_for_numerical_gate(modules, matrices.replicas.front(), matrices, 2u, stream);
    std::vector<float> batch_actual(2u * rows_to_check);
    for (std::size_t batch = 0u; batch < 2u; ++batch) {
        HIP_CHECK(hipMemcpyAsync(
            batch_actual.data() + batch * rows_to_check,
            static_cast<const char *>(matrices.sq8_output.get()) + batch * kRows * sizeof(float),
            rows_to_check * sizeof(float),
            hipMemcpyDeviceToHost,
            stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    thermal.check("numerical:SQ8_0_batch:after");
    std::vector<float> batch_expected = cpu_sq8_reference(matrices, rows_to_check, 0u);
    const std::vector<float> second_expected = cpu_sq8_reference(matrices, rows_to_check, 1u);
    batch_expected.insert(batch_expected.end(), second_expected.begin(), second_expected.end());
    const Error batch_error = error_against(batch_actual, batch_expected);
    const bool batch_passed = batch_error.max_abs <= 0.05f && batch_error.relative_l2 <= 1.0e-5;
    std::ostringstream batch_record;
    batch_record << std::setprecision(10)
                 << "{\"record\":\"numerical_gate\",\"path\":\"SQ8_0_batch\""
                 << ",\"batch_count\":2,\"rows_checked_per_batch\":8,\"cols\":5120,\"max_abs\":"
                 << batch_error.max_abs << ",\"relative_l2\":" << batch_error.relative_l2
                 << ",\"passed\":" << (batch_passed ? "true" : "false") << "}";
    emit(batch_record.str());
    if (!batch_passed) throw std::runtime_error("numerical gate failed for SQ8_0_batch");

    // The isolated prequant + 2-D batch prototype must agree with the same
    // CPU references before it is allowed to contribute any timing evidence.
    constexpr std::size_t prototype_batch_count = 2u;
    constexpr std::size_t prototype_rows_to_check = kRows;
    for (const Variant variant : {Variant::Sq8_1W8A16, Variant::Sq8_1W8A8}) {
        const char *name = variant == Variant::Sq8_1W8A16
            ? "SQ8_1_W8A16_batch_prototype"
            : "SQ8_1_W8A8_prequant_batch_prototype";
        thermal.check(std::string("numerical:") + name + ":before");
        launch_sq81_batch_prototype(
            modules, variant, matrices.replicas.front(), matrices, prototype_batch_count, stream);
        std::vector<float> prototype_actual(prototype_batch_count * prototype_rows_to_check);
        for (std::size_t batch = 0u; batch < prototype_batch_count; ++batch) {
            HIP_CHECK(hipMemcpyAsync(
                prototype_actual.data() + batch * prototype_rows_to_check,
                static_cast<const char *>(matrices.sq81_output.get()) + batch * kRows * sizeof(float),
                prototype_rows_to_check * sizeof(float),
                hipMemcpyDeviceToHost,
                stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        thermal.check(std::string("numerical:") + name + ":after");
        std::vector<float> prototype_expected = cpu_sq81_reference(
            matrices, variant == Variant::Sq8_1W8A8, prototype_rows_to_check, 0u);
        const std::vector<float> second_prototype_expected = cpu_sq81_reference(
            matrices, variant == Variant::Sq8_1W8A8, prototype_rows_to_check, 1u);
        prototype_expected.insert(
            prototype_expected.end(), second_prototype_expected.begin(), second_prototype_expected.end());
        const Error prototype_error = error_against(prototype_actual, prototype_expected);
        const bool prototype_passed =
            prototype_error.max_abs <= 0.05f && prototype_error.relative_l2 <= 1.0e-5;
        std::ostringstream prototype_record;
        prototype_record << std::setprecision(10)
                         << "{\"record\":\"numerical_gate\",\"path\":\"" << name
                         << "\",\"implementation_scope\":\"benchmark_only_prequant_2d_batch_prototype\""
                         << ",\"batch_count\":2,\"rows_checked_per_batch\":5120,\"cols\":5120,\"max_abs\":"
                         << prototype_error.max_abs << ",\"relative_l2\":" << prototype_error.relative_l2
                         << ",\"passed\":" << (prototype_passed ? "true" : "false") << "}";
        emit(prototype_record.str());
        if (!prototype_passed) throw std::runtime_error(std::string("numerical gate failed for ") + name);
    }
}

const char *variant_name(Variant variant) {
    switch (variant) {
        case Variant::Sq8_0: return "SQ8_0_E4M3_F32_block128_wave32";
        case Variant::Sq8_1W8A16: return "SQ8_1_W8A16_wave32_rows8";
        case Variant::Sq8_1W8A8: return "SQ8_1_W8A8_tiled_wave32_rows8";
    }
    return "unknown";
}

std::size_t weight_bytes(Variant variant) {
    if (variant == Variant::Sq8_0) return kRows * kCols + kSq8ScaleRows * kSq8ScaleCols * sizeof(float);
    return kRows * kSq81Stride + kRows * kSq81Groups * sizeof(std::uint16_t);
}

void observe(Timing *timing, double value) {
    timing->peak_c = std::max(timing->peak_c, value);
}

void emit_timing(
    const char *suite,
    int run,
    Variant variant,
    std::size_t m,
    const Timing &timing,
    const char *protocol,
    bool model_weight_per_m = true) {
    std::vector<float> sorted = timing.samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2u];
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double variance = 0.0;
    for (float value : sorted) variance += (value - mean) * (value - mean);
    const double seconds = median / 1000.0;
    const std::size_t bytes = weight_bytes(variant) * (model_weight_per_m ? m : 1u);
    std::ostringstream record;
    record << std::setprecision(10) << "{\"record\":\"timing\",\"suite\":\"" << suite
           << "\",\"run\":" << run << ",\"shape\":\"qwen3_14b_q_proj\""
           << ",\"rows\":5120,\"cols\":5120,\"m\":" << m
           << ",\"format_path\":\"" << variant_name(variant) << "\""
           << ",\"protocol\":\"" << protocol << "\""
           << ",\"median_ms\":" << median << ",\"mean_ms\":" << mean
           << ",\"stddev_ms\":" << std::sqrt(variance / sorted.size())
           << ",\"min_ms\":" << sorted.front() << ",\"max_ms\":" << sorted.back()
           << ",\"samples_ms\":[";
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        if (index) record << ',';
        record << sorted[index];
    }
    record << "]"
           << ",\"modeled_weight_bytes_per_measurement\":" << bytes
           << ",\"weight_byte_model\":\""
           << (model_weight_per_m
                ? "one_full_weight_stream_per_independent_M"
                : "one_resident_weight_copy_per_2d_batch_prototype")
           << "\""
           << ",\"modeled_weight_stream_GBps\":" << static_cast<double>(bytes) / seconds / 1.0e9
           << ",\"modeled_weight_stream_pct_of_512GBps\":" << static_cast<double>(bytes) / seconds / 1.0e9 / kPeakBandwidthGBps * 100.0
           << ",\"temperature_start_c\":" << timing.start_c
           << ",\"temperature_end_c\":" << timing.end_c
           << ",\"temperature_peak_c\":" << timing.peak_c << "}";
    emit(record.str());
}

void run_m1_co_dispatch(
    const Modules &modules,
    const Matrices &matrices,
    const Options &options,
    int run,
    hipStream_t stream,
    ThermalGuard &thermal) {
    const Variant variants[] = {Variant::Sq8_0, Variant::Sq8_1W8A16, Variant::Sq8_1W8A8};
    Timing timings[3];
    const std::string prefix = "m1_co_dispatch:r" + std::to_string(run);
    thermal.cooldown(prefix + ":before_warmup");
    for (Timing &timing : timings) {
        timing.start_c = thermal.check(prefix + ":warmup_start");
        observe(&timing, timing.start_c);
    }
    for (int warmup = 0; warmup < options.m1_warmups; ++warmup) {
        for (int order = 0; order < 3; ++order) {
            const int index = (warmup + order) % 3;
            const double before = thermal.check(prefix + ":warmup:" + std::to_string(warmup) + ":before:" + std::to_string(index));
            observe(&timings[index], before);
            launch(modules, variants[index], matrices.replicas[static_cast<std::size_t>(warmup) % matrices.replicas.size()], matrices, 0u, stream);
            HIP_CHECK(hipStreamSynchronize(stream));
            observe(&timings[index], thermal.check(prefix + ":warmup:" + std::to_string(warmup) + ":after:" + std::to_string(index)));
        }
    }
    thermal.cooldown(prefix + ":before_timing");
    for (Timing &timing : timings) {
        timing.start_c = thermal.check(prefix + ":timing_start");
        observe(&timing, timing.start_c);
    }
    hipEvent_t start{}, stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    try {
        for (int trial = 0; trial < options.m1_trials; ++trial) {
            for (int order = 0; order < 3; ++order) {
                const int index = (trial + order) % 3;
                observe(&timings[index], thermal.check(prefix + ":trial:" + std::to_string(trial) + ":before:" + std::to_string(index)));
                HIP_CHECK(hipEventRecord(start, stream));
                launch(modules, variants[index], matrices.replicas[static_cast<std::size_t>(trial) % matrices.replicas.size()], matrices, 0u, stream);
                HIP_CHECK(hipEventRecord(stop, stream));
                HIP_CHECK(hipEventSynchronize(stop));
                float elapsed = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
                timings[index].samples.push_back(elapsed);
                observe(&timings[index], thermal.check(prefix + ":trial:" + std::to_string(trial) + ":after:" + std::to_string(index)));
            }
        }
    } catch (...) {
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        throw;
    }
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    for (int index = 0; index < 3; ++index) {
        timings[index].end_c = thermal.check(prefix + ":timing_end:" + std::to_string(index));
        observe(&timings[index], timings[index].end_c);
        emit_timing("m1_fair_co_dispatch", run, variants[index], 1u, timings[index],
                    "same_process_rotating_order_exact_runtime_hiprtc");
    }
    thermal.cooldown(prefix + ":complete");
}

void launch_bundle(
    const Modules &modules,
    Variant variant,
    const Replica &replica,
    const Matrices &matrices,
    std::size_t m,
    hipStream_t stream) {
    for (std::size_t index = 0; index < m; ++index) launch(modules, variant, replica, matrices, index, stream);
}

void run_m_sweep(
    const Modules &modules,
    const Matrices &matrices,
    const Options &options,
    hipStream_t stream,
    ThermalGuard &thermal) {
    const Variant variants[] = {Variant::Sq8_1W8A16, Variant::Sq8_1W8A8};
    for (int run = 1; run <= options.sweep_runs; ++run) {
        for (std::size_t m : options.m_values) {
            Timing timings[2];
            const std::string prefix = "m_sweep:r" + std::to_string(run) + ":m" + std::to_string(m);
            thermal.cooldown(prefix + ":before_warmup");
            for (Timing &timing : timings) {
                timing.start_c = thermal.check(prefix + ":warmup_start");
                observe(&timing, timing.start_c);
            }
            for (int warmup = 0; warmup < options.sweep_warmups; ++warmup) {
                for (int order = 0; order < 2; ++order) {
                    const int index = (warmup + order) % 2;
                    observe(&timings[index], thermal.check(prefix + ":warmup:" + std::to_string(warmup) + ":before:" + std::to_string(index)));
                    launch_bundle(
                        modules,
                        variants[index],
                        matrices.replicas[static_cast<std::size_t>(run + warmup) % matrices.replicas.size()],
                        matrices,
                        m,
                        stream);
                    HIP_CHECK(hipStreamSynchronize(stream));
                    observe(&timings[index], thermal.check(prefix + ":warmup:" + std::to_string(warmup) + ":after:" + std::to_string(index)));
                }
            }
            thermal.cooldown(prefix + ":before_timing");
            for (Timing &timing : timings) {
                timing.start_c = thermal.check(prefix + ":timing_start");
                observe(&timing, timing.start_c);
            }
            hipEvent_t start{}, stop{};
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));
            try {
                for (int trial = 0; trial < options.sweep_trials; ++trial) {
                    for (int order = 0; order < 2; ++order) {
                        const int index = (trial + order) % 2;
                        observe(&timings[index], thermal.check(prefix + ":trial:" + std::to_string(trial) + ":before:" + std::to_string(index)));
                        HIP_CHECK(hipEventRecord(start, stream));
                        launch_bundle(
                            modules,
                            variants[index],
                            matrices.replicas[static_cast<std::size_t>(run + trial) % matrices.replicas.size()],
                            matrices,
                            m,
                            stream);
                        HIP_CHECK(hipEventRecord(stop, stream));
                        HIP_CHECK(hipEventSynchronize(stop));
                        float elapsed = 0.0f;
                        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
                        timings[index].samples.push_back(elapsed);
                        observe(&timings[index], thermal.check(prefix + ":trial:" + std::to_string(trial) + ":after:" + std::to_string(index)));
                    }
                }
            } catch (...) {
                (void)hipEventDestroy(stop);
                (void)hipEventDestroy(start);
                throw;
            }
            HIP_CHECK(hipEventDestroy(stop));
            HIP_CHECK(hipEventDestroy(start));
            for (int index = 0; index < 2; ++index) {
                timings[index].end_c = thermal.check(prefix + ":timing_end:" + std::to_string(index));
                observe(&timings[index], timings[index].end_c);
                emit_timing("m_sweep_exact_direct_bundle", run, variants[index], m, timings[index],
                            "same_process_rotating_order_M_independent_exact_runtime_matvec_launches");
            }
            thermal.cooldown(prefix + ":complete");
        }
    }
}

void run_m_sweep_prequant_batch_prototype(
    const Modules &modules,
    const Matrices &matrices,
    const Options &options,
    hipStream_t stream,
    ThermalGuard &thermal) {
    const Variant variants[] = {Variant::Sq8_1W8A16, Variant::Sq8_1W8A8};
    for (int run = 1; run <= options.sweep_runs; ++run) {
        for (std::size_t m : options.m_values) {
            Timing timings[2];
            const std::string prefix =
                "m_sweep_prequant_2d_batch:r" + std::to_string(run) + ":m" + std::to_string(m);
            thermal.cooldown(prefix + ":before_warmup");
            for (Timing &timing : timings) {
                timing.start_c = thermal.check(prefix + ":warmup_start");
                observe(&timing, timing.start_c);
            }
            for (int warmup = 0; warmup < options.sweep_warmups; ++warmup) {
                for (int order = 0; order < 2; ++order) {
                    const int index = (warmup + order) % 2;
                    observe(&timings[index], thermal.check(
                        prefix + ":warmup:" + std::to_string(warmup) + ":before:" + std::to_string(index)));
                    launch_sq81_batch_prototype(
                        modules,
                        variants[index],
                        matrices.replicas[static_cast<std::size_t>(run + warmup) % matrices.replicas.size()],
                        matrices,
                        m,
                        stream);
                    HIP_CHECK(hipStreamSynchronize(stream));
                    observe(&timings[index], thermal.check(
                        prefix + ":warmup:" + std::to_string(warmup) + ":after:" + std::to_string(index)));
                }
            }
            thermal.cooldown(prefix + ":before_timing");
            for (Timing &timing : timings) {
                timing.start_c = thermal.check(prefix + ":timing_start");
                observe(&timing, timing.start_c);
            }
            hipEvent_t start{}, stop{};
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));
            try {
                for (int trial = 0; trial < options.sweep_trials; ++trial) {
                    for (int order = 0; order < 2; ++order) {
                        const int index = (trial + order) % 2;
                        observe(&timings[index], thermal.check(
                            prefix + ":trial:" + std::to_string(trial) + ":before:" + std::to_string(index)));
                        HIP_CHECK(hipEventRecord(start, stream));
                        launch_sq81_batch_prototype(
                            modules,
                            variants[index],
                            matrices.replicas[static_cast<std::size_t>(run + trial) % matrices.replicas.size()],
                            matrices,
                            m,
                            stream);
                        HIP_CHECK(hipEventRecord(stop, stream));
                        HIP_CHECK(hipEventSynchronize(stop));
                        float elapsed = 0.0f;
                        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
                        timings[index].samples.push_back(elapsed);
                        observe(&timings[index], thermal.check(
                            prefix + ":trial:" + std::to_string(trial) + ":after:" + std::to_string(index)));
                    }
                }
            } catch (...) {
                (void)hipEventDestroy(stop);
                (void)hipEventDestroy(start);
                throw;
            }
            HIP_CHECK(hipEventDestroy(stop));
            HIP_CHECK(hipEventDestroy(start));
            for (int index = 0; index < 2; ++index) {
                timings[index].end_c = thermal.check(prefix + ":timing_end:" + std::to_string(index));
                observe(&timings[index], timings[index].end_c);
                emit_timing(
                    "m_sweep_prequant_2d_batch_prototype",
                    run,
                    variants[index],
                    m,
                    timings[index],
                    "same_process_rotating_order_prequant_plus_2d_batch_benchmark_only_no_runtime_abi",
                    false);
            }
            thermal.cooldown(prefix + ":complete");
        }
    }
}

int run(const Options &options) {
    const Device device = select_v620(options.bdf);
    ThermalGuard thermal(device, sensor_for_bdf(device.bdf));
    thermal.cooldown("startup");
    // The numerical gate also exercises the exact SQ8_0 batch symbol with
    // two activation vectors, even if a smoke run only asks to time M=1.
    const std::size_t max_m = std::max(
        std::size_t{2u},
        *std::max_element(options.m_values.begin(), options.m_values.end()));
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    try {
        const Modules modules = compile_modules();
        Matrices matrices = upload_matrices(max_m, stream, thermal);
        std::ostringstream metadata;
        metadata << "{\"record\":\"benchmark_metadata\",\"shape\":\"qwen3_14b_q_proj\""
                 << ",\"rows\":5120,\"cols\":5120,\"sq8_0_scale_kind\":2,\"sq8_0_scale_block\":128"
                 << ",\"sq8_1_group\":32,\"sq8_1_common_source_requantized_from_sq8_0\":true"
                 << ",\"m1_runs\":" << options.m1_runs << ",\"m1_warmups\":" << options.m1_warmups
                 << ",\"m1_trials\":" << options.m1_trials << ",\"sweep_runs\":" << options.sweep_runs
                 << ",\"sweep_warmups\":" << options.sweep_warmups
                 << ",\"sweep_trials\":" << options.sweep_trials << ",\"m_values\":[";
        for (std::size_t index = 0; index < options.m_values.size(); ++index) {
            if (index) metadata << ',';
            metadata << options.m_values[index];
        }
        metadata << "],\"m_sweep_current_direct_scope\":\"M independent exact SQ8_1 direct-matvec launches; no new batch ABI or fused GEMM is claimed\""
                 << ",\"m_sweep_prequant_batch_scope\":\"benchmark-only isolated two-stage prequant plus 2-D batch prototype; not a runtime ABI, dispatch, or release path\"}";
        emit(metadata.str());
        numerical_gate(modules, matrices, stream, thermal);
        for (int run_index = 1; run_index <= options.m1_runs; ++run_index) {
            run_m1_co_dispatch(modules, matrices, options, run_index, stream, thermal);
        }
        run_m_sweep(modules, matrices, options, stream, thermal);
        run_m_sweep_prequant_batch_prototype(modules, matrices, options, stream, thermal);
        HIP_CHECK(hipStreamDestroy(stream));
    } catch (...) {
        (void)hipStreamDestroy(stream);
        throw;
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        open_new(g_jsonl, options.jsonl);
        open_new(g_thermal, options.thermal);
        return run(options);
    } catch (const std::exception &error) {
        std::cerr << "bench-sq8_0-sq8_1-fair-comparison-hip: " << error.what() << '\n';
        return 1;
    }
}
