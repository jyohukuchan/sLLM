// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Isolated, thermal-guarded M=1 SQ8_1 timing for direct comparison with the
// SQ8_0 V620 q_proj protocol.  The timing kernel source is the exact HIPRTC
// source used by the runtime; no release artefact or service is involved.

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
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// This is intentionally included as a host-only source-string factory.  The
// program compiles that exact string through HIPRTC before loading the module.
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
constexpr std::size_t kGroups = kCols / 32u;
constexpr std::size_t kStride = kCols;
constexpr std::size_t kRowsPerBlock = 8u;
constexpr unsigned int kThreads = 256u;
constexpr int kReplicaCount = 6;
constexpr double kJunctionLimitC = 85.0;
constexpr double kCooldownC = 42.0;
constexpr double kPeakBandwidthGBps = 512.0;

struct Options {
    std::string bdf;
    std::filesystem::path jsonl;
    std::filesystem::path thermal;
    int warmups = 32;
    int trials = 31;
};

struct Sensor {
    std::string card;
    std::filesystem::path input;
};

struct Device {
    int ordinal = -1;
    std::string bdf;
    std::string arch;
    std::string name;
};

std::ofstream g_jsonl;
std::ofstream g_thermal;

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --pci-bus-id 0000:03:00.0 --jsonl-output /absolute/path"
              << " --thermal-output /absolute/path [--warmups N --trials N]\n";
    std::exit(2);
}

int positive_int(std::string_view text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(std::string(text).c_str(), &end, 10);
    if (text.empty() || end == nullptr || *end != '\0' || value <= 0 || value > 1000000) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
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
        else if (arg == "--warmups") options.warmups = positive_int(need(), "warmups");
        else if (arg == "--trials") options.trials = positive_int(need(), "trials");
        else usage(argv[0]);
    }
    if (options.bdf.empty() || !options.jsonl.is_absolute() || !options.thermal.is_absolute() ||
        options.jsonl == options.thermal) usage(argv[0]);
    return options;
}

void open_new(std::ofstream& output, const std::filesystem::path& path) {
    std::error_code error;
    if (std::filesystem::exists(path, error) || error) {
        throw std::runtime_error("refusing to overwrite output: " + path.string());
    }
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) throw std::runtime_error("failed to create output directory: " + error.message());
    output.open(path);
    if (!output) throw std::runtime_error("failed to open output: " + path.string());
}

void emit(const std::string& value, bool thermal = false) {
    std::cout << value << '\n';
    std::cout.flush();
    g_jsonl << value << '\n';
    g_jsonl.flush();
    if (thermal) {
        g_thermal << value << '\n';
        g_thermal.flush();
    }
}

std::string lower(std::string value) {
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

bool gfx1030(std::string_view arch) {
    return arch == "gfx1030" || (arch.size() > 8u && arch.substr(0u, 8u) == "gfx1030:");
}

Device select_v620(const std::string& requested_bdf) {
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
    for (const Device& device : devices) {
        if (gfx1030(device.arch) && lower(device.bdf) == lower(requested_bdf)) matched.push_back(device);
    }
    if (matched.size() != 1u) throw std::runtime_error("requested BDF did not select exactly one gfx1030");
    HIP_CHECK(hipSetDevice(matched.front().ordinal));
    return matched.front();
}

std::string read_line(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string value;
    if (!input || !std::getline(input, value)) throw std::runtime_error("failed to read " + path.string());
    return value;
}

Sensor sensor_for_bdf(const std::string& bdf) {
    std::vector<Sensor> matched;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", error)) {
        const std::string card = entry.path().filename().string();
        if (card.rfind("card", 0u) != 0u || card.size() == 4u) continue;
        const auto device = entry.path() / "device";
        const auto canonical = std::filesystem::weakly_canonical(device, error);
        if (error || lower(canonical.filename().string()) != lower(bdf)) { error.clear(); continue; }
        for (const auto& hwmon : std::filesystem::directory_iterator(device / "hwmon", error)) {
            if (error) break;
            const auto temp = hwmon.path() / "temp2_input";
            const auto label = hwmon.path() / "temp2_label";
            if (std::filesystem::is_regular_file(temp, error) && !error &&
                std::filesystem::is_regular_file(label, error) && !error && lower(read_line(label)) == "junction") {
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
        std::ostringstream out;
        out << "{\"record\":\"thermal_sensor\",\"hip_ordinal\":" << device_.ordinal
            << ",\"pci_bus_id\":\"" << device_.bdf << "\",\"drm_card\":\"" << sensor_.card
            << "\",\"temp2_input\":\"" << sensor_.input.string()
            << "\",\"junction_limit_c\":85.0,\"cooldown_target_c\":42.0}";
        emit(out.str(), true);
    }

    double check(std::string_view phase) {
        const std::string raw = read_line(sensor_.input);
        char* end = nullptr;
        const long milli = std::strtol(raw.c_str(), &end, 10);
        if (end == nullptr || *end != '\0' || milli <= 0 || milli > 200000) throw std::runtime_error("invalid junction reading");
        const double celsius = static_cast<double>(milli) / 1000.0;
        std::ostringstream out;
        out << std::setprecision(8) << "{\"record\":\"thermal\",\"phase\":\"" << phase
            << "\",\"pci_bus_id\":\"" << device_.bdf << "\",\"drm_card\":\"" << sensor_.card
            << "\",\"temperature_c\":" << celsius << ",\"junction_limit_c\":85.0}";
        emit(out.str(), true);
        if (celsius >= kJunctionLimitC) throw std::runtime_error("thermal guard reached 85 C");
        return celsius;
    }

    void cooldown(std::string_view phase) {
        for (int attempt = 0; attempt < 180; ++attempt) {
            const double value = check(std::string(phase) + ":poll");
            if (value <= kCooldownC) return;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        throw std::runtime_error("cooldown timed out");
    }

private:
    Device device_;
    Sensor sensor_;
};

class DeviceAllocation {
public:
    DeviceAllocation() = default;
    explicit DeviceAllocation(std::size_t bytes) { HIP_CHECK(hipMalloc(&ptr_, bytes)); }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    DeviceAllocation(DeviceAllocation&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
        if (this != &other) { reset(); ptr_ = other.ptr_; other.ptr_ = nullptr; }
        return *this;
    }
    ~DeviceAllocation() { reset(); }
    void* get() const { return ptr_; }
private:
    void reset() noexcept { if (ptr_ != nullptr) (void)hipFree(ptr_); ptr_ = nullptr; }
    void* ptr_ = nullptr;
};

struct Replica { DeviceAllocation payload; DeviceAllocation scales; };

struct Module {
    hipModule_t module{};
    hipFunction_t w8a16{};
    hipFunction_t w8a8{};
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&& other) noexcept : module(other.module), w8a16(other.w8a16), w8a8(other.w8a8) {
        other.module = nullptr;
        other.w8a16 = nullptr;
        other.w8a8 = nullptr;
    }
    Module& operator=(Module&& other) noexcept {
        if (this != &other) {
            if (module != nullptr) (void)hipModuleUnload(module);
            module = other.module;
            w8a16 = other.w8a16;
            w8a8 = other.w8a8;
            other.module = nullptr;
            other.w8a16 = nullptr;
            other.w8a8 = nullptr;
        }
        return *this;
    }
    ~Module() { if (module != nullptr) (void)hipModuleUnload(module); }
};

Module compile_exact_module() {
    hiprtcProgram program{};
    HIPRTC_CHECK(hiprtcCreateProgram(&program, sq8_1_matvec_kernel_source(), "sq8_1_v620_bench.hip", 0, nullptr, nullptr));
    const char* options[] = {"--offload-arch=gfx1030", "--std=c++17", "-O3"};
    const hiprtcResult compile = hiprtcCompileProgram(program, 3, options);
    if (compile != HIPRTC_SUCCESS) {
        std::size_t log_size = 0;
        (void)hiprtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size) (void)hiprtcGetProgramLog(program, log.data());
        (void)hiprtcDestroyProgram(&program);
        throw std::runtime_error("HIPRTC compilation failed: " + log);
    }
    std::size_t code_size = 0;
    HIPRTC_CHECK(hiprtcGetCodeSize(program, &code_size));
    std::vector<char> code(code_size);
    HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    Module result;
    HIP_CHECK(hipModuleLoadData(&result.module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&result.w8a16, result.module, "ullm_sq8_1_matvec_w8a16_f32_kernel"));
    HIP_CHECK(hipModuleGetFunction(&result.w8a8, result.module, "ullm_sq8_1_matvec_w8a8_explicit_f32_kernel"));
    return result;
}

std::uint32_t next(std::uint32_t& state) { state = state * 1664525u + 1013904223u; return state; }

std::vector<std::uint8_t> make_payload(int replica) {
    std::vector<std::uint8_t> payload(kRows * kStride);
    std::uint32_t state = 0x74190f2du ^ static_cast<std::uint32_t>(replica);
    for (std::uint8_t& value : payload) {
        value = static_cast<std::uint8_t>(static_cast<std::int8_t>(
            static_cast<int>(next(state) % 255u) - 127));
    }
    return payload;
}

void upload_replicas(
    std::vector<Replica>* replicas,
    std::vector<std::uint8_t>* first_payload,
    hipStream_t stream) {
    replicas->clear();
    replicas->reserve(kReplicaCount);
    for (int replica = 0; replica < kReplicaCount; ++replica) {
        std::vector<std::uint8_t> payload = make_payload(replica);
        std::vector<std::uint16_t> scales(kRows * kGroups, 0x3c00u);
        if (replica == 0) *first_payload = payload;
        Replica device{DeviceAllocation(payload.size()), DeviceAllocation(scales.size() * sizeof(std::uint16_t))};
        HIP_CHECK(hipMemcpyAsync(device.payload.get(), payload.data(), payload.size(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(device.scales.get(), scales.data(), scales.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream));
        replicas->push_back(std::move(device));
    }
}

std::vector<float> make_input() {
    std::vector<float> input(kCols);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>((index * 19u + 7u) % 253u) - 126);
    }
    return input;
}

float f16_bits_to_f32(std::uint16_t bits) {
    _Float16 value{};
    std::memcpy(&value, &bits, sizeof(bits));
    return static_cast<float>(value);
}

float ceil_f16(float value) {
    _Float16 narrowed = static_cast<_Float16>(value);
    std::uint16_t bits = 0u;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    float stored = f16_bits_to_f32(bits);
    if (bits == 0u) return f16_bits_to_f32(1u);
    if (stored < value) stored = f16_bits_to_f32(++bits);
    return stored;
}

void launch(const Module& module, bool w8a8, const Replica& replica, void* input, void* output, hipStream_t stream) {
    void* payload = replica.payload.get();
    void* scales = replica.scales.get();
    unsigned long long rows = kRows, cols = kCols, stride = kStride;
    void* params[] = {&payload, &scales, &input, &rows, &cols, &stride, &output};
    const unsigned int grid = static_cast<unsigned int>((kRows + kRowsPerBlock - 1u) / kRowsPerBlock);
    const unsigned int dynamic_shared = w8a8
        ? static_cast<unsigned int>(kStride + kGroups * sizeof(float))
        : 0u;
    HIP_CHECK(hipModuleLaunchKernel(
        w8a8 ? module.w8a8 : module.w8a16, grid, 1u, 1u, kThreads, 1u, 1u,
        dynamic_shared, stream, params, nullptr));
}

void numerical_gate(
    const Module& module,
    const Replica& replica,
    const std::vector<std::uint8_t>& payload,
    const std::vector<float>& input,
    void* input_device,
    void* output,
    hipStream_t stream,
    ThermalGuard& thermal) {
    constexpr std::size_t rows_to_check = 8u;
    std::vector<float> expected_w8a16(rows_to_check, 0.0f);
    std::vector<float> expected_w8a8(rows_to_check, 0.0f);
    std::vector<std::int8_t> activation(kCols);
    std::vector<float> activation_scales(kGroups);
    for (std::size_t group = 0; group < kGroups; ++group) {
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
    for (std::size_t row = 0; row < rows_to_check; ++row) {
        for (std::size_t group = 0; group < kGroups; ++group) {
            const std::size_t start = group * 32u;
            float w8a16_dot = 0.0f;
            std::int32_t w8a8_dot = 0;
            for (std::size_t index = 0; index < 32u; ++index) {
                const std::int8_t weight = static_cast<std::int8_t>(payload[row * kStride + start + index]);
                w8a16_dot += static_cast<float>(weight) * input[start + index];
                w8a8_dot += static_cast<std::int32_t>(weight) * activation[start + index];
            }
            expected_w8a16[row] += w8a16_dot;
            expected_w8a8[row] += static_cast<float>(w8a8_dot) * activation_scales[group];
        }
    }
    std::vector<float> actual(rows_to_check);
    const auto check = [&](bool w8a8, const std::vector<float>& expected) {
        const std::string path = w8a8 ? "numerical_gate:w8a8" : "numerical_gate:w8a16";
        thermal.check(path + ":before");
        launch(module, w8a8, replica, input_device, output, stream);
        HIP_CHECK(hipMemcpyAsync(actual.data(), output, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        thermal.check(path + ":after");
        double error = 0.0, reference = 0.0;
        float max_abs = 0.0f;
        for (std::size_t row = 0; row < rows_to_check; ++row) {
            const float delta = actual[row] - expected[row];
            max_abs = std::max(max_abs, std::fabs(delta));
            error += static_cast<double>(delta) * delta;
            reference += static_cast<double>(expected[row]) * expected[row];
        }
        const double relative_l2 = std::sqrt(error / reference);
        const bool passed = max_abs <= 1.0f && relative_l2 <= 1e-6;
        std::ostringstream out;
        out << std::setprecision(10) << "{\"record\":\"numerical_gate\",\"path\":\""
            << (w8a8 ? "W8A8" : "W8A16") << "\",\"rows_checked\":8,\"cols\":5120"
            << ",\"max_abs\":" << max_abs << ",\"relative_l2\":" << relative_l2
            << ",\"passed\":" << (passed ? "true" : "false") << "}";
        emit(out.str());
        if (!passed) throw std::runtime_error("wide numerical gate failed");
    };
    check(false, expected_w8a16);
    check(true, expected_w8a8);
}

struct Timing {
    std::vector<float> samples;
    double before = 0.0, after_warmup = 0.0, start = 0.0, end = 0.0, peak = 0.0;
};

void observe(Timing* timing, double value) {
    if (value > timing->peak) timing->peak = value;
}

template <typename Launch>
Timing time_point(const Options& options, ThermalGuard& thermal, std::string_view id, hipStream_t stream, Launch&& launch) {
    Timing timing;
    thermal.cooldown(std::string(id) + ":before_warmup");
    timing.before = thermal.check(std::string(id) + ":warmup_start"); observe(&timing, timing.before);
    for (int warmup = 0; warmup < options.warmups; ++warmup) {
        observe(&timing, thermal.check(std::string(id) + ":warmup:" + std::to_string(warmup) + ":before"));
        launch(warmup);
        HIP_CHECK(hipStreamSynchronize(stream));
        observe(&timing, thermal.check(std::string(id) + ":warmup:" + std::to_string(warmup) + ":after"));
    }
    timing.after_warmup = thermal.check(std::string(id) + ":warmup_complete"); observe(&timing, timing.after_warmup);
    thermal.cooldown(std::string(id) + ":before_timing");
    timing.start = thermal.check(std::string(id) + ":timing_start"); observe(&timing, timing.start);
    hipEvent_t start{}, stop{};
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    try {
        for (int trial = 0; trial < options.trials; ++trial) {
            observe(&timing, thermal.check(std::string(id) + ":trial:" + std::to_string(trial) + ":before"));
            HIP_CHECK(hipEventRecord(start, stream));
            launch(options.warmups + trial);
            HIP_CHECK(hipEventRecord(stop, stream)); HIP_CHECK(hipEventSynchronize(stop));
            float elapsed = 0.0f; HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
            timing.samples.push_back(elapsed);
            observe(&timing, thermal.check(std::string(id) + ":trial:" + std::to_string(trial) + ":after"));
        }
    } catch (...) { (void)hipEventDestroy(stop); (void)hipEventDestroy(start); throw; }
    HIP_CHECK(hipEventDestroy(stop)); HIP_CHECK(hipEventDestroy(start));
    timing.end = thermal.check(std::string(id) + ":timing_end"); observe(&timing, timing.end);
    return timing;
}

void emit_timing(const char* path, const Timing& timing) {
    std::vector<float> sorted = timing.samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2u];
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double variance = 0.0;
    for (float value : sorted) variance += (value - mean) * (value - mean);
    const double resident = static_cast<double>(kRows * kStride + kRows * kGroups * sizeof(std::uint16_t));
    const double seconds = median / 1000.0;
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"record\":\"timing\",\"kind\":\"dequant_plus_gemm\",\"shape\":\"qwen3_14b_q_proj\""
        << ",\"format\":\"SQ8_1\",\"variant\":\"" << path << "\""
        << ",\"rows\":" << kRows << ",\"cols\":" << kCols << ",\"m\":1"
        << ",\"weight_bytes_resident\":" << static_cast<std::size_t>(resident)
        << ",\"modeled_weight_bytes_per_launch\":" << static_cast<std::size_t>(resident)
        << ",\"median_ms\":" << median << ",\"mean_ms\":" << mean
        << ",\"stddev_ms\":" << std::sqrt(variance / sorted.size())
        << ",\"min_ms\":" << sorted.front() << ",\"max_ms\":" << sorted.back() << ",\"samples_ms\":[";
    for (std::size_t index = 0; index < sorted.size(); ++index) { if (index) out << ','; out << sorted[index]; }
    out << "]"
        << ",\"temperature_before_warmup_c\":" << timing.before
        << ",\"temperature_after_warmup_c\":" << timing.after_warmup
        << ",\"temperature_start_c\":" << timing.start << ",\"temperature_end_c\":" << timing.end
        << ",\"temperature_peak_c\":" << timing.peak
        << ",\"modeled_weight_stream_GBps\":" << resident / seconds / 1.0e9
        << ",\"modeled_weight_stream_pct_of_512GBps\":" << resident / seconds / 1.0e9 / kPeakBandwidthGBps * 100.0
        << ",\"gflops\":" << 2.0 * static_cast<double>(kRows) * kCols / seconds / 1.0e9
        << ",\"ns_per_logical_fma\":" << median * 1.0e6 / (static_cast<double>(kRows) * kCols)
        << "}";
    emit(out.str());
}

int run(const Options& options) {
    const Device device = select_v620(options.bdf);
    ThermalGuard thermal(device, sensor_for_bdf(device.bdf));
    thermal.cooldown("startup");
    hipStream_t stream{}; HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    try {
        const Module module = compile_exact_module();
        std::vector<Replica> replicas;
        std::vector<std::uint8_t> first_payload;
        upload_replicas(&replicas, &first_payload, stream);
        const std::vector<float> input = make_input();
        DeviceAllocation input_device(input.size() * sizeof(float));
        DeviceAllocation output(kRows * sizeof(float));
        HIP_CHECK(hipMemcpyAsync(input_device.get(), input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        numerical_gate(module, replicas.front(), first_payload, input, input_device.get(), output.get(), stream, thermal);
        emit("{\"record\":\"benchmark_metadata\",\"shape\":\"qwen3_14b_q_proj\",\"rows\":5120,\"cols\":5120,\"m\":1,\"warmups\":" + std::to_string(options.warmups) + ",\"trials\":" + std::to_string(options.trials) + ",\"launches_per_trial\":1,\"protocol\":\"SQ9 V620 M=1 thermal-normalized single-launch\",\"sq8_0_reference_median_ms\":0.6390069723}");
        for (const bool w8a8 : {false, true}) {
            const std::string name = std::string("sq8_1_m1_") + (w8a8 ? "w8a8_tiled" : "w8a16_wave32");
            const Timing timing = time_point(options, thermal, name, stream, [&](int sequence) {
                launch(module, w8a8, replicas[static_cast<std::size_t>(sequence) % replicas.size()],
                       input_device.get(), output.get(), stream);
            });
            emit_timing(w8a8 ? "w8a8_tiled_wave32_rows8" : "w8a16_wave32_rows8", timing);
            thermal.cooldown(name + ":after_timing");
        }
        HIP_CHECK(hipStreamDestroy(stream));
    } catch (...) { (void)hipStreamDestroy(stream); throw; }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        open_new(g_jsonl, options.jsonl); open_new(g_thermal, options.thermal);
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "bench-sq8_1-v620-optimization-hip: " << error.what() << '\n';
        return 1;
    }
}
