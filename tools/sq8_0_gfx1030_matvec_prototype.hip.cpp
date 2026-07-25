// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Isolated numerical prototype for the SQ8_0 gfx1030 generic matvec path.
// It intentionally has no runtime ABI, dispatch, candidate, or release
// integration.  The legacy and candidate symbols share the SQ8_0 E4M3/F32
// scale contract so that a V620 differential can precede runtime replacement.

#include <hip/hip_runtime.h>

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

#define HIP_CHECK(call) do { \
    const hipError_t status__ = (call); \
    if (status__ != hipSuccess) { \
        throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(status__)); \
    } \
} while (false)

typedef unsigned int ullm_sq8_0_proto_uint4 __attribute__((ext_vector_type(4)));
static_assert(sizeof(ullm_sq8_0_proto_uint4) == 16u, "prototype uint4 must be 16 bytes");
static_assert(alignof(ullm_sq8_0_proto_uint4) == 16u, "prototype uint4 must be aligned");

__device__ __forceinline__ float ullm_sq8_0_proto_e4m3fn_to_f32(unsigned char value) {
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
    return __uint_as_float((sign << 31u) | ((exponent + 120u) << 23u) | (mantissa << 20u));
}

__device__ __forceinline__ float ullm_sq8_0_proto_scale_at(
    const float *scales,
    unsigned long long row,
    unsigned long long col,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    unsigned long long blocks_per_row) {
    float scale = scales[0];
    if (scale_kind == 1u) {
        scale = scales[row];
    } else if (scale_kind == 2u) {
        scale = scales[(row / scale_block_rows) * blocks_per_row + (col / scale_block_cols)];
    }
    return scale;
}

extern "C" __global__ void ullm_sq8_0_matvec_legacy_prototype_kernel(
    const unsigned char *payload,
    const float *scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    float *output) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        unsigned long long blocks_per_row = 1ull;
        if (scale_kind == 2u) {
            blocks_per_row = 1ull + (cols - 1ull) / scale_block_cols;
        }
        for (unsigned long long col = tid; col < cols; col += blockDim.x) {
            const float scale = ullm_sq8_0_proto_scale_at(
                scales, row, col, scale_kind, scale_block_rows, scale_block_cols, blocks_per_row);
            sum += ullm_sq8_0_proto_e4m3fn_to_f32(payload[row_offset + col]) * scale * input[col];
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0u && row < rows) {
        output[row] = partial[0];
    }
}

constexpr unsigned int kUllmSq8_0ProtoWaveSize = 32u;

__device__ __forceinline__ float ullm_sq8_0_proto_wave32_sum(float value) {
#pragma unroll
    for (unsigned int offset = kUllmSq8_0ProtoWaveSize >> 1u; offset > 0u; offset >>= 1u) {
        value += __shfl_down(value, offset, kUllmSq8_0ProtoWaveSize);
    }
    return value;
}

__device__ __forceinline__ float ullm_sq8_0_proto_accumulate_uint4(
    ullm_sq8_0_proto_uint4 packed,
    const float *input,
    float scale,
    float sum) {
    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll 1
    for (unsigned int word = 0u; word < 4u; ++word) {
        unsigned int value = words[word];
#pragma unroll 1
        for (unsigned int byte = 0u; byte < 4u; ++byte) {
            sum = fmaf(
                ullm_sq8_0_proto_e4m3fn_to_f32(static_cast<unsigned char>(value & 0xffu)) * scale,
                input[word * 4u + byte], sum);
            value >>= 8u;
        }
    }
    return sum;
}

// One output row keeps all 256 threads active.  The old full LDS tree is
// replaced by wave32 shuffles plus eight LDS wave partials and one barrier.
// Each full K=16 segment uses an aligned 128-bit uint4 payload load.  This is
// deliberately a distinct tuning candidate from the earlier eight-row CTA:
// it preserves the legacy row-level parallelism while retaining the same
// reduction and wide-load techniques.
__device__ __forceinline__ void ullm_sq8_0_matvec_wave32_candidate_body(
    const unsigned char *payload,
    const float *scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    float *output) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kUllmSq8_0ProtoWaveSize - 1u);
    const unsigned int wave = tid >> 5u;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x);
    __shared__ float wave_partial[8];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned char *const row_payload = payload + row * cols;
        const unsigned long long blocks_per_row = scale_kind == 2u
            ? 1ull + (cols - 1ull) / scale_block_cols
            : 1ull;
        const bool segment_scale = scale_kind != 2u ||
            (scale_block_rows != 0ull && scale_block_cols >= 16ull &&
             (scale_block_cols & 15ull) == 0ull);
        const unsigned long long segments = (cols + 15ull) / 16ull;
        for (unsigned long long segment = tid; segment < segments; segment += blockDim.x) {
            const unsigned long long start = segment * 16ull;
            const unsigned int count = static_cast<unsigned int>(
                (cols - start) < 16ull ? (cols - start) : 16ull);
            const bool wide_aligned = count == 16u &&
                ((reinterpret_cast<unsigned long long>(row_payload + start) & 15ull) == 0ull);
            if (segment_scale && wide_aligned) {
                const float scale = ullm_sq8_0_proto_scale_at(
                    scales, row, start, scale_kind, scale_block_rows, scale_block_cols, blocks_per_row);
                const ullm_sq8_0_proto_uint4 packed =
                    *reinterpret_cast<const ullm_sq8_0_proto_uint4 *>(row_payload + start);
                sum = ullm_sq8_0_proto_accumulate_uint4(packed, input + start, scale, sum);
            } else {
                for (unsigned int index = 0u; index < count; ++index) {
                    const unsigned long long col = start + index;
                    const float scale = ullm_sq8_0_proto_scale_at(
                        scales, row, col, scale_kind, scale_block_rows, scale_block_cols, blocks_per_row);
                    sum = fmaf(ullm_sq8_0_proto_e4m3fn_to_f32(row_payload[col]) * scale, input[col], sum);
                }
            }
        }
    }
    const float reduced = ullm_sq8_0_proto_wave32_sum(sum);
    if (lane == 0u) {
        wave_partial[wave] = reduced;
    }
    __syncthreads();
    if (tid == 0u && row < rows) {
        output[row] = wave_partial[0] + wave_partial[1] + wave_partial[2] + wave_partial[3] +
            wave_partial[4] + wave_partial[5] + wave_partial[6] + wave_partial[7];
    }
}

extern "C" __global__ __launch_bounds__(256) void ullm_sq8_0_matvec_wave32_prototype_kernel(
    const unsigned char *payload,
    const float *scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    float *output) {
    ullm_sq8_0_matvec_wave32_candidate_body(
        payload, scales, input, rows, cols, scale_kind, scale_block_rows, scale_block_cols, output);
}

// Static-only alternative: force two resident workgroups in the compiler's
// launch-bound model.  It is never run; the metadata comparison records
// whether the pressure reduction is worth its codegen trade-off.
extern "C" __global__ __launch_bounds__(256, 2) void ullm_sq8_0_matvec_wave32_lb2_prototype_kernel(
    const unsigned char *payload,
    const float *scales,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    unsigned int scale_kind,
    unsigned long long scale_block_rows,
    unsigned long long scale_block_cols,
    float *output) {
    ullm_sq8_0_matvec_wave32_candidate_body(
        payload, scales, input, rows, cols, scale_kind, scale_block_rows, scale_block_cols, output);
}

namespace {

constexpr double kJunctionLimitC = 85.0;
constexpr double kCooldownC = 42.0;
constexpr unsigned int kThreads = 256u;

struct Options {
    std::string bdf;
    std::filesystem::path jsonl;
    std::filesystem::path thermal;
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

struct Case {
    const char *name;
    std::size_t rows;
    std::size_t cols;
    unsigned int scale_kind;
    std::size_t scale_block_rows;
    std::size_t scale_block_cols;
};

std::ofstream g_jsonl;
std::ofstream g_thermal;

[[noreturn]] void usage(const char *argv0) {
    std::cerr << "usage: " << argv0
              << " --pci-bus-id 0000:03:00.0 --jsonl-output /absolute/path"
              << " --thermal-output /absolute/path\n";
    std::exit(2);
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
    if (matched.size() != 1u) {
        throw std::runtime_error("requested BDF did not select exactly one gfx1030 device");
    }
    HIP_CHECK(hipSetDevice(matched.front().ordinal));
    return matched.front();
}

std::string read_line(const std::filesystem::path &path) {
    std::ifstream input(path);
    std::string value;
    if (!input || !std::getline(input, value)) {
        throw std::runtime_error("failed to read " + path.string());
    }
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
    if (matched.size() != 1u) {
        throw std::runtime_error("failed to map HIP BDF to exactly one own junction sensor");
    }
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
        const long value = std::strtol(raw.c_str(), &end, 10);
        if (end == nullptr || *end != '\0' || value <= 0 || value > 200000) {
            throw std::runtime_error("invalid junction temperature");
        }
        const double celsius = static_cast<double>(value) / 1000.0;
        std::ostringstream record;
        record << std::setprecision(8) << "{\"record\":\"thermal\",\"phase\":\"" << phase
               << "\",\"pci_bus_id\":\"" << device_.bdf << "\",\"drm_card\":\"" << sensor_.card
               << "\",\"temperature_c\":" << celsius << ",\"junction_limit_c\":85.0}";
        emit(record.str(), true);
        if (celsius >= kJunctionLimitC) {
            throw std::runtime_error("thermal guard reached 85 C");
        }
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

std::uint32_t next(std::uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

float e4m3fn_to_f32_host(std::uint8_t value) {
    const std::uint32_t raw = value;
    const std::uint32_t sign = raw >> 7u;
    const std::uint32_t exponent = (raw >> 3u) & 0x0fu;
    const std::uint32_t mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) return std::numeric_limits<float>::quiet_NaN();
    if (exponent == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0.001953125f;
        return sign == 0u ? magnitude : -magnitude;
    }
    std::uint32_t bits = (sign << 31u) | ((exponent + 120u) << 23u) | (mantissa << 20u);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float host_scale_at(
    const std::vector<float> &scales,
    std::size_t row,
    std::size_t col,
    const Case &test) {
    if (test.scale_kind == 0u) return scales[0];
    if (test.scale_kind == 1u) return scales[row];
    const std::size_t blocks_per_row = (test.cols + test.scale_block_cols - 1u) / test.scale_block_cols;
    return scales[(row / test.scale_block_rows) * blocks_per_row + (col / test.scale_block_cols)];
}

std::vector<float> cpu_reference(
    const Case &test,
    const std::vector<std::uint8_t> &payload,
    const std::vector<float> &scales,
    const std::vector<float> &input) {
    std::vector<float> output(test.rows, 0.0f);
    for (std::size_t row = 0; row < test.rows; ++row) {
        for (std::size_t col = 0; col < test.cols; ++col) {
            output[row] += e4m3fn_to_f32_host(payload[row * test.cols + col]) *
                host_scale_at(scales, row, col, test) * input[col];
        }
    }
    return output;
}

struct Error {
    float max_abs = 0.0f;
    double relative_l2 = 0.0;
};

Error error_against(const std::vector<float> &actual, const std::vector<float> &expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("differential size mismatch");
    double error = 0.0;
    double reference = 0.0;
    Error result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float delta = actual[index] - expected[index];
        result.max_abs = std::max(result.max_abs, std::fabs(delta));
        error += static_cast<double>(delta) * delta;
        reference += static_cast<double>(expected[index]) * expected[index];
    }
    result.relative_l2 = std::sqrt(error / std::max(reference, 1.0e-30));
    return result;
}

std::vector<float> launch_and_copy(
    bool candidate,
    const Case &test,
    DeviceAllocation &payload,
    DeviceAllocation &scales,
    DeviceAllocation &input,
    DeviceAllocation &output,
    hipStream_t stream,
    ThermalGuard &thermal) {
    const std::string name = std::string("numerical:") + test.name +
        (candidate ? ":wave32" : ":legacy");
    const unsigned long long rows = test.rows;
    const unsigned long long cols = test.cols;
    const unsigned long long scale_rows = test.scale_block_rows;
    const unsigned long long scale_cols = test.scale_block_cols;
    thermal.check(name + ":before");
    if (candidate) {
        hipLaunchKernelGGL(
            ullm_sq8_0_matvec_wave32_prototype_kernel,
            dim3(static_cast<unsigned int>(test.rows)), dim3(kThreads), 0, stream,
            static_cast<const unsigned char *>(payload.get()), static_cast<const float *>(scales.get()),
            static_cast<const float *>(input.get()), rows, cols, test.scale_kind,
            scale_rows, scale_cols, static_cast<float *>(output.get()));
    } else {
        hipLaunchKernelGGL(
            ullm_sq8_0_matvec_legacy_prototype_kernel,
            dim3(static_cast<unsigned int>(test.rows)), dim3(kThreads), 0, stream,
            static_cast<const unsigned char *>(payload.get()), static_cast<const float *>(scales.get()),
            static_cast<const float *>(input.get()), rows, cols, test.scale_kind,
            scale_rows, scale_cols, static_cast<float *>(output.get()));
    }
    HIP_CHECK(hipGetLastError());
    std::vector<float> result(test.rows);
    HIP_CHECK(hipMemcpyAsync(
        result.data(), output.get(), result.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    thermal.check(name + ":after");
    return result;
}

void run_case(const Case &test, hipStream_t stream, ThermalGuard &thermal) {
    std::uint32_t state = 0x5b8619cdu ^ static_cast<std::uint32_t>(test.rows) ^
        (static_cast<std::uint32_t>(test.cols) << 1u) ^ test.scale_kind;
    std::vector<std::uint8_t> payload(test.rows * test.cols);
    for (std::uint8_t &value : payload) {
        // exp=3..10 keeps every generated byte finite while exercising signs and mantissas.
        value = static_cast<std::uint8_t>(((next(state) & 1u) << 7u) |
            ((3u + (next(state) & 7u)) << 3u) | (next(state) & 7u));
    }
    const std::size_t scale_rows = test.scale_kind == 2u
        ? (test.rows + test.scale_block_rows - 1u) / test.scale_block_rows
        : (test.scale_kind == 1u ? test.rows : 1u);
    const std::size_t scale_cols = test.scale_kind == 2u
        ? (test.cols + test.scale_block_cols - 1u) / test.scale_block_cols
        : 1u;
    std::vector<float> scales(scale_rows * scale_cols);
    for (float &scale : scales) {
        scale = 0.0078125f * (1.0f + 0.125f * static_cast<float>(next(state) & 7u));
    }
    std::vector<float> input(test.cols);
    for (float &value : input) {
        value = static_cast<float>(static_cast<int>(next(state) % 255u) - 127) * (1.0f / 127.0f);
    }
    const std::vector<float> expected = cpu_reference(test, payload, scales, input);
    DeviceAllocation payload_device(payload.size());
    DeviceAllocation scale_device(scales.size() * sizeof(float));
    DeviceAllocation input_device(input.size() * sizeof(float));
    DeviceAllocation output_device(test.rows * sizeof(float));
    HIP_CHECK(hipMemcpyAsync(payload_device.get(), payload.data(), payload.size(), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(scale_device.get(), scales.data(), scales.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(input_device.get(), input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::vector<float> legacy = launch_and_copy(
        false, test, payload_device, scale_device, input_device, output_device, stream, thermal);
    const std::vector<float> candidate = launch_and_copy(
        true, test, payload_device, scale_device, input_device, output_device, stream, thermal);
    const Error legacy_error = error_against(legacy, expected);
    const Error candidate_error = error_against(candidate, expected);
    const Error differential = error_against(candidate, legacy);
    const bool passed = legacy_error.max_abs <= 0.05f && legacy_error.relative_l2 <= 1.0e-5 &&
        candidate_error.max_abs <= 0.05f && candidate_error.relative_l2 <= 1.0e-5 &&
        differential.max_abs <= 0.05f && differential.relative_l2 <= 1.0e-5;
    std::ostringstream record;
    record << std::setprecision(10)
           << "{\"record\":\"numerical_differential\",\"case\":\"" << test.name
           << "\",\"rows\":" << test.rows << ",\"cols\":" << test.cols
           << ",\"scale_kind\":" << test.scale_kind
           << ",\"scale_block_rows\":" << test.scale_block_rows
           << ",\"scale_block_cols\":" << test.scale_block_cols
           << ",\"legacy_vs_cpu_max_abs\":" << legacy_error.max_abs
           << ",\"legacy_vs_cpu_relative_l2\":" << legacy_error.relative_l2
           << ",\"candidate_vs_cpu_max_abs\":" << candidate_error.max_abs
           << ",\"candidate_vs_cpu_relative_l2\":" << candidate_error.relative_l2
           << ",\"candidate_vs_legacy_max_abs\":" << differential.max_abs
           << ",\"candidate_vs_legacy_relative_l2\":" << differential.relative_l2
           << ",\"passed\":" << (passed ? "true" : "false") << "}";
    emit(record.str());
    if (!passed) throw std::runtime_error(std::string("numerical differential failed: ") + test.name);
}

int run(const Options &options) {
    const Device device = select_v620(options.bdf);
    ThermalGuard thermal(device, sensor_for_bdf(device.bdf));
    thermal.cooldown("prototype:startup");
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    try {
        const std::vector<Case> cases = {
            {"tail_k1_scalar_scale", 3u, 1u, 0u, 1u, 1u},
            {"tail_k33_row_scale", 5u, 33u, 1u, 1u, 1u},
            {"unaligned_k65_block17", 7u, 65u, 2u, 2u, 17u},
            {"aligned_k128_block32", 9u, 128u, 2u, 3u, 32u},
            {"qwen3_14b_q_proj_5120_block128", 5120u, 5120u, 2u, 128u, 128u},
        };
        for (const Case &test : cases) run_case(test, stream, thermal);
        thermal.cooldown("prototype:complete");
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
        std::cerr << "sq8_0_gfx1030_matvec_prototype: " << error.what() << '\n';
        return 1;
    }
}
