// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Direct benchmark for the isolated SQ8_0 gfx1201 Phase 2 candidate. This
// executable links only the prototype in tools/ and has no runtime dispatch.

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" __global__ void ullm_sq_fp8_matvec_f32_kernel(
    const unsigned char*, const float*, const float*, unsigned long long, unsigned long long,
    unsigned int, unsigned long long, unsigned long long, float*);
extern "C" __global__ void ullm_sq_fp8_matvec_batch_f32_kernel(
    const unsigned char*, const float*, const float*, unsigned long long, unsigned long long,
    unsigned int, unsigned long long, unsigned long long, unsigned long long, float*);
extern "C" __global__ void ullm_sq_fp8_matvec_pair_f32_kernel(
    const unsigned char*, const float*, unsigned long long, unsigned int, unsigned long long,
    const unsigned char*, const float*, unsigned long long, unsigned int, unsigned long long,
    const float*, unsigned long long, float*, float*);
extern "C" __global__ void ullm_sq_fp8_matvec_triple_f32_kernel(
    const unsigned char*, const float*, unsigned long long, unsigned int, unsigned long long,
    const unsigned char*, const float*, unsigned long long, unsigned int, unsigned long long,
    const unsigned char*, const float*, unsigned long long, unsigned int, unsigned long long,
    const float*, unsigned long long, float*, float*, float*);

namespace {

#define HIP_CHECK(call) do { \
    const hipError_t status__ = (call); \
    if (status__ != hipSuccess) throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(status__)); \
} while (false)

constexpr unsigned int kThreads = 256u;
constexpr unsigned int kScaleKindBlock2d = 2u;
constexpr unsigned long long kCols = 5120ull;
constexpr unsigned long long kScaleBlock = 128ull;

struct Options {
    std::filesystem::path output;
    std::string pci_bus_id;
    unsigned int iterations = 10u;
};

struct Result {
    const char* kernel = nullptr;
    unsigned long long payload_bytes = 0ull;
    float milliseconds = 0.0f;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) { HIP_CHECK(hipMalloc(&ptr_, bytes)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    ~DeviceBuffer() { reset(); }
    void* get() const { return ptr_; }

private:
    void reset() noexcept {
        if (ptr_ != nullptr) (void)hipFree(ptr_);
        ptr_ = nullptr;
    }
    void* ptr_ = nullptr;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --output /absolute/path --pci-bus-id 0000:47:00.0 [--iterations N]\n";
    std::exit(2);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto need = [&]() -> std::string {
            if (++index >= argc) usage(argv[0]);
            return argv[index];
        };
        if (argument == "--output") options.output = need();
        else if (argument == "--pci-bus-id") options.pci_bus_id = need();
        else if (argument == "--iterations") {
            const std::string value = need();
            const unsigned long parsed = std::stoul(value);
            if (parsed == 0ul || parsed > 1000ul) usage(argv[0]);
            options.iterations = static_cast<unsigned int>(parsed);
        } else {
            usage(argv[0]);
        }
    }
    if (options.output.empty() || options.pci_bus_id.empty() || !options.output.is_absolute()) usage(argv[0]);
    return options;
}

int select_gfx1201(const std::string& expected_bdf) {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    for (int device = 0; device < count; ++device) {
        std::array<char, 32> bdf{};
        hipDeviceProp_t properties{};
        HIP_CHECK(hipDeviceGetPCIBusId(bdf.data(), static_cast<int>(bdf.size()), device));
        HIP_CHECK(hipGetDeviceProperties(&properties, device));
        if (expected_bdf == bdf.data()) {
            if (std::string(properties.gcnArchName).find("gfx1201") == std::string::npos) {
                throw std::runtime_error("requested BDF is not gfx1201: " + std::string(properties.gcnArchName));
            }
            return device;
        }
    }
    throw std::runtime_error("requested PCI BDF is not a HIP device: " + expected_bdf);
}

std::uint32_t next_random(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

std::vector<std::uint8_t> payload(std::size_t elements, std::uint32_t& state) {
    std::vector<std::uint8_t> values(elements);
    for (std::uint8_t& value : values) value = static_cast<std::uint8_t>(next_random(state) & 0x7eu);
    return values;
}

std::vector<float> floats(std::size_t elements, std::uint32_t& state) {
    std::vector<float> values(elements);
    for (float& value : values) {
        value = static_cast<float>(static_cast<int>(next_random(state) % 8192u) - 4096) / 4096.0f;
    }
    return values;
}

DeviceBuffer upload(const void* source, std::size_t bytes) {
    DeviceBuffer result(bytes);
    HIP_CHECK(hipMemcpy(result.get(), source, bytes, hipMemcpyHostToDevice));
    return result;
}

template <typename Launch>
float time_launches(Launch&& launch, unsigned int iterations) {
    hipEvent_t begin{};
    hipEvent_t end{};
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    for (unsigned int warmup = 0u; warmup < 3u; ++warmup) launch();
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(begin));
    for (unsigned int iteration = 0u; iteration < iterations; ++iteration) launch();
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float milliseconds = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    return milliseconds / static_cast<float>(iterations);
}

Result bench_single(unsigned int iterations, std::uint32_t& state) {
    constexpr unsigned long long rows = 5120ull;
    const std::vector<std::uint8_t> host_payload = payload(rows * kCols, state);
    const std::vector<float> host_scales = floats(40u * 40u, state);
    const std::vector<float> host_input = floats(kCols, state);
    DeviceBuffer d_payload = upload(host_payload.data(), host_payload.size());
    DeviceBuffer d_scales = upload(host_scales.data(), host_scales.size() * sizeof(float));
    DeviceBuffer d_input = upload(host_input.data(), host_input.size() * sizeof(float));
    DeviceBuffer d_output(rows * sizeof(float));
    const float milliseconds = time_launches([&] {
        hipLaunchKernelGGL(ullm_sq_fp8_matvec_f32_kernel, dim3(rows), dim3(kThreads), 0u, nullptr,
            static_cast<const unsigned char*>(d_payload.get()), static_cast<const float*>(d_scales.get()),
            static_cast<const float*>(d_input.get()), rows, kCols, kScaleKindBlock2d,
            kScaleBlock, kScaleBlock, static_cast<float*>(d_output.get()));
        HIP_CHECK(hipGetLastError());
    }, iterations);
    return {"single_q_or_o", rows * kCols, milliseconds};
}

Result bench_batch(unsigned int iterations, std::uint32_t& state) {
    constexpr unsigned long long rows = 5120ull;
    constexpr unsigned long long batches = 1ull;
    const std::vector<std::uint8_t> host_payload = payload(rows * kCols, state);
    const std::vector<float> host_scales = floats(40u * 40u, state);
    const std::vector<float> host_input = floats(batches * kCols, state);
    DeviceBuffer d_payload = upload(host_payload.data(), host_payload.size());
    DeviceBuffer d_scales = upload(host_scales.data(), host_scales.size() * sizeof(float));
    DeviceBuffer d_input = upload(host_input.data(), host_input.size() * sizeof(float));
    DeviceBuffer d_output(rows * batches * sizeof(float));
    const float milliseconds = time_launches([&] {
        hipLaunchKernelGGL(ullm_sq_fp8_matvec_batch_f32_kernel, dim3(rows, batches), dim3(kThreads), 0u, nullptr,
            static_cast<const unsigned char*>(d_payload.get()), static_cast<const float*>(d_scales.get()),
            static_cast<const float*>(d_input.get()), rows, kCols, kScaleKindBlock2d,
            kScaleBlock, kScaleBlock, batches, static_cast<float*>(d_output.get()));
        HIP_CHECK(hipGetLastError());
    }, iterations);
    return {"batch_q_or_o_m1", rows * kCols, milliseconds};
}

Result bench_pair(unsigned int iterations, std::uint32_t& state) {
    constexpr unsigned long long rows = 17408ull;
    const std::vector<std::uint8_t> left_payload = payload(rows * kCols, state);
    const std::vector<std::uint8_t> right_payload = payload(rows * kCols, state);
    const std::vector<float> left_scales = floats(rows * 40u, state);
    const std::vector<float> right_scales = floats(rows * 40u, state);
    const std::vector<float> host_input = floats(kCols, state);
    DeviceBuffer d_left_payload = upload(left_payload.data(), left_payload.size());
    DeviceBuffer d_right_payload = upload(right_payload.data(), right_payload.size());
    DeviceBuffer d_left_scales = upload(left_scales.data(), left_scales.size() * sizeof(float));
    DeviceBuffer d_right_scales = upload(right_scales.data(), right_scales.size() * sizeof(float));
    DeviceBuffer d_input = upload(host_input.data(), host_input.size() * sizeof(float));
    DeviceBuffer d_left_output(rows * sizeof(float));
    DeviceBuffer d_right_output(rows * sizeof(float));
    const float milliseconds = time_launches([&] {
        hipLaunchKernelGGL(ullm_sq_fp8_matvec_pair_f32_kernel, dim3(rows, 2u), dim3(kThreads), 0u, nullptr,
            static_cast<const unsigned char*>(d_left_payload.get()), static_cast<const float*>(d_left_scales.get()),
            rows, kScaleKindBlock2d, kScaleBlock,
            static_cast<const unsigned char*>(d_right_payload.get()), static_cast<const float*>(d_right_scales.get()),
            rows, kScaleKindBlock2d, kScaleBlock,
            static_cast<const float*>(d_input.get()), kCols,
            static_cast<float*>(d_left_output.get()), static_cast<float*>(d_right_output.get()));
        HIP_CHECK(hipGetLastError());
    }, iterations);
    return {"pair_gate_up", 2ull * rows * kCols, milliseconds};
}

Result bench_triple(unsigned int iterations, std::uint32_t& state) {
    constexpr unsigned long long first_rows = 5120ull;
    constexpr unsigned long long second_rows = 1024ull;
    constexpr unsigned long long third_rows = 1024ull;
    const std::vector<std::uint8_t> first_payload = payload(first_rows * kCols, state);
    const std::vector<std::uint8_t> second_payload = payload(second_rows * kCols, state);
    const std::vector<std::uint8_t> third_payload = payload(third_rows * kCols, state);
    const std::vector<float> first_scales = floats(first_rows * 40u, state);
    const std::vector<float> second_scales = floats(second_rows * 40u, state);
    const std::vector<float> third_scales = floats(third_rows * 40u, state);
    const std::vector<float> host_input = floats(kCols, state);
    DeviceBuffer d_first_payload = upload(first_payload.data(), first_payload.size());
    DeviceBuffer d_second_payload = upload(second_payload.data(), second_payload.size());
    DeviceBuffer d_third_payload = upload(third_payload.data(), third_payload.size());
    DeviceBuffer d_first_scales = upload(first_scales.data(), first_scales.size() * sizeof(float));
    DeviceBuffer d_second_scales = upload(second_scales.data(), second_scales.size() * sizeof(float));
    DeviceBuffer d_third_scales = upload(third_scales.data(), third_scales.size() * sizeof(float));
    DeviceBuffer d_input = upload(host_input.data(), host_input.size() * sizeof(float));
    DeviceBuffer d_first_output(first_rows * sizeof(float));
    DeviceBuffer d_second_output(second_rows * sizeof(float));
    DeviceBuffer d_third_output(third_rows * sizeof(float));
    const float milliseconds = time_launches([&] {
        hipLaunchKernelGGL(ullm_sq_fp8_matvec_triple_f32_kernel,
            dim3(first_rows, 3u), dim3(kThreads), 0u, nullptr,
            static_cast<const unsigned char*>(d_first_payload.get()), static_cast<const float*>(d_first_scales.get()),
            first_rows, kScaleKindBlock2d, kScaleBlock,
            static_cast<const unsigned char*>(d_second_payload.get()), static_cast<const float*>(d_second_scales.get()),
            second_rows, kScaleKindBlock2d, kScaleBlock,
            static_cast<const unsigned char*>(d_third_payload.get()), static_cast<const float*>(d_third_scales.get()),
            third_rows, kScaleKindBlock2d, kScaleBlock,
            static_cast<const float*>(d_input.get()), kCols,
            static_cast<float*>(d_first_output.get()), static_cast<float*>(d_second_output.get()),
            static_cast<float*>(d_third_output.get()));
        HIP_CHECK(hipGetLastError());
    }, iterations);
    return {"triple_qkv", (first_rows + second_rows + third_rows) * kCols, milliseconds};
}

void write_results(const Options& options, const std::vector<Result>& results) {
    std::filesystem::create_directories(options.output.parent_path());
    std::ofstream output(options.output);
    if (!output) throw std::runtime_error("cannot write " + options.output.string());
    output << "{\n  \"schema_version\": \"ullm.sq8_0.gfx1201.phase2.candidate.direct-speed.v1\",\n"
           << "  \"device_pci_bus_id\": \"" << options.pci_bus_id << "\",\n"
           << "  \"iterations\": " << options.iterations << ",\n"
           << "  \"numerical_gate\": \"not_evaluated_v0.2_reference_unavailable\",\n"
           << "  \"results\": [\n";
    for (std::size_t index = 0u; index < results.size(); ++index) {
        const Result& result = results[index];
        const double gbps = static_cast<double>(result.payload_bytes) / static_cast<double>(result.milliseconds) / 1.0e6;
        output << "    {\"kernel\":\"" << result.kernel << "\",\"payload_bytes\":" << result.payload_bytes
               << ",\"average_ms\":" << std::fixed << std::setprecision(6) << result.milliseconds
               << ",\"payload_gbps\":" << std::fixed << std::setprecision(3) << gbps << "}"
               << (index + 1u == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

int run(const Options& options) {
    HIP_CHECK(hipSetDevice(select_gfx1201(options.pci_bus_id)));
    std::uint32_t random_state = 0x7309a5e1u;
    const std::vector<Result> results{
        bench_single(options.iterations, random_state),
        bench_batch(options.iterations, random_state),
        bench_pair(options.iterations, random_state),
        bench_triple(options.iterations, random_state),
    };
    write_results(options, results);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
