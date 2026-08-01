// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Direct HIPRTC differential for the four generic SQ8_0 matvec entry points.
// It intentionally compiles two source snapshots and changes no runtime
// dispatch, served-model state, service, candidate, campaign, or authorization.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

#define HIP_CHECK(call) do { \
    const hipError_t status__ = (call); \
    if (status__ != hipSuccess) throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(status__)); \
} while (false)

#define HIPRTC_CHECK(call) do { \
    const hiprtcResult status__ = (call); \
    if (status__ != HIPRTC_SUCCESS) throw std::runtime_error(std::string(#call) + ": " + hiprtcGetErrorString(status__)); \
} while (false)

constexpr unsigned int kThreads = 256u;
constexpr std::size_t kCols = 5120u;

struct Options {
    std::filesystem::path baseline_source;
    std::filesystem::path candidate_source;
    std::filesystem::path output;
    std::string pci_bus_id;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        HIP_CHECK(hipMalloc(&ptr_, bytes));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0u;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0u;
        }
        return *this;
    }
    ~DeviceBuffer() { reset(); }
    void* get() const { return ptr_; }

private:
    void reset() noexcept {
        if (ptr_ != nullptr) (void)hipFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0u;
    }
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0u;
};

class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&& other) noexcept : module_(other.module_) { other.module_ = nullptr; }
    Module& operator=(Module&& other) noexcept {
        if (this != &other) {
            reset();
            module_ = other.module_;
            other.module_ = nullptr;
        }
        return *this;
    }
    ~Module() { reset(); }
    hipModule_t get() const { return module_; }
    void load(const std::vector<char>& code) { HIP_CHECK(hipModuleLoadData(&module_, code.data())); }

private:
    void reset() noexcept {
        if (module_ != nullptr) (void)hipModuleUnload(module_);
        module_ = nullptr;
    }
    hipModule_t module_ = nullptr;
};

struct CaseResult {
    std::string kernel;
    std::string scale_mode;
    std::size_t elements = 0u;
    std::size_t mismatches = 0u;
    std::uint64_t baseline_hash = 0u;
    std::uint64_t candidate_hash = 0u;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --baseline-source /absolute/path --candidate-source /absolute/path"
              << " --output /absolute/path --pci-bus-id 0000:47:00.0\n";
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
        if (argument == "--baseline-source") options.baseline_source = need();
        else if (argument == "--candidate-source") options.candidate_source = need();
        else if (argument == "--output") options.output = need();
        else if (argument == "--pci-bus-id") options.pci_bus_id = need();
        else usage(argv[0]);
    }
    if (options.baseline_source.empty() || options.candidate_source.empty() ||
        options.output.empty() || options.pci_bus_id.empty() ||
        !options.baseline_source.is_absolute() || !options.candidate_source.is_absolute() ||
        !options.output.is_absolute()) {
        usage(argv[0]);
    }
    return options;
}

std::string read_source(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    constexpr std::string_view kHostInclude = "#include <hip/hip_runtime.h>\n";
    if (source.rfind(kHostInclude, 0u) == 0u) source.erase(0u, kHostInclude.size());
    return source;
}

int select_device(const std::string& expected_bdf) {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    for (int device = 0; device < count; ++device) {
        std::array<char, 32> bdf{};
        HIP_CHECK(hipDeviceGetPCIBusId(bdf.data(), static_cast<int>(bdf.size()), device));
        if (expected_bdf == bdf.data()) return device;
    }
    throw std::runtime_error("requested PCI BDF is not a HIP device: " + expected_bdf);
}

Module compile_module(const std::string& source, const char* name) {
    hiprtcProgram program{};
    HIPRTC_CHECK(hiprtcCreateProgram(&program, source.c_str(), name, 0, nullptr, nullptr));
    const char* options[] = {"--offload-arch=gfx1201", "--std=c++17", "-O3"};
    const hiprtcResult compilation = hiprtcCompileProgram(program, 3, options);
    std::size_t log_size = 0u;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
    std::string log(log_size, '\0');
    if (log_size != 0u) HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
    if (compilation != HIPRTC_SUCCESS) {
        (void)hiprtcDestroyProgram(&program);
        throw std::runtime_error(std::string("HIPRTC compile failed for ") + name + ":\n" + log);
    }
    std::size_t code_size = 0u;
    HIPRTC_CHECK(hiprtcGetCodeSize(program, &code_size));
    std::vector<char> code(code_size);
    HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    Module module;
    module.load(code);
    return module;
}

std::uint32_t next_random(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

std::vector<std::uint8_t> make_payload(std::size_t rows, std::size_t cols, std::uint32_t& state) {
    std::vector<std::uint8_t> values(rows * cols);
    for (std::uint8_t& value : values) {
        value = static_cast<std::uint8_t>(next_random(state) & 0x7eu);
    }
    return values;
}

std::vector<float> make_floats(std::size_t count, std::uint32_t& state) {
    std::vector<float> values(count);
    for (float& value : values) {
        const int signed_value = static_cast<int>(next_random(state) % 8192u) - 4096;
        value = static_cast<float>(signed_value) * (1.0f / 4096.0f);
    }
    return values;
}

std::size_t scale_count_block2d(std::size_t rows, std::size_t cols, unsigned int kind,
                                std::size_t block_rows, std::size_t block_cols) {
    if (kind == 0u) return 1u;
    if (kind == 1u) return rows;
    return ((rows + block_rows - 1u) / block_rows) * ((cols + block_cols - 1u) / block_cols);
}

std::size_t scale_count_row_block(std::size_t rows, std::size_t cols, unsigned int kind,
                                  std::size_t block_cols) {
    if (kind == 0u) return 1u;
    if (kind == 1u) return rows;
    return rows * ((cols + block_cols - 1u) / block_cols);
}

DeviceBuffer copy_to_device(const void* source, std::size_t bytes) {
    DeviceBuffer result(bytes);
    HIP_CHECK(hipMemcpy(result.get(), source, bytes, hipMemcpyHostToDevice));
    return result;
}

std::vector<float> copy_from_device(const DeviceBuffer& source, std::size_t count) {
    std::vector<float> result(count);
    HIP_CHECK(hipMemcpy(result.data(), source.get(), count * sizeof(float), hipMemcpyDeviceToHost));
    return result;
}

std::uint64_t hash_bytes(const std::vector<float>& values) {
    constexpr std::uint64_t kOffset = 1469598103934665603ull;
    constexpr std::uint64_t kPrime = 1099511628211ull;
    std::uint64_t hash = kOffset;
    for (float value : values) {
        std::uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int byte = 0u; byte < 4u; ++byte) {
            hash ^= static_cast<std::uint8_t>(bits >> (byte * 8u));
            hash *= kPrime;
        }
    }
    return hash;
}

std::size_t mismatch_count(const std::vector<float>& baseline, const std::vector<float>& candidate) {
    if (baseline.size() != candidate.size()) throw std::runtime_error("output-size mismatch");
    std::size_t mismatches = 0u;
    for (std::size_t index = 0u; index < baseline.size(); ++index) {
        std::uint32_t lhs = 0u;
        std::uint32_t rhs = 0u;
        std::memcpy(&lhs, &baseline[index], sizeof(lhs));
        std::memcpy(&rhs, &candidate[index], sizeof(rhs));
        mismatches += lhs != rhs ? 1u : 0u;
    }
    return mismatches;
}

hipFunction_t function(hipModule_t module, const char* symbol) {
    hipFunction_t result{};
    HIP_CHECK(hipModuleGetFunction(&result, module, symbol));
    return result;
}

std::vector<float> launch_single(hipFunction_t kernel, const std::vector<std::uint8_t>& payload,
                                 const std::vector<float>& scales, const std::vector<float>& input,
                                 std::size_t rows, unsigned int kind, std::size_t block_rows,
                                 std::size_t block_cols) {
    DeviceBuffer d_payload = copy_to_device(payload.data(), payload.size());
    DeviceBuffer d_scales = copy_to_device(scales.data(), scales.size() * sizeof(float));
    DeviceBuffer d_input = copy_to_device(input.data(), input.size() * sizeof(float));
    DeviceBuffer d_output(rows * sizeof(float));
    void* payload_arg = d_payload.get();
    void* scales_arg = d_scales.get();
    void* input_arg = d_input.get();
    unsigned long long rows_arg = rows;
    unsigned long long cols_arg = kCols;
    unsigned int kind_arg = kind;
    unsigned long long block_rows_arg = block_rows;
    unsigned long long block_cols_arg = block_cols;
    void* output_arg = d_output.get();
    void* args[] = {&payload_arg, &scales_arg, &input_arg, &rows_arg, &cols_arg, &kind_arg,
                    &block_rows_arg, &block_cols_arg, &output_arg};
    HIP_CHECK(hipModuleLaunchKernel(kernel, static_cast<unsigned int>(rows), 1u, 1u,
                                    kThreads, 1u, 1u, 0u, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    return copy_from_device(d_output, rows);
}

std::vector<float> launch_batch(hipFunction_t kernel, const std::vector<std::uint8_t>& payload,
                                const std::vector<float>& scales, const std::vector<float>& input,
                                std::size_t rows, std::size_t batch_count, unsigned int kind,
                                std::size_t block_rows, std::size_t block_cols) {
    DeviceBuffer d_payload = copy_to_device(payload.data(), payload.size());
    DeviceBuffer d_scales = copy_to_device(scales.data(), scales.size() * sizeof(float));
    DeviceBuffer d_input = copy_to_device(input.data(), input.size() * sizeof(float));
    DeviceBuffer d_output(rows * batch_count * sizeof(float));
    void* payload_arg = d_payload.get();
    void* scales_arg = d_scales.get();
    void* input_arg = d_input.get();
    unsigned long long rows_arg = rows;
    unsigned long long cols_arg = kCols;
    unsigned int kind_arg = kind;
    unsigned long long block_rows_arg = block_rows;
    unsigned long long block_cols_arg = block_cols;
    unsigned long long batch_arg = batch_count;
    void* output_arg = d_output.get();
    void* args[] = {&payload_arg, &scales_arg, &input_arg, &rows_arg, &cols_arg, &kind_arg,
                    &block_rows_arg, &block_cols_arg, &batch_arg, &output_arg};
    HIP_CHECK(hipModuleLaunchKernel(kernel, static_cast<unsigned int>(rows),
                                    static_cast<unsigned int>(batch_count), 1u,
                                    kThreads, 1u, 1u, 0u, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    return copy_from_device(d_output, rows * batch_count);
}

std::vector<float> launch_pair(hipFunction_t kernel, const std::vector<std::uint8_t>& left_payload,
                               const std::vector<float>& left_scales, std::size_t left_rows,
                               const std::vector<std::uint8_t>& right_payload,
                               const std::vector<float>& right_scales, std::size_t right_rows,
                               const std::vector<float>& input, unsigned int kind,
                               std::size_t block_cols) {
    DeviceBuffer d_left_payload = copy_to_device(left_payload.data(), left_payload.size());
    DeviceBuffer d_left_scales = copy_to_device(left_scales.data(), left_scales.size() * sizeof(float));
    DeviceBuffer d_right_payload = copy_to_device(right_payload.data(), right_payload.size());
    DeviceBuffer d_right_scales = copy_to_device(right_scales.data(), right_scales.size() * sizeof(float));
    DeviceBuffer d_input = copy_to_device(input.data(), input.size() * sizeof(float));
    DeviceBuffer d_left_output(left_rows * sizeof(float));
    DeviceBuffer d_right_output(right_rows * sizeof(float));
    void* left_payload_arg = d_left_payload.get();
    void* left_scales_arg = d_left_scales.get();
    unsigned long long left_rows_arg = left_rows;
    unsigned int left_kind_arg = kind;
    unsigned long long left_block_cols_arg = block_cols;
    void* right_payload_arg = d_right_payload.get();
    void* right_scales_arg = d_right_scales.get();
    unsigned long long right_rows_arg = right_rows;
    unsigned int right_kind_arg = kind;
    unsigned long long right_block_cols_arg = block_cols;
    void* input_arg = d_input.get();
    unsigned long long cols_arg = kCols;
    void* left_output_arg = d_left_output.get();
    void* right_output_arg = d_right_output.get();
    void* args[] = {&left_payload_arg, &left_scales_arg, &left_rows_arg, &left_kind_arg,
                    &left_block_cols_arg, &right_payload_arg, &right_scales_arg, &right_rows_arg,
                    &right_kind_arg, &right_block_cols_arg, &input_arg, &cols_arg,
                    &left_output_arg, &right_output_arg};
    const std::size_t max_rows = std::max(left_rows, right_rows);
    HIP_CHECK(hipModuleLaunchKernel(kernel, static_cast<unsigned int>(max_rows), 2u, 1u,
                                    kThreads, 1u, 1u, 0u, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> result = copy_from_device(d_left_output, left_rows);
    const std::vector<float> right = copy_from_device(d_right_output, right_rows);
    result.insert(result.end(), right.begin(), right.end());
    return result;
}

std::vector<float> launch_triple(hipFunction_t kernel, const std::array<std::vector<std::uint8_t>, 3>& payloads,
                                 const std::array<std::vector<float>, 3>& scales,
                                 const std::array<std::size_t, 3>& rows, const std::vector<float>& input,
                                 unsigned int kind, std::size_t block_cols) {
    std::array<DeviceBuffer, 3> d_payloads = {
        copy_to_device(payloads[0].data(), payloads[0].size()),
        copy_to_device(payloads[1].data(), payloads[1].size()),
        copy_to_device(payloads[2].data(), payloads[2].size()),
    };
    std::array<DeviceBuffer, 3> d_scales = {
        copy_to_device(scales[0].data(), scales[0].size() * sizeof(float)),
        copy_to_device(scales[1].data(), scales[1].size() * sizeof(float)),
        copy_to_device(scales[2].data(), scales[2].size() * sizeof(float)),
    };
    DeviceBuffer d_input = copy_to_device(input.data(), input.size() * sizeof(float));
    std::array<DeviceBuffer, 3> d_outputs = {
        DeviceBuffer(rows[0] * sizeof(float)), DeviceBuffer(rows[1] * sizeof(float)),
        DeviceBuffer(rows[2] * sizeof(float)),
    };
    void* first_payload_arg = d_payloads[0].get();
    void* first_scales_arg = d_scales[0].get();
    unsigned long long first_rows_arg = rows[0];
    unsigned int first_kind_arg = kind;
    unsigned long long first_block_cols_arg = block_cols;
    void* second_payload_arg = d_payloads[1].get();
    void* second_scales_arg = d_scales[1].get();
    unsigned long long second_rows_arg = rows[1];
    unsigned int second_kind_arg = kind;
    unsigned long long second_block_cols_arg = block_cols;
    void* third_payload_arg = d_payloads[2].get();
    void* third_scales_arg = d_scales[2].get();
    unsigned long long third_rows_arg = rows[2];
    unsigned int third_kind_arg = kind;
    unsigned long long third_block_cols_arg = block_cols;
    void* input_arg = d_input.get();
    unsigned long long cols_arg = kCols;
    void* first_output_arg = d_outputs[0].get();
    void* second_output_arg = d_outputs[1].get();
    void* third_output_arg = d_outputs[2].get();
    void* args[] = {&first_payload_arg, &first_scales_arg, &first_rows_arg, &first_kind_arg,
                    &first_block_cols_arg, &second_payload_arg, &second_scales_arg, &second_rows_arg,
                    &second_kind_arg, &second_block_cols_arg, &third_payload_arg, &third_scales_arg,
                    &third_rows_arg, &third_kind_arg, &third_block_cols_arg, &input_arg, &cols_arg,
                    &first_output_arg, &second_output_arg, &third_output_arg};
    const std::size_t max_rows = std::max({rows[0], rows[1], rows[2]});
    HIP_CHECK(hipModuleLaunchKernel(kernel, static_cast<unsigned int>(max_rows), 3u, 1u,
                                    kThreads, 1u, 1u, 0u, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> result = copy_from_device(d_outputs[0], rows[0]);
    for (unsigned int index = 1u; index < 3u; ++index) {
        const std::vector<float> part = copy_from_device(d_outputs[index], rows[index]);
        result.insert(result.end(), part.begin(), part.end());
    }
    return result;
}

CaseResult compare_case(const std::string& kernel_name, const std::string& scale_mode,
                        const std::vector<float>& baseline, const std::vector<float>& candidate) {
    return CaseResult{kernel_name, scale_mode, baseline.size(), mismatch_count(baseline, candidate),
                      hash_bytes(baseline), hash_bytes(candidate)};
}

void write_results(const Options& options, const std::vector<CaseResult>& results) {
    std::filesystem::create_directories(options.output.parent_path());
    std::ofstream output(options.output);
    if (!output) throw std::runtime_error("cannot write " + options.output.string());
    const bool passed = std::all_of(results.begin(), results.end(),
        [](const CaseResult& result) { return result.mismatches == 0u; });
    output << "{\n  \"schema_version\": \"ullm.sq8_0.phase1.bitwise.v1\",\n"
           << "  \"device_pci_bus_id\": \"" << options.pci_bus_id << "\",\n"
           << "  \"baseline_source\": \"" << options.baseline_source.string() << "\",\n"
           << "  \"candidate_source\": \"" << options.candidate_source.string() << "\",\n"
           << "  \"passed\": " << (passed ? "true" : "false") << ",\n  \"cases\": [\n";
    for (std::size_t index = 0u; index < results.size(); ++index) {
        const CaseResult& result = results[index];
        output << "    {\"kernel\":\"" << result.kernel << "\",\"scale_mode\":\""
               << result.scale_mode << "\",\"elements\":" << result.elements
               << ",\"bit_mismatches\":" << result.mismatches << ",\"baseline_fnv1a64\":\"0x"
               << std::hex << result.baseline_hash << "\",\"candidate_fnv1a64\":\"0x"
               << result.candidate_hash << std::dec << "\"}";
        output << (index + 1u == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

int run(const Options& options) {
    const int device = select_device(options.pci_bus_id);
    HIP_CHECK(hipSetDevice(device));
    const Module baseline_module = compile_module(read_source(options.baseline_source), "sq8_phase1_baseline");
    const Module candidate_module = compile_module(read_source(options.candidate_source), "sq8_phase1_candidate");
    const hipFunction_t baseline_single = function(baseline_module.get(), "ullm_sq_fp8_matvec_f32_kernel");
    const hipFunction_t candidate_single = function(candidate_module.get(), "ullm_sq_fp8_matvec_f32_kernel");
    const hipFunction_t baseline_batch = function(baseline_module.get(), "ullm_sq_fp8_matvec_batch_f32_kernel");
    const hipFunction_t candidate_batch = function(candidate_module.get(), "ullm_sq_fp8_matvec_batch_f32_kernel");
    const hipFunction_t baseline_pair = function(baseline_module.get(), "ullm_sq_fp8_matvec_pair_f32_kernel");
    const hipFunction_t candidate_pair = function(candidate_module.get(), "ullm_sq_fp8_matvec_pair_f32_kernel");
    const hipFunction_t baseline_triple = function(baseline_module.get(), "ullm_sq_fp8_matvec_triple_f32_kernel");
    const hipFunction_t candidate_triple = function(candidate_module.get(), "ullm_sq_fp8_matvec_triple_f32_kernel");

    constexpr std::size_t kSingleRows = 257u;
    constexpr std::size_t kLeftRows = 257u;
    constexpr std::size_t kRightRows = 129u;
    constexpr std::array<std::size_t, 3> kTripleRows{257u, 129u, 9u};
    std::uint32_t random_state = 0x51a8e3d1u;
    const std::vector<float> input = make_floats(kCols * 3u, random_state);
    std::vector<CaseResult> results;

    struct ScaleMode { const char* name; unsigned int kind; std::size_t block_rows; std::size_t block_cols; };
    const std::array<ScaleMode, 4> modes{{
        {"scalar", 0u, 1u, 1u},
        {"row", 1u, 1u, 1u},
        {"block17_fallback", 2u, 17u, 17u},
        {"block128_phase1", 2u, 128u, 128u},
    }};
    for (const ScaleMode& mode : modes) {
        const std::vector<std::uint8_t> single_payload = make_payload(kSingleRows, kCols, random_state);
        const std::vector<float> single_scales = make_floats(
            scale_count_block2d(kSingleRows, kCols, mode.kind, mode.block_rows, mode.block_cols), random_state);
        results.push_back(compare_case("single", mode.name,
            launch_single(baseline_single, single_payload, single_scales, input, kSingleRows,
                          mode.kind, mode.block_rows, mode.block_cols),
            launch_single(candidate_single, single_payload, single_scales, input, kSingleRows,
                          mode.kind, mode.block_rows, mode.block_cols)));
        results.push_back(compare_case("batch", mode.name,
            launch_batch(baseline_batch, single_payload, single_scales, input, kSingleRows, 3u,
                         mode.kind, mode.block_rows, mode.block_cols),
            launch_batch(candidate_batch, single_payload, single_scales, input, kSingleRows, 3u,
                         mode.kind, mode.block_rows, mode.block_cols)));

        const std::vector<std::uint8_t> left_payload = make_payload(kLeftRows, kCols, random_state);
        const std::vector<std::uint8_t> right_payload = make_payload(kRightRows, kCols, random_state);
        const std::vector<float> left_scales = make_floats(
            scale_count_row_block(kLeftRows, kCols, mode.kind, mode.block_cols), random_state);
        const std::vector<float> right_scales = make_floats(
            scale_count_row_block(kRightRows, kCols, mode.kind, mode.block_cols), random_state);
        results.push_back(compare_case("pair", mode.name,
            launch_pair(baseline_pair, left_payload, left_scales, kLeftRows, right_payload, right_scales,
                        kRightRows, input, mode.kind, mode.block_cols),
            launch_pair(candidate_pair, left_payload, left_scales, kLeftRows, right_payload, right_scales,
                        kRightRows, input, mode.kind, mode.block_cols)));

        std::array<std::vector<std::uint8_t>, 3> triple_payloads{
            make_payload(kTripleRows[0], kCols, random_state),
            make_payload(kTripleRows[1], kCols, random_state),
            make_payload(kTripleRows[2], kCols, random_state),
        };
        std::array<std::vector<float>, 3> triple_scales{
            make_floats(scale_count_row_block(kTripleRows[0], kCols, mode.kind, mode.block_cols), random_state),
            make_floats(scale_count_row_block(kTripleRows[1], kCols, mode.kind, mode.block_cols), random_state),
            make_floats(scale_count_row_block(kTripleRows[2], kCols, mode.kind, mode.block_cols), random_state),
        };
        results.push_back(compare_case("triple", mode.name,
            launch_triple(baseline_triple, triple_payloads, triple_scales, kTripleRows, input,
                          mode.kind, mode.block_cols),
            launch_triple(candidate_triple, triple_payloads, triple_scales, kTripleRows, input,
                          mode.kind, mode.block_cols)));
    }
    write_results(options, results);
    return std::all_of(results.begin(), results.end(),
        [](const CaseResult& result) { return result.mismatches == 0u; }) ? 0 : 1;
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
