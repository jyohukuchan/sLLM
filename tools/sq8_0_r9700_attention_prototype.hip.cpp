// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Isolated SQ8_0 R9700 attention experiments. This binary deliberately uses
// HIPRTC modules and symbols that are separate from the runtime symbols. It is
// not a serving binary and refuses to run on any architecture other than
// gfx1201.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include <utility>
#include <vector>

namespace {

class ExitError : public std::runtime_error {
  public:
    ExitError(int code, std::string message) : std::runtime_error(std::move(message)), code_(code) {}
    int code() const { return code_; }

  private:
    int code_;
};

void hip_check(hipError_t status, std::string_view expression, const char* file, int line) {
    if (status == hipSuccess) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": "
            << hipGetErrorString(status) << " (" << static_cast<int>(status) << ')';
    throw std::runtime_error(message.str());
}

#define HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

void hiprtc_check(hiprtcResult status, std::string_view expression, const char* file, int line) {
    if (status == HIPRTC_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": "
            << hiprtcGetErrorString(status) << " (" << static_cast<int>(status) << ')';
    throw std::runtime_error(message.str());
}

#define HIPRTC_CHECK(expression) hiprtc_check((expression), #expression, __FILE__, __LINE__)

void cleanup_hip(hipError_t status, std::string_view operation) noexcept {
    if (status != hipSuccess) {
        std::cerr << operation << " during cleanup failed: " << hipGetErrorString(status) << '\n';
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes == 0) {
            throw std::runtime_error("zero-byte device allocation requested");
        }
        HIP_CHECK(hipMalloc(&pointer_, bytes));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cleanup_hip(hipFree(pointer_), "hipFree");
        }
    }

    void* get() { return pointer_; }
    const void* get() const { return pointer_; }
    std::size_t bytes() const { return bytes_; }

  private:
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0;
};

class HipStream {
  public:
    HipStream() { HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking)); }
    HipStream(const HipStream&) = delete;
    HipStream& operator=(const HipStream&) = delete;

    ~HipStream() {
        if (stream_ != nullptr) {
            cleanup_hip(hipStreamDestroy(stream_), "hipStreamDestroy");
        }
    }

    hipStream_t get() const { return stream_; }

  private:
    hipStream_t stream_ = nullptr;
};

class HipEvent {
  public:
    HipEvent() { HIP_CHECK(hipEventCreate(&event_)); }
    HipEvent(const HipEvent&) = delete;
    HipEvent& operator=(const HipEvent&) = delete;

    ~HipEvent() {
        if (event_ != nullptr) {
            cleanup_hip(hipEventDestroy(event_), "hipEventDestroy");
        }
    }

    hipEvent_t get() const { return event_; }

  private:
    hipEvent_t event_ = nullptr;
};

class HipRtcProgram {
  public:
    HipRtcProgram(const char* source, const char* name) {
        HIPRTC_CHECK(hiprtcCreateProgram(&program_, source, name, 0, nullptr, nullptr));
    }

    HipRtcProgram(const HipRtcProgram&) = delete;
    HipRtcProgram& operator=(const HipRtcProgram&) = delete;

    ~HipRtcProgram() {
        if (program_ != nullptr) {
            const hiprtcResult status = hiprtcDestroyProgram(&program_);
            if (status != HIPRTC_SUCCESS) {
                std::cerr << "hiprtcDestroyProgram during cleanup failed: "
                          << hiprtcGetErrorString(status) << '\n';
            }
        }
    }

    hiprtcProgram get() const { return program_; }

  private:
    hiprtcProgram program_ = nullptr;
};

class HipModule {
  public:
    explicit HipModule(const std::vector<char>& code) { HIP_CHECK(hipModuleLoadData(&module_, code.data())); }
    HipModule(const HipModule&) = delete;
    HipModule& operator=(const HipModule&) = delete;

    ~HipModule() {
        if (module_ != nullptr) {
            cleanup_hip(hipModuleUnload(module_), "hipModuleUnload");
        }
    }

    hipModule_t get() const { return module_; }

  private:
    hipModule_t module_ = nullptr;
};

enum class Mode {
    Compile,
    Flash2,
    PmcProbe,
    All,
};

struct Options {
    Mode mode = Mode::All;
    std::filesystem::path output_dir;
    int device = 0;
    int warmups = 3;
    int repeats = 20;
    std::size_t pmc_elements = 64ULL * 1024ULL * 1024ULL;
    int pmc_launches = 16;
};

struct Flash2Shape {
    std::string name;
    std::size_t cached_prefix_len;
    std::size_t new_tokens;
    std::size_t q_heads = 40;
    std::size_t kv_heads = 8;
    std::size_t head_dim = 128;
    std::size_t value_dim = 128;
    float amplitude = 1.0f;
};

struct Diff {
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t max_abs_index = 0;
    std::size_t nan_or_inf_count = 0;
};

struct Timing {
    float milliseconds = 0.0f;
    double launches_per_second = 0.0;
};

struct Flash2CaseResult {
    Flash2Shape shape;
    Diff legacy_vs_cpu;
    Diff qk_vs_legacy;
    Diff qk_max_vs_legacy;
    Diff all_staged_vs_legacy;
};

constexpr const char* kKernelSource = R"HIP(
__device__ __forceinline__ float ullm_sq8_0_reduce_sum_tree(float value, float* partial) {
    const unsigned int tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

__device__ __forceinline__ float ullm_sq8_0_reduce_max_tree(float value, float* partial) {
    const unsigned int tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = partial[tid] > partial[tid + stride] ? partial[tid] : partial[tid + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

__device__ __forceinline__ float ullm_sq8_0_reduce_sum_wave32(float value, float* partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5u;
    const unsigned int wave_count = (blockDim.x + 31u) >> 5u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    if (lane == 0u) {
        partial[wave] = value;
    }
    __syncthreads();
    if (wave == 0u) {
        value = lane < wave_count ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down(value, offset, 32);
        }
        if (lane == 0u) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float ullm_sq8_0_reduce_max_wave32(float value, float* partial) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5u;
    const unsigned int wave_count = (blockDim.x + 31u) >> 5u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down(value, offset, 32);
        value = value > other ? value : other;
    }
    if (lane == 0u) {
        partial[wave] = value;
    }
    __syncthreads();
    if (wave == 0u) {
        value = lane < wave_count ? partial[lane] : -3.4028234663852886e38f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down(value, offset, 32);
            value = value > other ? value : other;
        }
        if (lane == 0u) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

template <int Stage>
__device__ __forceinline__ float ullm_sq8_0_reduce_sum(float value, float* partial) {
    if constexpr (Stage == 3) {
        return ullm_sq8_0_reduce_sum_wave32(value, partial);
    }
    return ullm_sq8_0_reduce_sum_tree(value, partial);
}

template <int Stage>
__device__ __forceinline__ float ullm_sq8_0_reduce_qk(float value, float* partial) {
    if constexpr (Stage >= 1) {
        return ullm_sq8_0_reduce_sum_wave32(value, partial);
    }
    return ullm_sq8_0_reduce_sum_tree(value, partial);
}

template <int Stage>
__device__ __forceinline__ float ullm_sq8_0_reduce_max(float value, float* partial) {
    if constexpr (Stage >= 2) {
        return ullm_sq8_0_reduce_max_wave32(value, partial);
    }
    return ullm_sq8_0_reduce_max_tree(value, partial);
}

template <int Stage>
__device__ void ullm_sq8_0_flash2_body(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    unsigned long long cached_prefix_len,
    unsigned long long new_tokens,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float* output) {
    constexpr unsigned int kTileTokens = 64u;
    const unsigned long long q_head_index = (unsigned long long) blockIdx.x;
    const unsigned long long q_head_elements = new_tokens * q_heads;
    if (q_head_index >= q_head_elements) return;
    const unsigned int tid = threadIdx.x;
    if (value_dim > (unsigned long long) blockDim.x) return;
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long token_index = q_head_index / q_heads;
    const unsigned long long cache_len = cached_prefix_len + token_index + 1ull;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (token_index * q_heads + q_head) * head_dim;
    const unsigned long long value = (unsigned long long) tid;
    const bool value_active = value < value_dim;

    __shared__ float reduce[256];
    __shared__ float scores[kTileTokens];
    __shared__ float shared_tile_max;
    __shared__ float shared_tile_sum;
    __shared__ float shared_m_new;
    __shared__ float shared_alpha;
    float max_score = -3.4028234663852886e38f;
    float denominator = 0.0f;
    float weighted = 0.0f;

    for (unsigned long long tile_start = 0; tile_start < cache_len; tile_start += kTileTokens) {
        const unsigned long long remaining = cache_len - tile_start;
        const unsigned int tile_count = remaining < (unsigned long long) kTileTokens
            ? (unsigned int) remaining : kTileTokens;
        for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            const unsigned long long source_timestep = tile_start + (unsigned long long) tile_offset;
            const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
                partial += q[q_base + dim] * k_cache[k_base + dim];
            }
            const float score = ullm_sq8_0_reduce_qk<Stage>(partial, reduce) * softmax_scale;
            if (tid == 0u) scores[tile_offset] = score;
            __syncthreads();
        }

        float local_max = -3.4028234663852886e38f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float score = scores[tile_offset];
            local_max = score > local_max ? score : local_max;
        }
        const float tile_max = ullm_sq8_0_reduce_max<Stage>(local_max, reduce);
        if (tid == 0u) {
            shared_tile_max = tile_max;
            const float m_new = max_score > tile_max ? max_score : tile_max;
            shared_m_new = m_new;
            shared_alpha = max_score <= -3.0e38f ? 0.0f : expf(max_score - m_new);
        }
        __syncthreads();

        const float m_new = shared_m_new;
        float local_sum = 0.0f;
        for (unsigned int tile_offset = tid; tile_offset < tile_count; tile_offset += blockDim.x) {
            const float weight = expf(scores[tile_offset] - m_new);
            scores[tile_offset] = weight;
            local_sum += weight;
        }
        const float tile_sum = ullm_sq8_0_reduce_sum<Stage>(local_sum, reduce);
        if (tid == 0u) shared_tile_sum = tile_sum;
        __syncthreads();

        if (value_active) {
            float tile_weighted = 0.0f;
            for (unsigned int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const unsigned long long source_timestep = tile_start + (unsigned long long) tile_offset;
                const unsigned long long v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                tile_weighted += scores[tile_offset] * v_cache[v_index];
            }
            weighted = weighted * shared_alpha + tile_weighted;
        }
        denominator = denominator * shared_alpha + shared_tile_sum;
        max_score = m_new;
        __syncthreads();
    }
    if (value_active) output[q_head_index * value_dim + value] = weighted / denominator;
}

extern "C" __global__ void ullm_sq8_0_flash2_legacy_reference_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<0>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_flash2_qk_wave32_prototype_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<1>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<2>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_flash2_staged_wave32_prototype_kernel(
    const float* q, const float* k, const float* v, unsigned long long cached_prefix_len,
    unsigned long long new_tokens, unsigned long long q_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long value_dim, float softmax_scale, float* output) {
    ullm_sq8_0_flash2_body<3>(q, k, v, cached_prefix_len, new_tokens, q_heads, kv_heads,
                               head_dim, value_dim, softmax_scale, output);
}

extern "C" __global__ void ullm_sq8_0_pmc_probe_kernel(
    const float* input, float* output, unsigned long long count) {
    const unsigned long long tid = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long) gridDim.x * blockDim.x;
    float value = 0.0f;
    for (unsigned long long index = tid; index < count; index += stride) {
        const float loaded = ((volatile const float*) input)[index];
        value = fmaf(value, 1.000123f, loaded);
        value = fmaf(value, 0.999877f, loaded * 0.5f);
        value = fmaf(value, 1.000031f, loaded * 0.25f);
        value = fmaf(value, 0.999969f, loaded * 0.125f);
    }
    if (tid < count) output[tid] = value;
}
)HIP";

std::string mode_name(Mode mode) {
    switch (mode) {
        case Mode::Compile: return "compile";
        case Mode::Flash2: return "flash2";
        case Mode::PmcProbe: return "pmc-probe";
        case Mode::All: return "all";
    }
    return "unknown";
}

Mode parse_mode(std::string_view value) {
    if (value == "compile") return Mode::Compile;
    if (value == "flash2") return Mode::Flash2;
    if (value == "pmc-probe") return Mode::PmcProbe;
    if (value == "all") return Mode::All;
    throw ExitError(2, "--mode must be compile, flash2, pmc-probe, or all");
}

std::size_t parse_size(std::string_view value, std::string_view flag) {
    std::size_t parsed = 0;
    try {
        const std::string text(value);
        std::size_t position = 0;
        parsed = static_cast<std::size_t>(std::stoull(text, &position));
        if (position != text.size()) throw std::invalid_argument("suffix");
    } catch (const std::exception&) {
        throw ExitError(2, std::string(flag) + " must be a non-negative integer");
    }
    return parsed;
}

int parse_int(std::string_view value, std::string_view flag) {
    const std::size_t parsed = parse_size(value, flag);
    if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw ExitError(2, std::string(flag) + " is too large");
    }
    return static_cast<int>(parsed);
}

[[noreturn]] void usage() {
    throw ExitError(
        2,
        "usage: sq8_0_r9700_attention_prototype --output-dir DIR "
        "[--mode compile|flash2|pmc-probe|all] [--device N] [--warmups N] [--repeats N] "
        "[--pmc-elements N] [--pmc-launches N]");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&]() -> std::string_view {
            if (index + 1 >= argc) usage();
            return argv[++index];
        };
        if (argument == "--output-dir") {
            options.output_dir = std::filesystem::path(next());
        } else if (argument == "--mode") {
            options.mode = parse_mode(next());
        } else if (argument == "--device") {
            options.device = parse_int(next(), "--device");
        } else if (argument == "--warmups") {
            options.warmups = parse_int(next(), "--warmups");
        } else if (argument == "--repeats") {
            options.repeats = parse_int(next(), "--repeats");
        } else if (argument == "--pmc-elements") {
            options.pmc_elements = parse_size(next(), "--pmc-elements");
        } else if (argument == "--pmc-launches") {
            options.pmc_launches = parse_int(next(), "--pmc-launches");
        } else if (argument == "--help" || argument == "-h") {
            usage();
        } else {
            throw ExitError(2, "unknown argument: " + std::string(argument));
        }
    }
    if (options.output_dir.empty()) usage();
    if (options.device < 0 || options.warmups < 0 || options.repeats <= 0 ||
        options.pmc_elements == 0 || options.pmc_launches <= 0) {
        throw ExitError(2, "device/repeat/count arguments must be positive (warmups may be zero)");
    }
    return options;
}

void write_text(const std::filesystem::path& path, std::string_view content) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("failed to write " + path.string());
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!output) throw std::runtime_error("failed while writing " + path.string());
}

void write_binary(const std::filesystem::path& path, const std::vector<char>& content) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("failed to write " + path.string());
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!output) throw std::runtime_error("failed while writing " + path.string());
}

std::vector<char> compile_module(std::filesystem::path output_dir) {
    HipRtcProgram program(kKernelSource, "sq8_0_r9700_attention_prototype.hip");
    const std::string architecture = "--gpu-architecture=gfx1201";
    const char* options[] = {"--std=c++17", "-O3", architecture.c_str()};
    const hiprtcResult status =
        hiprtcCompileProgram(program.get(), static_cast<int>(std::size(options)), options);
    std::size_t log_size = 0;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(program.get(), &log_size));
    std::string log(log_size, '\0');
    if (log_size != 0) HIPRTC_CHECK(hiprtcGetProgramLog(program.get(), log.data()));
    write_text(output_dir / "hiprtc-compile.log", log);
    if (status != HIPRTC_SUCCESS) {
        throw std::runtime_error("HIPRTC compilation failed; see hiprtc-compile.log: " + log);
    }
    std::size_t code_size = 0;
    HIPRTC_CHECK(hiprtcGetCodeSize(program.get(), &code_size));
    std::vector<char> code(code_size);
    HIPRTC_CHECK(hiprtcGetCode(program.get(), code.data()));
    write_binary(output_dir / "sq8_0_r9700_attention_prototype.hsaco", code);
    write_text(output_dir / "sq8_0_r9700_attention_prototype.hip", kKernelSource);
    return code;
}

void require_gfx1201(int device) {
    HIP_CHECK(hipSetDevice(device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    const std::string architecture(properties.gcnArchName);
    if (architecture.find("gfx1201") == std::string::npos) {
        throw ExitError(
            3,
            "refusing to execute on non-R9700 device " + std::to_string(device) +
                " (gcnArchName=" + architecture + ")");
    }
}

std::string device_json(int device) {
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::ostringstream output;
    output << "{\"runtime_device\":" << device << ",\"name\":\"" << properties.name
           << "\",\"gcn_arch_name\":\"" << properties.gcnArchName
           << "\",\"multi_processor_count\":" << properties.multiProcessorCount
           << ",\"warp_size\":" << properties.warpSize << '}';
    return output.str();
}

void* function_for(hipModule_t module, const char* name) {
    hipFunction_t function = nullptr;
    HIP_CHECK(hipModuleGetFunction(&function, module, name));
    return reinterpret_cast<void*>(function);
}

void launch_flash2(
    void* function_pointer,
    const Flash2Shape& shape,
    const DeviceBuffer& q,
    const DeviceBuffer& k,
    const DeviceBuffer& v,
    DeviceBuffer* output,
    hipStream_t stream) {
    auto cached = static_cast<unsigned long long>(shape.cached_prefix_len);
    auto new_tokens = static_cast<unsigned long long>(shape.new_tokens);
    auto q_heads = static_cast<unsigned long long>(shape.q_heads);
    auto kv_heads = static_cast<unsigned long long>(shape.kv_heads);
    auto head_dim = static_cast<unsigned long long>(shape.head_dim);
    auto value_dim = static_cast<unsigned long long>(shape.value_dim);
    float softmax_scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    auto* q_pointer = static_cast<const float*>(q.get());
    auto* k_pointer = static_cast<const float*>(k.get());
    auto* v_pointer = static_cast<const float*>(v.get());
    auto* output_pointer = static_cast<float*>(output->get());
    void* parameters[] = {
        &q_pointer, &k_pointer, &v_pointer, &cached, &new_tokens, &q_heads, &kv_heads,
        &head_dim, &value_dim, &softmax_scale, &output_pointer,
    };
    HIP_CHECK(hipModuleLaunchKernel(
        reinterpret_cast<hipFunction_t>(function_pointer),
        static_cast<unsigned int>(shape.new_tokens * shape.q_heads), 1, 1,
        256, 1, 1, 0, stream, parameters, nullptr));
}

std::vector<float> values_for(std::size_t count, float amplitude, std::uint32_t salt) {
    std::vector<float> values(count);
    std::uint32_t state = 0x9e3779b9u ^ salt;
    for (std::size_t index = 0; index < count; ++index) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float centered = static_cast<float>(state & 0xffffu) / 32767.5f - 1.0f;
        values[index] = amplitude * centered;
    }
    return values;
}

std::vector<float> cpu_flash2_reference(
    const Flash2Shape& shape,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v) {
    const std::size_t output_count = shape.new_tokens * shape.q_heads * shape.value_dim;
    std::vector<float> output(output_count);
    const double softmax_scale = 1.0 / std::sqrt(static_cast<double>(shape.head_dim));
    const std::size_t q_per_kv = shape.q_heads / shape.kv_heads;
    for (std::size_t token = 0; token < shape.new_tokens; ++token) {
        const std::size_t cache_len = shape.cached_prefix_len + token + 1;
        for (std::size_t q_head = 0; q_head < shape.q_heads; ++q_head) {
            const std::size_t kv_head = q_head / q_per_kv;
            std::vector<double> scores(cache_len);
            double max_score = -std::numeric_limits<double>::infinity();
            for (std::size_t source = 0; source < cache_len; ++source) {
                double dot = 0.0;
                const std::size_t q_base = (token * shape.q_heads + q_head) * shape.head_dim;
                const std::size_t k_base = (source * shape.kv_heads + kv_head) * shape.head_dim;
                for (std::size_t dim = 0; dim < shape.head_dim; ++dim) {
                    dot += static_cast<double>(q[q_base + dim]) * static_cast<double>(k[k_base + dim]);
                }
                scores[source] = dot * softmax_scale;
                max_score = std::max(max_score, scores[source]);
            }
            double denominator = 0.0;
            for (double score : scores) denominator += std::exp(score - max_score);
            for (std::size_t value = 0; value < shape.value_dim; ++value) {
                double numerator = 0.0;
                for (std::size_t source = 0; source < cache_len; ++source) {
                    const std::size_t v_index =
                        (source * shape.kv_heads + kv_head) * shape.value_dim + value;
                    numerator += std::exp(scores[source] - max_score) * static_cast<double>(v[v_index]);
                }
                output[(token * shape.q_heads + q_head) * shape.value_dim + value] =
                    static_cast<float>(numerator / denominator);
            }
        }
    }
    return output;
}

Diff compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("differential length mismatch");
    Diff result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double observed = actual[index];
        const double reference = expected[index];
        if (!std::isfinite(observed) || !std::isfinite(reference)) {
            ++result.nan_or_inf_count;
            continue;
        }
        const double absolute = std::abs(observed - reference);
        const double relative = absolute / std::max(1.0e-30, std::abs(reference));
        if (absolute > result.max_abs) {
            result.max_abs = absolute;
            result.max_abs_index = index;
        }
        result.max_rel = std::max(result.max_rel, relative);
    }
    return result;
}

std::vector<float> copy_output(const DeviceBuffer& buffer, std::size_t elements, hipStream_t stream) {
    std::vector<float> output(elements);
    HIP_CHECK(hipMemcpyAsync(
        output.data(), buffer.get(), elements * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return output;
}

Timing time_flash2(
    void* function,
    const Flash2Shape& shape,
    const DeviceBuffer& q,
    const DeviceBuffer& k,
    const DeviceBuffer& v,
    DeviceBuffer* output,
    hipStream_t stream,
    int warmups,
    int repeats) {
    for (int iteration = 0; iteration < warmups; ++iteration) {
        launch_flash2(function, shape, q, k, v, output, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HipEvent start;
    HipEvent end;
    HIP_CHECK(hipEventRecord(start.get(), stream));
    for (int iteration = 0; iteration < repeats; ++iteration) {
        launch_flash2(function, shape, q, k, v, output, stream);
    }
    HIP_CHECK(hipEventRecord(end.get(), stream));
    HIP_CHECK(hipEventSynchronize(end.get()));
    float milliseconds = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start.get(), end.get()));
    return Timing{
        .milliseconds = milliseconds / static_cast<float>(repeats),
        .launches_per_second = static_cast<double>(repeats) * 1000.0 / static_cast<double>(milliseconds),
    };
}

Flash2CaseResult run_flash2_case(
    hipModule_t module,
    const Flash2Shape& shape,
    hipStream_t stream) {
    const std::size_t context = shape.cached_prefix_len + shape.new_tokens;
    const std::size_t q_elements = shape.new_tokens * shape.q_heads * shape.head_dim;
    const std::size_t k_elements = context * shape.kv_heads * shape.head_dim;
    const std::size_t v_elements = context * shape.kv_heads * shape.value_dim;
    const std::size_t output_elements = shape.new_tokens * shape.q_heads * shape.value_dim;
    const auto q_values = values_for(q_elements, shape.amplitude, 0x1001u);
    const auto k_values = values_for(k_elements, shape.amplitude, 0x1002u);
    const auto v_values = values_for(v_elements, shape.amplitude, 0x1003u);
    const auto cpu = cpu_flash2_reference(shape, q_values, k_values, v_values);
    DeviceBuffer q(q_elements * sizeof(float));
    DeviceBuffer k(k_elements * sizeof(float));
    DeviceBuffer v(v_elements * sizeof(float));
    DeviceBuffer legacy_output(output_elements * sizeof(float));
    DeviceBuffer qk_output(output_elements * sizeof(float));
    DeviceBuffer qk_max_output(output_elements * sizeof(float));
    DeviceBuffer staged_output(output_elements * sizeof(float));
    HIP_CHECK(hipMemcpyAsync(q.get(), q_values.data(), q.bytes(), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(k.get(), k_values.data(), k.bytes(), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(v.get(), v_values.data(), v.bytes(), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const void* legacy = function_for(module, "ullm_sq8_0_flash2_legacy_reference_kernel");
    const void* qk = function_for(module, "ullm_sq8_0_flash2_qk_wave32_prototype_kernel");
    const void* qk_max = function_for(module, "ullm_sq8_0_flash2_qk_max_wave32_prototype_kernel");
    const void* staged = function_for(module, "ullm_sq8_0_flash2_staged_wave32_prototype_kernel");
    launch_flash2(const_cast<void*>(legacy), shape, q, k, v, &legacy_output, stream);
    launch_flash2(const_cast<void*>(qk), shape, q, k, v, &qk_output, stream);
    launch_flash2(const_cast<void*>(qk_max), shape, q, k, v, &qk_max_output, stream);
    launch_flash2(const_cast<void*>(staged), shape, q, k, v, &staged_output, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto legacy_values = copy_output(legacy_output, output_elements, stream);
    const auto qk_values = copy_output(qk_output, output_elements, stream);
    const auto qk_max_values = copy_output(qk_max_output, output_elements, stream);
    const auto staged_values = copy_output(staged_output, output_elements, stream);
    return Flash2CaseResult{
        .shape = shape,
        .legacy_vs_cpu = compare(legacy_values, cpu),
        .qk_vs_legacy = compare(qk_values, legacy_values),
        .qk_max_vs_legacy = compare(qk_max_values, legacy_values),
        .all_staged_vs_legacy = compare(staged_values, legacy_values),
    };
}

std::string diff_json(const Diff& diff) {
    std::ostringstream output;
    output << std::setprecision(12) << "{\"max_abs\":" << diff.max_abs
           << ",\"max_rel\":" << diff.max_rel << ",\"max_abs_index\":" << diff.max_abs_index
           << ",\"nan_or_inf_count\":" << diff.nan_or_inf_count << '}';
    return output.str();
}

std::string flash2_case_json(const Flash2CaseResult& result) {
    std::ostringstream output;
    output << "{\"name\":\"" << result.shape.name << "\",\"cached_prefix_len\":"
           << result.shape.cached_prefix_len << ",\"new_tokens\":" << result.shape.new_tokens
           << ",\"q_heads\":" << result.shape.q_heads << ",\"kv_heads\":" << result.shape.kv_heads
           << ",\"head_dim\":" << result.shape.head_dim << ",\"value_dim\":"
           << result.shape.value_dim << ",\"amplitude\":" << std::setprecision(9)
           << result.shape.amplitude << ",\"legacy_vs_cpu\":" << diff_json(result.legacy_vs_cpu)
           << ",\"qk_vs_legacy\":" << diff_json(result.qk_vs_legacy)
           << ",\"qk_max_vs_legacy\":" << diff_json(result.qk_max_vs_legacy)
           << ",\"all_staged_vs_legacy\":" << diff_json(result.all_staged_vs_legacy) << '}';
    return output.str();
}

std::pair<Timing, Timing> time_full_flash2_stage(hipModule_t module, hipStream_t stream, int warmups, int repeats) {
    const Flash2Shape shape{
        .name = "production_shape_prefill_chunk_896_to_1024",
        .cached_prefix_len = 896,
        .new_tokens = 128,
    };
    const std::size_t context = shape.cached_prefix_len + shape.new_tokens;
    const std::size_t q_elements = shape.new_tokens * shape.q_heads * shape.head_dim;
    const std::size_t k_elements = context * shape.kv_heads * shape.head_dim;
    const std::size_t v_elements = context * shape.kv_heads * shape.value_dim;
    const std::size_t output_elements = shape.new_tokens * shape.q_heads * shape.value_dim;
    const auto q_values = values_for(q_elements, shape.amplitude, 0x2101u);
    const auto k_values = values_for(k_elements, shape.amplitude, 0x2102u);
    const auto v_values = values_for(v_elements, shape.amplitude, 0x2103u);
    DeviceBuffer q(q_elements * sizeof(float));
    DeviceBuffer k(k_elements * sizeof(float));
    DeviceBuffer v(v_elements * sizeof(float));
    DeviceBuffer output(output_elements * sizeof(float));
    HIP_CHECK(hipMemcpyAsync(q.get(), q_values.data(), q.bytes(), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(k.get(), k_values.data(), k.bytes(), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(v.get(), v_values.data(), v.bytes(), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto legacy = function_for(module, "ullm_sq8_0_flash2_legacy_reference_kernel");
    const auto staged = function_for(module, "ullm_sq8_0_flash2_staged_wave32_prototype_kernel");
    return {
        time_flash2(legacy, shape, q, k, v, &output, stream, warmups, repeats),
        time_flash2(staged, shape, q, k, v, &output, stream, warmups, repeats),
    };
}

void run_pmc_probe(hipModule_t module, hipStream_t stream, const Options& options) {
    const std::size_t bytes = options.pmc_elements * sizeof(float);
    DeviceBuffer input(bytes);
    DeviceBuffer output(bytes);
    const auto values = values_for(options.pmc_elements, 1.0f, 0x3141u);
    HIP_CHECK(hipMemcpyAsync(input.get(), values.data(), bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto function = function_for(module, "ullm_sq8_0_pmc_probe_kernel");
    auto count = static_cast<unsigned long long>(options.pmc_elements);
    auto* input_pointer = static_cast<const float*>(input.get());
    auto* output_pointer = static_cast<float*>(output.get());
    void* parameters[] = {&input_pointer, &output_pointer, &count};
    constexpr unsigned int blocks = 4096;
    constexpr unsigned int threads = 256;
    for (int launch = 0; launch < options.pmc_launches; ++launch) {
        HIP_CHECK(hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(function), blocks, 1, 1, threads, 1, 1, 0, stream,
            parameters, nullptr));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

std::string timing_json(const Timing& timing) {
    std::ostringstream output;
    output << std::setprecision(12) << "{\"milliseconds_per_launch\":" << timing.milliseconds
           << ",\"launches_per_second\":" << timing.launches_per_second << '}';
    return output.str();
}

int run(const Options& options) {
    std::filesystem::create_directories(options.output_dir);
    const auto code = compile_module(options.output_dir);
    const bool needs_device = options.mode != Mode::Compile;
    std::optional<HipModule> module;
    std::optional<HipStream> stream;
    std::string device;
    if (needs_device) {
        require_gfx1201(options.device);
        device = device_json(options.device);
        module.emplace(code);
        stream.emplace();
    }

    std::vector<Flash2CaseResult> differential;
    std::optional<Timing> legacy_timing;
    std::optional<Timing> staged_timing;
    if (options.mode == Mode::Flash2 || options.mode == Mode::All) {
        const std::array<Flash2Shape, 4> cases = {{
            {.name = "short", .cached_prefix_len = 0, .new_tokens = 7},
            {.name = "tile_tail_63_to_68", .cached_prefix_len = 63, .new_tokens = 5},
            {.name = "production_shape_prefill_chunk_896_to_1024", .cached_prefix_len = 896, .new_tokens = 128},
            {.name = "adversarial_score_range_63_to_68", .cached_prefix_len = 63, .new_tokens = 5, .amplitude = 7.0f},
        }};
        for (const auto& shape : cases) {
            differential.push_back(run_flash2_case(module->get(), shape, stream->get()));
        }
        const auto timings = time_full_flash2_stage(
            module->get(), stream->get(), options.warmups, options.repeats);
        legacy_timing = timings.first;
        staged_timing = timings.second;
    }
    if (options.mode == Mode::PmcProbe || options.mode == Mode::All) {
        run_pmc_probe(module->get(), stream->get(), options);
    }

    std::ostringstream summary;
    summary << "{\n"
            << "  \"schema_version\": \"ullm.sq8_0.r9700.attention_prototype.v0.1\",\n"
            << "  \"mode\": \"" << mode_name(options.mode) << "\",\n"
            << "  \"target_architecture\": \"gfx1201\",\n"
            << "  \"isolation\": \"HIPRTC separate prototype symbols; no runtime source or serving dispatch selected by this binary\",\n"
            << "  \"module\": \"sq8_0_r9700_attention_prototype.hsaco\",\n"
            << "  \"device\": " << (device.empty() ? "null" : device) << ",\n"
            << "  \"flash2_differential\": [";
    for (std::size_t index = 0; index < differential.size(); ++index) {
        if (index != 0) summary << ',';
        summary << '\n' << "    " << flash2_case_json(differential[index]);
    }
    if (!differential.empty()) summary << '\n' << "  ";
    summary << "],\n";
    if (legacy_timing.has_value() && staged_timing.has_value()) {
        const double speedup = static_cast<double>(legacy_timing->milliseconds) /
            static_cast<double>(staged_timing->milliseconds);
        summary << "  \"flash2_kernel_timing_unprofiled\": {\"scope\": \"standalone synthetic production-shape kernel; not serving throughput\",\"legacy\": "
                << timing_json(*legacy_timing) << ",\"staged_wave32\": " << timing_json(*staged_timing)
                << ",\"legacy_over_staged_speedup\": " << std::setprecision(12) << speedup << "},\n";
    } else {
        summary << "  \"flash2_kernel_timing_unprofiled\": null,\n";
    }
    summary << "  \"pmc_probe\": {\"ran\": "
            << ((options.mode == Mode::PmcProbe || options.mode == Mode::All) ? "true" : "false")
            << ",\"kernel\": \"ullm_sq8_0_pmc_probe_kernel\",\"elements\": "
            << options.pmc_elements << ",\"launches\": " << options.pmc_launches << "}\n"
            << "}\n";
    write_text(options.output_dir / "summary.json", summary.str());
    std::cout << summary.str();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const ExitError& error) {
        std::cerr << error.what() << '\n';
        return error.code();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
