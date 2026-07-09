#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
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
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/mxf8_utils.hpp"

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

void report_cleanup_error(hipError_t status, std::string_view operation) noexcept {
    if (status != hipSuccess) {
        std::cerr << operation << " during cleanup failed: " << hipGetErrorString(status) << '\n';
    }
}

using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<
    RowMajor,
    ColumnMajor,
    ck::Tuple<>,
    RowMajor,
    ck::f8_t,
    float,
    ck::f8_t,
    float,
    ck::Tuple<>,
    ck::bhalf_t,
    1,
    128,
    128,
    PassThrough,
    PassThrough,
    PassThrough>;

static_assert(DeviceOp::NumDTensor == 0);
static_assert(sizeof(ck::f8_t) == 1);
static_assert(sizeof(ck::bhalf_t) == 2);
static_assert(sizeof(float) == 4);

constexpr std::size_t kScaleBlock = 128;
constexpr std::size_t kMaxWorkingBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kCacheEvictionBytes = 256ULL * 1024ULL * 1024ULL;
constexpr unsigned int kCacheEvictionBlocks = 4096;
constexpr unsigned int kCacheEvictionThreads = 256;
constexpr double kRelativeL2Limit = 5.0e-3;
constexpr double kCosineLimit = 0.9999;

enum class CacheMode {
    Warm,
    TargetBuffersEvicted,
};

struct Options {
    int device = 0;
    std::size_t m = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    std::filesystem::path weight_path;
    std::filesystem::path weight_scale_path;
    std::filesystem::path activation_path;
    std::filesystem::path oracle_path;
    std::optional<std::filesystem::path> expected_activation_fp8_path;
    std::optional<std::filesystem::path> expected_activation_scale_path;
    int warmups = 5;
    int repeats = 20;
    CacheMode cache_mode = CacheMode::Warm;
};

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
            report_cleanup_error(hipFree(pointer_), "hipFree");
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
            report_cleanup_error(hipStreamDestroy(stream_), "hipStreamDestroy");
        }
    }

    hipStream_t get() const { return stream_; }

  private:
    hipStream_t stream_ = nullptr;
};

class HipEvent {
  public:
    HipEvent() { HIP_CHECK(hipEventCreateWithFlags(&event_, hipEventDefault)); }
    HipEvent(const HipEvent&) = delete;
    HipEvent& operator=(const HipEvent&) = delete;

    ~HipEvent() {
        if (event_ != nullptr) {
            report_cleanup_error(hipEventDestroy(event_), "hipEventDestroy");
        }
    }

    hipEvent_t get() const { return event_; }

  private:
    hipEvent_t event_ = nullptr;
};

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, std::string_view label);

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

const char* quantization_kernel_source() {
    return R"HIP(
__device__ float ullm_positive_ocp_e4m3_to_f32(unsigned int code) {
    const unsigned int exponent = (code >> 3U) & 15U;
    const unsigned int mantissa = code & 7U;
    if (exponent == 0U) {
        return (float)mantissa * 0.001953125f;
    }
    return ldexpf(1.0f + (float)mantissa * 0.125f, (int)exponent - 7);
}

__device__ unsigned char ullm_f32_to_ocp_e4m3_rne(float value, float scale) {
    union {
        float as_float;
        unsigned int as_bits;
    } bits = {value};
    const unsigned int sign = (bits.as_bits >> 24U) & 0x80U;
    const float magnitude = fabsf(value / scale);
    if (magnitude == 0.0f) {
        return (unsigned char)sign;
    }
    if (magnitude >= 448.0f) {
        return (unsigned char)(sign | 0x7eU);
    }

    unsigned int lower_bound = 0U;
    unsigned int upper_bound = 127U;
    while (lower_bound < upper_bound) {
        const unsigned int middle = lower_bound + (upper_bound - lower_bound) / 2U;
        if (ullm_positive_ocp_e4m3_to_f32(middle) < magnitude) {
            lower_bound = middle + 1U;
        } else {
            upper_bound = middle;
        }
    }
    const unsigned int upper = lower_bound;
    const float upper_value = ullm_positive_ocp_e4m3_to_f32(upper);
    if (upper_value == magnitude) {
        return (unsigned char)(sign | upper);
    }
    const unsigned int lower = upper - 1U;
    const double lower_distance =
        (double)magnitude - (double)ullm_positive_ocp_e4m3_to_f32(lower);
    const double upper_distance = (double)upper_value - (double)magnitude;
    const unsigned int encoded = lower_distance < upper_distance
                                     ? lower
                                     : (upper_distance < lower_distance
                                            ? upper
                                            : ((lower & 1U) == 0U ? lower : upper));
    return (unsigned char)(sign | encoded);
}

extern "C" __global__ void ullm_sq8_quantize_activation_block128(
    const float* input,
    unsigned char* output,
    float* scales,
    unsigned long k) {
    __shared__ float absolute_values[128];
    __shared__ float block_scale;

    const unsigned long block = (unsigned long)blockIdx.x;
    const unsigned long lane = (unsigned long)threadIdx.x;
    const unsigned long k_blocks = k / 128UL;
    const unsigned long row = block / k_blocks;
    const unsigned long block_k = block % k_blocks;
    const unsigned long element = row * k + block_k * 128UL + lane;

    absolute_values[lane] = fabsf(input[element]);
    __syncthreads();
    for (unsigned int stride = 64U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            absolute_values[lane] = fmaxf(absolute_values[lane], absolute_values[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0UL) {
        const float maximum = absolute_values[0];
        block_scale = maximum == 0.0f
                          ? 1.0f
                          : fmaxf(maximum / 448.0f, __uint_as_float(1U));
        scales[block] = block_scale;
    }
    __syncthreads();
    output[element] = ullm_f32_to_ocp_e4m3_rne(input[element], block_scale);
}
)HIP";
}

class QuantizationKernel {
  public:
    explicit QuantizationKernel(std::string_view arch) {
        HipRtcProgram program(quantization_kernel_source(), "ullm_sq8_quantize_activation.hip");
        const std::string architecture_option = "--gpu-architecture=" + std::string(arch);
        const char* options[] = {
            "--std=c++17",
            "-O3",
            architecture_option.c_str(),
        };
        const hiprtcResult compile_status =
            hiprtcCompileProgram(program.get(), static_cast<int>(std::size(options)), options);
        if (compile_status != HIPRTC_SUCCESS) {
            std::size_t log_size = 0;
            HIPRTC_CHECK(hiprtcGetProgramLogSize(program.get(), &log_size));
            std::string log(log_size, '\0');
            if (log_size != 0) {
                HIPRTC_CHECK(hiprtcGetProgramLog(program.get(), log.data()));
            }
            throw std::runtime_error("HIPRTC activation quantization compile failed: " + log);
        }

        std::size_t code_size = 0;
        HIPRTC_CHECK(hiprtcGetCodeSize(program.get(), &code_size));
        if (code_size == 0) {
            throw std::runtime_error("HIPRTC activation quantization produced empty code");
        }
        std::vector<char> code(code_size);
        HIPRTC_CHECK(hiprtcGetCode(program.get(), code.data()));
        HIP_CHECK(hipModuleLoadData(&module_, code.data()));
        const hipError_t function_status = hipModuleGetFunction(
            &function_, module_, "ullm_sq8_quantize_activation_block128");
        if (function_status != hipSuccess) {
            const hipError_t unload_status = hipModuleUnload(module_);
            module_ = nullptr;
            if (unload_status != hipSuccess) {
                report_cleanup_error(unload_status, "hipModuleUnload after get-function failure");
            }
            hip_check(function_status, "hipModuleGetFunction", __FILE__, __LINE__);
        }
    }

    QuantizationKernel(const QuantizationKernel&) = delete;
    QuantizationKernel& operator=(const QuantizationKernel&) = delete;

    ~QuantizationKernel() {
        if (module_ != nullptr) {
            report_cleanup_error(hipModuleUnload(module_), "hipModuleUnload");
        }
    }

    void launch(const DeviceBuffer& activation,
                DeviceBuffer& activation_fp8,
                DeviceBuffer& activation_scales,
                const Options& options,
                hipStream_t stream) const {
        const std::size_t blocks =
            checked_mul(options.m, options.k / kScaleBlock, "quant blocks");
        if (blocks > std::numeric_limits<unsigned int>::max()) {
            throw std::runtime_error("quantization grid exceeds dim3.x");
        }
        const float* input = static_cast<const float*>(activation.get());
        auto* output = static_cast<unsigned char*>(activation_fp8.get());
        auto* scales = static_cast<float*>(activation_scales.get());
        unsigned long k = static_cast<unsigned long>(options.k);
        void* arguments[] = {&input, &output, &scales, &k};
        HIP_CHECK(hipModuleLaunchKernel(function_,
                                        static_cast<unsigned int>(blocks),
                                        1,
                                        1,
                                        static_cast<unsigned int>(kScaleBlock),
                                        1,
                                        1,
                                        0,
                                        stream,
                                        arguments,
                                        nullptr));
    }

  private:
    hipModule_t module_ = nullptr;
    hipFunction_t function_ = nullptr;
};

std::uint32_t cache_eviction_hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

const char* cache_eviction_kernel_source() {
    return R"HIP(
__device__ unsigned int ullm_sq8_eviction_hash(unsigned int value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

extern "C" __global__ void ullm_sq8_initialize_eviction_buffer(
    uint4* output,
    unsigned long vectors) {
    unsigned long index = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long stride = (unsigned long)gridDim.x * blockDim.x;
    while (index < vectors) {
        const unsigned int base = (unsigned int)(index * 4UL);
        output[index] = make_uint4(ullm_sq8_eviction_hash(base),
                                   ullm_sq8_eviction_hash(base + 1U),
                                   ullm_sq8_eviction_hash(base + 2U),
                                   ullm_sq8_eviction_hash(base + 3U));
        index += stride;
    }
}

extern "C" __global__ void ullm_sq8_evict_cache(
    const uint4* input,
    unsigned long vectors,
    unsigned long long* block_sums) {
    __shared__ unsigned long long partial_sums[256];
    const unsigned int lane = threadIdx.x;
    unsigned long index = (unsigned long)blockIdx.x * blockDim.x + lane;
    const unsigned long stride = (unsigned long)gridDim.x * blockDim.x;
    unsigned long long sum = 0ULL;
    while (index < vectors) {
        const uint4 value = input[index];
        sum += (unsigned long long)value.x + value.y + value.z + value.w;
        index += stride;
    }
    partial_sums[lane] = sum;
    __syncthreads();
    for (unsigned int offset = 128U; offset != 0U; offset >>= 1U) {
        if (lane < offset) {
            partial_sums[lane] += partial_sums[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0U) {
        block_sums[blockIdx.x] = partial_sums[0];
    }
}
)HIP";
}

class CacheEvictionKernel {
  public:
    explicit CacheEvictionKernel(std::string_view arch) {
        HipRtcProgram program(cache_eviction_kernel_source(), "ullm_sq8_cache_evict.hip");
        const std::string architecture_option = "--gpu-architecture=" + std::string(arch);
        const char* options[] = {
            "--std=c++17",
            "-O3",
            architecture_option.c_str(),
        };
        const hiprtcResult compile_status =
            hiprtcCompileProgram(program.get(), static_cast<int>(std::size(options)), options);
        if (compile_status != HIPRTC_SUCCESS) {
            std::size_t log_size = 0;
            HIPRTC_CHECK(hiprtcGetProgramLogSize(program.get(), &log_size));
            std::string log(log_size, '\0');
            if (log_size != 0) {
                HIPRTC_CHECK(hiprtcGetProgramLog(program.get(), log.data()));
            }
            throw std::runtime_error("HIPRTC cache eviction compile failed: " + log);
        }

        std::size_t code_size = 0;
        HIPRTC_CHECK(hiprtcGetCodeSize(program.get(), &code_size));
        if (code_size == 0) {
            throw std::runtime_error("HIPRTC cache eviction produced empty code");
        }
        std::vector<char> code(code_size);
        HIPRTC_CHECK(hiprtcGetCode(program.get(), code.data()));
        HIP_CHECK(hipModuleLoadData(&module_, code.data()));
        hipError_t function_status = hipModuleGetFunction(
            &initialize_function_, module_, "ullm_sq8_initialize_eviction_buffer");
        if (function_status == hipSuccess) {
            function_status = hipModuleGetFunction(&evict_function_, module_, "ullm_sq8_evict_cache");
        }
        if (function_status != hipSuccess) {
            const hipError_t unload_status = hipModuleUnload(module_);
            module_ = nullptr;
            if (unload_status != hipSuccess) {
                report_cleanup_error(unload_status,
                                     "hipModuleUnload after cache get-function failure");
            }
            hip_check(function_status, "hipModuleGetFunction", __FILE__, __LINE__);
        }
    }

    CacheEvictionKernel(const CacheEvictionKernel&) = delete;
    CacheEvictionKernel& operator=(const CacheEvictionKernel&) = delete;

    ~CacheEvictionKernel() {
        if (module_ != nullptr) {
            report_cleanup_error(hipModuleUnload(module_), "hipModuleUnload cache eviction");
        }
    }

    void initialize(DeviceBuffer& output, hipStream_t stream) const {
        if (output.bytes() % sizeof(uint4) != 0) {
            throw std::runtime_error("cache eviction buffer is not uint4 aligned in size");
        }
        auto* output_pointer = static_cast<uint4*>(output.get());
        unsigned long vectors = static_cast<unsigned long>(output.bytes() / sizeof(uint4));
        void* arguments[] = {&output_pointer, &vectors};
        HIP_CHECK(hipModuleLaunchKernel(initialize_function_,
                                        kCacheEvictionBlocks,
                                        1,
                                        1,
                                        kCacheEvictionThreads,
                                        1,
                                        1,
                                        0,
                                        stream,
                                        arguments,
                                        nullptr));
    }

    void launch(const DeviceBuffer& input, DeviceBuffer& block_sums, hipStream_t stream) const {
        if (input.bytes() % sizeof(uint4) != 0) {
            throw std::runtime_error("cache eviction input is not uint4 aligned in size");
        }
        const auto* input_pointer = static_cast<const uint4*>(input.get());
        unsigned long vectors = static_cast<unsigned long>(input.bytes() / sizeof(uint4));
        auto* output_pointer = static_cast<unsigned long long*>(block_sums.get());
        void* arguments[] = {&input_pointer, &vectors, &output_pointer};
        HIP_CHECK(hipModuleLaunchKernel(evict_function_,
                                        kCacheEvictionBlocks,
                                        1,
                                        1,
                                        kCacheEvictionThreads,
                                        1,
                                        1,
                                        0,
                                        stream,
                                        arguments,
                                        nullptr));
    }

  private:
    hipModule_t module_ = nullptr;
    hipFunction_t initialize_function_ = nullptr;
    hipFunction_t evict_function_ = nullptr;
};

class CacheEvictionState {
  public:
    CacheEvictionState(std::string_view arch, hipStream_t stream)
        : kernel_(arch),
          input_(kCacheEvictionBytes),
          block_sums_(static_cast<std::size_t>(kCacheEvictionBlocks) *
                      sizeof(unsigned long long)) {
        kernel_.initialize(input_, stream);
        HIP_CHECK(hipMemsetAsync(block_sums_.get(), 0, block_sums_.bytes(), stream));
        HipEvent start;
        HipEvent stop;
        HIP_CHECK(hipEventRecord(start.get(), stream));
        kernel_.launch(input_, block_sums_, stream);
        HIP_CHECK(hipEventRecord(stop.get(), stream));
        HIP_CHECK(hipEventSynchronize(stop.get()));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            throw std::runtime_error("cache eviction validation returned an invalid duration");
        }
        validation_ms_ = elapsed_ms;
        std::vector<unsigned long long> host_sums(kCacheEvictionBlocks);
        HIP_CHECK(hipMemcpyAsync(host_sums.data(),
                                 block_sums_.get(),
                                 block_sums_.bytes(),
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for (const unsigned long long value : host_sums) {
            checksum_ += value;
        }
        std::uint64_t expected_checksum = 0;
        const std::size_t word_count = kCacheEvictionBytes / sizeof(std::uint32_t);
        for (std::size_t index = 0; index < word_count; ++index) {
            expected_checksum += cache_eviction_hash(static_cast<std::uint32_t>(index));
        }
        if (checksum_ != expected_checksum) {
            throw std::runtime_error("cache eviction validation checksum mismatch");
        }
    }

    CacheEvictionState(const CacheEvictionState&) = delete;
    CacheEvictionState& operator=(const CacheEvictionState&) = delete;

    void launch(hipStream_t stream) { kernel_.launch(input_, block_sums_, stream); }
    std::uint64_t checksum() const { return checksum_; }
    double validation_ms() const { return validation_ms_; }

  private:
    CacheEvictionKernel kernel_;
    DeviceBuffer input_;
    DeviceBuffer block_sums_;
    std::uint64_t checksum_ = 0;
    double validation_ms_ = 0.0;
};

struct Timing {
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

struct CandidateGroup {
    std::string name;
    std::vector<std::unique_ptr<DeviceOp>> instances;
};

struct ErrorMetrics {
    std::size_t nonfinite = 0;
    double max_abs = 0.0;
    double relative_l2 = std::numeric_limits<double>::infinity();
    double cosine = -1.0;
};

bool passes_numerical_gate(const ErrorMetrics& metrics) {
    return metrics.nonfinite == 0 && std::isfinite(metrics.relative_l2) &&
           metrics.relative_l2 <= kRelativeL2Limit && std::isfinite(metrics.cosine) &&
           metrics.cosine >= kCosineLimit;
}

struct CandidateMeasurement {
    std::string group;
    std::string instance;
    bool supported = false;
    bool runnable = false;
    bool numerically_valid = false;
    Timing gemm;
    std::optional<ErrorMetrics> correctness;
    std::string error;
    std::size_t group_index = 0;
    std::size_t instance_index = 0;
};

ErrorMetrics calculate_error_metrics(const std::vector<ck::bhalf_t>& output,
                                     const std::vector<float>& oracle);

struct QuantizationCheck {
    bool fp8_requested = false;
    bool fp8_exact = true;
    std::size_t fp8_mismatch_count = 0;
    std::size_t fp8_first_mismatch = 0;
    bool scale_requested = false;
    bool scale_exact = true;
    std::size_t scale_mismatch_count = 0;
    std::size_t scale_first_mismatch_byte = 0;

    bool passed() const {
        return (!fp8_requested || fp8_exact) && (!scale_requested || scale_exact);
    }
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --device N --m M --n N --k K"
        << " --weight FILE --weight-scale FILE --activation FILE --oracle FILE"
        << " [--expected-activation-fp8 FILE] [--expected-activation-scales FILE]"
        << " [--warmups N] [--repeats N] [--cache-mode warm|evicted]\n";
    std::exit(2);
}

std::size_t parse_size(std::string_view text, std::string_view option) {
    if (text.empty() || text.front() == '-') {
        throw ExitError(2, std::string(option) + " must be a non-negative integer");
    }
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(owned.c_str(), &end, 10);
    if (errno == ERANGE || end == owned.c_str() || *end != '\0') {
        throw ExitError(2, "invalid integer for " + std::string(option));
    }
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw ExitError(2, std::string(option) + " exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&]() -> std::string_view {
            if (++index >= argc) {
                throw ExitError(2, "missing value for " + std::string(argument));
            }
            return argv[index];
        };

        if (argument == "--device") {
            const std::size_t parsed = parse_size(value(), argument);
            if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw ExitError(2, "--device exceeds int");
            }
            options.device = static_cast<int>(parsed);
        } else if (argument == "--m") {
            options.m = parse_size(value(), argument);
        } else if (argument == "--n") {
            options.n = parse_size(value(), argument);
        } else if (argument == "--k") {
            options.k = parse_size(value(), argument);
        } else if (argument == "--weight") {
            options.weight_path = value();
        } else if (argument == "--weight-scale") {
            options.weight_scale_path = value();
        } else if (argument == "--activation") {
            options.activation_path = value();
        } else if (argument == "--oracle") {
            options.oracle_path = value();
        } else if (argument == "--expected-activation-fp8") {
            options.expected_activation_fp8_path = std::filesystem::path(value());
        } else if (argument == "--expected-activation-scales") {
            options.expected_activation_scale_path = std::filesystem::path(value());
        } else if (argument == "--warmups") {
            const std::size_t parsed = parse_size(value(), argument);
            if (parsed > 100000) {
                throw ExitError(2, "--warmups exceeds 100000");
            }
            options.warmups = static_cast<int>(parsed);
        } else if (argument == "--repeats") {
            const std::size_t parsed = parse_size(value(), argument);
            if (parsed == 0 || parsed > 100000) {
                throw ExitError(2, "--repeats must be in [1, 100000]");
            }
            options.repeats = static_cast<int>(parsed);
        } else if (argument == "--cache-mode") {
            const std::string_view mode = value();
            if (mode == "warm") {
                options.cache_mode = CacheMode::Warm;
            } else if (mode == "evicted") {
                options.cache_mode = CacheMode::TargetBuffersEvicted;
            } else {
                throw ExitError(2, "--cache-mode must be warm or evicted");
            }
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
        } else {
            throw ExitError(2, "unknown option: " + std::string(argument));
        }
    }

    if (options.m == 0 || options.n == 0 || options.k == 0) {
        throw ExitError(2, "M, N, and K must be positive");
    }
    if (options.n % kScaleBlock != 0 || options.k % kScaleBlock != 0) {
        throw ExitError(2, "N and K must be divisible by 128");
    }
    if (options.weight_path.empty() || options.weight_scale_path.empty() ||
        options.activation_path.empty() || options.oracle_path.empty()) {
        throw ExitError(2, "--weight, --weight-scale, --activation, and --oracle are required");
    }
    const auto ck_max = static_cast<std::size_t>(std::numeric_limits<ck::index_t>::max());
    if (options.m > ck_max || options.n > ck_max || options.k > ck_max) {
        throw ExitError(2, "M, N, or K exceeds ck::index_t");
    }
    return options;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, std::string_view label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw ExitError(2, std::string(label) + " overflows size_t");
    }
    return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, std::string_view label) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw ExitError(2, std::string(label) + " overflows size_t");
    }
    return lhs + rhs;
}

std::size_t estimated_working_bytes(std::size_t activation_elements,
                                    std::size_t weight_elements,
                                    std::size_t output_elements,
                                    std::size_t activation_scale_elements,
                                    std::size_t weight_scale_elements) {
    std::size_t bytes = checked_mul(weight_elements, 2, "weight working bytes");
    bytes = checked_add(bytes,
                        checked_mul(weight_scale_elements, 10, "weight scale working bytes"),
                        "component working bytes");
    bytes = checked_add(bytes,
                        checked_mul(activation_elements, 15, "activation working bytes"),
                        "component working bytes");
    bytes = checked_add(bytes,
                        checked_mul(output_elements, 12, "output working bytes"),
                        "component working bytes");
    return checked_add(
        bytes,
        checked_mul(activation_scale_elements, 16, "activation scale working bytes"),
        "component working bytes");
}

std::vector<std::uint8_t> read_exact_bytes(const std::filesystem::path& path,
                                           std::size_t expected_bytes,
                                           std::string_view label) {
    std::error_code file_size_error;
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path, file_size_error);
    if (file_size_error) {
        throw ExitError(2,
                        std::string(label) + " file_size failed for " + path.string() + ": " +
                            file_size_error.message());
    }
    if (actual_bytes != expected_bytes) {
        throw ExitError(2,
                        std::string(label) + " byte count mismatch: expected " +
                            std::to_string(expected_bytes) + ", got " +
                            std::to_string(actual_bytes));
    }
    if (expected_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw ExitError(2, std::string(label) + " exceeds std::streamsize");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw ExitError(2, "cannot open " + std::string(label) + ": " + path.string());
    }
    std::vector<std::uint8_t> bytes(expected_bytes);
    if (expected_bytes != 0) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(expected_bytes));
        if (input.gcount() != static_cast<std::streamsize>(expected_bytes) || !input) {
            throw ExitError(2, std::string(label) + " ended before the declared byte count");
        }
    }
    char trailing = 0;
    input.read(&trailing, 1);
    if (input.gcount() != 0 || !input.eof()) {
        throw ExitError(2, std::string(label) + " has trailing data or an I/O error");
    }
    return bytes;
}

std::uint16_t decode_u16_le(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t decode_u32_le(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::vector<float> decode_f32_le(const std::vector<std::uint8_t>& bytes,
                                 std::string_view label) {
    if (bytes.size() % sizeof(float) != 0) {
        throw ExitError(2, std::string(label) + " byte count is not divisible by four");
    }
    std::vector<float> values(bytes.size() / sizeof(float));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = std::bit_cast<float>(decode_u32_le(bytes.data() + index * 4));
        if (!std::isfinite(values[index])) {
            throw ExitError(2,
                            std::string(label) + " contains a non-finite value at element " +
                                std::to_string(index));
        }
    }
    return values;
}

std::vector<float> decode_bf16_le(const std::vector<std::uint8_t>& bytes,
                                  std::string_view label) {
    if (bytes.size() % sizeof(std::uint16_t) != 0) {
        throw ExitError(2, std::string(label) + " byte count is not divisible by two");
    }
    std::vector<float> values(bytes.size() / sizeof(std::uint16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(
                                       decode_u16_le(bytes.data() + index * 2))
                                   << 16U;
        values[index] = std::bit_cast<float>(bits);
        if (!std::isfinite(values[index])) {
            throw ExitError(2,
                            std::string(label) + " contains a non-finite value at element " +
                                std::to_string(index));
        }
    }
    return values;
}

void validate_fp8_ocp(const std::vector<std::uint8_t>& bytes, std::string_view label) {
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        const ck::f8_t value{static_cast<ck::fp8_storage_t>(bytes[index])};
        if (!std::isfinite(static_cast<float>(value))) {
            throw ExitError(2,
                            std::string(label) + " contains a non-finite OCP E4M3 value at byte " +
                                std::to_string(index));
        }
    }
}

std::vector<std::uint8_t> encode_f32_le(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(checked_mul(values.size(), sizeof(float), "F32 bytes"));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(values[index]);
        bytes[index * 4] = static_cast<std::uint8_t>(bits);
        bytes[index * 4 + 1] = static_cast<std::uint8_t>(bits >> 8U);
        bytes[index * 4 + 2] = static_cast<std::uint8_t>(bits >> 16U);
        bytes[index * 4 + 3] = static_cast<std::uint8_t>(bits >> 24U);
    }
    return bytes;
}

std::string json_string(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}

std::string json_number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

std::string normalized_arch(const hipDeviceProp_t& properties) {
    std::string arch(properties.gcnArchName);
    if (const std::size_t separator = arch.find(':'); separator != std::string::npos) {
        arch.resize(separator);
    }
    return arch;
}

std::optional<std::string> selected_visible_device_token() {
    const char* token = std::getenv("ULLM_CK_COMPONENT_VISIBLE_DEVICE_TOKEN");
    if (token == nullptr || token[0] == '\0') {
        token = std::getenv("HIP_VISIBLE_DEVICES");
    }
    if (token == nullptr || token[0] == '\0') {
        return std::nullopt;
    }
    return std::string(token);
}

double percentile(std::vector<float> samples, double probability) {
    if (samples.empty()) {
        throw std::runtime_error("cannot compute percentile of an empty sample");
    }
    std::sort(samples.begin(), samples.end());
    const double position = probability * static_cast<double>(samples.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return static_cast<double>(samples[lower]) * (1.0 - fraction) +
           static_cast<double>(samples[upper]) * fraction;
}

Timing summarize_timing(const std::vector<float>& samples) {
    return Timing{percentile(samples, 0.50), percentile(samples, 0.95)};
}

template <typename Prepare, typename Launch>
std::vector<float> measure_gpu_events(int warmups,
                                      int repeats,
                                      hipStream_t stream,
                                      Prepare&& prepare,
                                      Launch&& launch) {
    for (int iteration = 0; iteration < warmups; ++iteration) {
        prepare();
        launch();
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    HipEvent start;
    HipEvent stop;
    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int iteration = 0; iteration < repeats; ++iteration) {
        prepare();
        HIP_CHECK(hipEventRecord(start.get(), stream));
        launch();
        HIP_CHECK(hipEventRecord(stop.get(), stream));
        HIP_CHECK(hipEventSynchronize(stop.get()));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        if (!std::isfinite(elapsed_ms) || elapsed_ms < 0.0f) {
            throw std::runtime_error("HIP event returned an invalid duration");
        }
        samples.push_back(elapsed_ms);
    }
    return samples;
}

std::vector<CandidateGroup> make_candidate_groups() {
    using namespace ck::tensor_operation::device::instance;
    std::vector<CandidateGroup> groups;
    groups.push_back({"comp_default", {}});
    add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
        groups.back().instances);
    groups.push_back({"comp_kpadding", {}});
    add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_kpadding_instances(
        groups.back().instances);
    groups.push_back({"mem_v1_default", {}});
    add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
        groups.back().instances);
    groups.push_back({"mem_v1_kpadding", {}});
    add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
        groups.back().instances);
    return groups;
}

std::unique_ptr<ck::tensor_operation::device::BaseArgument> make_argument(
    DeviceOp& operation,
    const DeviceBuffer& activation_fp8,
    const DeviceBuffer& weights,
    DeviceBuffer& output,
    const DeviceBuffer& activation_scales,
    const DeviceBuffer& weight_scales,
    const Options& options) {
    const auto m = static_cast<ck::index_t>(options.m);
    const auto n = static_cast<ck::index_t>(options.n);
    const auto k = static_cast<ck::index_t>(options.k);
    return operation.MakeArgumentPointer(activation_fp8.get(),
                                         weights.get(),
                                         std::array<const void*, 0>{},
                                         output.get(),
                                         m,
                                         n,
                                         k,
                                         k,
                                         k,
                                         std::array<ck::index_t, 0>{},
                                         n,
                                         activation_scales.get(),
                                         weight_scales.get(),
                                         PassThrough{},
                                         PassThrough{},
                                         PassThrough{});
}

StreamConfig stream_config(hipStream_t stream) {
    StreamConfig config;
    config.stream_id_ = stream;
    config.time_kernel_ = false;
    config.log_level_ = 0;
    config.flush_cache = false;
    return config;
}

CandidateMeasurement measure_candidate(std::size_t group_index,
                                       std::size_t instance_index,
                                       CandidateGroup& group,
                                       ck::tensor_operation::device::BaseArgument& argument,
                                       DeviceBuffer& output,
                                       std::size_t output_elements,
                                       const std::vector<float>& oracle,
                                       const Options& options,
                                       CacheEvictionState* cache_eviction,
                                       hipStream_t stream) {
    CandidateMeasurement measurement;
    measurement.group = group.name;
    measurement.instance = group.instances[instance_index]->GetTypeString();
    measurement.supported = true;
    measurement.group_index = group_index;
    measurement.instance_index = instance_index;
    try {
        auto invoker = group.instances[instance_index]->MakeInvokerPointer();
        const StreamConfig config = stream_config(stream);
        const auto samples = measure_gpu_events(
            options.warmups,
            options.repeats,
            stream,
            [&]() {
                if (cache_eviction != nullptr) {
                    cache_eviction->launch(stream);
                }
            },
            [&]() {
                (void)invoker->Run(&argument, config);
                HIP_CHECK(hipGetLastError());
            });
        measurement.gemm = summarize_timing(samples);
        std::vector<ck::bhalf_t> host_output(output_elements);
        HIP_CHECK(hipMemcpyAsync(host_output.data(),
                                 output.get(),
                                 output.bytes(),
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        measurement.correctness = calculate_error_metrics(host_output, oracle);
        measurement.numerically_valid = passes_numerical_gate(*measurement.correctness);
        measurement.runnable = true;
    } catch (const std::exception& error) {
        measurement.error = error.what();
        const hipError_t pending_error = hipGetLastError();
        if (pending_error != hipSuccess) {
            measurement.error += "; pending HIP error: ";
            measurement.error += hipGetErrorString(pending_error);
        }
    }
    return measurement;
}

ErrorMetrics calculate_error_metrics(const std::vector<ck::bhalf_t>& output,
                                     const std::vector<float>& oracle) {
    if (output.size() != oracle.size()) {
        throw std::runtime_error("output and oracle element counts differ");
    }
    ErrorMetrics metrics;
    long double squared_error = 0.0L;
    long double squared_oracle = 0.0L;
    long double squared_output = 0.0L;
    long double dot = 0.0L;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float actual = ck::type_convert<float>(output[index]);
        const float expected = oracle[index];
        if (!std::isfinite(actual)) {
            ++metrics.nonfinite;
            continue;
        }
        const long double actual_ld = actual;
        const long double expected_ld = expected;
        const long double difference = actual_ld - expected_ld;
        metrics.max_abs = std::max(metrics.max_abs, std::abs(static_cast<double>(difference)));
        squared_error += difference * difference;
        squared_oracle += expected_ld * expected_ld;
        squared_output += actual_ld * actual_ld;
        dot += actual_ld * expected_ld;
    }
    if (metrics.nonfinite != 0) {
        return metrics;
    }
    if (squared_oracle == 0.0L) {
        metrics.relative_l2 = squared_error == 0.0L ? 0.0 : std::numeric_limits<double>::infinity();
    } else {
        metrics.relative_l2 = static_cast<double>(std::sqrt(squared_error / squared_oracle));
    }
    if (squared_oracle == 0.0L || squared_output == 0.0L) {
        metrics.cosine = squared_oracle == 0.0L && squared_output == 0.0L ? 1.0 : 0.0;
    } else {
        metrics.cosine = static_cast<double>(dot / std::sqrt(squared_oracle * squared_output));
        metrics.cosine = std::clamp(metrics.cosine, -1.0, 1.0);
    }
    return metrics;
}

struct ByteComparison {
    bool exact = true;
    std::size_t mismatch_count = 0;
    std::size_t first_mismatch = 0;
};

ByteComparison compare_bytes(const std::vector<std::uint8_t>& actual,
                             const std::vector<std::uint8_t>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("internal byte comparison length mismatch");
    }
    ByteComparison result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            if (result.exact) {
                result.first_mismatch = index;
            }
            result.exact = false;
            ++result.mismatch_count;
        }
    }
    return result;
}

void print_error_json(std::string_view error) {
    std::cout << "{\"schema_version\":\"ullm.sq8.ck_component.v2\","
              << "\"status\":\"error\",\"fallback\":\"not_used\",\"error\":"
              << json_string(error) << "}\n";
}

void print_result_json(const Options& options,
                       const hipDeviceProp_t& properties,
                       const std::string& arch,
                       std::size_t candidate_count,
                       std::size_t supported_count,
                       const std::vector<CandidateMeasurement>& candidates,
                       const CandidateMeasurement* selected,
                       const Timing& quant_timing,
                       const std::optional<Timing>& combined_timing,
                       const std::optional<ErrorMetrics>& error_metrics,
                       const QuantizationCheck& quant_check,
                       const CacheEvictionState* cache_eviction,
                       std::size_t working_bytes,
                       std::size_t device_allocation_bytes,
                       std::size_t free_device_bytes,
                       bool passed,
                       std::string_view failure_reason) {
    const double operations = 2.0 * static_cast<double>(options.m) *
                              static_cast<double>(options.n) * static_cast<double>(options.k);
    const auto tflops = [&](double milliseconds) {
        return milliseconds > 0.0 ? operations / (milliseconds * 1.0e9) : 0.0;
    };
    const std::optional<std::string> visible_device_token = selected_visible_device_token();

    std::cout << std::setprecision(17);
    std::cout << "{\n"
              << "  \"schema_version\": \"ullm.sq8.ck_component.v2\",\n"
              << "  \"status\": " << json_string(passed ? "passed" : "failed") << ",\n"
              << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
              << "  \"fallback\": \"not_used\",\n"
              << "  \"failure_reason\": "
              << (failure_reason.empty() ? "null" : json_string(failure_reason)) << ",\n"
              << "  \"device\": {\"hip_device\": " << options.device << ", \"name\": "
              << json_string(properties.name) << ", \"arch\": " << json_string(arch)
              << ", \"visible_device_token\": "
              << (visible_device_token.has_value() ? json_string(*visible_device_token) : "null")
              << "},\n"
              << "  \"shape\": {\"m\": " << options.m << ", \"n\": " << options.n
              << ", \"k\": " << options.k << "},\n"
              << "  \"memory\": {\"estimated_working_bytes\": " << working_bytes
              << ", \"device_allocation_bytes\": " << device_allocation_bytes
              << ", \"free_device_bytes_at_check\": " << free_device_bytes
              << ", \"limit_bytes\": " << kMaxWorkingBytes << "},\n"
              << "  \"contract\": {\"a_type\": \"fp8_e4m3_ocp\", \"b_type\": "
                 "\"fp8_e4m3_ocp\", \"output_type\": \"bf16\", \"a_layout\": "
                 "\"row_major_mk\", \"b_layout\": \"column_major_kn\", \"b_storage\": "
                 "\"canonical_row_major_nk_bytes\", \"scale_type\": \"f32\", "
                 "\"scale_block\": {\"m\": 1, \"n\": 128, \"k\": 128}},\n"
              << "  \"inputs\": {\"weight\": " << json_string(options.weight_path.string())
              << ", \"weight_scale_bf16\": " << json_string(options.weight_scale_path.string())
              << ", \"activation_f32\": " << json_string(options.activation_path.string())
              << ", \"oracle_f32\": " << json_string(options.oracle_path.string()) << "},\n"
              << "  \"instance_count\": " << candidate_count << ",\n"
              << "  \"supported_count\": " << supported_count << ",\n"
              << "  \"candidate_measurements\": [\n";
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const CandidateMeasurement& candidate = candidates[index];
        std::cout << "    {\"group\": " << json_string(candidate.group)
                  << ", \"instance\": " << json_string(candidate.instance)
                  << ", \"supported\": true, \"runnable\": "
                  << (candidate.runnable ? "true" : "false");
        if (candidate.runnable) {
            std::cout << ", \"gemm_p50_ms\": " << candidate.gemm.p50_ms
                      << ", \"gemm_p95_ms\": " << candidate.gemm.p95_ms
                      << ", \"numerically_valid\": "
                      << (candidate.numerically_valid ? "true" : "false");
            if (candidate.correctness.has_value()) {
                std::cout << ", \"relative_l2\": "
                          << json_number(candidate.correctness->relative_l2)
                          << ", \"cosine\": " << json_number(candidate.correctness->cosine)
                          << ", \"nonfinite\": " << candidate.correctness->nonfinite;
            }
        }
        if (!candidate.error.empty()) {
            std::cout << ", \"error\": " << json_string(candidate.error);
        }
        std::cout << '}' << (index + 1 == candidates.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n  \"selected_instance\": ";
    if (selected == nullptr) {
        std::cout << "null,\n";
    } else {
        std::cout << "{\"group\": " << json_string(selected->group) << ", \"instance\": "
                  << json_string(selected->instance) << "},\n";
    }
    const bool target_buffers_evicted =
        options.cache_mode == CacheMode::TargetBuffersEvicted;
    std::cout << "  \"timing\": {\"source\": \"hip_event\", \"cache_state\": "
              << json_string(target_buffers_evicted ? "target_buffers_evicted"
                                                    : "warm_repeated_same_buffers")
              << ", \"cache_evict_bytes\": "
              << (target_buffers_evicted ? kCacheEvictionBytes : 0)
              << ", \"cache_evict_in_timed_region\": false, \"quant_only_cache_state\": "
              << json_string("warm_repeated_same_buffers") << ", \"warmups\": "
              << options.warmups << ", \"repeats\": " << options.repeats
              << ", \"quant_only\": {\"p50_ms\": " << quant_timing.p50_ms
              << ", \"p95_ms\": " << quant_timing.p95_ms << "}, \"gemm_only\": ";
    if (selected == nullptr) {
        std::cout << "null, \"quant_plus_gemm\": null},\n";
    } else {
        std::cout << "{\"p50_ms\": " << selected->gemm.p50_ms
                  << ", \"p95_ms\": " << selected->gemm.p95_ms
                  << ", \"tflops_p50\": " << tflops(selected->gemm.p50_ms)
                  << "}, \"quant_plus_gemm\": {\"p50_ms\": " << combined_timing->p50_ms
                  << ", \"p95_ms\": " << combined_timing->p95_ms
                  << ", \"tflops_p50\": " << tflops(combined_timing->p50_ms) << "}},\n";
    }
    std::cout << "  \"cache_eviction\": {\"enabled\": "
              << (cache_eviction == nullptr ? "false" : "true")
              << ", \"bytes\": " << (cache_eviction == nullptr ? 0 : kCacheEvictionBytes)
              << ", \"passes_per_gemm_sample\": " << (cache_eviction == nullptr ? 0 : 1)
              << ", \"outside_timed_region\": true, \"device_l2_cache_bytes\": "
              << properties.l2CacheSize << ", \"validation_checksum_u64\": ";
    if (cache_eviction == nullptr) {
        std::cout << "null, \"validation_checksum_matches\": null, \"validation_ms\": null, "
                     "\"validation_bandwidth_gbps\": null},\n";
    } else {
        const double validation_ms = cache_eviction->validation_ms();
        const double bandwidth_gbps =
            static_cast<double>(kCacheEvictionBytes) / (validation_ms * 1.0e6);
        std::cout << json_string(std::to_string(cache_eviction->checksum()))
                  << ", \"validation_checksum_matches\": true, \"validation_ms\": "
                  << validation_ms
                  << ", \"validation_bandwidth_gbps\": " << bandwidth_gbps << "},\n";
    }
    std::cout << "  \"activation_quantization_check\": {\"passed\": "
              << (quant_check.passed() ? "true" : "false")
              << ", \"fp8_expected\": " << (quant_check.fp8_requested ? "true" : "false")
              << ", \"fp8_byte_exact\": " << (quant_check.fp8_exact ? "true" : "false")
              << ", \"fp8_mismatch_count\": " << quant_check.fp8_mismatch_count
              << ", \"fp8_first_mismatch\": "
              << (quant_check.fp8_exact ? "null" : std::to_string(quant_check.fp8_first_mismatch))
              << ", \"scales_expected\": "
              << (quant_check.scale_requested ? "true" : "false")
              << ", \"scale_bit_exact\": " << (quant_check.scale_exact ? "true" : "false")
              << ", \"scale_mismatch_count\": " << quant_check.scale_mismatch_count
              << ", \"scale_first_mismatch_byte\": "
              << (quant_check.scale_exact ? "null"
                                          : std::to_string(quant_check.scale_first_mismatch_byte))
              << "},\n  \"correctness\": ";
    if (!error_metrics.has_value()) {
        std::cout << "null,\n";
    } else {
        std::cout << "{\"output_converted_from\": \"bf16\", \"nonfinite\": "
                  << error_metrics->nonfinite << ", \"max_abs\": " << error_metrics->max_abs
                  << ", \"relative_l2\": " << json_number(error_metrics->relative_l2)
                  << ", \"cosine\": " << json_number(error_metrics->cosine)
                  << ", \"limits\": {\"relative_l2_max\": " << kRelativeL2Limit
                  << ", \"cosine_min\": " << kCosineLimit << "}},\n";
    }
    std::cout << "  \"throughput\": {\"operation_count\": " << operations
              << ", \"gemm_tflops_p50\": "
              << (selected == nullptr ? 0.0 : tflops(selected->gemm.p50_ms))
              << ", \"quant_plus_gemm_tflops_p50\": "
              << (!combined_timing.has_value() ? 0.0 : tflops(combined_timing->p50_ms)) << "}\n"
              << "}\n";
}

int run(const Options& options) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count != 1) {
        throw ExitError(
            4,
            "SQ8 CK component requires an isolated R9700; use tools/run-sq8-ck-component.sh "
            "or set HIP_VISIBLE_DEVICES to the R9700 HIP ordinal");
    }
    if (options.device < 0 || options.device >= device_count) {
        throw ExitError(2, "--device is outside HIP enumeration");
    }
    HIP_CHECK(hipSetDevice(options.device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, options.device));
    const std::string arch = normalized_arch(properties);
    if (arch != "gfx1201") {
        throw ExitError(4, "SQ8 CK component requires gfx1201; selected device is " + arch);
    }

    const std::size_t activation_elements = checked_mul(options.m, options.k, "activation elements");
    const std::size_t weight_elements = checked_mul(options.n, options.k, "weight elements");
    const std::size_t output_elements = checked_mul(options.m, options.n, "output elements");
    const std::size_t activation_scale_elements =
        checked_mul(options.m, options.k / kScaleBlock, "activation scale elements");
    const std::size_t weight_scale_elements =
        checked_mul(options.n / kScaleBlock, options.k / kScaleBlock, "weight scale elements");
    std::size_t working_bytes = estimated_working_bytes(activation_elements,
                                                        weight_elements,
                                                        output_elements,
                                                        activation_scale_elements,
                                                        weight_scale_elements);
    if (options.cache_mode == CacheMode::TargetBuffersEvicted) {
        working_bytes = checked_add(working_bytes, kCacheEvictionBytes, "component working bytes");
        working_bytes = checked_add(
            working_bytes,
            static_cast<std::size_t>(kCacheEvictionBlocks) * sizeof(unsigned long long),
            "component working bytes");
    }
    if (working_bytes > kMaxWorkingBytes) {
        throw ExitError(2,
                        "SQ8 CK component estimated working set " +
                            std::to_string(working_bytes) + " exceeds " +
                            std::to_string(kMaxWorkingBytes) + " bytes");
    }
    std::size_t device_allocation_bytes = 0;
    const auto add_device_bytes = [&](std::size_t bytes) {
        device_allocation_bytes =
            checked_add(device_allocation_bytes, bytes, "device allocation bytes");
    };
    add_device_bytes(checked_mul(activation_elements, sizeof(float), "device activation bytes"));
    add_device_bytes(activation_elements);
    add_device_bytes(checked_mul(
        activation_scale_elements, sizeof(float), "device activation scale bytes"));
    add_device_bytes(weight_elements);
    add_device_bytes(
        checked_mul(weight_scale_elements, sizeof(float), "device weight scale bytes"));
    add_device_bytes(
        checked_mul(output_elements, sizeof(ck::bhalf_t), "device output bytes"));
    if (options.cache_mode == CacheMode::TargetBuffersEvicted) {
        add_device_bytes(kCacheEvictionBytes);
        add_device_bytes(static_cast<std::size_t>(kCacheEvictionBlocks) *
                         sizeof(unsigned long long));
    }
    std::size_t free_device_bytes = 0;
    std::size_t total_device_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_device_bytes, &total_device_bytes));
    if (device_allocation_bytes > free_device_bytes) {
        throw ExitError(2,
                        "SQ8 CK component requires " +
                            std::to_string(device_allocation_bytes) +
                            " device bytes but only " + std::to_string(free_device_bytes) +
                            " are free");
    }

    auto weight_bytes = read_exact_bytes(options.weight_path, weight_elements, "weight");
    validate_fp8_ocp(weight_bytes, "weight");
    auto weight_scale_bf16 = read_exact_bytes(options.weight_scale_path,
                                              checked_mul(weight_scale_elements,
                                                          sizeof(std::uint16_t),
                                                          "weight scale bytes"),
                                              "weight scale");
    std::vector<float> weight_scales = decode_bf16_le(weight_scale_bf16, "weight scale");
    for (std::size_t index = 0; index < weight_scales.size(); ++index) {
        if (weight_scales[index] <= 0.0f) {
            throw ExitError(2,
                            "weight scale must be positive at element " +
                                std::to_string(index));
        }
    }
    std::vector<std::uint8_t>().swap(weight_scale_bf16);
    auto activation_bytes = read_exact_bytes(options.activation_path,
                                             checked_mul(activation_elements,
                                                         sizeof(float),
                                                         "activation bytes"),
                                             "activation");
    std::vector<float> activation = decode_f32_le(activation_bytes, "activation");
    std::vector<std::uint8_t>().swap(activation_bytes);
    auto oracle_bytes = read_exact_bytes(options.oracle_path,
                                         checked_mul(output_elements,
                                                     sizeof(float),
                                                     "oracle bytes"),
                                         "oracle");
    const std::vector<float> oracle = decode_f32_le(oracle_bytes, "oracle");
    std::vector<std::uint8_t>().swap(oracle_bytes);

    std::optional<std::vector<std::uint8_t>> expected_activation_fp8;
    if (options.expected_activation_fp8_path.has_value()) {
        expected_activation_fp8 = read_exact_bytes(*options.expected_activation_fp8_path,
                                                   activation_elements,
                                                   "expected activation FP8");
        validate_fp8_ocp(*expected_activation_fp8, "expected activation FP8");
    }
    std::optional<std::vector<std::uint8_t>> expected_activation_scales;
    if (options.expected_activation_scale_path.has_value()) {
        expected_activation_scales = read_exact_bytes(
            *options.expected_activation_scale_path,
            checked_mul(activation_scale_elements, sizeof(float), "expected activation scale bytes"),
            "expected activation scales");
        (void)decode_f32_le(*expected_activation_scales, "expected activation scales");
    }

    DeviceBuffer device_activation(
        checked_mul(activation_elements, sizeof(float), "device activation bytes"));
    DeviceBuffer device_activation_fp8(activation_elements);
    DeviceBuffer device_activation_scales(
        checked_mul(activation_scale_elements, sizeof(float), "device activation scale bytes"));
    DeviceBuffer device_weights(weight_bytes.size());
    DeviceBuffer device_weight_scales(
        checked_mul(weight_scale_elements, sizeof(float), "device weight scale bytes"));
    DeviceBuffer device_output(
        checked_mul(output_elements, sizeof(ck::bhalf_t), "device output bytes"));
    HipStream stream;
    QuantizationKernel quantization_kernel(arch);
    std::unique_ptr<CacheEvictionState> cache_eviction;
    if (options.cache_mode == CacheMode::TargetBuffersEvicted) {
        cache_eviction = std::make_unique<CacheEvictionState>(arch, stream.get());
    }

    HIP_CHECK(hipMemcpyAsync(device_activation.get(),
                             activation.data(),
                             device_activation.bytes(),
                             hipMemcpyHostToDevice,
                             stream.get()));
    HIP_CHECK(hipMemcpyAsync(device_weights.get(),
                             weight_bytes.data(),
                             device_weights.bytes(),
                             hipMemcpyHostToDevice,
                             stream.get()));
    HIP_CHECK(hipMemcpyAsync(device_weight_scales.get(),
                             weight_scales.data(),
                             device_weight_scales.bytes(),
                             hipMemcpyHostToDevice,
                             stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    std::vector<float>().swap(activation);
    std::vector<std::uint8_t>().swap(weight_bytes);
    std::vector<float>().swap(weight_scales);

    const auto quant_samples = measure_gpu_events(
        options.warmups,
        options.repeats,
        stream.get(),
        []() {},
        [&]() {
            quantization_kernel.launch(device_activation,
                                       device_activation_fp8,
                                       device_activation_scales,
                                       options,
                                       stream.get());
        });
    const Timing quant_timing = summarize_timing(quant_samples);

    std::vector<std::uint8_t> actual_activation_fp8(activation_elements);
    std::vector<float> actual_activation_scales(activation_scale_elements);
    HIP_CHECK(hipMemcpyAsync(actual_activation_fp8.data(),
                             device_activation_fp8.get(),
                             actual_activation_fp8.size(),
                             hipMemcpyDeviceToHost,
                             stream.get()));
    HIP_CHECK(hipMemcpyAsync(actual_activation_scales.data(),
                             device_activation_scales.get(),
                             device_activation_scales.bytes(),
                             hipMemcpyDeviceToHost,
                             stream.get()));
    HIP_CHECK(hipStreamSynchronize(stream.get()));
    for (std::size_t index = 0; index < actual_activation_scales.size(); ++index) {
        if (!std::isfinite(actual_activation_scales[index]) || actual_activation_scales[index] <= 0.0f) {
            throw std::runtime_error("GPU activation quantizer produced an invalid scale at element " +
                                     std::to_string(index));
        }
    }
    validate_fp8_ocp(actual_activation_fp8, "GPU activation FP8");

    QuantizationCheck quant_check;
    if (expected_activation_fp8.has_value()) {
        quant_check.fp8_requested = true;
        const ByteComparison comparison =
            compare_bytes(actual_activation_fp8, *expected_activation_fp8);
        quant_check.fp8_exact = comparison.exact;
        quant_check.fp8_mismatch_count = comparison.mismatch_count;
        quant_check.fp8_first_mismatch = comparison.first_mismatch;
    }
    if (expected_activation_scales.has_value()) {
        quant_check.scale_requested = true;
        const ByteComparison comparison =
            compare_bytes(encode_f32_le(actual_activation_scales), *expected_activation_scales);
        quant_check.scale_exact = comparison.exact;
        quant_check.scale_mismatch_count = comparison.mismatch_count;
        quant_check.scale_first_mismatch_byte = comparison.first_mismatch;
    }

    auto groups = make_candidate_groups();
    std::size_t candidate_count = 0;
    std::size_t supported_count = 0;
    std::vector<CandidateMeasurement> candidate_measurements;
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        CandidateGroup& group = groups[group_index];
        candidate_count += group.instances.size();
        for (std::size_t instance_index = 0; instance_index < group.instances.size();
             ++instance_index) {
            auto argument = make_argument(*group.instances[instance_index],
                                          device_activation_fp8,
                                          device_weights,
                                          device_output,
                                          device_activation_scales,
                                          device_weight_scales,
                                          options);
            if (!group.instances[instance_index]->IsSupportedArgument(argument.get())) {
                continue;
            }
            ++supported_count;
            candidate_measurements.push_back(measure_candidate(group_index,
                                                               instance_index,
                                                               group,
                                                               *argument,
                                                               device_output,
                                                               output_elements,
                                                               oracle,
                                                               options,
                                                               cache_eviction.get(),
                                                               stream.get()));
        }
    }

    CandidateMeasurement* selected = nullptr;
    for (CandidateMeasurement& candidate : candidate_measurements) {
        if (!candidate.runnable || !candidate.numerically_valid) {
            continue;
        }
        if (selected == nullptr || candidate.gemm.p50_ms < selected->gemm.p50_ms) {
            selected = &candidate;
        }
    }

    std::optional<Timing> combined_timing;
    std::optional<ErrorMetrics> error_metrics;
    if (selected != nullptr) {
        CandidateGroup& group = groups[selected->group_index];
        DeviceOp& operation = *group.instances[selected->instance_index];
        auto argument = make_argument(operation,
                                      device_activation_fp8,
                                      device_weights,
                                      device_output,
                                      device_activation_scales,
                                      device_weight_scales,
                                      options);
        auto invoker = operation.MakeInvokerPointer();
        const StreamConfig config = stream_config(stream.get());
        const auto combined_samples = measure_gpu_events(
            options.warmups,
            options.repeats,
            stream.get(),
            [&]() {
                if (cache_eviction != nullptr) {
                    cache_eviction->launch(stream.get());
                }
            },
            [&]() {
                quantization_kernel.launch(device_activation,
                                           device_activation_fp8,
                                           device_activation_scales,
                                           options,
                                           stream.get());
                (void)invoker->Run(argument.get(), config);
                HIP_CHECK(hipGetLastError());
            });
        combined_timing = summarize_timing(combined_samples);

        (void)invoker->Run(argument.get(), config);
        HIP_CHECK(hipGetLastError());
        std::vector<ck::bhalf_t> output(output_elements);
        HIP_CHECK(hipMemcpyAsync(output.data(),
                                 device_output.get(),
                                 device_output.bytes(),
                                 hipMemcpyDeviceToHost,
                                 stream.get()));
        HIP_CHECK(hipStreamSynchronize(stream.get()));
        error_metrics = calculate_error_metrics(output, oracle);
    }

    bool passed = selected != nullptr && error_metrics.has_value() && quant_check.passed();
    std::string failure_reason;
    if (supported_count == 0) {
        passed = false;
        failure_reason = "CK reported no supported ABScale instance for the requested shape";
    } else if (selected == nullptr &&
               std::none_of(candidate_measurements.begin(),
                            candidate_measurements.end(),
                            [](const CandidateMeasurement& candidate) {
                                return candidate.runnable;
                            })) {
        passed = false;
        failure_reason = "no supported CK ABScale instance completed event measurement";
    } else if (selected == nullptr) {
        passed = false;
        failure_reason = "no runnable CK ABScale instance passed the numerical gate";
    } else if (!quant_check.passed()) {
        passed = false;
        failure_reason = "GPU activation quantization differs from expected bytes";
    } else if (error_metrics->nonfinite != 0) {
        passed = false;
        failure_reason = "BF16 output contains non-finite values";
    } else if (!std::isfinite(error_metrics->relative_l2) ||
               error_metrics->relative_l2 > kRelativeL2Limit) {
        passed = false;
        failure_reason = "relative L2 exceeds 5e-3";
    } else if (!std::isfinite(error_metrics->cosine) || error_metrics->cosine < kCosineLimit) {
        passed = false;
        failure_reason = "cosine similarity is below 0.9999";
    }

    print_result_json(options,
                      properties,
                      arch,
                      candidate_count,
                      supported_count,
                      candidate_measurements,
                      selected,
                      quant_timing,
                      combined_timing,
                      error_metrics,
                      quant_check,
                      cache_eviction.get(),
                      working_bytes,
                      device_allocation_bytes,
                      free_device_bytes,
                      passed,
                      failure_reason);
    return passed ? 0 : 5;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const ExitError& error) {
        print_error_json(error.what());
        return error.code();
    } catch (const std::exception& error) {
        print_error_json(error.what());
        return 3;
    }
}
