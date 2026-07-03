// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "ullm_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace {

thread_local std::string last_error;

void set_error(const char *message) {
    last_error = message == nullptr ? "" : message;
}

void copy_cstr(char *dst, size_t dst_len, const std::string &src) {
    if (dst == nullptr || dst_len == 0) {
        return;
    }
    const size_t copy_len = std::min(dst_len - 1, src.size());
    std::memcpy(dst, src.data(), copy_len);
    dst[copy_len] = '\0';
}

class HipRuntime {
public:
    int device_count() {
        load_once();
        if (hip_get_device_count_ == nullptr) {
            return 0;
        }
        int count = 0;
        const int status = hip_get_device_count_(&count);
        if (status != 0 || count < 0) {
            return 0;
        }
        return count;
    }

    int runtime_version() {
        load_once();
        if (hip_runtime_get_version_ == nullptr) {
            return 0;
        }
        int version = 0;
        if (hip_runtime_get_version_(&version) != 0) {
            return 0;
        }
        return version;
    }

    std::string device_name(int device_id) {
        load_once();
        if (hip_device_get_name_ == nullptr) {
            return "HIP device " + std::to_string(device_id);
        }
        std::array<char, 128> name{};
        if (hip_device_get_name_(name.data(), static_cast<int>(name.size()), device_id) != 0) {
            return "HIP device " + std::to_string(device_id);
        }
        name.back() = '\0';
        return std::string(name.data());
    }

    uint64_t device_total_mem(int device_id) {
        load_once();
        if (hip_device_total_mem_ == nullptr) {
            return 0;
        }
        size_t bytes = 0;
        if (hip_device_total_mem_(&bytes, device_id) != 0) {
            return 0;
        }
        return static_cast<uint64_t>(bytes);
    }

    void device_compute_capability(int device_id, int *major, int *minor) {
        load_once();
        if (major == nullptr || minor == nullptr) {
            return;
        }
        *major = 0;
        *minor = 0;
        if (hip_device_compute_capability_ == nullptr) {
            return;
        }
        if (hip_device_compute_capability_(major, minor, device_id) != 0) {
            *major = 0;
            *minor = 0;
        }
    }

    bool set_device(int device_id) {
        load_once();
        if (hip_set_device_ == nullptr) {
            return false;
        }
        return hip_set_device_(device_id) == 0;
    }

    void *malloc_device(size_t bytes, int device_id) {
        load_once();
        if (hip_malloc_ == nullptr || !set_device(device_id)) {
            return nullptr;
        }
        void *ptr = nullptr;
        if (hip_malloc_(&ptr, bytes) != 0) {
            return nullptr;
        }
        return ptr;
    }

    bool free_device(void *ptr, int device_id) {
        load_once();
        if (ptr == nullptr) {
            return true;
        }
        if (hip_free_ == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_free_(ptr) == 0;
    }

    void *create_stream(int device_id) {
        load_once();
        if (hip_stream_create_ == nullptr || !set_device(device_id)) {
            return nullptr;
        }
        void *stream = nullptr;
        if (hip_stream_create_(&stream) != 0) {
            return nullptr;
        }
        return stream;
    }

    bool destroy_stream(void *stream, int device_id) {
        load_once();
        if (stream == nullptr) {
            return true;
        }
        if (hip_stream_destroy_ == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_stream_destroy_(stream) == 0;
    }

    bool synchronize_stream(void *stream, int device_id) {
        load_once();
        if (stream == nullptr) {
            return true;
        }
        if (hip_stream_synchronize_ == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_stream_synchronize_(stream) == 0;
    }

    bool synchronize_device(int device_id) {
        load_once();
        if (hip_device_synchronize_ == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_device_synchronize_() == 0;
    }

    bool copy_async(void *dst, const void *src, size_t bytes, int kind, void *stream, int device_id) {
        load_once();
        if (bytes == 0) {
            return true;
        }
        if (hip_memcpy_async_ == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_memcpy_async_(dst, src, bytes, kind, stream) == 0;
    }

    bool module_load_data(void **module, const void *image, int device_id) {
        load_once();
        if (hip_module_load_data_ == nullptr || module == nullptr || image == nullptr ||
            !set_device(device_id)) {
            return false;
        }
        return hip_module_load_data_(module, image) == 0;
    }

    bool module_get_function(void **function, void *module, const char *name, int device_id) {
        load_once();
        if (hip_module_get_function_ == nullptr || function == nullptr || module == nullptr ||
            name == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_module_get_function_(function, module, name) == 0;
    }

    bool module_launch_kernel(
        void *function,
        unsigned int grid_x,
        unsigned int block_x,
        void **kernel_params,
        void *stream,
        int device_id) {
        load_once();
        if (hip_module_launch_kernel_ == nullptr || function == nullptr || kernel_params == nullptr ||
            !set_device(device_id)) {
            return false;
        }
        return hip_module_launch_kernel_(
                   function,
                   grid_x,
                   1,
                   1,
                   block_x,
                   1,
                   1,
                   0,
                   stream,
                   kernel_params,
                   nullptr) == 0;
    }

    bool module_unload(void *module, int device_id) {
        load_once();
        if (module == nullptr) {
            return true;
        }
        if (hip_module_unload_ == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_module_unload_(module) == 0;
    }

private:
    using hip_get_device_count_fn = int (*)(int *);
    using hip_runtime_get_version_fn = int (*)(int *);
    using hip_device_get_name_fn = int (*)(char *, int, int);
    using hip_device_total_mem_fn = int (*)(size_t *, int);
    using hip_device_compute_capability_fn = int (*)(int *, int *, int);
    using hip_set_device_fn = int (*)(int);
    using hip_malloc_fn = int (*)(void **, size_t);
    using hip_free_fn = int (*)(void *);
    using hip_stream_create_fn = int (*)(void **);
    using hip_stream_destroy_fn = int (*)(void *);
    using hip_stream_synchronize_fn = int (*)(void *);
    using hip_device_synchronize_fn = int (*)();
    using hip_memcpy_async_fn = int (*)(void *, const void *, size_t, int, void *);
    using hip_module_load_data_fn = int (*)(void **, const void *);
    using hip_module_get_function_fn = int (*)(void **, void *, const char *);
    using hip_module_launch_kernel_fn = int (*)(
        void *,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        void *,
        void **,
        void **);
    using hip_module_unload_fn = int (*)(void *);

    void load_once() {
        std::call_once(load_flag_, [this]() {
#if defined(__linux__)
            constexpr std::array<const char *, 3> candidates = {
                "libamdhip64.so",
                "libamdhip64.so.6",
                "libhiprtc.so",
            };
            for (const char *candidate : candidates) {
                handle_ = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
                if (handle_ != nullptr) {
                    break;
                }
            }
            if (handle_ == nullptr) {
                return;
            }
            hip_get_device_count_ =
                reinterpret_cast<hip_get_device_count_fn>(dlsym(handle_, "hipGetDeviceCount"));
            hip_runtime_get_version_ =
                reinterpret_cast<hip_runtime_get_version_fn>(dlsym(handle_, "hipRuntimeGetVersion"));
            hip_device_get_name_ =
                reinterpret_cast<hip_device_get_name_fn>(dlsym(handle_, "hipDeviceGetName"));
            hip_device_total_mem_ =
                reinterpret_cast<hip_device_total_mem_fn>(dlsym(handle_, "hipDeviceTotalMem"));
            hip_device_compute_capability_ = reinterpret_cast<hip_device_compute_capability_fn>(
                dlsym(handle_, "hipDeviceComputeCapability"));
            hip_set_device_ = reinterpret_cast<hip_set_device_fn>(dlsym(handle_, "hipSetDevice"));
            hip_malloc_ = reinterpret_cast<hip_malloc_fn>(dlsym(handle_, "hipMalloc"));
            hip_free_ = reinterpret_cast<hip_free_fn>(dlsym(handle_, "hipFree"));
            hip_stream_create_ = reinterpret_cast<hip_stream_create_fn>(dlsym(handle_, "hipStreamCreate"));
            hip_stream_destroy_ = reinterpret_cast<hip_stream_destroy_fn>(dlsym(handle_, "hipStreamDestroy"));
            hip_stream_synchronize_ = reinterpret_cast<hip_stream_synchronize_fn>(
                dlsym(handle_, "hipStreamSynchronize"));
            hip_device_synchronize_ =
                reinterpret_cast<hip_device_synchronize_fn>(dlsym(handle_, "hipDeviceSynchronize"));
            hip_memcpy_async_ =
                reinterpret_cast<hip_memcpy_async_fn>(dlsym(handle_, "hipMemcpyAsync"));
            hip_module_load_data_ =
                reinterpret_cast<hip_module_load_data_fn>(dlsym(handle_, "hipModuleLoadData"));
            hip_module_get_function_ =
                reinterpret_cast<hip_module_get_function_fn>(dlsym(handle_, "hipModuleGetFunction"));
            hip_module_launch_kernel_ = reinterpret_cast<hip_module_launch_kernel_fn>(
                dlsym(handle_, "hipModuleLaunchKernel"));
            hip_module_unload_ =
                reinterpret_cast<hip_module_unload_fn>(dlsym(handle_, "hipModuleUnload"));
#endif
        });
    }

    std::once_flag load_flag_;
    void *handle_ = nullptr;
    hip_get_device_count_fn hip_get_device_count_ = nullptr;
    hip_runtime_get_version_fn hip_runtime_get_version_ = nullptr;
    hip_device_get_name_fn hip_device_get_name_ = nullptr;
    hip_device_total_mem_fn hip_device_total_mem_ = nullptr;
    hip_device_compute_capability_fn hip_device_compute_capability_ = nullptr;
    hip_set_device_fn hip_set_device_ = nullptr;
    hip_malloc_fn hip_malloc_ = nullptr;
    hip_free_fn hip_free_ = nullptr;
    hip_stream_create_fn hip_stream_create_ = nullptr;
    hip_stream_destroy_fn hip_stream_destroy_ = nullptr;
    hip_stream_synchronize_fn hip_stream_synchronize_ = nullptr;
    hip_device_synchronize_fn hip_device_synchronize_ = nullptr;
    hip_memcpy_async_fn hip_memcpy_async_ = nullptr;
    hip_module_load_data_fn hip_module_load_data_ = nullptr;
    hip_module_get_function_fn hip_module_get_function_ = nullptr;
    hip_module_launch_kernel_fn hip_module_launch_kernel_ = nullptr;
    hip_module_unload_fn hip_module_unload_ = nullptr;
};

class HipRtcRuntime {
public:
    bool compile_aq4_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, aq4_kernel_source(), "ullm_aq4_dequant_f32.hip", code, error);
    }

    bool compile_matvec_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, matvec_kernel_source(), "ullm_matvec_f32.hip", code, error);
    }

    bool compile_rmsnorm_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, rmsnorm_kernel_source(), "ullm_rmsnorm_f32.hip", code, error);
    }

    bool compile_silu_mul_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, silu_mul_kernel_source(), "ullm_silu_mul_f32.hip", code, error);
    }

    bool compile_depthwise_conv1d_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            depthwise_conv1d_kernel_source(),
            "ullm_depthwise_conv1d_f32.hip",
            code,
            error);
    }

private:
    using hiprtc_create_program_fn =
        int (*)(void **, const char *, const char *, int, const char *const *, const char *const *);
    using hiprtc_compile_program_fn = int (*)(void *, int, const char *const *);
    using hiprtc_get_code_size_fn = int (*)(void *, size_t *);
    using hiprtc_get_code_fn = int (*)(void *, char *);
    using hiprtc_get_log_size_fn = int (*)(void *, size_t *);
    using hiprtc_get_log_fn = int (*)(void *, char *);
    using hiprtc_destroy_program_fn = int (*)(void **);
    using hiprtc_get_error_string_fn = const char *(*)(int);

    bool compile_kernel(
        const std::string &arch,
        const char *source,
        const char *name,
        std::vector<char> *code,
        std::string *error) {
        load_once();
        if (!available()) {
            append_error(error, "HIPRTC is not available");
            return false;
        }
        if (code == nullptr) {
            append_error(error, "HIPRTC output code pointer is null");
            return false;
        }

        void *program = nullptr;
        int status = hiprtc_create_program_(
            &program,
            source,
            name,
            0,
            nullptr,
            nullptr);
        if (status != 0 || program == nullptr) {
            append_error(error, "hiprtcCreateProgram failed: " + error_string(status));
            return false;
        }

        const auto destroy_program = [&]() {
            if (program != nullptr) {
                hiprtc_destroy_program_(&program);
            }
        };

        const std::string arch_option = "--offload-arch=" + arch;
        const std::array<const char *, 3> options = {
            arch_option.c_str(),
            "--std=c++17",
            "-O3",
        };
        status = hiprtc_compile_program_(
            program,
            static_cast<int>(options.size()),
            options.data());
        if (status != 0) {
            append_error(
                error,
                "hiprtcCompileProgram failed for " + arch + ": " + error_string(status) +
                    "\n" + program_log(program));
            destroy_program();
            return false;
        }

        size_t code_size = 0;
        status = hiprtc_get_code_size_(program, &code_size);
        if (status != 0 || code_size == 0) {
            append_error(error, "hiprtcGetCodeSize failed: " + error_string(status));
            destroy_program();
            return false;
        }
        code->assign(code_size, '\0');
        status = hiprtc_get_code_(program, code->data());
        if (status != 0) {
            append_error(error, "hiprtcGetCode failed: " + error_string(status));
            destroy_program();
            return false;
        }
        destroy_program();
        return true;
    }

    static const char *aq4_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_dequant_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long elements,
    float *output,
    unsigned int *error_out) {
    const unsigned long long element =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element >= elements) {
        return;
    }
    const unsigned char packed = indices[element >> 1];
    const unsigned char codebook_index =
        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const unsigned long long group = element / group_size;
    const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
    if (scale_index >= scale_count) {
        atomicOr(error_out, 1u);
        output[element] = 0.0f;
        return;
    }
    output[element] = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
}
)";
    }

    static const char *matvec_kernel_source() {
        return R"(
extern "C" __global__ void ullm_matvec_f32_kernel(
    const float *matrix,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        for (unsigned long long col = tid; col < cols; col += blockDim.x) {
            sum += matrix[row_offset + col] * input[col];
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0 && row < rows) {
        output[row] = partial[0];
    }
}
)";
    }

    static const char *rmsnorm_kernel_source() {
        return R"(
extern "C" __global__ void ullm_rmsnorm_f32_kernel(
    const float *input,
    const float *weight,
    unsigned long long elements,
    float epsilon,
    float *output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned long long index = tid; index < elements; index += blockDim.x) {
        const float value = input[index];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(elements) + epsilon);
    for (unsigned long long index = tid; index < elements; index += blockDim.x) {
        output[index] = input[index] * inv_rms * weight[index];
    }
}
)";
    }

    static const char *silu_mul_kernel_source() {
        return R"(
extern "C" __global__ void ullm_silu_mul_f32_kernel(
    const float *gate,
    const float *up,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float gate_value = gate[index];
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    output[index] = gate_value * sigmoid * up[index];
}
)";
    }

    static const char *depthwise_conv1d_kernel_source() {
        return R"(
extern "C" __global__ void ullm_depthwise_conv1d_f32_kernel(
    const float *input,
    const float *weight,
    unsigned long long channels,
    unsigned long long sequence_len,
    unsigned long long kernel_size,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = channels * sequence_len;
    if (index >= elements) {
        return;
    }
    const unsigned long long timestep = index / channels;
    const unsigned long long channel = index - timestep * channels;
    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        if (timestep < kernel) {
            break;
        }
        const unsigned long long source_timestep = timestep - kernel;
        sum += input[source_timestep * channels + channel] *
               weight[channel * kernel_size + kernel];
    }
    output[index] = sum;
}
)";
    }

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    bool available() const {
        return hiprtc_create_program_ != nullptr && hiprtc_compile_program_ != nullptr &&
               hiprtc_get_code_size_ != nullptr && hiprtc_get_code_ != nullptr &&
               hiprtc_get_log_size_ != nullptr && hiprtc_get_log_ != nullptr &&
               hiprtc_destroy_program_ != nullptr;
    }

    std::string error_string(int status) const {
        if (hiprtc_get_error_string_ == nullptr) {
            return std::to_string(status);
        }
        const char *message = hiprtc_get_error_string_(status);
        if (message == nullptr) {
            return std::to_string(status);
        }
        return std::string(message);
    }

    std::string program_log(void *program) const {
        if (program == nullptr || hiprtc_get_log_size_ == nullptr || hiprtc_get_log_ == nullptr) {
            return "";
        }
        size_t log_size = 0;
        if (hiprtc_get_log_size_(program, &log_size) != 0 || log_size == 0) {
            return "";
        }
        std::string log(log_size, '\0');
        if (hiprtc_get_log_(program, log.data()) != 0) {
            return "";
        }
        while (!log.empty() && log.back() == '\0') {
            log.pop_back();
        }
        return log;
    }

    void load_once() {
        std::call_once(load_flag_, [this]() {
#if defined(__linux__)
            constexpr std::array<const char *, 3> candidates = {
                "libhiprtc.so",
                "libhiprtc.so.7",
                "libhiprtc.so.6",
            };
            for (const char *candidate : candidates) {
                handle_ = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
                if (handle_ != nullptr) {
                    break;
                }
            }
            if (handle_ == nullptr) {
                return;
            }
            hiprtc_create_program_ = reinterpret_cast<hiprtc_create_program_fn>(
                dlsym(handle_, "hiprtcCreateProgram"));
            hiprtc_compile_program_ = reinterpret_cast<hiprtc_compile_program_fn>(
                dlsym(handle_, "hiprtcCompileProgram"));
            hiprtc_get_code_size_ = reinterpret_cast<hiprtc_get_code_size_fn>(
                dlsym(handle_, "hiprtcGetCodeSize"));
            hiprtc_get_code_ =
                reinterpret_cast<hiprtc_get_code_fn>(dlsym(handle_, "hiprtcGetCode"));
            hiprtc_get_log_size_ = reinterpret_cast<hiprtc_get_log_size_fn>(
                dlsym(handle_, "hiprtcGetProgramLogSize"));
            hiprtc_get_log_ =
                reinterpret_cast<hiprtc_get_log_fn>(dlsym(handle_, "hiprtcGetProgramLog"));
            hiprtc_destroy_program_ = reinterpret_cast<hiprtc_destroy_program_fn>(
                dlsym(handle_, "hiprtcDestroyProgram"));
            hiprtc_get_error_string_ = reinterpret_cast<hiprtc_get_error_string_fn>(
                dlsym(handle_, "hiprtcGetErrorString"));
#endif
        });
    }

    std::once_flag load_flag_;
    void *handle_ = nullptr;
    hiprtc_create_program_fn hiprtc_create_program_ = nullptr;
    hiprtc_compile_program_fn hiprtc_compile_program_ = nullptr;
    hiprtc_get_code_size_fn hiprtc_get_code_size_ = nullptr;
    hiprtc_get_code_fn hiprtc_get_code_ = nullptr;
    hiprtc_get_log_size_fn hiprtc_get_log_size_ = nullptr;
    hiprtc_get_log_fn hiprtc_get_log_ = nullptr;
    hiprtc_destroy_program_fn hiprtc_destroy_program_ = nullptr;
    hiprtc_get_error_string_fn hiprtc_get_error_string_ = nullptr;
};

HipRuntime &hip_runtime() {
    static HipRuntime runtime;
    return runtime;
}

HipRtcRuntime &hiprtc_runtime() {
    static HipRtcRuntime runtime;
    return runtime;
}

uint32_t total_device_count() {
    return static_cast<uint32_t>(1 + hip_runtime().device_count());
}

void fill_cpu_device(ullm_device_info *info) {
    info->device_id = 0;
    copy_cstr(info->backend, sizeof(info->backend), "cpu");
    copy_cstr(info->name, sizeof(info->name), "host CPU fallback");
    info->total_global_mem = 0;
    info->compute_major = 0;
    info->compute_minor = 0;
    copy_cstr(info->gcn_arch_name, sizeof(info->gcn_arch_name), "");
    info->flags = 1u;
}

void fill_hip_device(uint32_t index, ullm_device_info *info) {
    const int hip_index = static_cast<int>(index - 1);
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(hip_index, &major, &minor);
    info->device_id = hip_index;
    copy_cstr(info->backend, sizeof(info->backend), "hip");
    copy_cstr(info->name, sizeof(info->name), hip_runtime().device_name(hip_index));
    info->total_global_mem = hip_runtime().device_total_mem(hip_index);
    info->compute_major = major;
    info->compute_minor = minor;
    copy_cstr(info->gcn_arch_name, sizeof(info->gcn_arch_name), "");
    info->flags = static_cast<uint32_t>(hip_runtime().runtime_version());
}

enum class BackendKind : uint32_t {
    Cpu = 0,
    Hip = 1,
};

constexpr int HIP_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int HIP_MEMCPY_DEVICE_TO_HOST = 2;

} // namespace

struct ullm_runtime_context {
    uint32_t device_index = 0;
    BackendKind backend = BackendKind::Cpu;
    int hip_device_id = -1;
};

struct ullm_runtime_buffer {
    BackendKind backend = BackendKind::Cpu;
    int hip_device_id = -1;
    void *ptr = nullptr;
    size_t bytes = 0;
};

struct ullm_runtime_stream {
    BackendKind backend = BackendKind::Cpu;
    int hip_device_id = -1;
    void *stream = nullptr;
};

namespace {

bool checked_range(size_t offset, size_t bytes, size_t total) {
    return offset <= total && bytes <= total - offset;
}

bool stream_matches_buffer(const ullm_runtime_buffer *buffer, const ullm_runtime_stream *stream) {
    if (stream == nullptr) {
        return true;
    }
    return buffer->backend == stream->backend && buffer->hip_device_id == stream->hip_device_id;
}

bool buffers_share_backend(
    const ullm_runtime_buffer *lhs,
    const ullm_runtime_buffer *rhs) {
    return lhs->backend == rhs->backend && lhs->hip_device_id == rhs->hip_device_id;
}

bool synchronize_hip_staging(const ullm_runtime_stream *stream, int device_id) {
    if (stream != nullptr) {
        return hip_runtime().synchronize_stream(stream->stream, device_id);
    }
    return hip_runtime().synchronize_device(device_id);
}

bool aq4_dequant_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    float *output) {
    for (size_t element = 0; element < elements; ++element) {
        const std::uint8_t packed = indices[element / 2];
        const std::uint8_t codebook_index =
            (element % 2 == 0) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
        const size_t group = element / group_size;
        const size_t scale_index = static_cast<size_t>(scale_indices[group]);
        if (scale_index >= scale_count) {
            set_error("AQ4 dequant scale index is out of range");
            return false;
        }
        output[element] = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
    }
    return true;
}

std::vector<std::string> hip_arch_candidates(int device_id) {
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    std::vector<std::string> candidates;
    if (major == 12 && minor == 0) {
        candidates.emplace_back("gfx1201");
        candidates.emplace_back("gfx1200");
    } else if (major == 10 && minor == 3) {
        candidates.emplace_back("gfx1030");
    } else if (major == 9) {
        candidates.emplace_back("gfx9" + std::to_string(minor) + "0");
    } else if (major > 0) {
        candidates.emplace_back("gfx" + std::to_string(major) + std::to_string(minor) + "0");
    }
    return candidates;
}

class HipAq4KernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return nullptr;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_aq4_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *function = nullptr;
            if (!hip_runtime().module_get_function(
                    &function,
                    module,
                    "ullm_aq4_dequant_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(error, compile_errors.empty() ? "failed to build AQ4 HIP kernel" : compile_errors);
        return nullptr;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
        std::string arch;
    };

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipAq4KernelCache &hip_aq4_kernel_cache() {
    static HipAq4KernelCache cache;
    return cache;
}

enum class HipAq4LaunchResult {
    Ok,
    InvalidArgument,
    RuntimeError,
};

HipAq4LaunchResult aq4_dequant_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return HipAq4LaunchResult::RuntimeError;
    }

    const size_t scale_value_bytes = scale_count * sizeof(float);
    void *device_scale_values = hip_runtime().malloc_device(scale_value_bytes, device_id);
    void *device_error = hip_runtime().malloc_device(sizeof(std::uint32_t), device_id);
    if (device_scale_values == nullptr || device_error == nullptr) {
        if (device_scale_values != nullptr) {
            hip_runtime().free_device(device_scale_values, device_id);
        }
        if (device_error != nullptr) {
            hip_runtime().free_device(device_error, device_id);
        }
        if (error != nullptr) {
            *error = "failed to allocate AQ4 HIP kernel temporary buffers";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const std::uint32_t zero = 0;
    if (!hip_runtime().copy_async(
            device_scale_values,
            scale_values,
            scale_value_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            device_error,
            &zero,
            sizeof(zero),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        hip_runtime().free_device(device_scale_values, device_id);
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to upload AQ4 HIP kernel temporary buffers";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    const unsigned int grid_x =
        static_cast<unsigned int>((elements + block_size - 1) / block_size);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &device_scale_values,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_elements,
        &output_ptr,
        &device_error,
    };
    if (!hip_runtime().module_launch_kernel(
            function,
            grid_x,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        hip_runtime().free_device(device_scale_values, device_id);
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 dequant";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    std::uint32_t host_error = 0;
    if (!hip_runtime().copy_async(
            &host_error,
            device_error,
            sizeof(host_error),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !synchronize_hip_staging(stream, device_id)) {
        hip_runtime().free_device(device_scale_values, device_id);
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to read AQ4 HIP kernel status";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    hip_runtime().free_device(device_scale_values, device_id);
    hip_runtime().free_device(device_error, device_id);
    if (host_error != 0) {
        if (error != nullptr) {
            *error = "AQ4 dequant scale index is out of range";
        }
        return HipAq4LaunchResult::InvalidArgument;
    }
    (void)required_output_bytes;
    return HipAq4LaunchResult::Ok;
}

ullm_status aq4_dequant_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    size_t required_index_bytes,
    size_t groups,
    size_t required_output_bytes,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_indices.data(),
            index_buffer->ptr,
            required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_indices.data(),
            scale_buffer->ptr,
            groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_codebook.data(),
            codebook_buffer->ptr,
            codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_dequant_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            scale_values,
            scale_count,
            group_size,
            tensor_scale,
            elements,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 materialized output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void matvec_f32_host(
    const float *matrix,
    const float *input,
    size_t rows,
    size_t cols,
    float *output) {
    for (size_t row = 0; row < rows; ++row) {
        const float *row_values = matrix + row * cols;
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            sum += row_values[col] * input[col];
        }
        output[row] = sum;
    }
}

class HipMatvecKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return nullptr;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_matvec_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *function = nullptr;
            if (!hip_runtime().module_get_function(
                    &function,
                    module,
                    "ullm_matvec_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(error, compile_errors.empty() ? "failed to build matvec HIP kernel" : compile_errors);
        return nullptr;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
        std::string arch;
    };

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipMatvecKernelCache &hip_matvec_kernel_cache() {
    static HipMatvecKernelCache cache;
    return cache;
}

bool matvec_f32_hip_kernel(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = matrix_buffer->hip_device_id;
    void *function = hip_matvec_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "matvec row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *matrix_ptr = matrix_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &matrix_ptr,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(rows),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 matvec";
        }
        return false;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        if (error != nullptr) {
            *error = "failed to synchronize f32 matvec HIP kernel";
        }
        return false;
    }
    return true;
}

ullm_status matvec_f32_hip_staging(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t required_matrix_bytes,
    size_t required_input_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_matrix(required_matrix_bytes / sizeof(float));
    std::vector<float> host_input(cols);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = matrix_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_matrix.data(),
            matrix_buffer->ptr,
            required_matrix_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 matvec HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 matvec HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    matvec_f32_host(host_matrix.data(), host_input.data(), rows, cols, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 matvec output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 matvec HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void rmsnorm_f32_host(
    const float *input,
    const float *weight,
    size_t elements,
    float epsilon,
    float *output) {
    float sum_squares = 0.0f;
    for (size_t index = 0; index < elements; ++index) {
        sum_squares += input[index] * input[index];
    }
    const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(elements) + epsilon);
    for (size_t index = 0; index < elements; ++index) {
        output[index] = input[index] * inv_rms * weight[index];
    }
}

class HipRmsNormKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return nullptr;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_rmsnorm_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *function = nullptr;
            if (!hip_runtime().module_get_function(
                    &function,
                    module,
                    "ullm_rmsnorm_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(error, compile_errors.empty() ? "failed to build RMSNorm HIP kernel" : compile_errors);
        return nullptr;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
        std::string arch;
    };

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipRmsNormKernelCache &hip_rmsnorm_kernel_cache() {
    static HipRmsNormKernelCache cache;
    return cache;
}

bool rmsnorm_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_rmsnorm_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &kernel_elements,
        &epsilon,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1,
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 RMSNorm";
        }
        return false;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        if (error != nullptr) {
            *error = "failed to synchronize f32 RMSNorm HIP kernel";
        }
        return false;
    }
    return true;
}

ullm_status rmsnorm_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(elements);
    std::vector<float> host_weight(elements);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RMSNorm HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RMSNorm HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    rmsnorm_f32_host(host_input.data(), host_weight.data(), elements, epsilon, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RMSNorm output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RMSNorm HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void silu_mul_f32_host(
    const float *gate,
    const float *up,
    size_t elements,
    float *output) {
    for (size_t index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
        output[index] = gate_value * sigmoid * up[index];
    }
}

class HipSiluMulKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return nullptr;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_silu_mul_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *function = nullptr;
            if (!hip_runtime().module_get_function(
                    &function,
                    module,
                    "ullm_silu_mul_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(error, compile_errors.empty() ? "failed to build SiLU-mul HIP kernel" : compile_errors);
        return nullptr;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
        std::string arch;
    };

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipSiluMulKernelCache &hip_silu_mul_kernel_cache() {
    static HipSiluMulKernelCache cache;
    return cache;
}

bool silu_mul_f32_hip_kernel(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = gate_buffer->hip_device_id;
    void *function = hip_silu_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "SiLU-mul element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *gate_ptr = gate_buffer->ptr;
    void *up_ptr = up_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &gate_ptr,
        &up_ptr,
        &kernel_elements,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 SiLU-mul";
        }
        return false;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        if (error != nullptr) {
            *error = "failed to synchronize f32 SiLU-mul HIP kernel";
        }
        return false;
    }
    return true;
}

ullm_status silu_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_gate(elements);
    std::vector<float> host_up(elements);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = gate_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_up.data(),
            up_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 SiLU-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 SiLU-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    silu_mul_f32_host(host_gate.data(), host_up.data(), elements, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 SiLU-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 SiLU-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void depthwise_conv1d_f32_host(
    const float *input,
    const float *weight,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    float *output) {
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t channel = 0; channel < channels; ++channel) {
            float sum = 0.0f;
            for (size_t kernel = 0; kernel < kernel_size; ++kernel) {
                if (timestep < kernel) {
                    break;
                }
                const size_t source_timestep = timestep - kernel;
                sum += input[source_timestep * channels + channel] *
                       weight[channel * kernel_size + kernel];
            }
            output[timestep * channels + channel] = sum;
        }
    }
}

class HipDepthwiseConv1dKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return nullptr;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_depthwise_conv1d_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *function = nullptr;
            if (!hip_runtime().module_get_function(
                    &function,
                    module,
                    "ullm_depthwise_conv1d_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->function = function;
            loaded->arch = arch;
            void *result = loaded->function;
            modules_.emplace(device_id, std::move(loaded));
            return result;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build depthwise conv1d HIP kernel" : compile_errors);
        return nullptr;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *function = nullptr;
        std::string arch;
    };

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<LoadedModule>> modules_;
};

HipDepthwiseConv1dKernelCache &hip_depthwise_conv1d_kernel_cache() {
    static HipDepthwiseConv1dKernelCache cache;
    return cache;
}

bool depthwise_conv1d_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_depthwise_conv1d_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t elements = channels * sequence_len;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "depthwise conv1d element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_channels = static_cast<unsigned long long>(channels);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_size_arg = static_cast<unsigned long long>(kernel_size);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &kernel_channels,
        &kernel_sequence_len,
        &kernel_size_arg,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 depthwise conv1d";
        }
        return false;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        if (error != nullptr) {
            *error = "failed to synchronize f32 depthwise conv1d HIP kernel";
        }
        return false;
    }
    return true;
}

ullm_status depthwise_conv1d_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    size_t required_input_bytes,
    size_t required_weight_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(required_input_bytes / sizeof(float));
    std::vector<float> host_weight(required_weight_bytes / sizeof(float));
    std::vector<float> host_output(required_output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            required_weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 depthwise conv1d HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 depthwise conv1d HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    depthwise_conv1d_f32_host(
        host_input.data(),
        host_weight.data(),
        channels,
        sequence_len,
        kernel_size,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 depthwise conv1d output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 depthwise conv1d HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

} // namespace

uint32_t ullm_runtime_abi_version(void) {
    return ULLM_RUNTIME_ABI_VERSION;
}

ullm_status ullm_runtime_get_last_error(char *buffer, size_t *buffer_len) {
    if (buffer_len == nullptr) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const std::string &message = last_error.empty() ? std::string("ok") : last_error;
    const size_t required = message.size() + 1;
    if (buffer == nullptr || *buffer_len < required) {
        *buffer_len = required;
        return ULLM_STATUS_BUFFER_TOO_SMALL;
    }
    copy_cstr(buffer, *buffer_len, message);
    *buffer_len = required;
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_get_device_count(uint32_t *count) {
    if (count == nullptr) {
        set_error("device count output pointer is null");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    *count = total_device_count();
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_get_device_info(uint32_t index, ullm_device_info *info) {
    if (info == nullptr) {
        set_error("device info output pointer is null");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (index >= total_device_count()) {
        set_error("device index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    std::memset(info, 0, sizeof(*info));
    if (index == 0) {
        fill_cpu_device(info);
    } else {
        fill_hip_device(index, info);
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_context_create(uint32_t device_index, ullm_runtime_context **context) {
    if (context == nullptr) {
        set_error("context output pointer is null");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    *context = nullptr;
    if (device_index >= total_device_count()) {
        set_error("context device index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    auto *created = new (std::nothrow) ullm_runtime_context();
    if (created == nullptr) {
        set_error("failed to allocate runtime context");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    created->device_index = device_index;
    if (device_index == 0) {
        created->backend = BackendKind::Cpu;
        created->hip_device_id = -1;
    } else {
        created->backend = BackendKind::Hip;
        created->hip_device_id = static_cast<int>(device_index - 1);
        if (!hip_runtime().set_device(created->hip_device_id)) {
            delete created;
            set_error("failed to select HIP device");
            return ULLM_STATUS_RUNTIME_ERROR;
        }
    }
    *context = created;
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_context_destroy(ullm_runtime_context *context) {
    delete context;
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_context_device_info(
    const ullm_runtime_context *context,
    ullm_device_info *info) {
    if (context == nullptr || info == nullptr) {
        set_error("context or device info output pointer is null");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return ullm_runtime_get_device_info(context->device_index, info);
}

ullm_status ullm_runtime_buffer_alloc(
    ullm_runtime_context *context,
    size_t bytes,
    ullm_runtime_buffer **buffer) {
    if (context == nullptr || buffer == nullptr) {
        set_error("buffer allocation received a null context or output pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    *buffer = nullptr;
    auto *created = new (std::nothrow) ullm_runtime_buffer();
    if (created == nullptr) {
        set_error("failed to allocate buffer handle");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    created->backend = context->backend;
    created->hip_device_id = context->hip_device_id;
    created->bytes = bytes;

    if (bytes == 0) {
        *buffer = created;
        set_error("");
        return ULLM_STATUS_OK;
    }

    if (context->backend == BackendKind::Cpu) {
        created->ptr = std::malloc(bytes);
        if (created->ptr == nullptr) {
            delete created;
            set_error("failed to allocate CPU buffer");
            return ULLM_STATUS_RUNTIME_ERROR;
        }
    } else {
        created->ptr = hip_runtime().malloc_device(bytes, context->hip_device_id);
        if (created->ptr == nullptr) {
            delete created;
            set_error("failed to allocate HIP buffer");
            return ULLM_STATUS_RUNTIME_ERROR;
        }
    }

    *buffer = created;
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_buffer_destroy(ullm_runtime_buffer *buffer) {
    if (buffer == nullptr) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    bool ok = true;
    if (buffer->backend == BackendKind::Cpu) {
        std::free(buffer->ptr);
    } else {
        ok = hip_runtime().free_device(buffer->ptr, buffer->hip_device_id);
    }
    delete buffer;
    if (!ok) {
        set_error("failed to destroy runtime buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_buffer_size(
    const ullm_runtime_buffer *buffer,
    size_t *bytes) {
    if (buffer == nullptr || bytes == nullptr) {
        set_error("buffer size received a null buffer or output pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    *bytes = buffer->bytes;
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_buffer_copy_from_host(
    ullm_runtime_buffer *buffer,
    size_t offset,
    const void *src,
    size_t bytes,
    ullm_runtime_stream *stream) {
    if (buffer == nullptr || (bytes > 0 && src == nullptr)) {
        set_error("copy from host received a null buffer or source pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_range(offset, bytes, buffer->bytes)) {
        set_error("copy from host range is out of bounds");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(buffer, stream)) {
        set_error("copy from host stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (bytes == 0) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    auto *dst = static_cast<unsigned char *>(buffer->ptr) + offset;
    if (buffer->backend == BackendKind::Cpu) {
        std::memcpy(dst, src, bytes);
    } else if (!hip_runtime().copy_async(
                   dst,
                   src,
                   bytes,
                   HIP_MEMCPY_HOST_TO_DEVICE,
                   stream == nullptr ? nullptr : stream->stream,
                   buffer->hip_device_id)) {
        set_error("failed to copy host data to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_buffer_copy_to_host(
    const ullm_runtime_buffer *buffer,
    size_t offset,
    void *dst,
    size_t bytes,
    ullm_runtime_stream *stream) {
    if (buffer == nullptr || (bytes > 0 && dst == nullptr)) {
        set_error("copy to host received a null buffer or destination pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!checked_range(offset, bytes, buffer->bytes)) {
        set_error("copy to host range is out of bounds");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(buffer, stream)) {
        set_error("copy to host stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (bytes == 0) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const auto *src = static_cast<const unsigned char *>(buffer->ptr) + offset;
    if (buffer->backend == BackendKind::Cpu) {
        std::memcpy(dst, src, bytes);
    } else if (!hip_runtime().copy_async(
                   dst,
                   src,
                   bytes,
                   HIP_MEMCPY_DEVICE_TO_HOST,
                   stream == nullptr ? nullptr : stream->stream,
                   buffer->hip_device_id)) {
        set_error("failed to copy HIP buffer data to host");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_stream_create(
    ullm_runtime_context *context,
    ullm_runtime_stream **stream) {
    if (context == nullptr || stream == nullptr) {
        set_error("stream creation received a null context or output pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    *stream = nullptr;
    auto *created = new (std::nothrow) ullm_runtime_stream();
    if (created == nullptr) {
        set_error("failed to allocate stream handle");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    created->backend = context->backend;
    created->hip_device_id = context->hip_device_id;
    if (context->backend == BackendKind::Hip) {
        created->stream = hip_runtime().create_stream(context->hip_device_id);
        if (created->stream == nullptr) {
            delete created;
            set_error("failed to create HIP stream");
            return ULLM_STATUS_RUNTIME_ERROR;
        }
    }
    *stream = created;
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_stream_destroy(ullm_runtime_stream *stream) {
    if (stream == nullptr) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    bool ok = true;
    if (stream->backend == BackendKind::Hip) {
        ok = hip_runtime().destroy_stream(stream->stream, stream->hip_device_id);
    }
    delete stream;
    if (!ok) {
        set_error("failed to destroy runtime stream");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_stream_synchronize(ullm_runtime_stream *stream) {
    if (stream == nullptr) {
        set_error("stream synchronize received a null stream");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (stream->backend == BackendKind::Hip &&
        !hip_runtime().synchronize_stream(stream->stream, stream->hip_device_id)) {
        set_error("failed to synchronize HIP stream");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_aq4_dequant_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        output_buffer == nullptr || (scale_count > 0 && scale_values == nullptr)) {
        set_error("AQ4 dequant received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 dequant group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 dequant scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer)) {
        set_error("AQ4 dequant buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 dequant stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements > (static_cast<size_t>(-1) / sizeof(float))) {
        set_error("AQ4 dequant output byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    const size_t required_output_bytes = elements * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 dequant index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 dequant scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 dequant output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 dequant codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 dequant requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_dequant_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                elements,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    const HipAq4LaunchResult launch_result = aq4_dequant_hip_kernel(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values,
        scale_count,
        group_size,
        tensor_scale,
        elements,
        required_output_bytes,
        output_buffer,
        stream,
        &hip_kernel_error);
    if (launch_result == HipAq4LaunchResult::Ok) {
        set_error("");
        return ULLM_STATUS_OK;
    }
    if (launch_result == HipAq4LaunchResult::InvalidArgument) {
        set_error(hip_kernel_error.c_str());
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "AQ4 HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_dequant_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values,
        scale_count,
        group_size,
        tensor_scale,
        elements,
        required_index_bytes,
        groups,
        required_output_bytes,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_matvec_f32(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (matrix_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("f32 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(matrix_buffer, input_buffer) ||
        !buffers_share_backend(matrix_buffer, output_buffer)) {
        set_error("f32 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("f32 matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    if (matrix_elements > max_size / sizeof(float) ||
        cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float)) {
        set_error("f32 matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_matrix_bytes = matrix_elements * sizeof(float);
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    if (matrix_buffer->bytes < required_matrix_bytes) {
        set_error("f32 matvec matrix buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("f32 matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("f32 matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (matrix_buffer->backend == BackendKind::Cpu) {
        const auto *matrix = static_cast<const float *>(matrix_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        matvec_f32_host(matrix, input, rows, cols, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (matvec_f32_hip_kernel(
            matrix_buffer,
            input_buffer,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 matvec HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return matvec_f32_hip_staging(
        matrix_buffer,
        input_buffer,
        rows,
        cols,
        required_matrix_bytes,
        required_input_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_rmsnorm_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 RMSNorm received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 RMSNorm elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("f32 RMSNorm epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 RMSNorm buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 RMSNorm stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 RMSNorm byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (input_buffer->bytes < required_bytes) {
        set_error("f32 RMSNorm input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < required_bytes) {
        set_error("f32 RMSNorm weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 RMSNorm output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *weight = static_cast<const float *>(weight_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        rmsnorm_f32_host(input, weight, elements, epsilon, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (rmsnorm_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            elements,
            epsilon,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_RMSNORM_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 RMSNorm HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return rmsnorm_f32_hip_staging(
        input_buffer,
        weight_buffer,
        elements,
        epsilon,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_silu_mul_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr || up_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 SiLU-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 SiLU-mul elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(gate_buffer, up_buffer) ||
        !buffers_share_backend(gate_buffer, output_buffer)) {
        set_error("f32 SiLU-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 SiLU-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 SiLU-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (gate_buffer->bytes < required_bytes) {
        set_error("f32 SiLU-mul gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (up_buffer->bytes < required_bytes) {
        set_error("f32 SiLU-mul up buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 SiLU-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (gate_buffer->backend == BackendKind::Cpu) {
        const auto *gate = static_cast<const float *>(gate_buffer->ptr);
        const auto *up = static_cast<const float *>(up_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        silu_mul_f32_host(gate, up, elements, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (silu_mul_f32_hip_kernel(
            gate_buffer,
            up_buffer,
            elements,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_SILU_MUL_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 SiLU-mul HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return silu_mul_f32_hip_staging(
        gate_buffer,
        up_buffer,
        elements,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_depthwise_conv1d_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 depthwise conv1d received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (channels == 0 || sequence_len == 0 || kernel_size == 0) {
        set_error("f32 depthwise conv1d channels, sequence length, and kernel size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 depthwise conv1d buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 depthwise conv1d stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (sequence_len > max_size / channels) {
        set_error("f32 depthwise conv1d input element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (kernel_size > max_size / channels) {
        set_error("f32 depthwise conv1d weight element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_elements = channels * sequence_len;
    const size_t weight_elements = channels * kernel_size;
    if (input_elements > max_size / sizeof(float) ||
        weight_elements > max_size / sizeof(float)) {
        set_error("f32 depthwise conv1d byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = input_elements * sizeof(float);
    const size_t required_weight_bytes = weight_elements * sizeof(float);
    const size_t required_output_bytes = required_input_bytes;
    if (input_buffer->bytes < required_input_bytes) {
        set_error("f32 depthwise conv1d input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < required_weight_bytes) {
        set_error("f32 depthwise conv1d weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("f32 depthwise conv1d output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *weight = static_cast<const float *>(weight_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        depthwise_conv1d_f32_host(
            input,
            weight,
            channels,
            sequence_len,
            kernel_size,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (depthwise_conv1d_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            channels,
            sequence_len,
            kernel_size,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_DEPTHWISE_CONV1D_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 depthwise conv1d HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return depthwise_conv1d_f32_hip_staging(
        input_buffer,
        weight_buffer,
        channels,
        sequence_len,
        kernel_size,
        required_input_bytes,
        required_weight_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_smoke_add_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    size_t count) {
    if (count > 0 && (lhs == nullptr || rhs == nullptr || out == nullptr)) {
        set_error("smoke add received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; ++i) {
        out[i] = lhs[i] + rhs[i];
    }
    set_error("");
    return ULLM_STATUS_OK;
}
