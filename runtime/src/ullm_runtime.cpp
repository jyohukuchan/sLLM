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

    bool compile_sigmoid_mul_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, sigmoid_mul_kernel_source(), "ullm_sigmoid_mul_f32.hip", code, error);
    }

    bool compile_add_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, add_kernel_source(), "ullm_add_f32.hip", code, error);
    }

    bool compile_rope_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, rope_kernel_source(), "ullm_rope_f32.hip", code, error);
    }

    bool compile_causal_attn_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_kernel_source(),
            "ullm_causal_attn_f32.hip",
            code,
            error);
    }

    bool compile_decode_attn_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            decode_attn_kernel_source(),
            "ullm_decode_attn_f32.hip",
            code,
            error);
    }

    bool compile_paged_decode_attn_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            paged_decode_attn_kernel_source(),
            "ullm_paged_decode_attn_f32.hip",
            code,
            error);
    }

    bool compile_paged_kv_write_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            paged_kv_write_kernel_source(),
            "ullm_paged_kv_write_f32.hip",
            code,
            error);
    }

    bool compile_depthwise_conv1d_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            depthwise_conv1d_kernel_source(),
            "ullm_depthwise_conv1d_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_gate_beta_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_gate_beta_kernel_source(),
            "ullm_linear_attn_gate_beta_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_recurrent_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_recurrent_kernel_source(),
            "ullm_linear_attn_recurrent_f32.hip",
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

    static const char *sigmoid_mul_kernel_source() {
        return R"(
extern "C" __global__ void ullm_sigmoid_mul_f32_kernel(
    const float *gate,
    const float *input,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float gate_value = gate[index];
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    output[index] = sigmoid * input[index];
}
)";
    }

    static const char *add_kernel_source() {
        return R"(
extern "C" __global__ void ullm_add_f32_kernel(
    const float *lhs,
    const float *rhs,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    output[index] = lhs[index] + rhs[index];
}
)";
    }

    static const char *rope_kernel_source() {
        return R"(
extern "C" __global__ void ullm_rope_f32_kernel(
    const float *input,
    unsigned long long sequence_len,
    unsigned long long heads,
    unsigned long long head_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long half = rotary_dim >> 1;
    const unsigned long long work_dim = half + (head_dim - rotary_dim);
    const unsigned long long work_items = sequence_len * heads * work_dim;
    if (index >= work_items) {
        return;
    }
    const unsigned long long local_dim = index % work_dim;
    const unsigned long long head_index = index / work_dim;
    const unsigned long long base = head_index * head_dim;
    if (local_dim >= half) {
        const unsigned long long dim = rotary_dim + (local_dim - half);
        output[base + dim] = input[base + dim];
        return;
    }
    const unsigned long long pair_dim = local_dim;
    const unsigned long long timestep = head_index / heads;
    const float position = static_cast<float>(position_offset + timestep);
    const float exponent =
        (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
    const float theta = position / powf(rope_base, exponent);
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float first = input[base + pair_dim];
    const float second = input[base + half + pair_dim];
    output[base + pair_dim] = first * c - second * s;
    output[base + half + pair_dim] = second * c + first * s;
}
)";
    }

    static const char *causal_attn_kernel_source() {
        return R"(
extern "C" __global__ void ullm_causal_attn_f32_kernel(
    const float *q,
    const float *k,
    const float *v,
    unsigned long long sequence_len,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = sequence_len * q_heads * value_dim;
    if (index >= elements) {
        return;
    }
    const unsigned long long value = index % value_dim;
    const unsigned long long q_head_index = index / value_dim;
    const unsigned long long q_head = q_head_index % q_heads;
    const unsigned long long timestep = q_head_index / q_heads;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = (timestep * q_heads + q_head) * head_dim;

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k[k_base + dim];
        }
        score *= softmax_scale;
        max_score = score > max_score ? score : max_score;
    }

    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k[k_base + dim];
        }
        const float weight = expf(score * softmax_scale - max_score);
        denominator += weight;
        const unsigned long long v_index =
            (source_timestep * kv_heads + kv_head) * value_dim + value;
        weighted += weight * v[v_index];
    }
    output[index] = weighted / denominator;
}
)";
    }

    static const char *decode_attn_kernel_source() {
        return R"(
extern "C" __global__ void ullm_decode_attn_f32_kernel(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    unsigned long long cache_len,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = q_heads * value_dim;
    if (index >= elements) {
        return;
    }
    const unsigned long long value = index % value_dim;
    const unsigned long long q_head = index / value_dim;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        score *= softmax_scale;
        max_score = score > max_score ? score : max_score;
    }

    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long k_base = (source_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        const float weight = expf(score * softmax_scale - max_score);
        denominator += weight;
        const unsigned long long v_index =
            (source_timestep * kv_heads + kv_head) * value_dim + value;
        weighted += weight * v_cache[v_index];
    }
    output[index] = weighted / denominator;
}
)";
    }

    static const char *paged_decode_attn_kernel_source() {
        return R"(
extern "C" __global__ void ullm_paged_decode_attn_f32_kernel(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const unsigned int *block_table,
    unsigned long long cache_len,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float softmax_scale,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = q_heads * value_dim;
    if (index >= elements) {
        return;
    }
    const unsigned long long value = index % value_dim;
    const unsigned long long q_head = index / value_dim;
    const unsigned long long q_per_kv = q_heads / kv_heads;
    const unsigned long long kv_head = q_head / q_per_kv;
    const unsigned long long q_base = q_head * head_dim;

    float max_score = -3.4028234663852886e38f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long block_index = source_timestep / block_size;
        const unsigned long long block_offset = source_timestep - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical_timestep = block_id * block_size + block_offset;
        const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        score *= softmax_scale;
        max_score = score > max_score ? score : max_score;
    }

    float denominator = 0.0f;
    float weighted = 0.0f;
    for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
        const unsigned long long block_index = source_timestep / block_size;
        const unsigned long long block_offset = source_timestep - block_index * block_size;
        const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
        if (block_id >= cache_blocks) {
            output[index] = 0.0f;
            return;
        }
        const unsigned long long physical_timestep = block_id * block_size + block_offset;
        const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            score += q[q_base + dim] * k_cache[k_base + dim];
        }
        const float weight = expf(score * softmax_scale - max_score);
        denominator += weight;
        const unsigned long long v_index =
            (physical_timestep * kv_heads + kv_head) * value_dim + value;
        weighted += weight * v_cache[v_index];
    }
    output[index] = weighted / denominator;
}
)";
    }

    static const char *paged_kv_write_kernel_source() {
        return R"(
extern "C" __global__ void ullm_paged_kv_write_f32_kernel(
    const float *k,
    const float *v,
    const unsigned int *block_table,
    unsigned long long cache_position,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    float *k_cache,
    float *v_cache) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long k_elements = kv_heads * head_dim;
    const unsigned long long v_elements = kv_heads * value_dim;
    const unsigned long long total_elements = k_elements + v_elements;
    if (index >= total_elements) {
        return;
    }
    const unsigned long long block_index = cache_position / block_size;
    const unsigned long long block_offset = cache_position - block_index * block_size;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) {
        return;
    }
    const unsigned long long physical_timestep = block_id * block_size + block_offset;
    if (index < k_elements) {
        const unsigned long long kv_head = index / head_dim;
        const unsigned long long dim = index - kv_head * head_dim;
        k_cache[(physical_timestep * kv_heads + kv_head) * head_dim + dim] = k[index];
        return;
    }
    const unsigned long long v_index = index - k_elements;
    const unsigned long long kv_head = v_index / value_dim;
    const unsigned long long dim = v_index - kv_head * value_dim;
    v_cache[(physical_timestep * kv_heads + kv_head) * value_dim + dim] = v[v_index];
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
        const unsigned long long left_padding = kernel_size - 1 - kernel;
        if (timestep < left_padding) {
            continue;
        }
        const unsigned long long source_timestep = timestep - left_padding;
        sum += input[source_timestep * channels + channel] *
               weight[channel * kernel_size + kernel];
    }
    output[index] = sum;
}
)";
    }

    static const char *linear_attn_gate_beta_kernel_source() {
        return R"(
extern "C" __global__ void ullm_linear_attn_gate_beta_f32_kernel(
    const float *a,
    const float *b,
    const float *a_log,
    const float *dt_bias,
    unsigned long long heads,
    unsigned long long sequence_len,
    float *gate_output,
    float *beta_output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = heads * sequence_len;
    if (index >= elements) {
        return;
    }
    const unsigned long long head = index - (index / heads) * heads;
    const float x = a[index] + dt_bias[head];
    const float softplus = x <= 20.0f ? logf(1.0f + expf(x)) : x;
    gate_output[index] = -expf(a_log[head]) * softplus;
    const float b_value = b[index];
    beta_output[index] = 1.0f / (1.0f + expf(-b_value));
}
)";
    }

    static const char *linear_attn_recurrent_kernel_source() {
        return R"(
extern "C" __global__ void ullm_linear_attn_recurrent_f32_kernel(
    const float *q,
    const float *k,
    const float *v,
    const float *gate,
    const float *beta,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long sequence_len,
    unsigned long long key_dim,
    unsigned long long value_dim,
    float *state,
    float *output) {
    const unsigned long long value_head = blockIdx.x;
    if (value_head >= value_heads || threadIdx.x != 0) {
        return;
    }
    const unsigned long long key_head_group = value_heads / key_heads;
    const unsigned long long key_head = value_head / key_head_group;
    const unsigned long long state_head_offset = value_head * key_dim * value_dim;
    for (unsigned long long timestep = 0; timestep < sequence_len; ++timestep) {
        const unsigned long long value_head_index = timestep * value_heads + value_head;
        const unsigned long long key_head_index = timestep * key_heads + key_head;
        const unsigned long long qk_base = key_head_index * key_dim;
        const unsigned long long v_base = value_head_index * value_dim;
        const float decay = expf(gate[value_head_index]);
        const float beta_value = beta[value_head_index];

        for (unsigned long long key = 0; key < key_dim; ++key) {
            const unsigned long long state_key_offset = state_head_offset + key * value_dim;
            for (unsigned long long value = 0; value < value_dim; ++value) {
                state[state_key_offset + value] *= decay;
            }
        }

        for (unsigned long long value = 0; value < value_dim; ++value) {
            float current = 0.0f;
            for (unsigned long long key = 0; key < key_dim; ++key) {
                current += state[state_head_offset + key * value_dim + value] *
                           k[qk_base + key];
            }
            const float v_prime = (v[v_base + value] - current) * beta_value;
            for (unsigned long long key = 0; key < key_dim; ++key) {
                state[state_head_offset + key * value_dim + value] +=
                    k[qk_base + key] * v_prime;
            }
        }

        for (unsigned long long value = 0; value < value_dim; ++value) {
            float sum = 0.0f;
            for (unsigned long long key = 0; key < key_dim; ++key) {
                sum += state[state_head_offset + key * value_dim + value] *
                       q[qk_base + key];
            }
            output[v_base + value] = sum;
        }
    }
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

class RocBlasRuntime {
public:
    bool sgemv_row_major(
        int device_id,
        void *stream,
        const void *matrix_ptr,
        const void *input_ptr,
        size_t rows,
        size_t cols,
        void *output_ptr,
        std::string *error) {
        load_once();
        if (!available()) {
            append_error(error, "rocBLAS is not available");
            return false;
        }
        if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max())) {
            append_error(error, "rocBLAS SGEMV dimensions exceed rocblas_int range");
            return false;
        }
        if (!hip_runtime().set_device(device_id)) {
            append_error(error, "failed to select HIP device for rocBLAS SGEMV");
            return false;
        }

        void *handle = handle_for_device(device_id, error);
        if (handle == nullptr) {
            return false;
        }
        if (rocblas_set_stream_(handle, stream) != 0) {
            append_error(error, "rocblas_set_stream failed for SGEMV");
            return false;
        }

        constexpr int rocblas_operation_transpose = 112;
        const int m = static_cast<int>(cols);
        const int n = static_cast<int>(rows);
        const int lda = static_cast<int>(cols);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int status = rocblas_sgemv_(
            handle,
            rocblas_operation_transpose,
            m,
            n,
            &alpha,
            static_cast<const float *>(matrix_ptr),
            lda,
            static_cast<const float *>(input_ptr),
            1,
            &beta,
            static_cast<float *>(output_ptr),
            1);
        if (status != 0) {
            append_error(error, "rocblas_sgemv failed with status " + std::to_string(status));
            return false;
        }
        return true;
    }

private:
    using rocblas_create_handle_fn = int (*)(void **);
    using rocblas_destroy_handle_fn = int (*)(void *);
    using rocblas_set_stream_fn = int (*)(void *, void *);
    using rocblas_sgemv_fn = int (*)(
        void *,
        int,
        int,
        int,
        const float *,
        const float *,
        int,
        const float *,
        int,
        const float *,
        float *,
        int);

    struct HandleEntry {
        void *handle = nullptr;
    };

    bool available() const {
        return rocblas_create_handle_ != nullptr && rocblas_destroy_handle_ != nullptr &&
               rocblas_set_stream_ != nullptr && rocblas_sgemv_ != nullptr;
    }

    void *handle_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = handles_.find(device_id);
        if (found != handles_.end()) {
            return found->second.handle;
        }
        void *handle = nullptr;
        if (rocblas_create_handle_(&handle) != 0 || handle == nullptr) {
            append_error(error, "rocblas_create_handle failed");
            return nullptr;
        }
        handles_.emplace(device_id, HandleEntry{handle});
        return handle;
    }

    static void append_error(std::string *error, const std::string &message) {
        if (error == nullptr || message.empty()) {
            return;
        }
        if (!error->empty()) {
            error->append("\n");
        }
        error->append(message);
    }

    void load_once() {
        std::call_once(load_flag_, [this]() {
#if defined(__linux__)
            constexpr std::array<const char *, 3> candidates = {
                "librocblas.so",
                "librocblas.so.5",
                "librocblas.so.4",
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
            rocblas_create_handle_ = reinterpret_cast<rocblas_create_handle_fn>(
                dlsym(handle_, "rocblas_create_handle"));
            rocblas_destroy_handle_ = reinterpret_cast<rocblas_destroy_handle_fn>(
                dlsym(handle_, "rocblas_destroy_handle"));
            rocblas_set_stream_ =
                reinterpret_cast<rocblas_set_stream_fn>(dlsym(handle_, "rocblas_set_stream"));
            rocblas_sgemv_ = reinterpret_cast<rocblas_sgemv_fn>(dlsym(handle_, "rocblas_sgemv"));
#endif
        });
    }

    std::once_flag load_flag_;
    void *handle_ = nullptr;
    rocblas_create_handle_fn rocblas_create_handle_ = nullptr;
    rocblas_destroy_handle_fn rocblas_destroy_handle_ = nullptr;
    rocblas_set_stream_fn rocblas_set_stream_ = nullptr;
    rocblas_sgemv_fn rocblas_sgemv_ = nullptr;
    std::mutex mutex_;
    std::unordered_map<int, HandleEntry> handles_;
};

RocBlasRuntime &rocblas_runtime() {
    static RocBlasRuntime runtime;
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

void sigmoid_mul_f32_host(
    const float *gate,
    const float *input,
    size_t elements,
    float *output) {
    for (size_t index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
        output[index] = sigmoid * input[index];
    }
}

void add_f32_host(
    const float *lhs,
    const float *rhs,
    size_t elements,
    float *output) {
    for (size_t index = 0; index < elements; ++index) {
        output[index] = lhs[index] + rhs[index];
    }
}

void rope_f32_host(
    const float *input,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float *output) {
    const size_t half = rotary_dim / 2;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        const float position = static_cast<float>(position_offset + timestep);
        for (size_t head = 0; head < heads; ++head) {
            const size_t base = (timestep * heads + head) * head_dim;
            for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
                const float exponent =
                    (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
                const float theta = position / std::pow(rope_base, exponent);
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const float first = input[base + pair_dim];
                const float second = input[base + half + pair_dim];
                output[base + pair_dim] = first * c - second * s;
                output[base + half + pair_dim] = second * c + first * s;
            }
            for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
                output[base + dim] = input[base + dim];
            }
        }
    }
}

void causal_attn_f32_host(
    const float *q,
    const float *k,
    const float *v,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (timestep * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                score *= softmax_scale;
                max_score = std::max(max_score, score);
            }

            float denominator = 0.0f;
            for (size_t source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                denominator += std::exp(score * softmax_scale - max_score);
            }

            const size_t output_base = (timestep * q_heads + q_head) * value_dim;
            for (size_t value = 0; value < value_dim; ++value) {
                float weighted = 0.0f;
                for (size_t source_timestep = 0; source_timestep <= timestep; ++source_timestep) {
                    const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    float score = 0.0f;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        score += q[q_base + dim] * k[k_base + dim];
                    }
                    const float weight = std::exp(score * softmax_scale - max_score);
                    const size_t v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted += weight * v[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
}

void decode_attn_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t q_head = 0; q_head < q_heads; ++q_head) {
        const size_t kv_head = q_head / q_per_kv;
        const size_t q_base = q_head * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            score *= softmax_scale;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0f;
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            denominator += std::exp(score * softmax_scale - max_score);
        }

        const size_t output_base = q_head * value_dim;
        for (size_t value = 0; value < value_dim; ++value) {
            float weighted = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                const float weight = std::exp(score * softmax_scale - max_score);
                const size_t v_index =
                    (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
}

void paged_decode_attn_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const std::uint32_t *block_table,
    size_t cache_len,
    size_t block_size,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t q_head = 0; q_head < q_heads; ++q_head) {
        const size_t kv_head = q_head / q_per_kv;
        const size_t q_base = q_head * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t block_index = source_timestep / block_size;
            const size_t block_offset = source_timestep - block_index * block_size;
            const size_t physical_timestep =
                static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
            const size_t k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            score *= softmax_scale;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0f;
        for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const size_t block_index = source_timestep / block_size;
            const size_t block_offset = source_timestep - block_index * block_size;
            const size_t physical_timestep =
                static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
            const size_t k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                score += q[q_base + dim] * k_cache[k_base + dim];
            }
            denominator += std::exp(score * softmax_scale - max_score);
        }

        const size_t output_base = q_head * value_dim;
        for (size_t value = 0; value < value_dim; ++value) {
            float weighted = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t block_index = source_timestep / block_size;
                const size_t block_offset = source_timestep - block_index * block_size;
                const size_t physical_timestep =
                    static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
                const size_t k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                const float weight = std::exp(score * softmax_scale - max_score);
                const size_t v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
}

void paged_kv_write_f32_host(
    const float *k,
    const float *v,
    const std::uint32_t *block_table,
    size_t cache_position,
    size_t block_size,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float *k_cache,
    float *v_cache) {
    const size_t block_index = cache_position / block_size;
    const size_t block_offset = cache_position - block_index * block_size;
    const size_t physical_timestep =
        static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
    for (size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const size_t k_src = kv_head * head_dim;
        const size_t k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
        std::copy(k + k_src, k + k_src + head_dim, k_cache + k_dst);

        const size_t v_src = kv_head * value_dim;
        const size_t v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
        std::copy(v + v_src, v + v_src + value_dim, v_cache + v_dst);
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

class HipSigmoidMulKernelCache {
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
            if (!hiprtc_runtime().compile_sigmoid_mul_kernel(arch, &code, &compile_error)) {
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
                    "ullm_sigmoid_mul_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build sigmoid-mul HIP kernel" : compile_errors);
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

HipSigmoidMulKernelCache &hip_sigmoid_mul_kernel_cache() {
    static HipSigmoidMulKernelCache cache;
    return cache;
}

bool sigmoid_mul_f32_hip_kernel(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = gate_buffer->hip_device_id;
    void *function = hip_sigmoid_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "sigmoid-mul element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *gate_ptr = gate_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &gate_ptr,
        &input_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 sigmoid-mul";
        }
        return false;
    }
    return true;
}

ullm_status sigmoid_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_gate(elements);
    std::vector<float> host_input(elements);
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
            host_input.data(),
            input_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 sigmoid-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 sigmoid-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    sigmoid_mul_f32_host(host_gate.data(), host_input.data(), elements, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 sigmoid-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 sigmoid-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipAddKernelCache {
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
            if (!hiprtc_runtime().compile_add_kernel(arch, &code, &compile_error)) {
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
                    "ullm_add_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build f32 add HIP kernel" : compile_errors);
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

HipAddKernelCache &hip_add_kernel_cache() {
    static HipAddKernelCache cache;
    return cache;
}

bool add_f32_hip_kernel(
    const ullm_runtime_buffer *lhs_buffer,
    const ullm_runtime_buffer *rhs_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = lhs_buffer->hip_device_id;
    void *function = hip_add_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "f32 add element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *lhs_ptr = lhs_buffer->ptr;
    void *rhs_ptr = rhs_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &lhs_ptr,
        &rhs_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 add";
        }
        return false;
    }
    return true;
}

ullm_status add_f32_hip_staging(
    const ullm_runtime_buffer *lhs_buffer,
    const ullm_runtime_buffer *rhs_buffer,
    size_t elements,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_lhs(elements);
    std::vector<float> host_rhs(elements);
    std::vector<float> host_output(elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = lhs_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_lhs.data(),
            lhs_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_rhs.data(),
            rhs_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 add HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 add HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    add_f32_host(host_lhs.data(), host_rhs.data(), elements, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 add output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 add HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipRopeKernelCache {
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
            if (!hiprtc_runtime().compile_rope_kernel(arch, &code, &compile_error)) {
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
                    "ullm_rope_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build RoPE HIP kernel" : compile_errors);
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

HipRopeKernelCache &hip_rope_kernel_cache() {
    static HipRopeKernelCache cache;
    return cache;
}

bool rope_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_rope_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t work_dim = (rotary_dim / 2) + (head_dim - rotary_dim);
    const size_t work_items = sequence_len * heads * work_dim;
    const size_t grid_size = (work_items + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "RoPE element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_rotary_dim = static_cast<unsigned long long>(rotary_dim);
    unsigned long long kernel_position_offset = static_cast<unsigned long long>(position_offset);
    void *input_ptr = input_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &kernel_sequence_len,
        &kernel_heads,
        &kernel_head_dim,
        &kernel_rotary_dim,
        &kernel_position_offset,
        &rope_base,
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
            *error = "hipModuleLaunchKernel failed for f32 RoPE";
        }
        return false;
    }
    return true;
}

ullm_status rope_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    size_t required_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(required_bytes / sizeof(float));
    std::vector<float> host_output(required_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RoPE HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RoPE HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    rope_f32_host(
        host_input.data(),
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 RoPE output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 RoPE HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipCausalAttnKernelCache {
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
            if (!hiprtc_runtime().compile_causal_attn_kernel(arch, &code, &compile_error)) {
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
                    "ullm_causal_attn_f32_kernel",
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
            compile_errors.empty() ? "failed to build causal attention HIP kernel" : compile_errors);
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

HipCausalAttnKernelCache &hip_causal_attn_kernel_cache() {
    static HipCausalAttnKernelCache cache;
    return cache;
}

bool causal_attn_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_causal_attn_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t output_elements = sequence_len * q_heads * value_dim;
    const size_t grid_size = (output_elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "causal attention output element count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_sequence_len,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
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
            *error = "hipModuleLaunchKernel failed for f32 causal attention";
        }
        return false;
    }
    return true;
}

ullm_status causal_attn_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            q_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k.data(),
            k_buffer->ptr,
            k_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v.data(),
            v_buffer->ptr,
            v_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 causal attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 causal attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    causal_attn_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 causal attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 causal attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipDecodeAttnKernelCache {
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
            if (!hiprtc_runtime().compile_decode_attn_kernel(arch, &code, &compile_error)) {
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
                    "ullm_decode_attn_f32_kernel",
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
            compile_errors.empty() ? "failed to build decode attention HIP kernel" : compile_errors);
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

HipDecodeAttnKernelCache &hip_decode_attn_kernel_cache() {
    static HipDecodeAttnKernelCache cache;
    return cache;
}

bool decode_attn_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_decode_attn_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t output_elements = q_heads * value_dim;
    const size_t grid_size = (output_elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "decode attention output element count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &kernel_cache_len,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
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
            *error = "hipModuleLaunchKernel failed for f32 decode attention";
        }
        return false;
    }
    return true;
}

ullm_status decode_attn_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            q_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k.data(),
            k_cache_buffer->ptr,
            k_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v.data(),
            v_cache_buffer->ptr,
            v_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 decode attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 decode attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    decode_attn_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 decode attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 decode attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipPagedDecodeAttnKernelCache {
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
            if (!hiprtc_runtime().compile_paged_decode_attn_kernel(arch, &code, &compile_error)) {
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
                    "ullm_paged_decode_attn_f32_kernel",
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
            compile_errors.empty() ? "failed to build paged decode attention HIP kernel" : compile_errors);
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

HipPagedDecodeAttnKernelCache &hip_paged_decode_attn_kernel_cache() {
    static HipPagedDecodeAttnKernelCache cache;
    return cache;
}

bool paged_decode_attn_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_paged_decode_attn_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int launch_block_size = 256;
    const size_t output_elements = q_heads * value_dim;
    const size_t grid_size = (output_elements + launch_block_size - 1) / launch_block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged decode attention output element count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cache_len = static_cast<unsigned long long>(cache_len);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &block_table_ptr,
        &kernel_cache_len,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &softmax_scale,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 paged decode attention";
        }
        return false;
    }
    return true;
}

ullm_status paged_decode_attn_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    size_t block_table_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<std::uint32_t> host_block_table(block_table_bytes / sizeof(std::uint32_t));
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            q_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k.data(),
            k_cache_buffer->ptr,
            k_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v.data(),
            v_cache_buffer->ptr,
            v_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_block_table.data(),
            block_table_buffer->ptr,
            block_table_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 paged decode attention HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged decode attention HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    for (std::uint32_t block_id : host_block_table) {
        if (static_cast<size_t>(block_id) >= cache_blocks) {
            set_error("f32 paged decode attention block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }

    paged_decode_attn_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        host_block_table.data(),
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 paged decode attention output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged decode attention HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipPagedKvWriteKernelCache {
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
            if (!hiprtc_runtime().compile_paged_kv_write_kernel(arch, &code, &compile_error)) {
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
                    "ullm_paged_kv_write_f32_kernel",
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
            compile_errors.empty() ? "failed to build paged KV write HIP kernel" : compile_errors);
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

HipPagedKvWriteKernelCache &hip_paged_kv_write_kernel_cache() {
    static HipPagedKvWriteKernelCache cache;
    return cache;
}

bool paged_kv_write_f32_hip_kernel(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = k_buffer->hip_device_id;
    void *function = hip_paged_kv_write_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int launch_block_size = 256;
    const size_t elements = kv_heads * head_dim + kv_heads * value_dim;
    const size_t grid_size = (elements + launch_block_size - 1) / launch_block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged KV write element count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_cache_position = static_cast<unsigned long long>(cache_position);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *k_cache_ptr = k_cache_buffer->ptr;
    void *v_cache_ptr = v_cache_buffer->ptr;
    void *kernel_params[] = {
        &k_ptr,
        &v_ptr,
        &block_table_ptr,
        &kernel_cache_position,
        &kernel_block_size,
        &kernel_cache_blocks,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &k_cache_ptr,
        &v_cache_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 paged KV write";
        }
        return false;
    }
    return true;
}

ullm_status paged_kv_write_f32_hip_staging(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t k_bytes,
    size_t v_bytes,
    size_t block_table_bytes,
    size_t k_cache_bytes,
    size_t v_cache_bytes,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<std::uint32_t> host_block_table(block_table_bytes / sizeof(std::uint32_t));
    std::vector<float> host_k_cache(k_cache_bytes / sizeof(float));
    std::vector<float> host_v_cache(v_cache_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = k_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_k.data(),
            k_buffer->ptr,
            k_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v.data(),
            v_buffer->ptr,
            v_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_block_table.data(),
            block_table_buffer->ptr,
            block_table_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_cache.data(),
            k_cache_buffer->ptr,
            k_cache_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v_cache.data(),
            v_cache_buffer->ptr,
            v_cache_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 paged KV write HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged KV write HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    const size_t block_index = cache_position / block_size;
    const std::uint32_t block_id = host_block_table[block_index];
    if (static_cast<size_t>(block_id) >= cache_blocks) {
        set_error("f32 paged KV write block table contains an out-of-range block id");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    paged_kv_write_f32_host(
        host_k.data(),
        host_v.data(),
        host_block_table.data(),
        cache_position,
        block_size,
        kv_heads,
        head_dim,
        value_dim,
        host_k_cache.data(),
        host_v_cache.data());

    if (!hip_runtime().copy_async(
            k_cache_buffer->ptr,
            host_k_cache.data(),
            k_cache_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            v_cache_buffer->ptr,
            host_v_cache.data(),
            v_cache_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 paged KV write caches back to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 paged KV write HIP output staging copy");
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
                const size_t left_padding = kernel_size - 1 - kernel;
                if (timestep < left_padding) {
                    continue;
                }
                const size_t source_timestep = timestep - left_padding;
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

void linear_attn_gate_beta_f32_host(
    const float *a,
    const float *b,
    const float *a_log,
    const float *dt_bias,
    size_t heads,
    size_t sequence_len,
    float *gate_output,
    float *beta_output) {
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t head = 0; head < heads; ++head) {
            const size_t index = timestep * heads + head;
            const float x = a[index] + dt_bias[head];
            const float softplus = x <= 20.0f ? std::log1p(std::exp(x)) : x;
            gate_output[index] = -std::exp(a_log[head]) * softplus;
            const float b_value = b[index];
            beta_output[index] = 1.0f / (1.0f + std::exp(-b_value));
        }
    }
}

class HipLinearAttnGateBetaKernelCache {
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
            if (!hiprtc_runtime().compile_linear_attn_gate_beta_kernel(arch, &code, &compile_error)) {
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
                    "ullm_linear_attn_gate_beta_f32_kernel",
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
            compile_errors.empty() ? "failed to build linear attention gate beta HIP kernel" :
                                     compile_errors);
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

HipLinearAttnGateBetaKernelCache &hip_linear_attn_gate_beta_kernel_cache() {
    static HipLinearAttnGateBetaKernelCache cache;
    return cache;
}

bool linear_attn_gate_beta_f32_hip_kernel(
    const ullm_runtime_buffer *a_buffer,
    const ullm_runtime_buffer *b_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t sequence_len,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = a_buffer->hip_device_id;
    void *function = hip_linear_attn_gate_beta_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t elements = heads * sequence_len;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention gate beta element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    void *a_ptr = a_buffer->ptr;
    void *b_ptr = b_buffer->ptr;
    void *a_log_ptr = a_log_buffer->ptr;
    void *dt_bias_ptr = dt_bias_buffer->ptr;
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *beta_output_ptr = beta_output_buffer->ptr;
    void *kernel_params[] = {
        &a_ptr,
        &b_ptr,
        &a_log_ptr,
        &dt_bias_ptr,
        &kernel_heads,
        &kernel_sequence_len,
        &gate_output_ptr,
        &beta_output_ptr,
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
            *error = "hipModuleLaunchKernel failed for f32 linear attention gate beta";
        }
        return false;
    }
    return true;
}

ullm_status linear_attn_gate_beta_f32_hip_staging(
    const ullm_runtime_buffer *a_buffer,
    const ullm_runtime_buffer *b_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t sequence_len,
    size_t sequence_bytes,
    size_t head_bytes,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_a(sequence_bytes / sizeof(float));
    std::vector<float> host_b(sequence_bytes / sizeof(float));
    std::vector<float> host_a_log(head_bytes / sizeof(float));
    std::vector<float> host_dt_bias(head_bytes / sizeof(float));
    std::vector<float> host_gate(sequence_bytes / sizeof(float));
    std::vector<float> host_beta(sequence_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = a_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_a.data(),
            a_buffer->ptr,
            sequence_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b.data(),
            b_buffer->ptr,
            sequence_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_log.data(),
            a_log_buffer->ptr,
            head_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_dt_bias.data(),
            dt_bias_buffer->ptr,
            head_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention gate beta HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention gate beta HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    linear_attn_gate_beta_f32_host(
        host_a.data(),
        host_b.data(),
        host_a_log.data(),
        host_dt_bias.data(),
        heads,
        sequence_len,
        host_gate.data(),
        host_beta.data());
    if (!hip_runtime().copy_async(
            gate_output_buffer->ptr,
            host_gate.data(),
            sequence_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            beta_output_buffer->ptr,
            host_beta.data(),
            sequence_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention gate beta outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention gate beta HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void linear_attn_recurrent_f32_host(
    const float *q,
    const float *k,
    const float *v,
    const float *gate,
    const float *beta,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    float *state,
    float *output) {
    const size_t key_head_group = value_heads / key_heads;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        for (size_t value_head = 0; value_head < value_heads; ++value_head) {
            const size_t key_head = value_head / key_head_group;
            const size_t value_head_index = timestep * value_heads + value_head;
            const size_t key_head_index = timestep * key_heads + key_head;
            const size_t qk_base = key_head_index * key_dim;
            const size_t v_base = value_head_index * value_dim;
            const size_t state_head_offset = value_head * key_dim * value_dim;
            const float decay = std::exp(gate[value_head_index]);
            const float beta_value = beta[value_head_index];

            for (size_t key = 0; key < key_dim; ++key) {
                const size_t state_key_offset = state_head_offset + key * value_dim;
                for (size_t value = 0; value < value_dim; ++value) {
                    state[state_key_offset + value] *= decay;
                }
            }

            for (size_t value = 0; value < value_dim; ++value) {
                float current = 0.0f;
                for (size_t key = 0; key < key_dim; ++key) {
                    current += state[state_head_offset + key * value_dim + value] *
                               k[qk_base + key];
                }
                const float v_prime = (v[v_base + value] - current) * beta_value;
                for (size_t key = 0; key < key_dim; ++key) {
                    state[state_head_offset + key * value_dim + value] +=
                        k[qk_base + key] * v_prime;
                }
            }

            for (size_t value = 0; value < value_dim; ++value) {
                float sum = 0.0f;
                for (size_t key = 0; key < key_dim; ++key) {
                    sum += state[state_head_offset + key * value_dim + value] *
                           q[qk_base + key];
                }
                output[v_base + value] = sum;
            }
        }
    }
}

class HipLinearAttnRecurrentKernelCache {
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
            if (!hiprtc_runtime().compile_linear_attn_recurrent_kernel(arch, &code, &compile_error)) {
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
                    "ullm_linear_attn_recurrent_f32_kernel",
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
            compile_errors.empty() ? "failed to build linear attention recurrent HIP kernel" :
                                     compile_errors);
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

HipLinearAttnRecurrentKernelCache &hip_linear_attn_recurrent_kernel_cache() {
    static HipLinearAttnRecurrentKernelCache cache;
    return cache;
}

bool linear_attn_recurrent_f32_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *beta_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    ullm_runtime_buffer *state_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_buffer->hip_device_id;
    void *function = hip_linear_attn_recurrent_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    if (value_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention recurrent value head count exceeds HIP grid limit";
        }
        return false;
    }

    unsigned long long kernel_key_heads = static_cast<unsigned long long>(key_heads);
    unsigned long long kernel_value_heads = static_cast<unsigned long long>(value_heads);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    unsigned long long kernel_key_dim = static_cast<unsigned long long>(key_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    void *q_ptr = q_buffer->ptr;
    void *k_ptr = k_buffer->ptr;
    void *v_ptr = v_buffer->ptr;
    void *gate_ptr = gate_buffer->ptr;
    void *beta_ptr = beta_buffer->ptr;
    void *state_ptr = state_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &v_ptr,
        &gate_ptr,
        &beta_ptr,
        &kernel_key_heads,
        &kernel_value_heads,
        &kernel_sequence_len,
        &kernel_key_dim,
        &kernel_value_dim,
        &state_ptr,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(value_heads),
            1,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 linear attention recurrent";
        }
        return false;
    }
    return true;
}

ullm_status linear_attn_recurrent_f32_hip_staging(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *beta_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    size_t qk_bytes,
    size_t v_bytes,
    size_t gate_beta_bytes,
    size_t state_bytes,
    ullm_runtime_buffer *state_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(qk_bytes / sizeof(float));
    std::vector<float> host_k(qk_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<float> host_gate(gate_beta_bytes / sizeof(float));
    std::vector<float> host_beta(gate_beta_bytes / sizeof(float));
    std::vector<float> host_state(state_bytes / sizeof(float));
    std::vector<float> host_output(v_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            qk_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k.data(),
            k_buffer->ptr,
            qk_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v.data(),
            v_buffer->ptr,
            v_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            gate_beta_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_beta.data(),
            beta_buffer->ptr,
            gate_beta_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_state.data(),
            state_buffer->ptr,
            state_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention recurrent HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention recurrent HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    linear_attn_recurrent_f32_host(
        host_q.data(),
        host_k.data(),
        host_v.data(),
        host_gate.data(),
        host_beta.data(),
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        host_state.data(),
        host_output.data());
    if (!hip_runtime().copy_async(
            state_buffer->ptr,
            host_state.data(),
            state_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            v_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 linear attention recurrent outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 linear attention recurrent HIP output staging copy");
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

    const bool disable_rocblas = std::getenv("ULLM_DISABLE_ROCBLAS_MATVEC") != nullptr;
    const bool require_rocblas = std::getenv("ULLM_REQUIRE_ROCBLAS_MATVEC") != nullptr;
    const bool enable_rocblas =
        require_rocblas || std::getenv("ULLM_ENABLE_ROCBLAS_MATVEC") != nullptr;
    if (disable_rocblas && require_rocblas) {
        set_error("ULLM_DISABLE_ROCBLAS_MATVEC and ULLM_REQUIRE_ROCBLAS_MATVEC are both set");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    if (enable_rocblas && !disable_rocblas) {
        std::string rocblas_error;
        if (rocblas_runtime().sgemv_row_major(
                matrix_buffer->hip_device_id,
                stream == nullptr ? nullptr : stream->stream,
                matrix_buffer->ptr,
                input_buffer->ptr,
                rows,
                cols,
                output_buffer->ptr,
                &rocblas_error)) {
            set_error("");
            return ULLM_STATUS_OK;
        }
        if (require_rocblas) {
            set_error(rocblas_error.empty() ? "rocBLAS SGEMV is unavailable" : rocblas_error.c_str());
            return ULLM_STATUS_RUNTIME_ERROR;
        }
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

ullm_status ullm_runtime_sigmoid_mul_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 Sigmoid-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 Sigmoid-mul elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(gate_buffer, input_buffer) ||
        !buffers_share_backend(gate_buffer, output_buffer)) {
        set_error("f32 Sigmoid-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 Sigmoid-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 Sigmoid-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (gate_buffer->bytes < required_bytes) {
        set_error("f32 Sigmoid-mul gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_bytes) {
        set_error("f32 Sigmoid-mul input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 Sigmoid-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (gate_buffer->backend == BackendKind::Cpu) {
        const auto *gate = static_cast<const float *>(gate_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        sigmoid_mul_f32_host(gate, input, elements, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (sigmoid_mul_f32_hip_kernel(
            gate_buffer,
            input_buffer,
            elements,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 Sigmoid-mul HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return sigmoid_mul_f32_hip_staging(
        gate_buffer,
        input_buffer,
        elements,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_add_f32(
    const ullm_runtime_buffer *lhs_buffer,
    const ullm_runtime_buffer *rhs_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (lhs_buffer == nullptr || rhs_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 add received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0) {
        set_error("f32 add elements must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(lhs_buffer, rhs_buffer) ||
        !buffers_share_backend(lhs_buffer, output_buffer)) {
        set_error("f32 add buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 add stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float)) {
        set_error("f32 add byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (lhs_buffer->bytes < required_bytes) {
        set_error("f32 add lhs buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rhs_buffer->bytes < required_bytes) {
        set_error("f32 add rhs buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 add output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (lhs_buffer->backend == BackendKind::Cpu) {
        const auto *lhs = static_cast<const float *>(lhs_buffer->ptr);
        const auto *rhs = static_cast<const float *>(rhs_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        add_f32_host(lhs, rhs, elements, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (add_f32_hip_kernel(
            lhs_buffer,
            rhs_buffer,
            elements,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_ADD_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 add HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return add_f32_hip_staging(
        lhs_buffer,
        rhs_buffer,
        elements,
        required_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_rope_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 RoPE received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (sequence_len == 0 || heads == 0 || head_dim == 0 || rotary_dim == 0) {
        set_error("f32 RoPE sequence_len, heads, head_dim, and rotary_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        set_error("f32 RoPE rotary_dim must be even and no greater than head_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(rope_base) || rope_base <= 1.0f) {
        set_error("f32 RoPE base must be finite and greater than one");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 RoPE buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 RoPE stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (heads > max_size / sequence_len) {
        set_error("f32 RoPE head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t head_sequence = heads * sequence_len;
    if (head_dim > max_size / head_sequence) {
        set_error("f32 RoPE element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = head_sequence * head_dim;
    if (elements > max_size / sizeof(float)) {
        set_error("f32 RoPE byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_bytes = elements * sizeof(float);
    if (input_buffer->bytes < required_bytes) {
        set_error("f32 RoPE input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_bytes) {
        set_error("f32 RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        rope_f32_host(
            input,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (rope_f32_hip_kernel(
            input_buffer,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_ROPE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 RoPE HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return rope_f32_hip_staging(
        input_buffer,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
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

ullm_status ullm_runtime_causal_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 causal attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (sequence_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 causal attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 causal attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 causal attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, v_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 causal attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 causal attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (q_heads > max_size / sequence_len) {
        set_error("f32 causal attention q head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_head_sequence = sequence_len * q_heads;
    if (head_dim > max_size / q_head_sequence) {
        set_error("f32 causal attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_head_sequence * head_dim;

    if (kv_heads > max_size / sequence_len) {
        set_error("f32 causal attention kv head-sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_sequence = sequence_len * kv_heads;
    if (head_dim > max_size / kv_head_sequence ||
        value_dim > max_size / kv_head_sequence) {
        set_error("f32 causal attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_sequence * head_dim;
    const size_t v_elements = kv_head_sequence * value_dim;
    if (value_dim > max_size / q_head_sequence) {
        set_error("f32 causal attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_head_sequence * value_dim;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 causal attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 causal attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->bytes < k_bytes) {
        set_error("f32 causal attention k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 causal attention v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 causal attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        causal_attn_f32_host(
            q,
            k,
            v,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (causal_attn_f32_hip_kernel(
            q_buffer,
            k_buffer,
            v_buffer,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 causal attention HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return causal_attn_f32_hip_staging(
        q_buffer,
        k_buffer,
        v_buffer,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_decode_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 decode attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 decode attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 decode attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 decode attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 decode attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 decode attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (head_dim > max_size / q_heads) {
        set_error("f32 decode attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_heads * head_dim;

    if (kv_heads > max_size / cache_len) {
        set_error("f32 decode attention kv head-cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = cache_len * kv_heads;
    if (head_dim > max_size / kv_head_cache ||
        value_dim > max_size / kv_head_cache) {
        set_error("f32 decode attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_cache * head_dim;
    const size_t v_elements = kv_head_cache * value_dim;
    if (value_dim > max_size / q_heads) {
        set_error("f32 decode attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_heads * value_dim;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("f32 decode attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 decode attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("f32 decode attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("f32 decode attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 decode attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k_cache = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const float *>(v_cache_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        decode_attn_f32_host(
            q,
            k_cache,
            v_cache,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (decode_attn_f32_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 decode attention HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return decode_attn_f32_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_decode_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_cache_buffer == nullptr || v_cache_buffer == nullptr ||
        block_table_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 paged decode attention received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (cache_len == 0 || block_size == 0 || cache_blocks == 0 || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || value_dim == 0) {
        set_error("f32 paged decode attention dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        set_error("f32 paged decode attention q_heads must be a multiple of kv_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        set_error("f32 paged decode attention softmax scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_buffer, v_cache_buffer) ||
        !buffers_share_backend(q_buffer, block_table_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 paged decode attention buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 paged decode attention stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t block_table_entries = ((cache_len - 1) / block_size) + 1;
    if (cache_blocks > max_size / block_size) {
        set_error("f32 paged decode attention physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (block_table_entries > cache_blocks || cache_len > physical_tokens) {
        set_error("f32 paged decode attention cache_len exceeds physical cache block capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (head_dim > max_size / q_heads) {
        set_error("f32 paged decode attention q element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = q_heads * head_dim;

    if (kv_heads > max_size / physical_tokens) {
        set_error("f32 paged decode attention kv head-cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = physical_tokens * kv_heads;
    if (head_dim > max_size / kv_head_cache ||
        value_dim > max_size / kv_head_cache) {
        set_error("f32 paged decode attention kv element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_head_cache * head_dim;
    const size_t v_elements = kv_head_cache * value_dim;
    if (value_dim > max_size / q_heads) {
        set_error("f32 paged decode attention output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_heads * value_dim;
    if (q_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float) ||
        block_table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("f32 paged decode attention byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 paged decode attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_bytes) {
        set_error("f32 paged decode attention k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_bytes) {
        set_error("f32 paged decode attention v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_table_buffer->bytes < block_table_bytes) {
        set_error("f32 paged decode attention block table buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < output_bytes) {
        set_error("f32 paged decode attention output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k_cache = static_cast<const float *>(k_cache_buffer->ptr);
        const auto *v_cache = static_cast<const float *>(v_cache_buffer->ptr);
        const auto *block_table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        for (size_t entry = 0; entry < block_table_entries; ++entry) {
            if (static_cast<size_t>(block_table[entry]) >= cache_blocks) {
                set_error("f32 paged decode attention block table contains an out-of-range block id");
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        auto *output = static_cast<float *>(output_buffer->ptr);
        paged_decode_attn_f32_host(
            q,
            k_cache,
            v_cache,
            block_table,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_decode_attn_f32_hip_kernel(
            q_buffer,
            k_cache_buffer,
            v_cache_buffer,
            block_table_buffer,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 paged decode attention HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return paged_decode_attn_f32_hip_staging(
        q_buffer,
        k_cache_buffer,
        v_cache_buffer,
        block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_bytes,
        k_bytes,
        v_bytes,
        block_table_bytes,
        output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_paged_kv_write_f32(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    if (k_buffer == nullptr || v_buffer == nullptr || block_table_buffer == nullptr ||
        k_cache_buffer == nullptr || v_cache_buffer == nullptr) {
        set_error("f32 paged KV write received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_size == 0 || cache_blocks == 0 || kv_heads == 0 ||
        head_dim == 0 || value_dim == 0) {
        set_error("f32 paged KV write dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(k_buffer, v_buffer) ||
        !buffers_share_backend(k_buffer, block_table_buffer) ||
        !buffers_share_backend(k_buffer, k_cache_buffer) ||
        !buffers_share_backend(k_buffer, v_cache_buffer)) {
        set_error("f32 paged KV write buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(k_cache_buffer, stream) ||
        !stream_matches_buffer(v_cache_buffer, stream)) {
        set_error("f32 paged KV write stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size) {
        set_error("f32 paged KV write physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (cache_position >= physical_tokens) {
        set_error("f32 paged KV write cache_position exceeds physical cache capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t block_index = cache_position / block_size;
    const size_t block_table_entries = block_index + 1;

    if (head_dim > max_size / kv_heads || value_dim > max_size / kv_heads) {
        set_error("f32 paged KV write token element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_elements = kv_heads * head_dim;
    const size_t v_elements = kv_heads * value_dim;
    if (kv_heads > max_size / physical_tokens) {
        set_error("f32 paged KV write kv head-cache count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = physical_tokens * kv_heads;
    if (head_dim > max_size / kv_head_cache ||
        value_dim > max_size / kv_head_cache) {
        set_error("f32 paged KV write cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_cache_elements = kv_head_cache * head_dim;
    const size_t v_cache_elements = kv_head_cache * value_dim;
    if (k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        k_cache_elements > max_size / sizeof(float) ||
        v_cache_elements > max_size / sizeof(float) ||
        block_table_entries > max_size / sizeof(std::uint32_t)) {
        set_error("f32 paged KV write byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t k_cache_bytes = k_cache_elements * sizeof(float);
    const size_t v_cache_bytes = v_cache_elements * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);

    if (k_buffer->bytes < k_bytes) {
        set_error("f32 paged KV write k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 paged KV write v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_table_buffer->bytes < block_table_bytes) {
        set_error("f32 paged KV write block table buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_cache_bytes) {
        set_error("f32 paged KV write k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_cache_bytes) {
        set_error("f32 paged KV write v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (k_buffer->backend == BackendKind::Cpu) {
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        const auto *block_table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        if (static_cast<size_t>(block_table[block_index]) >= cache_blocks) {
            set_error("f32 paged KV write block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        auto *k_cache = static_cast<float *>(k_cache_buffer->ptr);
        auto *v_cache = static_cast<float *>(v_cache_buffer->ptr);
        paged_kv_write_f32_host(
            k,
            v,
            block_table,
            cache_position,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            k_cache,
            v_cache);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_kv_write_f32_hip_kernel(
            k_buffer,
            v_buffer,
            block_table_buffer,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            k_cache_buffer,
            v_cache_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 paged KV write HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return paged_kv_write_f32_hip_staging(
        k_buffer,
        v_buffer,
        block_table_buffer,
        cache_position,
        block_size,
        cache_blocks,
        kv_heads,
        head_dim,
        value_dim,
        k_bytes,
        v_bytes,
        block_table_bytes,
        k_cache_bytes,
        v_cache_bytes,
        k_cache_buffer,
        v_cache_buffer,
        stream);
}

ullm_status ullm_runtime_linear_attn_gate_beta_f32(
    const ullm_runtime_buffer *a_buffer,
    const ullm_runtime_buffer *b_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t sequence_len,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    if (a_buffer == nullptr || b_buffer == nullptr || a_log_buffer == nullptr ||
        dt_bias_buffer == nullptr || gate_output_buffer == nullptr ||
        beta_output_buffer == nullptr) {
        set_error("f32 linear attention gate beta received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (heads == 0 || sequence_len == 0) {
        set_error("f32 linear attention gate beta heads and sequence length must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(a_buffer, b_buffer) ||
        !buffers_share_backend(a_buffer, a_log_buffer) ||
        !buffers_share_backend(a_buffer, dt_bias_buffer) ||
        !buffers_share_backend(a_buffer, gate_output_buffer) ||
        !buffers_share_backend(a_buffer, beta_output_buffer)) {
        set_error("f32 linear attention gate beta buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(gate_output_buffer, stream) ||
        !stream_matches_buffer(beta_output_buffer, stream)) {
        set_error("f32 linear attention gate beta stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (sequence_len > max_size / heads) {
        set_error("f32 linear attention gate beta element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t sequence_elements = heads * sequence_len;
    if (sequence_elements > max_size / sizeof(float) ||
        heads > max_size / sizeof(float)) {
        set_error("f32 linear attention gate beta byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t sequence_bytes = sequence_elements * sizeof(float);
    const size_t head_bytes = heads * sizeof(float);
    if (a_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta a buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (b_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta b buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_log_buffer->bytes < head_bytes) {
        set_error("f32 linear attention gate beta A_log buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (dt_bias_buffer->bytes < head_bytes) {
        set_error("f32 linear attention gate beta dt_bias buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_output_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (beta_output_buffer->bytes < sequence_bytes) {
        set_error("f32 linear attention gate beta beta output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (a_buffer->backend == BackendKind::Cpu) {
        const auto *a = static_cast<const float *>(a_buffer->ptr);
        const auto *b = static_cast<const float *>(b_buffer->ptr);
        const auto *a_log = static_cast<const float *>(a_log_buffer->ptr);
        const auto *dt_bias = static_cast<const float *>(dt_bias_buffer->ptr);
        auto *gate = static_cast<float *>(gate_output_buffer->ptr);
        auto *beta = static_cast<float *>(beta_output_buffer->ptr);
        linear_attn_gate_beta_f32_host(
            a,
            b,
            a_log,
            dt_bias,
            heads,
            sequence_len,
            gate,
            beta);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_gate_beta_f32_hip_kernel(
            a_buffer,
            b_buffer,
            a_log_buffer,
            dt_bias_buffer,
            heads,
            sequence_len,
            gate_output_buffer,
            beta_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "f32 linear attention gate beta HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_gate_beta_f32_hip_staging(
        a_buffer,
        b_buffer,
        a_log_buffer,
        dt_bias_buffer,
        heads,
        sequence_len,
        sequence_bytes,
        head_bytes,
        gate_output_buffer,
        beta_output_buffer,
        stream);
}

ullm_status ullm_runtime_linear_attn_recurrent_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *beta_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    ullm_runtime_buffer *state_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr ||
        gate_buffer == nullptr || beta_buffer == nullptr || state_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 linear attention recurrent received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads == 0 || value_heads == 0 || sequence_len == 0 || key_dim == 0 ||
        value_dim == 0) {
        set_error(
            "f32 linear attention recurrent key_heads, value_heads, sequence length, key_dim, and value_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (value_heads % key_heads != 0) {
        set_error("f32 linear attention recurrent value_heads must be a multiple of key_heads");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_buffer, k_buffer) ||
        !buffers_share_backend(q_buffer, v_buffer) ||
        !buffers_share_backend(q_buffer, gate_buffer) ||
        !buffers_share_backend(q_buffer, beta_buffer) ||
        !buffers_share_backend(q_buffer, state_buffer) ||
        !buffers_share_backend(q_buffer, output_buffer)) {
        set_error("f32 linear attention recurrent buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(state_buffer, stream) ||
        !stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 linear attention recurrent stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (sequence_len > max_size / key_heads) {
        set_error("f32 linear attention recurrent key head sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t key_head_sequence_elements = key_heads * sequence_len;
    if (sequence_len > max_size / value_heads) {
        set_error("f32 linear attention recurrent value head sequence element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t value_head_sequence_elements = value_heads * sequence_len;
    if (key_head_sequence_elements > max_size / key_dim) {
        set_error("f32 linear attention recurrent q/k element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qk_elements = key_head_sequence_elements * key_dim;
    if (value_head_sequence_elements > max_size / value_dim) {
        set_error("f32 linear attention recurrent v/output element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t v_elements = value_head_sequence_elements * value_dim;
    if (value_heads > max_size / key_dim) {
        set_error("f32 linear attention recurrent state key element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t state_key_elements = value_heads * key_dim;
    if (state_key_elements > max_size / value_dim) {
        set_error("f32 linear attention recurrent state element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t state_elements = state_key_elements * value_dim;
    if (qk_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        value_head_sequence_elements > max_size / sizeof(float) ||
        state_elements > max_size / sizeof(float)) {
        set_error("f32 linear attention recurrent byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qk_bytes = qk_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t gate_beta_bytes = value_head_sequence_elements * sizeof(float);
    const size_t state_bytes = state_elements * sizeof(float);

    if (q_buffer->bytes < qk_bytes) {
        set_error("f32 linear attention recurrent q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_buffer->bytes < qk_bytes) {
        set_error("f32 linear attention recurrent k buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_buffer->bytes < v_bytes) {
        set_error("f32 linear attention recurrent v buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer->bytes < gate_beta_bytes) {
        set_error("f32 linear attention recurrent gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (beta_buffer->bytes < gate_beta_bytes) {
        set_error("f32 linear attention recurrent beta buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (state_buffer->bytes < state_bytes) {
        set_error("f32 linear attention recurrent state buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < v_bytes) {
        set_error("f32 linear attention recurrent output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_buffer->backend == BackendKind::Cpu) {
        const auto *q = static_cast<const float *>(q_buffer->ptr);
        const auto *k = static_cast<const float *>(k_buffer->ptr);
        const auto *v = static_cast<const float *>(v_buffer->ptr);
        const auto *gate = static_cast<const float *>(gate_buffer->ptr);
        const auto *beta = static_cast<const float *>(beta_buffer->ptr);
        auto *state = static_cast<float *>(state_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        linear_attn_recurrent_f32_host(
            q,
            k,
            v,
            gate,
            beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            state,
            output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_recurrent_f32_hip_kernel(
            q_buffer,
            k_buffer,
            v_buffer,
            gate_buffer,
            beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            state_buffer,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ?
                "f32 linear attention recurrent HIP kernel is unavailable" :
                hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_recurrent_f32_hip_staging(
        q_buffer,
        k_buffer,
        v_buffer,
        gate_buffer,
        beta_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        qk_bytes,
        v_bytes,
        gate_beta_bytes,
        state_bytes,
        state_buffer,
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
