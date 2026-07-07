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

constexpr unsigned int kAq4MatvecTop1RowsPerBlock = 8u;

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

bool aq4_rows_per_block_is_valid(unsigned long value) {
    return value >= 1ul && value <= 32ul && 256ul % value == 0ul;
}

unsigned int aq4_rows_per_block_from_env(
    const char *primary_env_name,
    const char *fallback_env_name,
    unsigned int fallback) {
    auto parse = [](const char *env_name, unsigned int *out) {
        if (env_name == nullptr || out == nullptr) {
            return false;
        }
        const char *raw = std::getenv(env_name);
        if (raw == nullptr || raw[0] == '\0') {
            return false;
        }
        char *end = nullptr;
        const unsigned long value = std::strtoul(raw, &end, 10);
        if (end == raw || *end != '\0' || !aq4_rows_per_block_is_valid(value)) {
            return false;
        }
        *out = static_cast<unsigned int>(value);
        return true;
    };

    unsigned int value = fallback;
    if (parse(primary_env_name, &value)) {
        return value;
    }
    if (parse(fallback_env_name, &value)) {
        return value;
    }
    return fallback;
}

unsigned int aq4_matvec_top1_rows_per_block_from_env() {
    return aq4_rows_per_block_from_env(
        "ULLM_AQ4_MATVEC_TOP1_RPB",
        nullptr,
        kAq4MatvecTop1RowsPerBlock);
}

unsigned int block_size_from_env(const char *env_name, unsigned int fallback) {
    const char *raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 32ul || value > 256ul ||
        (value & (value - 1ul)) != 0ul) {
        return fallback;
    }
    return static_cast<unsigned int>(value);
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

    bool module_launch_kernel_2d(
        void *function,
        unsigned int grid_x,
        unsigned int grid_y,
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
                   grid_y,
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

    bool compile_aq4_row_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, aq4_row_kernel_source(), "ullm_aq4_row_f32.hip", code, error);
    }

    bool compile_aq4_matvec_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        const std::string source = aq4_matvec_kernel_source_for_arch(arch);
        return compile_kernel(arch, source.c_str(), "ullm_aq4_matvec_f32.hip", code, error);
    }

    bool compile_aq4_matvec_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_batch_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_batch_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_top1_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_top1_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_top1_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_add_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_add_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_add_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_pair_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_pair_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_pair_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_triple_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_triple_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_triple_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_qkv_z_gate_beta_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_qkv_z_gate_beta_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_qkv_z_gate_beta_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_silu_mul_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_silu_mul_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_silu_mul_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_gate_beta_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_gate_beta_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_gate_beta_f32.hip",
            code,
            error);
    }

    bool compile_matvec_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, matvec_kernel_source(), "ullm_matvec_f32.hip", code, error);
    }

    bool compile_matvec_bf16_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            matvec_bf16_kernel_source(),
            "ullm_matvec_bf16_f32.hip",
            code,
            error);
    }

    bool compile_bf16_row_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            bf16_row_kernel_source(),
            "ullm_bf16_row_f32.hip",
            code,
            error);
    }

    bool compile_top1_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, top1_kernel_source(), "ullm_top1_f32.hip", code, error);
    }

    bool compile_top1_pairs_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            top1_pairs_kernel_source(),
            "ullm_top1_pairs_f32.hip",
            code,
            error);
    }

    bool compile_rmsnorm_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, rmsnorm_kernel_source(), "ullm_rmsnorm_f32.hip", code, error);
    }

    bool compile_segmented_rmsnorm_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            segmented_rmsnorm_kernel_source(),
            "ullm_segmented_rmsnorm_f32.hip",
            code,
            error);
    }

    bool compile_segmented_rmsnorm_silu_mul_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            segmented_rmsnorm_silu_mul_kernel_source(),
            "ullm_segmented_rmsnorm_silu_mul_f32.hip",
            code,
            error);
    }

    bool compile_silu_mul_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, silu_mul_kernel_source(), "ullm_silu_mul_f32.hip", code, error);
    }

    bool compile_sigmoid_mul_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(arch, sigmoid_mul_kernel_source(), "ullm_sigmoid_mul_f32.hip", code, error);
    }

    bool compile_qwen35_split_q_gate_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_split_q_gate_kernel_source(),
            "ullm_qwen35_split_q_gate_f32.hip",
            code,
            error);
    }

    bool compile_qwen35_qk_norm_rope_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_qk_norm_rope_kernel_source(),
            "ullm_qwen35_qk_norm_rope_f32.hip",
            code,
            error);
    }

    bool compile_qwen35_qk_norm_rope_paged_kv_write_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_qk_norm_rope_paged_kv_write_kernel_source(),
            "ullm_qwen35_qk_norm_rope_paged_kv_write_f32.hip",
            code,
            error);
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
            paged_decode_attn_kernel_source_for_arch(arch).c_str(),
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

    bool compile_linear_attn_qkv_prepare_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_qkv_prepare_kernel_source(),
            "ullm_linear_attn_qkv_prepare_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_qkv_prepare_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_qkv_prepare_batch_kernel_source(),
            "ullm_linear_attn_qkv_prepare_batch_f32.hip",
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

    static const char *aq4_row_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_row_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long row_index,
    float *output,
    unsigned int *error_out) {
    const unsigned long long col =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= cols || row_index >= rows) {
        return;
    }
    const unsigned long long element = row_index * cols + col;
    const unsigned char packed = indices[element >> 1];
    const unsigned char codebook_index =
        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const unsigned long long group = element / group_size;
    const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
    if (scale_index >= scale_count) {
        atomicOr(error_out, 1u);
        output[col] = 0.0f;
        return;
    }
    float value = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
    if (row_scales != nullptr && row_index < row_scale_count) {
        value *= row_scales[row_index];
    }
    output[col] = value;
}
)";
    }

    static unsigned int aq4_rows_per_block_for_arch(
        const std::string &arch,
        const char *env_name = nullptr) {
        const unsigned int fallback = arch.rfind("gfx12", 0) == 0 ? 32u : 1u;
        return aq4_rows_per_block_from_env(env_name, nullptr, fallback);
    }

    static std::string aq4_rows_per_block_preamble_for_rows(unsigned int rows_per_block) {
        return "#define ULLM_AQ4_ROWS_PER_BLOCK " + std::to_string(rows_per_block) + "\n";
    }

    static std::string aq4_rows_per_block_preamble(
        const std::string &arch,
        const char *env_name = nullptr) {
        return aq4_rows_per_block_preamble_for_rows(
            aq4_rows_per_block_for_arch(arch, env_name));
    }

    static unsigned int aq4_fused_rows_per_block_for_arch(
        const std::string &arch,
        unsigned int rdna4_rows_per_block,
        unsigned int rdna2_rows_per_block = 2u,
        const char *env_name = nullptr) {
        // RDNA2 loses too much per-row parallelism above 2 rows/block on fused AQ4 kernels.
        const unsigned int fallback =
            arch.rfind("gfx12", 0) == 0 ? rdna4_rows_per_block : rdna2_rows_per_block;
        return aq4_rows_per_block_from_env(env_name, "ULLM_AQ4_FUSED_RPB", fallback);
    }

    static std::string aq4_fused_rows_per_block_preamble(
        const std::string &arch,
        unsigned int rdna4_rows_per_block,
        unsigned int rdna2_rows_per_block = 2u,
        const char *env_name = nullptr) {
        return aq4_rows_per_block_preamble_for_rows(
            aq4_fused_rows_per_block_for_arch(
                arch,
                rdna4_rows_per_block,
                rdna2_rows_per_block,
                env_name));
    }

    static std::string aq4_matvec_kernel_source_for_arch(const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_RPB") +
               aq4_matvec_kernel_source();
    }

    static std::string aq4_matvec_batch_kernel_source_for_arch(const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_BATCH_RPB") +
               aq4_matvec_batch_kernel_source();
    }

    static std::string aq4_matvec_top1_kernel_source_for_arch(const std::string &arch) {
        std::string preamble =
            aq4_rows_per_block_preamble_for_rows(aq4_matvec_top1_rows_per_block_from_env());
        if (arch.rfind("gfx12", 0) != 0 ||
            std::getenv("ULLM_DISABLE_AQ4_MATVEC_TOP1_WARP_REDUCE") != nullptr) {
            preamble += "#define ULLM_AQ4_MATVEC_TOP1_USE_SHARED_REDUCE 1\n";
        }
        return preamble + aq4_matvec_top1_kernel_source();
    }

    static std::string aq4_matvec_add_kernel_source_for_arch(const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_ADD_RPB") +
               aq4_matvec_add_kernel_source();
    }

    static std::string aq4_matvec_pair_kernel_source_for_arch(const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   16u,
                   4u,
                   "ULLM_AQ4_MATVEC_PAIR_RPB") +
               aq4_matvec_pair_kernel_source();
    }

    static std::string aq4_matvec_triple_kernel_source_for_arch(const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   4u,
                   "ULLM_AQ4_MATVEC_TRIPLE_RPB") +
               aq4_matvec_triple_kernel_source();
    }

    static std::string aq4_matvec_qkv_z_gate_beta_kernel_source_for_arch(
        const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   4u,
                   2u,
                   "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB") +
               aq4_matvec_qkv_z_gate_beta_kernel_source();
    }

    static std::string aq4_matvec_silu_mul_kernel_source_for_arch(const std::string &arch) {
        return aq4_fused_rows_per_block_preamble(
                   arch,
                   8u,
                   2u,
                   "ULLM_AQ4_MATVEC_SILU_MUL_RPB") +
               aq4_matvec_silu_mul_kernel_source();
    }

    static std::string aq4_matvec_gate_beta_kernel_source_for_arch(const std::string &arch) {
        return aq4_rows_per_block_preamble(arch, "ULLM_AQ4_MATVEC_RPB") +
               aq4_matvec_gate_beta_kernel_source();
    }

    static const char *aq4_matvec_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_matvec_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        if ((cols % group_size) == 0ull) {
            const unsigned long long groups_per_row = cols / group_size;
            for (unsigned long long group_in_row = lane; group_in_row < groups_per_row;
                 group_in_row += threads_per_row) {
                const unsigned long long group = static_cast<unsigned long long>(row) * groups_per_row +
                    group_in_row;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                float raw_sum = 0.0f;
                const unsigned long long col_start = group_in_row * group_size;
                if (group_size == 16ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if (group_size == 8ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if ((group_size & 1ull) == 0ull) {
                    for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else {
                    for (unsigned long long offset = 0; offset < group_size; ++offset) {
                        const unsigned long long col = col_start + offset;
                        const unsigned long long element = row_offset + col;
                        const unsigned char packed = indices[element >> 1];
                        const unsigned char codebook_index =
                            (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                        raw_sum += codebook[codebook_index] * input[col];
                    }
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        } else {
            for (unsigned long long col = lane; col < cols; col += threads_per_row) {
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                const unsigned long long group = element / group_size;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index < scale_count) {
                    const float value =
                        codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                    sum += value * input[col];
                }
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[row] = value;
    }
}
)";
    }

    static const char *aq4_matvec_batch_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_matvec_batch_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long batch_count,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned long long batch = static_cast<unsigned long long>(blockIdx.y);
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows && batch < batch_count) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        const float *batch_input = input + batch * cols;
        if ((cols % group_size) == 0ull) {
            const unsigned long long groups_per_row = cols / group_size;
            for (unsigned long long group_in_row = lane; group_in_row < groups_per_row;
                 group_in_row += threads_per_row) {
                const unsigned long long group = static_cast<unsigned long long>(row) * groups_per_row +
                    group_in_row;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                float raw_sum = 0.0f;
                const unsigned long long col_start = group_in_row * group_size;
                if (group_size == 16ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * batch_input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * batch_input[col + 1ull];
                    }
                } else if (group_size == 8ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * batch_input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * batch_input[col + 1ull];
                    }
                } else if ((group_size & 1ull) == 0ull) {
                    for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * batch_input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * batch_input[col + 1ull];
                    }
                } else {
                    for (unsigned long long offset = 0; offset < group_size; ++offset) {
                        const unsigned long long col = col_start + offset;
                        const unsigned long long element = row_offset + col;
                        const unsigned char packed = indices[element >> 1];
                        const unsigned char codebook_index =
                            (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                        raw_sum += codebook[codebook_index] * batch_input[col];
                    }
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        } else {
            for (unsigned long long col = lane; col < cols; col += threads_per_row) {
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                const unsigned long long group = element / group_size;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index < scale_count) {
                    const float value =
                        codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                    sum += value * batch_input[col];
                }
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows && batch < batch_count) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[batch * rows + row] = value;
    }
}
)";
    }

    static const char *aq4_matvec_top1_kernel_source() {
        return R"(
extern "C" __global__ void ullm_aq4_matvec_top1_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *partial_values,
    unsigned int *partial_indices) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    __shared__ float row_values[32];
    __shared__ unsigned int row_indices[32];
    float sum = 0.0f;
    if (row < rows) {
        const unsigned long long row_offset = static_cast<unsigned long long>(row) * cols;
        if ((cols % group_size) == 0ull) {
            const unsigned long long groups_per_row = cols / group_size;
            for (unsigned long long group_in_row = lane; group_in_row < groups_per_row;
                 group_in_row += threads_per_row) {
                const unsigned long long group =
                    static_cast<unsigned long long>(row) * groups_per_row + group_in_row;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index >= scale_count) {
                    continue;
                }
                float raw_sum = 0.0f;
                const unsigned long long col_start = group_in_row * group_size;
                if (group_size == 16ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if (group_size == 8ull) {
#pragma unroll
                    for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else if ((group_size & 1ull) == 0ull) {
                    for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                        const unsigned long long col = col_start + offset;
                        const unsigned char packed = indices[(row_offset + col) >> 1];
                        raw_sum += codebook[packed & 0x0f] * input[col];
                        raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                    }
                } else {
                    for (unsigned long long offset = 0; offset < group_size; ++offset) {
                        const unsigned long long col = col_start + offset;
                        const unsigned long long element = row_offset + col;
                        const unsigned char packed = indices[element >> 1];
                        const unsigned char codebook_index =
                            (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                        raw_sum += codebook[codebook_index] * input[col];
                    }
                }
                sum += raw_sum * scale_values[scale_index] * tensor_scale;
            }
        } else {
            for (unsigned long long col = lane; col < cols; col += threads_per_row) {
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                const unsigned long long group = element / group_size;
                const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
                if (scale_index < scale_count) {
                    const float value =
                        codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                    sum += value * input[col];
                }
            }
        }
    }
    float row_sum = sum;
#if defined(ULLM_AQ4_MATVEC_TOP1_USE_SHARED_REDUCE)
    partial[tid] = row_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    row_sum = partial[partial_offset];
#else
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        row_sum += __shfl_down(row_sum, offset, threads_per_row);
    }
#endif
    if (lane == 0) {
        float value = -3.4028234663852886e38f;
        unsigned int token_index = 0xffffffffu;
        if (row < rows) {
            value = row_sum;
            if (row_scales != nullptr && row < row_scale_count) {
                value *= row_scales[row];
            }
            if (!(value == value)) {
                value = -3.4028234663852886e38f;
            }
            token_index = static_cast<unsigned int>(row);
        }
        row_values[row_in_block] = value;
        row_indices[row_in_block] = token_index;
    }
    __syncthreads();
    if (tid < rows_per_block) {
        for (unsigned int offset = rows_per_block >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float right_value = row_values[tid + offset];
                const unsigned int right_index = row_indices[tid + offset];
                const float left_value = row_values[tid];
                const unsigned int left_index = row_indices[tid];
                if (right_value > left_value ||
                    (right_value == left_value && right_index < left_index)) {
                    row_values[tid] = right_value;
                    row_indices[tid] = right_index;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            partial_values[blockIdx.x] = row_values[0];
            partial_indices[blockIdx.x] = row_indices[0];
        }
    }
}
)";
    }

    static const char *aq4_matvec_add_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_add_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if ((group_size & 1ull) == 0ull) {
                for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

extern "C" __global__ void ullm_aq4_matvec_add_f32_kernel(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    const float *residual,
    const float *row_scales,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row_scale_count,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;
    if (row < rows) {
        sum = ullm_aq4_matvec_add_thread_sum(
            indices,
            scale_indices,
            codebook,
            scale_values,
            input,
            scale_count,
            group_size,
            tensor_scale,
            row,
            cols,
            lane,
            threads_per_row);
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];
        if (row_scales != nullptr && row < row_scale_count) {
            value *= row_scales[row];
        }
        output[row] = residual[row] + value;
    }
}
)";
    }

    static const char *aq4_matvec_pair_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_pair_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if ((group_size & 1ull) == 0ull) {
                for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_matvec_pair_thread_sums(
    const unsigned char *left_indices,
    const unsigned char *left_scale_indices,
    const float *left_codebook,
    const float *left_scale_values,
    unsigned long long left_scale_count,
    float left_tensor_scale,
    const unsigned char *right_indices,
    const unsigned char *right_scale_indices,
    const float *right_codebook,
    const float *right_scale_values,
    unsigned long long right_scale_count,
    float right_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *left_sum_output,
    float *right_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float left_sum = 0.0f;
    float right_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int left_scale_index =
            static_cast<unsigned int>(left_scale_indices[group]);
        const unsigned int right_scale_index =
            static_cast<unsigned int>(right_scale_indices[group]);
        float left_raw_sum = 0.0f;
        float right_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char left_packed = left_indices[packed_index];
                const unsigned char right_packed = right_indices[packed_index];
                left_raw_sum += left_codebook[left_packed & 0x0f] * input_low;
                right_raw_sum += right_codebook[right_packed & 0x0f] * input_low;
                left_raw_sum += left_codebook[(left_packed >> 4) & 0x0f] * input_high;
                right_raw_sum += right_codebook[(right_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char left_packed = left_indices[packed_index];
                const unsigned char right_packed = right_indices[packed_index];
                left_raw_sum += left_codebook[left_packed & 0x0f] * input_low;
                right_raw_sum += right_codebook[right_packed & 0x0f] * input_low;
                left_raw_sum += left_codebook[(left_packed >> 4) & 0x0f] * input_high;
                right_raw_sum += right_codebook[(right_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char left_packed = left_indices[packed_index];
                const unsigned char right_packed = right_indices[packed_index];
                left_raw_sum += left_codebook[left_packed & 0x0f] * input_low;
                right_raw_sum += right_codebook[right_packed & 0x0f] * input_low;
                left_raw_sum += left_codebook[(left_packed >> 4) & 0x0f] * input_high;
                right_raw_sum += right_codebook[(right_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char left_packed = left_indices[element >> 1];
                const unsigned char left_codebook_index =
                    (element & 1ull) == 0ull ? (left_packed & 0x0f) :
                                               ((left_packed >> 4) & 0x0f);
                const unsigned char right_packed = right_indices[element >> 1];
                const unsigned char right_codebook_index =
                    (element & 1ull) == 0ull ? (right_packed & 0x0f) :
                                               ((right_packed >> 4) & 0x0f);
                left_raw_sum += left_codebook[left_codebook_index] * input_value;
                right_raw_sum += right_codebook[right_codebook_index] * input_value;
            }
        }
        if (left_scale_index < left_scale_count) {
            left_sum += left_raw_sum * left_scale_values[left_scale_index] * left_tensor_scale;
        }
        if (right_scale_index < right_scale_count) {
            right_sum +=
                right_raw_sum * right_scale_values[right_scale_index] * right_tensor_scale;
        }
    }
    *left_sum_output = left_sum;
    *right_sum_output = right_sum;
}

extern "C" __global__ void ullm_aq4_matvec_pair_f32_kernel(
    const unsigned char *left_indices,
    const unsigned char *left_scale_indices,
    const float *left_codebook,
    const float *left_scale_values,
    const float *left_row_scales,
    unsigned long long left_scale_count,
    unsigned long long left_group_size,
    float left_tensor_scale,
    unsigned long long left_row_scale_count,
    const unsigned char *right_indices,
    const unsigned char *right_scale_indices,
    const float *right_codebook,
    const float *right_scale_values,
    const float *right_row_scales,
    unsigned long long right_scale_count,
    unsigned long long right_group_size,
    float right_tensor_scale,
    unsigned long long right_row_scale_count,
    const float *input,
    unsigned long long left_rows,
    unsigned long long right_rows,
    unsigned long long cols,
    float *left_output,
    float *right_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned long long work_rows = left_rows > right_rows ? left_rows : right_rows;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float left_partial[256];
    __shared__ float right_partial[256];
    float left_sum = 0.0f;
    float right_sum = 0.0f;
    if (row < work_rows) {
        if (row < left_rows && row < right_rows && left_group_size == right_group_size &&
            (cols % left_group_size) == 0ull) {
            ullm_aq4_matvec_pair_thread_sums(
                left_indices,
                left_scale_indices,
                left_codebook,
                left_scale_values,
                left_scale_count,
                left_tensor_scale,
                right_indices,
                right_scale_indices,
                right_codebook,
                right_scale_values,
                right_scale_count,
                right_tensor_scale,
                input,
                left_group_size,
                row,
                cols,
                lane,
                threads_per_row,
                &left_sum,
                &right_sum);
        } else {
            if (row < left_rows) {
                left_sum = ullm_aq4_matvec_pair_thread_sum(
                    left_indices,
                    left_scale_indices,
                    left_codebook,
                    left_scale_values,
                    input,
                    left_scale_count,
                    left_group_size,
                    left_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (row < right_rows) {
                right_sum = ullm_aq4_matvec_pair_thread_sum(
                    right_indices,
                    right_scale_indices,
                    right_codebook,
                    right_scale_values,
                    input,
                    right_scale_count,
                    right_group_size,
                    right_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
        }
    }
    left_partial[tid] = left_sum;
    right_partial[tid] = right_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            left_partial[partial_offset + lane] += left_partial[partial_offset + lane + offset];
            right_partial[partial_offset + lane] +=
                right_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        if (row < left_rows) {
            float left_value = left_partial[partial_offset];
            if (left_row_scales != nullptr && row < left_row_scale_count) {
                left_value *= left_row_scales[row];
            }
            left_output[row] = left_value;
        }
        if (row < right_rows) {
            float right_value = right_partial[partial_offset];
            if (right_row_scales != nullptr && row < right_row_scale_count) {
                right_value *= right_row_scales[row];
            }
            right_output[row] = right_value;
        }
    }
}
)";
    }

    static const char *aq4_matvec_triple_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_triple_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if ((group_size & 1ull) == 0ull) {
                for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_matvec_triple_thread_sums(
    const unsigned char *first_indices,
    const unsigned char *first_scale_indices,
    const float *first_codebook,
    const float *first_scale_values,
    unsigned long long first_scale_count,
    float first_tensor_scale,
    const unsigned char *second_indices,
    const unsigned char *second_scale_indices,
    const float *second_codebook,
    const float *second_scale_values,
    unsigned long long second_scale_count,
    float second_tensor_scale,
    const unsigned char *third_indices,
    const unsigned char *third_scale_indices,
    const float *third_codebook,
    const float *third_scale_values,
    unsigned long long third_scale_count,
    float third_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *first_sum_output,
    float *second_sum_output,
    float *third_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float third_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int first_scale_index =
            static_cast<unsigned int>(first_scale_indices[group]);
        const unsigned int second_scale_index =
            static_cast<unsigned int>(second_scale_indices[group]);
        const unsigned int third_scale_index =
            static_cast<unsigned int>(third_scale_indices[group]);
        float first_raw_sum = 0.0f;
        float second_raw_sum = 0.0f;
        float third_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char first_packed = first_indices[packed_index];
                const unsigned char second_packed = second_indices[packed_index];
                const unsigned char third_packed = third_indices[packed_index];
                first_raw_sum += first_codebook[first_packed & 0x0f] * input_low;
                second_raw_sum += second_codebook[second_packed & 0x0f] * input_low;
                third_raw_sum += third_codebook[third_packed & 0x0f] * input_low;
                first_raw_sum += first_codebook[(first_packed >> 4) & 0x0f] * input_high;
                second_raw_sum += second_codebook[(second_packed >> 4) & 0x0f] * input_high;
                third_raw_sum += third_codebook[(third_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char first_packed = first_indices[packed_index];
                const unsigned char second_packed = second_indices[packed_index];
                const unsigned char third_packed = third_indices[packed_index];
                first_raw_sum += first_codebook[first_packed & 0x0f] * input_low;
                second_raw_sum += second_codebook[second_packed & 0x0f] * input_low;
                third_raw_sum += third_codebook[third_packed & 0x0f] * input_low;
                first_raw_sum += first_codebook[(first_packed >> 4) & 0x0f] * input_high;
                second_raw_sum += second_codebook[(second_packed >> 4) & 0x0f] * input_high;
                third_raw_sum += third_codebook[(third_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char first_packed = first_indices[packed_index];
                const unsigned char second_packed = second_indices[packed_index];
                const unsigned char third_packed = third_indices[packed_index];
                first_raw_sum += first_codebook[first_packed & 0x0f] * input_low;
                second_raw_sum += second_codebook[second_packed & 0x0f] * input_low;
                third_raw_sum += third_codebook[third_packed & 0x0f] * input_low;
                first_raw_sum += first_codebook[(first_packed >> 4) & 0x0f] * input_high;
                second_raw_sum += second_codebook[(second_packed >> 4) & 0x0f] * input_high;
                third_raw_sum += third_codebook[(third_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char first_packed = first_indices[element >> 1];
                const unsigned char first_codebook_index =
                    (element & 1ull) == 0ull ? (first_packed & 0x0f) :
                                               ((first_packed >> 4) & 0x0f);
                const unsigned char second_packed = second_indices[element >> 1];
                const unsigned char second_codebook_index =
                    (element & 1ull) == 0ull ? (second_packed & 0x0f) :
                                               ((second_packed >> 4) & 0x0f);
                const unsigned char third_packed = third_indices[element >> 1];
                const unsigned char third_codebook_index =
                    (element & 1ull) == 0ull ? (third_packed & 0x0f) :
                                               ((third_packed >> 4) & 0x0f);
                first_raw_sum += first_codebook[first_codebook_index] * input_value;
                second_raw_sum += second_codebook[second_codebook_index] * input_value;
                third_raw_sum += third_codebook[third_codebook_index] * input_value;
            }
        }
        if (first_scale_index < first_scale_count) {
            first_sum += first_raw_sum * first_scale_values[first_scale_index] *
                         first_tensor_scale;
        }
        if (second_scale_index < second_scale_count) {
            second_sum += second_raw_sum * second_scale_values[second_scale_index] *
                          second_tensor_scale;
        }
        if (third_scale_index < third_scale_count) {
            third_sum += third_raw_sum * third_scale_values[third_scale_index] *
                         third_tensor_scale;
        }
    }
    *first_sum_output = first_sum;
    *second_sum_output = second_sum;
    *third_sum_output = third_sum;
}

extern "C" __global__ void ullm_aq4_matvec_triple_f32_kernel(
    const unsigned char *first_indices,
    const unsigned char *first_scale_indices,
    const float *first_codebook,
    const float *first_scale_values,
    const float *first_row_scales,
    unsigned long long first_scale_count,
    unsigned long long first_group_size,
    float first_tensor_scale,
    unsigned long long first_row_scale_count,
    const unsigned char *second_indices,
    const unsigned char *second_scale_indices,
    const float *second_codebook,
    const float *second_scale_values,
    const float *second_row_scales,
    unsigned long long second_scale_count,
    unsigned long long second_group_size,
    float second_tensor_scale,
    unsigned long long second_row_scale_count,
    const unsigned char *third_indices,
    const unsigned char *third_scale_indices,
    const float *third_codebook,
    const float *third_scale_values,
    const float *third_row_scales,
    unsigned long long third_scale_count,
    unsigned long long third_group_size,
    float third_tensor_scale,
    unsigned long long third_row_scale_count,
    const float *input,
    unsigned long long first_rows,
    unsigned long long second_rows,
    unsigned long long third_rows,
    unsigned long long cols,
    float *first_output,
    float *second_output,
    float *third_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    unsigned long long work_rows = first_rows > second_rows ? first_rows : second_rows;
    work_rows = work_rows > third_rows ? work_rows : third_rows;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float first_partial[256];
    __shared__ float second_partial[256];
    __shared__ float third_partial[256];
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float third_sum = 0.0f;
    if (row < work_rows) {
        if (row < first_rows && row < second_rows && row < third_rows &&
            first_group_size == second_group_size && first_group_size == third_group_size &&
            (cols % first_group_size) == 0ull) {
            ullm_aq4_matvec_triple_thread_sums(
                first_indices,
                first_scale_indices,
                first_codebook,
                first_scale_values,
                first_scale_count,
                first_tensor_scale,
                second_indices,
                second_scale_indices,
                second_codebook,
                second_scale_values,
                second_scale_count,
                second_tensor_scale,
                third_indices,
                third_scale_indices,
                third_codebook,
                third_scale_values,
                third_scale_count,
                third_tensor_scale,
                input,
                first_group_size,
                row,
                cols,
                lane,
                threads_per_row,
                &first_sum,
                &second_sum,
                &third_sum);
        } else {
            if (row < first_rows) {
                first_sum = ullm_aq4_matvec_triple_thread_sum(
                    first_indices,
                    first_scale_indices,
                    first_codebook,
                    first_scale_values,
                    input,
                    first_scale_count,
                    first_group_size,
                    first_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (row < second_rows) {
                second_sum = ullm_aq4_matvec_triple_thread_sum(
                    second_indices,
                    second_scale_indices,
                    second_codebook,
                    second_scale_values,
                    input,
                    second_scale_count,
                    second_group_size,
                    second_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (row < third_rows) {
                third_sum = ullm_aq4_matvec_triple_thread_sum(
                    third_indices,
                    third_scale_indices,
                    third_codebook,
                    third_scale_values,
                    input,
                    third_scale_count,
                    third_group_size,
                    third_tensor_scale,
                    row,
                    cols,
                    lane,
                    threads_per_row);
            }
        }
    }
    first_partial[tid] = first_sum;
    second_partial[tid] = second_sum;
    third_partial[tid] = third_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            first_partial[partial_offset + lane] +=
                first_partial[partial_offset + lane + offset];
            second_partial[partial_offset + lane] +=
                second_partial[partial_offset + lane + offset];
            third_partial[partial_offset + lane] +=
                third_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        if (row < first_rows) {
            float first_value = first_partial[partial_offset];
            if (first_row_scales != nullptr && row < first_row_scale_count) {
                first_value *= first_row_scales[row];
            }
            first_output[row] = first_value;
        }
        if (row < second_rows) {
            float second_value = second_partial[partial_offset];
            if (second_row_scales != nullptr && row < second_row_scale_count) {
                second_value *= second_row_scales[row];
            }
            second_output[row] = second_value;
        }
        if (row < third_rows) {
            float third_value = third_partial[partial_offset];
            if (third_row_scales != nullptr && row < third_row_scale_count) {
                third_value *= third_row_scales[row];
            }
            third_output[row] = third_value;
        }
    }
}
)";
    }

    static const char *aq4_matvec_qkv_z_gate_beta_kernel_source() {
        return R"(
__device__ float ullm_aq4_qkv_z_gate_beta_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            if (group_size == 16ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else if (group_size == 8ull) {
#pragma unroll
                for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                    const unsigned long long col = col_start + offset;
                    const unsigned char packed = indices[(row_offset + col) >> 1];
                    raw_sum += codebook[packed & 0x0f] * input[col];
                    raw_sum += codebook[(packed >> 4) & 0x0f] * input[col + 1ull];
                }
            } else {
                for (unsigned long long offset = 0; offset < group_size; ++offset) {
                    const unsigned long long col = col_start + offset;
                    const unsigned long long element = row_offset + col;
                    const unsigned char packed = indices[element >> 1];
                    const unsigned char codebook_index =
                        (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                    raw_sum += codebook[codebook_index] * input[col];
                }
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_qkv_z_gate_beta_pair_thread_sums(
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    unsigned long long a_scale_count,
    float a_tensor_scale,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    unsigned long long b_scale_count,
    float b_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *a_sum_output,
    float *b_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float a_sum = 0.0f;
    float b_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int a_scale_index = static_cast<unsigned int>(a_scale_indices[group]);
        const unsigned int b_scale_index = static_cast<unsigned int>(b_scale_indices[group]);
        float a_raw_sum = 0.0f;
        float b_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char a_packed = a_indices[element >> 1];
                const unsigned char a_codebook_index =
                    (element & 1ull) == 0ull ? (a_packed & 0x0f) :
                                               ((a_packed >> 4) & 0x0f);
                const unsigned char b_packed = b_indices[element >> 1];
                const unsigned char b_codebook_index =
                    (element & 1ull) == 0ull ? (b_packed & 0x0f) :
                                               ((b_packed >> 4) & 0x0f);
                a_raw_sum += a_codebook[a_codebook_index] * input_value;
                b_raw_sum += b_codebook[b_codebook_index] * input_value;
            }
        }
        if (a_scale_index < a_scale_count) {
            a_sum += a_raw_sum * a_scale_values[a_scale_index] * a_tensor_scale;
        }
        if (b_scale_index < b_scale_count) {
            b_sum += b_raw_sum * b_scale_values[b_scale_index] * b_tensor_scale;
        }
    }
    *a_sum_output = a_sum;
    *b_sum_output = b_sum;
}

extern "C" __global__ void ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel(
    const unsigned char *qkv_indices,
    const unsigned char *qkv_scale_indices,
    const float *qkv_codebook,
    const float *qkv_scale_values,
    const float *qkv_row_scales,
    unsigned long long qkv_scale_count,
    unsigned long long qkv_group_size,
    float qkv_tensor_scale,
    unsigned long long qkv_row_scale_count,
    const unsigned char *z_indices,
    const unsigned char *z_scale_indices,
    const float *z_codebook,
    const float *z_scale_values,
    const float *z_row_scales,
    unsigned long long z_scale_count,
    unsigned long long z_group_size,
    float z_tensor_scale,
    unsigned long long z_row_scale_count,
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    const float *a_row_scales,
    unsigned long long a_scale_count,
    unsigned long long a_group_size,
    float a_tensor_scale,
    unsigned long long a_row_scale_count,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    const float *b_row_scales,
    unsigned long long b_scale_count,
    unsigned long long b_group_size,
    float b_tensor_scale,
    unsigned long long b_row_scale_count,
    const float *input,
    const float *a_log,
    const float *dt_bias,
    unsigned long long qkv_rows,
    unsigned long long z_rows,
    unsigned long long heads,
    unsigned long long cols,
    float *qkv_output,
    float *z_output,
    float *gate_output,
    float *beta_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long logical_row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned long long projection_rows = qkv_rows > z_rows ? qkv_rows : z_rows;
    const unsigned long long total_rows = projection_rows + heads;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float primary_partial[256];
    __shared__ float secondary_partial[256];
    float primary_sum = 0.0f;
    float secondary_sum = 0.0f;
    if (logical_row < projection_rows) {
        if (logical_row < qkv_rows && logical_row < z_rows &&
            qkv_group_size == z_group_size && (cols % qkv_group_size) == 0ull) {
            ullm_aq4_qkv_z_gate_beta_pair_thread_sums(
                qkv_indices,
                qkv_scale_indices,
                qkv_codebook,
                qkv_scale_values,
                qkv_scale_count,
                qkv_tensor_scale,
                z_indices,
                z_scale_indices,
                z_codebook,
                z_scale_values,
                z_scale_count,
                z_tensor_scale,
                input,
                qkv_group_size,
                logical_row,
                cols,
                lane,
                threads_per_row,
                &primary_sum,
                &secondary_sum);
        } else {
            if (logical_row < qkv_rows) {
                primary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                    qkv_indices,
                    qkv_scale_indices,
                    qkv_codebook,
                    qkv_scale_values,
                    input,
                    qkv_scale_count,
                    qkv_group_size,
                    qkv_tensor_scale,
                    logical_row,
                    cols,
                    lane,
                    threads_per_row);
            }
            if (logical_row < z_rows) {
                secondary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                    z_indices,
                    z_scale_indices,
                    z_codebook,
                    z_scale_values,
                    input,
                    z_scale_count,
                    z_group_size,
                    z_tensor_scale,
                    logical_row,
                    cols,
                    lane,
                    threads_per_row);
            }
        }
    } else if (logical_row < total_rows) {
        const unsigned long long head = logical_row - projection_rows;
        if (a_group_size == b_group_size && (cols % a_group_size) == 0ull) {
            ullm_aq4_qkv_z_gate_beta_pair_thread_sums(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                a_scale_count,
                a_tensor_scale,
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                b_scale_count,
                b_tensor_scale,
                input,
                a_group_size,
                head,
                cols,
                lane,
                threads_per_row,
                &primary_sum,
                &secondary_sum);
        } else {
            primary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                input,
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
            secondary_sum = ullm_aq4_qkv_z_gate_beta_thread_sum(
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                input,
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
        }
    }
    primary_partial[tid] = primary_sum;
    secondary_partial[tid] = secondary_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            primary_partial[partial_offset + lane] += primary_partial[partial_offset + lane + offset];
            secondary_partial[partial_offset + lane] +=
                secondary_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && logical_row < total_rows) {
        float value = primary_partial[partial_offset];
        if (logical_row < projection_rows) {
            if (logical_row < qkv_rows) {
                if (qkv_row_scales != nullptr && logical_row < qkv_row_scale_count) {
                    value *= qkv_row_scales[logical_row];
                }
                qkv_output[logical_row] = value;
            }
            if (logical_row < z_rows) {
                float z_value = secondary_partial[partial_offset];
                if (z_row_scales != nullptr && logical_row < z_row_scale_count) {
                    z_value *= z_row_scales[logical_row];
                }
                z_output[logical_row] = z_value;
            }
        } else {
            const unsigned long long head = logical_row - projection_rows;
            float b_value = secondary_partial[partial_offset];
            if (a_row_scales != nullptr && head < a_row_scale_count) {
                value *= a_row_scales[head];
            }
            if (b_row_scales != nullptr && head < b_row_scale_count) {
                b_value *= b_row_scales[head];
            }
            const float x = value + dt_bias[head];
            const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
            gate_output[head] = -expf(a_log[head]) * softplus;
            beta_output[head] = 1.0f / (1.0f + expf(-b_value));
        }
    }
}
)";
    }

    static const char *aq4_matvec_silu_mul_kernel_source() {
        return R"(
__device__ float ullm_aq4_matvec_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                raw_sum += codebook[codebook_index] * input[col];
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_matvec_silu_mul_thread_sums(
    const unsigned char *gate_indices,
    const unsigned char *gate_scale_indices,
    const float *gate_codebook,
    const float *gate_scale_values,
    unsigned long long gate_scale_count,
    float gate_tensor_scale,
    const unsigned char *up_indices,
    const unsigned char *up_scale_indices,
    const float *up_codebook,
    const float *up_scale_values,
    unsigned long long up_scale_count,
    float up_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *gate_sum_output,
    float *up_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int gate_scale_index =
            static_cast<unsigned int>(gate_scale_indices[group]);
        const unsigned int up_scale_index = static_cast<unsigned int>(up_scale_indices[group]);
        float gate_raw_sum = 0.0f;
        float up_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char gate_packed = gate_indices[packed_index];
                const unsigned char up_packed = up_indices[packed_index];
                gate_raw_sum += gate_codebook[gate_packed & 0x0f] * input_low;
                up_raw_sum += up_codebook[up_packed & 0x0f] * input_low;
                gate_raw_sum += gate_codebook[(gate_packed >> 4) & 0x0f] * input_high;
                up_raw_sum += up_codebook[(up_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char gate_packed = gate_indices[packed_index];
                const unsigned char up_packed = up_indices[packed_index];
                gate_raw_sum += gate_codebook[gate_packed & 0x0f] * input_low;
                up_raw_sum += up_codebook[up_packed & 0x0f] * input_low;
                gate_raw_sum += gate_codebook[(gate_packed >> 4) & 0x0f] * input_high;
                up_raw_sum += up_codebook[(up_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char gate_packed = gate_indices[packed_index];
                const unsigned char up_packed = up_indices[packed_index];
                gate_raw_sum += gate_codebook[gate_packed & 0x0f] * input_low;
                up_raw_sum += up_codebook[up_packed & 0x0f] * input_low;
                gate_raw_sum += gate_codebook[(gate_packed >> 4) & 0x0f] * input_high;
                up_raw_sum += up_codebook[(up_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char gate_packed = gate_indices[element >> 1];
                const unsigned char gate_codebook_index =
                    (element & 1ull) == 0ull ? (gate_packed & 0x0f) :
                                               ((gate_packed >> 4) & 0x0f);
                const unsigned char up_packed = up_indices[element >> 1];
                const unsigned char up_codebook_index =
                    (element & 1ull) == 0ull ? (up_packed & 0x0f) : ((up_packed >> 4) & 0x0f);
                gate_raw_sum += gate_codebook[gate_codebook_index] * input_value;
                up_raw_sum += up_codebook[up_codebook_index] * input_value;
            }
        }
        if (gate_scale_index < gate_scale_count) {
            gate_sum += gate_raw_sum * gate_scale_values[gate_scale_index] * gate_tensor_scale;
        }
        if (up_scale_index < up_scale_count) {
            up_sum += up_raw_sum * up_scale_values[up_scale_index] * up_tensor_scale;
        }
    }
    *gate_sum_output = gate_sum;
    *up_sum_output = up_sum;
}

extern "C" __global__ void ullm_aq4_matvec_silu_mul_f32_kernel(
    const unsigned char *gate_indices,
    const unsigned char *gate_scale_indices,
    const float *gate_codebook,
    const float *gate_scale_values,
    const float *gate_row_scales,
    unsigned long long gate_scale_count,
    unsigned long long gate_group_size,
    float gate_tensor_scale,
    unsigned long long gate_row_scale_count,
    const unsigned char *up_indices,
    const unsigned char *up_scale_indices,
    const float *up_codebook,
    const float *up_scale_values,
    const float *up_row_scales,
    unsigned long long up_scale_count,
    unsigned long long up_group_size,
    float up_tensor_scale,
    unsigned long long up_row_scale_count,
    const float *input,
    unsigned long long rows,
    unsigned long long cols,
    float *output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float gate_partial[256];
    __shared__ float up_partial[256];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (row < rows) {
        if (gate_group_size == up_group_size && (cols % gate_group_size) == 0ull) {
            ullm_aq4_matvec_silu_mul_thread_sums(
                gate_indices,
                gate_scale_indices,
                gate_codebook,
                gate_scale_values,
                gate_scale_count,
                gate_tensor_scale,
                up_indices,
                up_scale_indices,
                up_codebook,
                up_scale_values,
                up_scale_count,
                up_tensor_scale,
                input,
                gate_group_size,
                row,
                cols,
                lane,
                threads_per_row,
                &gate_sum,
                &up_sum);
        } else {
            gate_sum = ullm_aq4_matvec_thread_sum(
                gate_indices,
                gate_scale_indices,
                gate_codebook,
                gate_scale_values,
                input,
                gate_scale_count,
                gate_group_size,
                gate_tensor_scale,
                row,
                cols,
                lane,
                threads_per_row);
            up_sum = ullm_aq4_matvec_thread_sum(
                up_indices,
                up_scale_indices,
                up_codebook,
                up_scale_values,
                input,
                up_scale_count,
                up_group_size,
                up_tensor_scale,
                row,
                cols,
                lane,
                threads_per_row);
        }
    }
    gate_partial[tid] = gate_sum;
    up_partial[tid] = up_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            gate_partial[partial_offset + lane] += gate_partial[partial_offset + lane + offset];
            up_partial[partial_offset + lane] += up_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float gate_value = gate_partial[partial_offset];
        float up_value = up_partial[partial_offset];
        if (gate_row_scales != nullptr && row < gate_row_scale_count) {
            gate_value *= gate_row_scales[row];
        }
        if (up_row_scales != nullptr && row < up_row_scale_count) {
            up_value *= up_row_scales[row];
        }
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        output[row] = gate_value * sigmoid * up_value;
    }
}
)";
    }

    static const char *aq4_matvec_gate_beta_kernel_source() {
        return R"(
__device__ float ullm_aq4_gate_beta_thread_sum(
    const unsigned char *indices,
    const unsigned char *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *input,
    unsigned long long scale_count,
    unsigned long long group_size,
    float tensor_scale,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size) {
    const unsigned long long row_offset = row * cols;
    float sum = 0.0f;
    if ((cols % group_size) == 0ull) {
        const unsigned long long groups_per_row = cols / group_size;
        for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
             group_in_row += block_size) {
            const unsigned long long group = row * groups_per_row + group_in_row;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index >= scale_count) {
                continue;
            }
            float raw_sum = 0.0f;
            const unsigned long long col_start = group_in_row * group_size;
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const unsigned char packed = indices[element >> 1];
                const unsigned char codebook_index =
                    (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                raw_sum += codebook[codebook_index] * input[col];
            }
            sum += raw_sum * scale_values[scale_index] * tensor_scale;
        }
    } else {
        for (unsigned long long col = tid; col < cols; col += block_size) {
            const unsigned long long element = row_offset + col;
            const unsigned char packed = indices[element >> 1];
            const unsigned char codebook_index =
                (element & 1ull) == 0ull ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const unsigned long long group = element / group_size;
            const unsigned int scale_index = static_cast<unsigned int>(scale_indices[group]);
            if (scale_index < scale_count) {
                const float value =
                    codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
                sum += value * input[col];
            }
        }
    }
    return sum;
}

__device__ void ullm_aq4_gate_beta_thread_sums(
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    unsigned long long a_scale_count,
    float a_tensor_scale,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    unsigned long long b_scale_count,
    float b_tensor_scale,
    const float *input,
    unsigned long long group_size,
    unsigned long long row,
    unsigned long long cols,
    unsigned int tid,
    unsigned int block_size,
    float *a_sum_output,
    float *b_sum_output) {
    const unsigned long long row_offset = row * cols;
    const unsigned long long groups_per_row = cols / group_size;
    float a_sum = 0.0f;
    float b_sum = 0.0f;
    for (unsigned long long group_in_row = tid; group_in_row < groups_per_row;
         group_in_row += block_size) {
        const unsigned long long group = row * groups_per_row + group_in_row;
        const unsigned int a_scale_index = static_cast<unsigned int>(a_scale_indices[group]);
        const unsigned int b_scale_index = static_cast<unsigned int>(b_scale_indices[group]);
        float a_raw_sum = 0.0f;
        float b_raw_sum = 0.0f;
        const unsigned long long col_start = group_in_row * group_size;
        if (group_size == 16ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 16u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if (group_size == 8ull) {
#pragma unroll
            for (unsigned int offset = 0; offset < 8u; offset += 2u) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else if ((group_size & 1ull) == 0ull) {
            for (unsigned long long offset = 0; offset < group_size; offset += 2ull) {
                const unsigned long long col = col_start + offset;
                const unsigned long long packed_index = (row_offset + col) >> 1;
                const float input_low = input[col];
                const float input_high = input[col + 1ull];
                const unsigned char a_packed = a_indices[packed_index];
                const unsigned char b_packed = b_indices[packed_index];
                a_raw_sum += a_codebook[a_packed & 0x0f] * input_low;
                b_raw_sum += b_codebook[b_packed & 0x0f] * input_low;
                a_raw_sum += a_codebook[(a_packed >> 4) & 0x0f] * input_high;
                b_raw_sum += b_codebook[(b_packed >> 4) & 0x0f] * input_high;
            }
        } else {
            for (unsigned long long offset = 0; offset < group_size; ++offset) {
                const unsigned long long col = col_start + offset;
                const unsigned long long element = row_offset + col;
                const float input_value = input[col];
                const unsigned char a_packed = a_indices[element >> 1];
                const unsigned char a_codebook_index =
                    (element & 1ull) == 0ull ? (a_packed & 0x0f) :
                                               ((a_packed >> 4) & 0x0f);
                const unsigned char b_packed = b_indices[element >> 1];
                const unsigned char b_codebook_index =
                    (element & 1ull) == 0ull ? (b_packed & 0x0f) :
                                               ((b_packed >> 4) & 0x0f);
                a_raw_sum += a_codebook[a_codebook_index] * input_value;
                b_raw_sum += b_codebook[b_codebook_index] * input_value;
            }
        }
        if (a_scale_index < a_scale_count) {
            a_sum += a_raw_sum * a_scale_values[a_scale_index] * a_tensor_scale;
        }
        if (b_scale_index < b_scale_count) {
            b_sum += b_raw_sum * b_scale_values[b_scale_index] * b_tensor_scale;
        }
    }
    *a_sum_output = a_sum;
    *b_sum_output = b_sum;
}

extern "C" __global__ void ullm_aq4_matvec_gate_beta_f32_kernel(
    const unsigned char *a_indices,
    const unsigned char *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    const float *a_row_scales,
    unsigned long long a_scale_count,
    unsigned long long a_group_size,
    float a_tensor_scale,
    unsigned long long a_row_scale_count,
    const unsigned char *b_indices,
    const unsigned char *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    const float *b_row_scales,
    unsigned long long b_scale_count,
    unsigned long long b_group_size,
    float b_tensor_scale,
    unsigned long long b_row_scale_count,
    const float *input,
    const float *a_log,
    const float *dt_bias,
    unsigned long long heads,
    unsigned long long cols,
    float *gate_output,
    float *beta_output) {
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int rows_per_block = ULLM_AQ4_ROWS_PER_BLOCK;
    constexpr unsigned int threads_per_row = 256u / rows_per_block;
    const unsigned int row_in_block = tid / threads_per_row;
    const unsigned int lane = tid - row_in_block * threads_per_row;
    const unsigned long long head =
        static_cast<unsigned long long>(blockIdx.x) * rows_per_block + row_in_block;
    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float a_partial[256];
    __shared__ float b_partial[256];
    float a_sum = 0.0f;
    float b_sum = 0.0f;
    if (head < heads) {
        if (a_group_size == b_group_size && (cols % a_group_size) == 0ull) {
            ullm_aq4_gate_beta_thread_sums(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                a_scale_count,
                a_tensor_scale,
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                b_scale_count,
                b_tensor_scale,
                input,
                a_group_size,
                head,
                cols,
                lane,
                threads_per_row,
                &a_sum,
                &b_sum);
        } else {
            a_sum = ullm_aq4_gate_beta_thread_sum(
                a_indices,
                a_scale_indices,
                a_codebook,
                a_scale_values,
                input,
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
            b_sum = ullm_aq4_gate_beta_thread_sum(
                b_indices,
                b_scale_indices,
                b_codebook,
                b_scale_values,
                input,
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                head,
                cols,
                lane,
                threads_per_row);
        }
    }
    a_partial[tid] = a_sum;
    b_partial[tid] = b_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            a_partial[partial_offset + lane] += a_partial[partial_offset + lane + offset];
            b_partial[partial_offset + lane] += b_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && head < heads) {
        float a_value = a_partial[partial_offset];
        float b_value = b_partial[partial_offset];
        if (a_row_scales != nullptr && head < a_row_scale_count) {
            a_value *= a_row_scales[head];
        }
        if (b_row_scales != nullptr && head < b_row_scale_count) {
            b_value *= b_row_scales[head];
        }
        const float x = a_value + dt_bias[head];
        const float softplus = x <= 20.0f ? logf(1.0f + expf(x)) : x;
        gate_output[head] = -expf(a_log[head]) * softplus;
        beta_output[head] = 1.0f / (1.0f + expf(-b_value));
    }
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

    static const char *matvec_bf16_kernel_source() {
        return R"(
__device__ float ullm_bf16_to_f32(unsigned short value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ float ullm_bf16_pair_low_to_f32(unsigned int value) {
    return __uint_as_float((value & 0xffffu) << 16);
}

__device__ float ullm_bf16_pair_high_to_f32(unsigned int value) {
    return __uint_as_float((value & 0xffff0000u));
}

extern "C" __global__ void ullm_matvec_bf16_f32_kernel(
    const unsigned short *matrix,
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
        if ((cols & 1ull) == 0ull) {
            const unsigned int *matrix_pairs =
                reinterpret_cast<const unsigned int *>(matrix + row_offset);
            const unsigned long long pair_cols = cols >> 1;
            for (unsigned long long pair_col = tid; pair_col < pair_cols;
                 pair_col += blockDim.x) {
                const unsigned int packed = matrix_pairs[pair_col];
                const unsigned long long col = pair_col << 1;
                sum += ullm_bf16_pair_low_to_f32(packed) * input[col];
                sum += ullm_bf16_pair_high_to_f32(packed) * input[col + 1ull];
            }
        } else {
            for (unsigned long long col = tid; col < cols; col += blockDim.x) {
                sum += ullm_bf16_to_f32(matrix[row_offset + col]) * input[col];
            }
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

    static const char *bf16_row_kernel_source() {
        return R"(
__device__ float ullm_bf16_to_f32(unsigned short value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

extern "C" __global__ void ullm_bf16_row_f32_kernel(
    const unsigned short *matrix,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long row_index,
    float *output) {
    const unsigned int tid = threadIdx.x;
    if (row_index >= rows) {
        return;
    }
    const unsigned long long row_offset = row_index * cols;
    for (unsigned long long col = tid; col < cols; col += blockDim.x) {
        output[col] = ullm_bf16_to_f32(matrix[row_offset + col]);
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

    static const char *top1_kernel_source() {
        return R"(
extern "C" __global__ void ullm_top1_f32_kernel(
    const float *input,
    unsigned long long elements,
    float *partial_values,
    unsigned int *partial_indices) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + tid;
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    float value = -3.4028234663852886e38f;
    unsigned int token_index = 0xffffffffu;
    if (index < elements) {
        value = input[index];
        if (!(value == value)) {
            value = -3.4028234663852886e38f;
        }
        token_index = static_cast<unsigned int>(index);
    }
    values[tid] = value;
    indices[tid] = token_index;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float right_value = values[tid + offset];
            const unsigned int right_index = indices[tid + offset];
            const float left_value = values[tid];
            const unsigned int left_index = indices[tid];
            if (right_value > left_value ||
                (right_value == left_value && right_index < left_index)) {
                values[tid] = right_value;
                indices[tid] = right_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}
)";
    }

    static const char *top1_pairs_kernel_source() {
        return R"(
extern "C" __global__ void ullm_top1_pairs_f32_kernel(
    const float *input_values,
    const unsigned int *input_indices,
    unsigned long long elements,
    float *partial_values,
    unsigned int *partial_indices) {
    const unsigned int tid = threadIdx.x;
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + tid;
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    float value = -3.4028234663852886e38f;
    unsigned int token_index = 0xffffffffu;
    if (index < elements) {
        value = input_values[index];
        if (!(value == value)) {
            value = -3.4028234663852886e38f;
        }
        token_index = input_indices[index];
    }
    values[tid] = value;
    indices[tid] = token_index;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float right_value = values[tid + offset];
            const unsigned int right_index = indices[tid + offset];
            const float left_value = values[tid];
            const unsigned int left_index = indices[tid];
            if (right_value > left_value ||
                (right_value == left_value && right_index < left_index)) {
                values[tid] = right_value;
                indices[tid] = right_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}
)";
    }

    static const char *segmented_rmsnorm_kernel_source() {
        return R"(
extern "C" __global__ void ullm_segmented_rmsnorm_f32_kernel(
    const float *input,
    const float *weight,
    unsigned long long segments,
    unsigned long long segment_size,
    float epsilon,
    float *output) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < segments) {
        const unsigned long long base = segment * segment_size;
        for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
            const float value = input[base + dim];
            sum_squares += value * value;
        }
    }
    partial[tid] = sum_squares;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (segment >= segments) {
        return;
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(segment_size) + epsilon);
    const unsigned long long base = segment * segment_size;
    for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
        output[base + dim] = input[base + dim] * inv_rms * weight[dim];
    }
}
)";
    }

    static const char *segmented_rmsnorm_silu_mul_kernel_source() {
        return R"(
extern "C" __global__ void ullm_segmented_rmsnorm_silu_mul_f32_kernel(
    const float *input,
    const float *weight,
    const float *gate,
    unsigned long long segments,
    unsigned long long segment_size,
    float epsilon,
    float *output) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < segments) {
        const unsigned long long base = segment * segment_size;
        for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
            const float value = input[base + dim];
            sum_squares += value * value;
        }
    }
    partial[tid] = sum_squares;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (segment >= segments) {
        return;
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(segment_size) + epsilon);
    const unsigned long long base = segment * segment_size;
    for (unsigned long long dim = tid; dim < segment_size; dim += blockDim.x) {
        const unsigned long long index = base + dim;
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float normed = input[index] * inv_rms * weight[dim];
        output[index] = gate_value * sigmoid * normed;
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

    static const char *qwen35_split_q_gate_kernel_source() {
        return R"(
extern "C" __global__ void ullm_qwen35_split_q_gate_f32_kernel(
    const float *projected,
    unsigned long long q_heads,
    unsigned long long head_dim,
    float *query_output,
    float *gate_output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = q_heads * head_dim;
    if (index >= elements) {
        return;
    }
    const unsigned long long dim = index % head_dim;
    const unsigned long long head = index / head_dim;
    const unsigned long long source_base = head * 2ULL * head_dim;
    query_output[index] = projected[source_base + dim];
    gate_output[index] = projected[source_base + head_dim + dim];
}
)";
    }

    static const char *qwen35_qk_norm_rope_kernel_source() {
        return R"(
extern "C" __global__ void ullm_qwen35_qk_norm_rope_f32_kernel(
    const float *q_projected,
    const float *k_projected,
    const float *q_weight,
    const float *k_weight,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float epsilon,
    float *q_gate_output,
    float *q_rope_output,
    float *k_rope_output) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long total_segments = q_heads + kv_heads;
    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = q_projected[source_base + dim];
            q_gate_output[segment * head_dim + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
    } else if (segment < total_segments) {
        const unsigned long long k_segment = segment - q_heads;
        const unsigned long long source_base = k_segment * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = k_projected[source_base + dim];
            sum_squares += value * value;
        }
    }
    partial[tid] = sum_squares;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    if (segment >= total_segments) {
        return;
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
    const unsigned long long half = rotary_dim >> 1;
    const float position = static_cast<float>(position_offset);
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        const unsigned long long output_base = segment * head_dim;
        for (unsigned long long pair_dim = tid; pair_dim < half; pair_dim += blockDim.x) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float c = cosf(theta);
            const float s = sinf(theta);
            const float first = q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const unsigned long long second_dim = half + pair_dim;
            const float second =
                q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * c - second * s;
            q_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            q_rope_output[output_base + dim] =
                q_projected[source_base + dim] * inv_rms * q_weight[dim];
        }
    } else {
        const unsigned long long k_segment = segment - q_heads;
        const unsigned long long source_base = k_segment * head_dim;
        const unsigned long long output_base = k_segment * head_dim;
        for (unsigned long long pair_dim = tid; pair_dim < half; pair_dim += blockDim.x) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float c = cosf(theta);
            const float s = sinf(theta);
            const float first = k_projected[source_base + pair_dim] * inv_rms * k_weight[pair_dim];
            const unsigned long long second_dim = half + pair_dim;
            const float second =
                k_projected[source_base + second_dim] * inv_rms * k_weight[second_dim];
            k_rope_output[output_base + pair_dim] = first * c - second * s;
            k_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            k_rope_output[output_base + dim] =
                k_projected[source_base + dim] * inv_rms * k_weight[dim];
        }
    }
}
)";
    }

    static const char *qwen35_qk_norm_rope_paged_kv_write_kernel_source() {
        return R"(
extern "C" __global__ void ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel(
    const float *q_projected,
    const float *k_projected,
    const float *v_projected,
    const float *q_weight,
    const float *k_weight,
    const unsigned int *block_table,
    unsigned long long q_heads,
    unsigned long long kv_heads,
    unsigned long long head_dim,
    unsigned long long value_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float epsilon,
    unsigned long long cache_position,
    unsigned long long block_size,
    unsigned long long cache_blocks,
    float *q_gate_output,
    float *q_rope_output,
    float *k_cache,
    float *v_cache) {
    const unsigned long long segment = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned long long k_segment_start = q_heads;
    const unsigned long long v_segment_start = q_heads + kv_heads;
    const unsigned long long total_segments = q_heads + kv_heads + kv_heads;
    if (segment >= total_segments) {
        return;
    }
    const unsigned long long block_index = cache_position / block_size;
    const unsigned long long block_offset = cache_position - block_index * block_size;
    const unsigned long long block_id = static_cast<unsigned long long>(block_table[block_index]);
    if (block_id >= cache_blocks) {
        return;
    }
    const unsigned long long physical_timestep = block_id * block_size + block_offset;
    if (segment >= v_segment_start) {
        const unsigned long long kv_head = segment - v_segment_start;
        const unsigned long long source_base = kv_head * value_dim;
        const unsigned long long cache_base =
            (physical_timestep * kv_heads + kv_head) * value_dim;
        for (unsigned long long dim = tid; dim < value_dim; dim += blockDim.x) {
            v_cache[cache_base + dim] = v_projected[source_base + dim];
        }
        return;
    }

    __shared__ float partial[256];
    float sum_squares = 0.0f;
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = q_projected[source_base + dim];
            q_gate_output[segment * head_dim + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
    } else {
        const unsigned long long kv_head = segment - k_segment_start;
        const unsigned long long source_base = kv_head * head_dim;
        for (unsigned long long dim = tid; dim < head_dim; dim += blockDim.x) {
            const float value = k_projected[source_base + dim];
            sum_squares += value * value;
        }
    }
    partial[tid] = sum_squares;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
    const unsigned long long half = rotary_dim >> 1;
    const float position = static_cast<float>(position_offset);
    if (segment < q_heads) {
        const unsigned long long source_base = segment * 2ull * head_dim;
        const unsigned long long output_base = segment * head_dim;
        for (unsigned long long pair_dim = tid; pair_dim < half; pair_dim += blockDim.x) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float c = cosf(theta);
            const float s = sinf(theta);
            const float first = q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const unsigned long long second_dim = half + pair_dim;
            const float second =
                q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * c - second * s;
            q_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            q_rope_output[output_base + dim] =
                q_projected[source_base + dim] * inv_rms * q_weight[dim];
        }
    } else {
        const unsigned long long kv_head = segment - k_segment_start;
        const unsigned long long source_base = kv_head * head_dim;
        const unsigned long long cache_base =
            (physical_timestep * kv_heads + kv_head) * head_dim;
        for (unsigned long long pair_dim = tid; pair_dim < half; pair_dim += blockDim.x) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / powf(rope_base, exponent);
            const float c = cosf(theta);
            const float s = sinf(theta);
            const float first = k_projected[source_base + pair_dim] * inv_rms * k_weight[pair_dim];
            const unsigned long long second_dim = half + pair_dim;
            const float second =
                k_projected[source_base + second_dim] * inv_rms * k_weight[second_dim];
            k_cache[cache_base + pair_dim] = first * c - second * s;
            k_cache[cache_base + second_dim] = second * c + first * s;
        }
        for (unsigned long long dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            k_cache[cache_base + dim] =
                k_projected[source_base + dim] * inv_rms * k_weight[dim];
        }
    }
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

    static std::string paged_decode_attn_kernel_source_for_arch(const std::string &arch) {
        (void)arch;
        std::string preamble;
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_SHARED_REDUCE 1\n";
        }
        if (std::getenv("ULLM_DISABLE_PAGED_DECODE_ONLINE_SOFTMAX") != nullptr) {
            preamble += "#define ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX 1\n";
        }
        return preamble + paged_decode_attn_kernel_source();
    }

    static const char *paged_decode_attn_kernel_source() {
        return R"(
__device__ float ullm_paged_decode_reduce_sum_256(float value, float *partial) {
#if defined(ULLM_PAGED_DECODE_USE_SHARED_REDUCE)
    const unsigned int tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        __syncthreads();
    }
    return partial[0];
#else
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % warpSize;
    const unsigned int wave = tid / warpSize;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, warpSize);
    }
    if (lane == 0) {
        partial[wave] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    const unsigned int wave_count = (blockDim.x + warpSize - 1u) / warpSize;
    if (wave == 0) {
        block_sum = lane < wave_count ? partial[lane] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_sum += __shfl_down(block_sum, offset, warpSize);
        }
        if (lane == 0) {
            partial[0] = block_sum;
        }
    }
    __syncthreads();
    return partial[0];
#endif
}

extern "C" __global__ void ullm_paged_decode_attn_f32_kernel(
    const float *q,
    const float *gate,
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
    __shared__ float partial[256];
    if (gridDim.x == q_heads && blockDim.x == 256u && head_dim <= 256ull &&
        value_dim <= 256ull) {
        const unsigned long long q_head = blockIdx.x;
        const unsigned int tid = threadIdx.x;
        const unsigned long long q_per_kv = q_heads / kv_heads;
        const unsigned long long kv_head = q_head / q_per_kv;
        const unsigned long long q_base = q_head * head_dim;

#if defined(ULLM_PAGED_DECODE_USE_TWO_PASS_SOFTMAX)
        float max_score = -3.4028234663852886e38f;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const unsigned long long block_index = source_timestep / block_size;
            const unsigned long long block_offset = source_timestep - block_index * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[block_index]);
            if (block_id >= cache_blocks) {
                if (tid < value_dim) {
                    output[q_head * value_dim + tid] = 0.0f;
                }
                return;
            }
            const unsigned long long physical_timestep = block_id * block_size + block_offset;
            const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            const float local =
                tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
            const float score =
                ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            max_score = score > max_score ? score : max_score;
        }

        float denominator = 0.0f;
        float weighted = 0.0f;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            const unsigned long long block_index = source_timestep / block_size;
            const unsigned long long block_offset = source_timestep - block_index * block_size;
            const unsigned long long block_id =
                static_cast<unsigned long long>(block_table[block_index]);
            if (block_id >= cache_blocks) {
                if (tid < value_dim) {
                    output[q_head * value_dim + tid] = 0.0f;
                }
                return;
            }
            const unsigned long long physical_timestep = block_id * block_size + block_offset;
            const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            const float local =
                tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
            const float score =
                ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            const float weight = expf(score - max_score);
            denominator += weight;
            if (tid < value_dim) {
                const unsigned long long v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + tid;
                weighted += weight * v_cache[v_index];
            }
        }
#else
        float max_score = -3.4028234663852886e38f;
        float denominator = 0.0f;
        float weighted = 0.0f;
        unsigned long long block_index = 0ull;
        unsigned long long block_end = block_size;
        unsigned long long block_id = static_cast<unsigned long long>(block_table[0]);
        if (block_id >= cache_blocks) {
            if (tid < value_dim) {
                output[q_head * value_dim + tid] = 0.0f;
            }
            return;
        }
        unsigned long long physical_timestep = block_id * block_size;
        for (unsigned long long source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
            if (source_timestep == block_end) {
                ++block_index;
                block_end += block_size;
                block_id = static_cast<unsigned long long>(block_table[block_index]);
                if (block_id >= cache_blocks) {
                    if (tid < value_dim) {
                        output[q_head * value_dim + tid] = 0.0f;
                    }
                    return;
                }
                physical_timestep = block_id * block_size;
            }
            const unsigned long long k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            const float local =
                tid < head_dim ? q[q_base + tid] * k_cache[k_base + tid] : 0.0f;
            const float score =
                ullm_paged_decode_reduce_sum_256(local, partial) * softmax_scale;
            if (tid < value_dim) {
                const unsigned long long v_index =
                    (physical_timestep * kv_heads + kv_head) * value_dim + tid;
                const float v = v_cache[v_index];
                if (score > max_score) {
                    const float scale = expf(max_score - score);
                    weighted = weighted * scale + v;
                    denominator = denominator * scale + 1.0f;
                    max_score = score;
                } else {
                    const float weight = expf(score - max_score);
                    weighted += weight * v;
                    denominator += weight;
                }
            }
            ++physical_timestep;
        }
#endif
        if (tid < value_dim) {
            const unsigned long long output_index = q_head * value_dim + tid;
            const float decoded = weighted / denominator;
            if (gate != nullptr) {
                const float gate_value = gate[output_index];
                const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
                output[output_index] = sigmoid * decoded;
            } else {
                output[output_index] = decoded;
            }
        }
        return;
    }

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
    const float decoded = weighted / denominator;
    if (gate != nullptr) {
        const float gate_value = gate[index];
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        output[index] = sigmoid * decoded;
    } else {
        output[index] = decoded;
    }
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

    static const char *linear_attn_qkv_prepare_kernel_source() {
        return R"(
__device__ float ullm_linear_attn_qkv_conv_step_value(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long channels,
    unsigned long long kernel_size,
    unsigned long long channel) {
    for (unsigned long long kernel = 0; kernel + 1 < kernel_size; ++kernel) {
        conv_history[kernel * channels + channel] =
            conv_history[(kernel + 1ull) * channels + channel];
    }
    conv_history[(kernel_size - 1ull) * channels + channel] = qkv[channel];

    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        sum += conv_history[kernel * channels + channel] *
               conv_weight[channel * kernel_size + kernel];
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sum));
    return sum * sigmoid;
}

extern "C" __global__ void ullm_linear_attn_qkv_prepare_f32_kernel(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long key_dim,
    unsigned long long value_dim,
    unsigned long long kernel_size,
    float q_scale,
    int qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const unsigned long long q_elements = key_heads * key_dim;
    const unsigned long long k_base = q_elements;
    const unsigned long long v_base = q_elements * 2ull;
    const unsigned long long v_elements = value_heads * value_dim;
    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;

    if (block < key_heads) {
        const unsigned long long head = block;
        const unsigned long long source_base = head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_step_value(
                qkv,
                conv_weight,
                conv_history,
                channels,
                kernel_size,
                channel);
            conv_output[channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[channel];
            q_output[head * key_dim + dim] =
                qk_l2_norm != 0 ? (value / norm) * q_scale : value * q_scale;
        }
        return;
    }

    if (block < key_heads * 2ull) {
        const unsigned long long head = block - key_heads;
        const unsigned long long source_base = k_base + head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_step_value(
                qkv,
                conv_weight,
                conv_history,
                channels,
                kernel_size,
                channel);
            conv_output[channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[channel];
            k_output[head * key_dim + dim] = qk_l2_norm != 0 ? value / norm : value;
        }
        return;
    }

    const unsigned long long v_block = block - key_heads * 2ull;
    const unsigned long long v_index = v_block * blockDim.x + tid;
    if (v_index < v_elements) {
        const unsigned long long channel = v_base + v_index;
        const float value = ullm_linear_attn_qkv_conv_step_value(
            qkv,
            conv_weight,
            conv_history,
            channels,
            kernel_size,
            channel);
        conv_output[channel] = value;
        v_output[v_index] = value;
    }
}

extern "C" __global__ void ullm_linear_attn_qkv_conv_step_silu_f32_kernel(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long channels,
    unsigned long long kernel_size,
    float *conv_output) {
    const unsigned long long channel =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    for (unsigned long long kernel = 0; kernel + 1 < kernel_size; ++kernel) {
        conv_history[kernel * channels + channel] =
            conv_history[(kernel + 1ull) * channels + channel];
    }
    conv_history[(kernel_size - 1ull) * channels + channel] = qkv[channel];

    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        sum += conv_history[kernel * channels + channel] *
               conv_weight[channel * kernel_size + kernel];
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sum));
    conv_output[channel] = sum * sigmoid;
}

extern "C" __global__ void ullm_linear_attn_qkv_split_l2norm_f32_kernel(
    const float *conv_output,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long key_dim,
    unsigned long long value_dim,
    float q_scale,
    int qk_l2_norm,
    float *q_output,
    float *k_output,
    float *v_output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const unsigned long long q_elements = key_heads * key_dim;
    const unsigned long long k_base = q_elements;
    const unsigned long long v_base = q_elements * 2ull;
    const unsigned long long v_elements = value_heads * value_dim;
    const unsigned long long block = blockIdx.x;

    if (block < key_heads) {
        const unsigned long long head = block;
        const unsigned long long source_base = head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            q_output[head * key_dim + dim] =
                qk_l2_norm != 0 ? (value / norm) * q_scale : value * q_scale;
        }
        return;
    }

    if (block < key_heads * 2ull) {
        const unsigned long long head = block - key_heads;
        const unsigned long long source_base = k_base + head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const float value = conv_output[source_base + dim];
            k_output[head * key_dim + dim] = qk_l2_norm != 0 ? value / norm : value;
        }
        return;
    }

    const unsigned long long v_block = block - key_heads * 2ull;
    const unsigned long long v_index = v_block * blockDim.x + tid;
    if (v_index < v_elements) {
        v_output[v_index] = conv_output[v_base + v_index];
    }
}
)";
    }

    static const char *linear_attn_qkv_prepare_batch_kernel_source() {
        return R"(
__device__ float ullm_linear_attn_qkv_conv_batch_value(
    const float *qkv,
    const float *conv_weight,
    const float *conv_history,
    unsigned long long sequence_len,
    unsigned long long channels,
    unsigned long long kernel_size,
    unsigned long long token,
    unsigned long long channel) {
    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
        const long long source_token = static_cast<long long>(token) -
            static_cast<long long>(kernel_size - 1ull - kernel);
        float source = 0.0f;
        if (source_token >= 0) {
            source = qkv[static_cast<unsigned long long>(source_token) * channels + channel];
        } else {
            const unsigned long long history_index = kernel + token + 1ull;
            source = history_index < kernel_size ? conv_history[history_index * channels + channel] : 0.0f;
        }
        sum += source * conv_weight[channel * kernel_size + kernel];
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sum));
    return sum * sigmoid;
}

__device__ void ullm_linear_attn_qkv_conv_batch_update_history(
    const float *qkv,
    float *conv_history,
    unsigned long long sequence_len,
    unsigned long long channels,
    unsigned long long kernel_size,
    unsigned long long channel) {
    if (sequence_len >= kernel_size) {
        const unsigned long long first_token = sequence_len - kernel_size;
        for (unsigned long long kernel = 0; kernel < kernel_size; ++kernel) {
            conv_history[kernel * channels + channel] =
                qkv[(first_token + kernel) * channels + channel];
        }
        return;
    }
    const unsigned long long preserved = kernel_size - sequence_len;
    for (unsigned long long kernel = 0; kernel < preserved; ++kernel) {
        conv_history[kernel * channels + channel] =
            conv_history[(kernel + sequence_len) * channels + channel];
    }
    for (unsigned long long token = 0; token < sequence_len; ++token) {
        conv_history[(preserved + token) * channels + channel] =
            qkv[token * channels + channel];
    }
}

extern "C" __global__ void ullm_linear_attn_qkv_prepare_batch_update_history_f32_kernel(
    const float *qkv,
    float *conv_history,
    unsigned long long sequence_len,
    unsigned long long channels,
    unsigned long long kernel_size) {
    const unsigned long long channel =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    ullm_linear_attn_qkv_conv_batch_update_history(
        qkv,
        conv_history,
        sequence_len,
        channels,
        kernel_size,
        channel);
}

extern "C" __global__ void ullm_linear_attn_qkv_prepare_batch_f32_kernel(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    unsigned long long key_heads,
    unsigned long long value_heads,
    unsigned long long key_dim,
    unsigned long long value_dim,
    unsigned long long kernel_size,
    unsigned long long sequence_len,
    float q_scale,
    int qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const unsigned int tid = threadIdx.x;
    __shared__ float partial[256];
    const unsigned long long q_elements = key_heads * key_dim;
    const unsigned long long k_base = q_elements;
    const unsigned long long v_base = q_elements * 2ull;
    const unsigned long long v_elements = value_heads * value_dim;
    const unsigned long long channels = q_elements + q_elements + v_elements;
    const unsigned long long block = blockIdx.x;
    const unsigned long long token = blockIdx.y;
    if (token >= sequence_len) {
        return;
    }

    if (block < key_heads) {
        const unsigned long long head = block;
        const unsigned long long source_base = head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_batch_value(
                qkv,
                conv_weight,
                conv_history,
                sequence_len,
                channels,
                kernel_size,
                token,
                channel);
            conv_output[token * channels + channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[token * channels + channel];
            q_output[token * q_elements + head * key_dim + dim] =
                qk_l2_norm != 0 ? (value / norm) * q_scale : value * q_scale;
        }
        return;
    }

    if (block < key_heads * 2ull) {
        const unsigned long long head = block - key_heads;
        const unsigned long long source_base = k_base + head * key_dim;
        float sum_squares = 0.0f;
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = ullm_linear_attn_qkv_conv_batch_value(
                qkv,
                conv_weight,
                conv_history,
                sequence_len,
                channels,
                kernel_size,
                token,
                channel);
            conv_output[token * channels + channel] = value;
            sum_squares += value * value;
        }
        partial[tid] = sum_squares;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        const float norm = sqrtf(partial[0] + 1.0e-6f);
        for (unsigned long long dim = tid; dim < key_dim; dim += blockDim.x) {
            const unsigned long long channel = source_base + dim;
            const float value = conv_output[token * channels + channel];
            k_output[token * q_elements + head * key_dim + dim] =
                qk_l2_norm != 0 ? value / norm : value;
        }
        return;
    }

    const unsigned long long v_block = block - key_heads * 2ull;
    const unsigned long long v_index = v_block * blockDim.x + tid;
    if (v_index < v_elements) {
        const unsigned long long channel = v_base + v_index;
        const float value = ullm_linear_attn_qkv_conv_batch_value(
            qkv,
            conv_weight,
            conv_history,
            sequence_len,
            channels,
            kernel_size,
            token,
            channel);
        conv_output[token * channels + channel] = value;
        v_output[token * v_elements + v_index] = value;
    }
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
    if (sequence_len == 1ull && blockDim.x > 1u) {
        const unsigned long long block = blockIdx.x;
        const unsigned long long value_head = block / value_dim;
        const unsigned long long value = block - value_head * value_dim;
        const unsigned int tid = threadIdx.x;
        __shared__ float partial[256];
        __shared__ float v_prime_shared;
        if (value_head >= value_heads || value >= value_dim) {
            return;
        }
        const unsigned long long key_head_group = value_heads / key_heads;
        const unsigned long long key_head = value_head / key_head_group;
        const unsigned long long qk_base = key_head * key_dim;
        const unsigned long long v_base = value_head * value_dim;
        const unsigned long long state_head_offset = value_head * key_dim * value_dim;
        const float decay = expf(gate[value_head]);
        const float beta_value = beta[value_head];

        if (key_dim <= static_cast<unsigned long long>(blockDim.x)) {
            float current = 0.0f;
            float decayed = 0.0f;
            float key_value = 0.0f;
            float query_value = 0.0f;
            unsigned long long state_index = 0ull;
            const bool active = static_cast<unsigned long long>(tid) < key_dim;
            if (active) {
                const unsigned long long key = tid;
                state_index = state_head_offset + key * value_dim + value;
                key_value = k[qk_base + key];
                query_value = q[qk_base + key];
                decayed = state[state_index] * decay;
                current = decayed * key_value;
            }
            partial[tid] = current;
            __syncthreads();
            for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    partial[tid] += partial[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                v_prime_shared = (v[v_base + value] - partial[0]) * beta_value;
            }
            __syncthreads();

            float sum = 0.0f;
            if (active) {
                const float updated = decayed + key_value * v_prime_shared;
                state[state_index] = updated;
                sum = updated * query_value;
            }
            partial[tid] = sum;
            __syncthreads();
            for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    partial[tid] += partial[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                output[v_base + value] = partial[0];
            }
            return;
        }

        float current = 0.0f;
        for (unsigned long long key = tid; key < key_dim; key += blockDim.x) {
            const unsigned long long state_index = state_head_offset + key * value_dim + value;
            const float decayed = state[state_index] * decay;
            state[state_index] = decayed;
            current += decayed * k[qk_base + key];
        }
        partial[tid] = current;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            v_prime_shared = (v[v_base + value] - partial[0]) * beta_value;
        }
        __syncthreads();

        float sum = 0.0f;
        for (unsigned long long key = tid; key < key_dim; key += blockDim.x) {
            const unsigned long long state_index = state_head_offset + key * value_dim + value;
            const float updated = state[state_index] + k[qk_base + key] * v_prime_shared;
            state[state_index] = updated;
            sum += updated * q[qk_base + key];
        }
        partial[tid] = sum;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[v_base + value] = partial[0];
        }
        return;
    }

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

bool aq4_row_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    const float *row_scales,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t row_index,
    float *output) {
    if (row_index >= rows) {
        set_error("AQ4 row index is out of range");
        return false;
    }
    const size_t row_offset = row_index * cols;
    for (size_t col = 0; col < cols; ++col) {
        const size_t element = row_offset + col;
        const std::uint8_t packed = indices[element / 2];
        const std::uint8_t codebook_index =
            (element % 2 == 0) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
        const size_t group = element / group_size;
        const size_t scale_index = static_cast<size_t>(scale_indices[group]);
        if (scale_index >= scale_count) {
            set_error("AQ4 row scale index is out of range");
            return false;
        }
        float value = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
        if (row_scales != nullptr && row_index < row_scale_count) {
            value *= row_scales[row_index];
        }
        output[col] = value;
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

struct Aq4MatvecLaunchConfig {
    unsigned int block_size = 256;
    unsigned int rows_per_block = 1;
};

Aq4MatvecLaunchConfig aq4_matvec_launch_config_for_device(int device_id) {
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    const unsigned int fallback = major >= 12 ? 32u : 1u;
    return Aq4MatvecLaunchConfig{
        256u,
        aq4_rows_per_block_from_env("ULLM_AQ4_MATVEC_RPB", nullptr, fallback)};
}

Aq4MatvecLaunchConfig aq4_matvec_launch_config_for_fused_kernel(
    int device_id,
    unsigned int rdna4_rows_per_block,
    unsigned int rdna2_rows_per_block = 2u,
    const char *env_name = nullptr) {
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const unsigned int fallback = major >= 12 ? rdna4_rows_per_block : rdna2_rows_per_block;
    launch_config.rows_per_block =
        aq4_rows_per_block_from_env(env_name, "ULLM_AQ4_FUSED_RPB", fallback);
    return launch_config;
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

class HipAq4RowKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_row_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_row_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build AQ4 row HIP kernel" : compile_errors);
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

HipAq4RowKernelCache &hip_aq4_row_kernel_cache() {
    static HipAq4RowKernelCache cache;
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

HipAq4LaunchResult aq4_row_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_row_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return HipAq4LaunchResult::RuntimeError;
    }

    void *device_error = hip_runtime().malloc_device(sizeof(std::uint32_t), device_id);
    if (device_error == nullptr) {
        if (error != nullptr) {
            *error = "failed to allocate AQ4 row HIP kernel status buffer";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const std::uint32_t zero = 0;
    if (!hip_runtime().copy_async(
            device_error,
            &zero,
            sizeof(zero),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to upload AQ4 row HIP kernel status buffer";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_index = static_cast<unsigned long long>(row_index);
    const unsigned int grid_x =
        static_cast<unsigned int>((cols + block_size - 1) / block_size);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *row_scales_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &row_scales_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_row_index,
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
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 row";
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
        hip_runtime().free_device(device_error, device_id);
        if (error != nullptr) {
            *error = "failed to read AQ4 row HIP kernel status";
        }
        return HipAq4LaunchResult::RuntimeError;
    }

    hip_runtime().free_device(device_error, device_id);
    if (host_error != 0) {
        if (error != nullptr) {
            *error = "AQ4 row scale index is out of range";
        }
        return HipAq4LaunchResult::InvalidArgument;
    }
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

bool aq4_matvec_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    float *output) {
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const size_t row_offset = row * cols;
        for (size_t col = 0; col < cols; ++col) {
            const size_t element = row_offset + col;
            const std::uint8_t packed = indices[element / 2];
            const std::uint8_t codebook_index =
                (element % 2 == 0) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const size_t group = element / group_size;
            const size_t scale_index = static_cast<size_t>(scale_indices[group]);
            if (scale_index >= scale_count) {
                set_error("AQ4 matvec scale index is out of range");
                return false;
            }
            const float value = codebook[codebook_index] * scale_values[scale_index] * tensor_scale;
            sum += value * input[col];
        }
        output[row] = row_scales != nullptr && row < row_scale_count ? sum * row_scales[row] : sum;
    }
    return true;
}

bool aq4_matvec_batch_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    float *output) {
    for (size_t batch = 0; batch < batch_count; ++batch) {
        if (!aq4_matvec_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input + batch * cols,
                row_scales,
                row_scale_count,
                rows,
                cols,
                output + batch * rows)) {
            return false;
        }
    }
    return true;
}

size_t aq4_matvec_top1_partial_count(size_t rows) {
    const unsigned int rows_per_block = aq4_matvec_top1_rows_per_block_from_env();
    return (rows + rows_per_block - 1) / rows_per_block;
}

bool aq4_matvec_top1_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t partial_count,
    float *partial_values,
    uint32_t *partial_indices) {
    std::vector<float> output(rows);
    if (!aq4_matvec_f32_host(
            indices,
            scale_indices,
            codebook,
            scale_values,
            scale_count,
            group_size,
            tensor_scale,
            input,
            row_scales,
            row_scale_count,
            rows,
            cols,
            output.data())) {
        return false;
    }
    for (size_t partial = 0; partial < partial_count; ++partial) {
        const size_t start = partial * kAq4MatvecTop1RowsPerBlock;
        const size_t end = std::min(start + kAq4MatvecTop1RowsPerBlock, rows);
        float best_value = -std::numeric_limits<float>::max();
        uint32_t best_index = std::numeric_limits<uint32_t>::max();
        for (size_t row = start; row < end; ++row) {
            float value = output[row];
            if (value != value) {
                value = -std::numeric_limits<float>::max();
            }
            const auto token_index = static_cast<uint32_t>(row);
            if (value > best_value || (value == best_value && token_index < best_index)) {
                best_value = value;
                best_index = token_index;
            }
        }
        partial_values[partial] = best_value;
        partial_indices[partial] = best_index;
    }
    return true;
}

bool aq4_matvec_add_f32_host(
    const std::uint8_t *indices,
    const std::uint8_t *scale_indices,
    const float *codebook,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    const float *input,
    const float *residual,
    const float *row_scales,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    float *output) {
    std::vector<float> matvec_output(rows);
    if (!aq4_matvec_f32_host(
            indices,
            scale_indices,
            codebook,
            scale_values,
            scale_count,
            group_size,
            tensor_scale,
            input,
            row_scales,
            row_scale_count,
            rows,
            cols,
            matvec_output.data())) {
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        output[row] = residual[row] + matvec_output[row];
    }
    return true;
}

bool aq4_matvec_silu_mul_f32_host(
    const std::uint8_t *gate_indices,
    const std::uint8_t *gate_scale_indices,
    const float *gate_codebook,
    const float *gate_scale_values,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    const float *gate_row_scales,
    size_t gate_row_scale_count,
    const std::uint8_t *up_indices,
    const std::uint8_t *up_scale_indices,
    const float *up_codebook,
    const float *up_scale_values,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    const float *up_row_scales,
    size_t up_row_scale_count,
    const float *input,
    size_t rows,
    size_t cols,
    float *output) {
    std::vector<float> gate(rows);
    std::vector<float> up(rows);
    if (!aq4_matvec_f32_host(
            gate_indices,
            gate_scale_indices,
            gate_codebook,
            gate_scale_values,
            gate_scale_count,
            gate_group_size,
            gate_tensor_scale,
            input,
            gate_row_scales,
            gate_row_scale_count,
            rows,
            cols,
            gate.data())) {
        return false;
    }
    if (!aq4_matvec_f32_host(
            up_indices,
            up_scale_indices,
            up_codebook,
            up_scale_values,
            up_scale_count,
            up_group_size,
            up_tensor_scale,
            input,
            up_row_scales,
            up_row_scale_count,
            rows,
            cols,
            up.data())) {
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        const float gate_value = gate[row];
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
        output[row] = gate_value * sigmoid * up[row];
    }
    return true;
}

bool aq4_matvec_gate_beta_f32_host(
    const std::uint8_t *a_indices,
    const std::uint8_t *a_scale_indices,
    const float *a_codebook,
    const float *a_scale_values,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    const float *a_row_scales,
    size_t a_row_scale_count,
    const std::uint8_t *b_indices,
    const std::uint8_t *b_scale_indices,
    const float *b_codebook,
    const float *b_scale_values,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    const float *b_row_scales,
    size_t b_row_scale_count,
    const float *input,
    const float *a_log,
    const float *dt_bias,
    size_t heads,
    size_t cols,
    float *gate_output,
    float *beta_output) {
    std::vector<float> a(heads);
    std::vector<float> b(heads);
    if (!aq4_matvec_f32_host(
            a_indices,
            a_scale_indices,
            a_codebook,
            a_scale_values,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            input,
            a_row_scales,
            a_row_scale_count,
            heads,
            cols,
            a.data())) {
        return false;
    }
    if (!aq4_matvec_f32_host(
            b_indices,
            b_scale_indices,
            b_codebook,
            b_scale_values,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            input,
            b_row_scales,
            b_row_scale_count,
            heads,
            cols,
            b.data())) {
        return false;
    }
    for (size_t head = 0; head < heads; ++head) {
        const float x = a[head] + dt_bias[head];
        const float softplus = x <= 20.0f ? std::log1p(std::exp(x)) : x;
        gate_output[head] = -std::exp(a_log[head]) * softplus;
        beta_output[head] = 1.0f / (1.0f + std::exp(-b[head]));
    }
    return true;
}

class HipAq4MatvecKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build AQ4 matvec HIP kernel" : compile_errors);
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

HipAq4MatvecKernelCache &hip_aq4_matvec_kernel_cache() {
    static HipAq4MatvecKernelCache cache;
    return cache;
}

class HipAq4MatvecBatchKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_batch_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_aq4_matvec_batch_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec batch HIP kernel" :
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

HipAq4MatvecBatchKernelCache &hip_aq4_matvec_batch_kernel_cache() {
    static HipAq4MatvecBatchKernelCache cache;
    return cache;
}

class HipAq4MatvecTop1KernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_top1_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_top1_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec top1 HIP kernel"
                                   : compile_errors);
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

HipAq4MatvecTop1KernelCache &hip_aq4_matvec_top1_kernel_cache() {
    static HipAq4MatvecTop1KernelCache cache;
    return cache;
}

class HipAq4MatvecAddKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_add_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_add_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec add HIP kernel" :
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

HipAq4MatvecAddKernelCache &hip_aq4_matvec_add_kernel_cache() {
    static HipAq4MatvecAddKernelCache cache;
    return cache;
}

class HipAq4MatvecPairKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_pair_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_pair_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec pair HIP kernel" :
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

HipAq4MatvecPairKernelCache &hip_aq4_matvec_pair_kernel_cache() {
    static HipAq4MatvecPairKernelCache cache;
    return cache;
}

class HipAq4MatvecTripleKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_triple_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_aq4_matvec_triple_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec triple HIP kernel" :
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

HipAq4MatvecTripleKernelCache &hip_aq4_matvec_triple_kernel_cache() {
    static HipAq4MatvecTripleKernelCache cache;
    return cache;
}

class HipAq4MatvecQkvZGateBetaKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_qkv_z_gate_beta_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 qkv/z gate/beta HIP kernel" :
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

HipAq4MatvecQkvZGateBetaKernelCache &hip_aq4_matvec_qkv_z_gate_beta_kernel_cache() {
    static HipAq4MatvecQkvZGateBetaKernelCache cache;
    return cache;
}

bool aq4_matvec_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_batch_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_batch_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const size_t grid_x =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        batch_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec batch dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &kernel_batch_count,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            function,
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(batch_count),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec batch";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_top1_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_top1_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const unsigned int rows_per_block = aq4_matvec_top1_rows_per_block_from_env();
    const size_t grid_size = (rows + rows_per_block - 1) / rows_per_block;
    if (grid_size != partial_count) {
        if (error != nullptr) {
            *error = "AQ4 matvec top1 partial count does not match row block count";
        }
        return false;
    }
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec top1 row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *partial_values_ptr = partial_values_buffer->ptr;
    void *partial_indices_ptr = partial_indices_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &partial_values_ptr,
        &partial_indices_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            256u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec top1";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_add_f32_hip_kernel(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_add_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_ADD_RPB");
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec add row count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *residual_ptr = residual_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *kernel_params[] = {
        &index_ptr,
        &scale_ptr,
        &codebook_ptr,
        &scale_values_ptr,
        &input_ptr,
        &residual_ptr,
        &row_scale_ptr,
        &kernel_scale_count,
        &kernel_group_size,
        &tensor_scale,
        &kernel_row_scale_count,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec add";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_pair_f32_hip_kernel(
    const ullm_runtime_buffer *left_index_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    const ullm_runtime_buffer *left_codebook_buffer,
    const ullm_runtime_buffer *left_scale_values_buffer,
    const ullm_runtime_buffer *left_row_scale_buffer,
    size_t left_scale_count,
    size_t left_group_size,
    float left_tensor_scale,
    size_t left_row_scale_count,
    const ullm_runtime_buffer *right_index_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    const ullm_runtime_buffer *right_codebook_buffer,
    const ullm_runtime_buffer *right_scale_values_buffer,
    const ullm_runtime_buffer *right_row_scale_buffer,
    size_t right_scale_count,
    size_t right_group_size,
    float right_tensor_scale,
    size_t right_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = left_index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_pair_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            16u,
            4u,
            "ULLM_AQ4_MATVEC_PAIR_RPB");
    const size_t work_rows = std::max(left_rows, right_rows);
    const size_t grid_size =
        (work_rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec pair row count exceeds HIP grid limit";
        }
        return false;
    }
    void *left_index_ptr = left_index_buffer->ptr;
    void *left_scale_ptr = left_scale_buffer->ptr;
    void *left_codebook_ptr = left_codebook_buffer->ptr;
    void *left_scale_values_ptr = left_scale_values_buffer->ptr;
    void *left_row_scale_ptr =
        left_row_scale_buffer == nullptr ? nullptr : left_row_scale_buffer->ptr;
    unsigned long long kernel_left_scale_count =
        static_cast<unsigned long long>(left_scale_count);
    unsigned long long kernel_left_group_size =
        static_cast<unsigned long long>(left_group_size);
    unsigned long long kernel_left_row_scale_count =
        static_cast<unsigned long long>(left_row_scale_count);
    void *right_index_ptr = right_index_buffer->ptr;
    void *right_scale_ptr = right_scale_buffer->ptr;
    void *right_codebook_ptr = right_codebook_buffer->ptr;
    void *right_scale_values_ptr = right_scale_values_buffer->ptr;
    void *right_row_scale_ptr =
        right_row_scale_buffer == nullptr ? nullptr : right_row_scale_buffer->ptr;
    unsigned long long kernel_right_scale_count =
        static_cast<unsigned long long>(right_scale_count);
    unsigned long long kernel_right_group_size =
        static_cast<unsigned long long>(right_group_size);
    unsigned long long kernel_right_row_scale_count =
        static_cast<unsigned long long>(right_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    unsigned long long kernel_left_rows = static_cast<unsigned long long>(left_rows);
    unsigned long long kernel_right_rows = static_cast<unsigned long long>(right_rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *left_output_ptr = left_output_buffer->ptr;
    void *right_output_ptr = right_output_buffer->ptr;
    void *kernel_params[] = {
        &left_index_ptr,
        &left_scale_ptr,
        &left_codebook_ptr,
        &left_scale_values_ptr,
        &left_row_scale_ptr,
        &kernel_left_scale_count,
        &kernel_left_group_size,
        &left_tensor_scale,
        &kernel_left_row_scale_count,
        &right_index_ptr,
        &right_scale_ptr,
        &right_codebook_ptr,
        &right_scale_values_ptr,
        &right_row_scale_ptr,
        &kernel_right_scale_count,
        &kernel_right_group_size,
        &right_tensor_scale,
        &kernel_right_row_scale_count,
        &input_ptr,
        &kernel_left_rows,
        &kernel_right_rows,
        &kernel_cols,
        &left_output_ptr,
        &right_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec pair";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_triple_f32_hip_kernel(
    const ullm_runtime_buffer *first_index_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    const ullm_runtime_buffer *first_codebook_buffer,
    const ullm_runtime_buffer *first_scale_values_buffer,
    const ullm_runtime_buffer *first_row_scale_buffer,
    size_t first_scale_count,
    size_t first_group_size,
    float first_tensor_scale,
    size_t first_row_scale_count,
    const ullm_runtime_buffer *second_index_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    const ullm_runtime_buffer *second_codebook_buffer,
    const ullm_runtime_buffer *second_scale_values_buffer,
    const ullm_runtime_buffer *second_row_scale_buffer,
    size_t second_scale_count,
    size_t second_group_size,
    float second_tensor_scale,
    size_t second_row_scale_count,
    const ullm_runtime_buffer *third_index_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    const ullm_runtime_buffer *third_codebook_buffer,
    const ullm_runtime_buffer *third_scale_values_buffer,
    const ullm_runtime_buffer *third_row_scale_buffer,
    size_t third_scale_count,
    size_t third_group_size,
    float third_tensor_scale,
    size_t third_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = first_index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_triple_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            4u,
            "ULLM_AQ4_MATVEC_TRIPLE_RPB");
    const size_t work_rows = std::max(first_rows, std::max(second_rows, third_rows));
    const size_t grid_size =
        (work_rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec triple row count exceeds HIP grid limit";
        }
        return false;
    }
    void *first_index_ptr = first_index_buffer->ptr;
    void *first_scale_ptr = first_scale_buffer->ptr;
    void *first_codebook_ptr = first_codebook_buffer->ptr;
    void *first_scale_values_ptr = first_scale_values_buffer->ptr;
    void *first_row_scale_ptr =
        first_row_scale_buffer == nullptr ? nullptr : first_row_scale_buffer->ptr;
    unsigned long long kernel_first_scale_count =
        static_cast<unsigned long long>(first_scale_count);
    unsigned long long kernel_first_group_size =
        static_cast<unsigned long long>(first_group_size);
    unsigned long long kernel_first_row_scale_count =
        static_cast<unsigned long long>(first_row_scale_count);
    void *second_index_ptr = second_index_buffer->ptr;
    void *second_scale_ptr = second_scale_buffer->ptr;
    void *second_codebook_ptr = second_codebook_buffer->ptr;
    void *second_scale_values_ptr = second_scale_values_buffer->ptr;
    void *second_row_scale_ptr =
        second_row_scale_buffer == nullptr ? nullptr : second_row_scale_buffer->ptr;
    unsigned long long kernel_second_scale_count =
        static_cast<unsigned long long>(second_scale_count);
    unsigned long long kernel_second_group_size =
        static_cast<unsigned long long>(second_group_size);
    unsigned long long kernel_second_row_scale_count =
        static_cast<unsigned long long>(second_row_scale_count);
    void *third_index_ptr = third_index_buffer->ptr;
    void *third_scale_ptr = third_scale_buffer->ptr;
    void *third_codebook_ptr = third_codebook_buffer->ptr;
    void *third_scale_values_ptr = third_scale_values_buffer->ptr;
    void *third_row_scale_ptr =
        third_row_scale_buffer == nullptr ? nullptr : third_row_scale_buffer->ptr;
    unsigned long long kernel_third_scale_count =
        static_cast<unsigned long long>(third_scale_count);
    unsigned long long kernel_third_group_size =
        static_cast<unsigned long long>(third_group_size);
    unsigned long long kernel_third_row_scale_count =
        static_cast<unsigned long long>(third_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    unsigned long long kernel_first_rows = static_cast<unsigned long long>(first_rows);
    unsigned long long kernel_second_rows = static_cast<unsigned long long>(second_rows);
    unsigned long long kernel_third_rows = static_cast<unsigned long long>(third_rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *first_output_ptr = first_output_buffer->ptr;
    void *second_output_ptr = second_output_buffer->ptr;
    void *third_output_ptr = third_output_buffer->ptr;
    void *kernel_params[] = {
        &first_index_ptr,
        &first_scale_ptr,
        &first_codebook_ptr,
        &first_scale_values_ptr,
        &first_row_scale_ptr,
        &kernel_first_scale_count,
        &kernel_first_group_size,
        &first_tensor_scale,
        &kernel_first_row_scale_count,
        &second_index_ptr,
        &second_scale_ptr,
        &second_codebook_ptr,
        &second_scale_values_ptr,
        &second_row_scale_ptr,
        &kernel_second_scale_count,
        &kernel_second_group_size,
        &second_tensor_scale,
        &kernel_second_row_scale_count,
        &third_index_ptr,
        &third_scale_ptr,
        &third_codebook_ptr,
        &third_scale_values_ptr,
        &third_row_scale_ptr,
        &kernel_third_scale_count,
        &kernel_third_group_size,
        &third_tensor_scale,
        &kernel_third_row_scale_count,
        &input_ptr,
        &kernel_first_rows,
        &kernel_second_rows,
        &kernel_third_rows,
        &kernel_cols,
        &first_output_ptr,
        &second_output_ptr,
        &third_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec triple";
        }
        return false;
    }
    return true;
}

bool aq4_matvec_qkv_z_gate_beta_f32_hip_kernel(
    const ullm_runtime_buffer *qkv_index_buffer,
    const ullm_runtime_buffer *qkv_scale_buffer,
    const ullm_runtime_buffer *qkv_codebook_buffer,
    const ullm_runtime_buffer *qkv_scale_values_buffer,
    const ullm_runtime_buffer *qkv_row_scale_buffer,
    size_t qkv_scale_count,
    size_t qkv_group_size,
    float qkv_tensor_scale,
    size_t qkv_row_scale_count,
    const ullm_runtime_buffer *z_index_buffer,
    const ullm_runtime_buffer *z_scale_buffer,
    const ullm_runtime_buffer *z_codebook_buffer,
    const ullm_runtime_buffer *z_scale_values_buffer,
    const ullm_runtime_buffer *z_row_scale_buffer,
    size_t z_scale_count,
    size_t z_group_size,
    float z_tensor_scale,
    size_t z_row_scale_count,
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t qkv_rows,
    size_t z_rows,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *qkv_output_buffer,
    ullm_runtime_buffer *z_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = qkv_index_buffer->hip_device_id;
    void *function =
        hip_aq4_matvec_qkv_z_gate_beta_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            4u,
            2u,
            "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB");
    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t projection_rows = std::max(qkv_rows, z_rows);
    if (projection_rows > max_size - heads) {
        if (error != nullptr) {
            *error = "AQ4 qkv/z gate/beta row count overflows";
        }
        return false;
    }
    const size_t total_rows = projection_rows + heads;
    const size_t grid_size =
        (total_rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 qkv/z gate/beta row count exceeds HIP grid limit";
        }
        return false;
    }

    void *qkv_index_ptr = qkv_index_buffer->ptr;
    void *qkv_scale_ptr = qkv_scale_buffer->ptr;
    void *qkv_codebook_ptr = qkv_codebook_buffer->ptr;
    void *qkv_scale_values_ptr = qkv_scale_values_buffer->ptr;
    void *qkv_row_scale_ptr =
        qkv_row_scale_buffer == nullptr ? nullptr : qkv_row_scale_buffer->ptr;
    unsigned long long kernel_qkv_scale_count =
        static_cast<unsigned long long>(qkv_scale_count);
    unsigned long long kernel_qkv_group_size =
        static_cast<unsigned long long>(qkv_group_size);
    unsigned long long kernel_qkv_row_scale_count =
        static_cast<unsigned long long>(qkv_row_scale_count);
    void *z_index_ptr = z_index_buffer->ptr;
    void *z_scale_ptr = z_scale_buffer->ptr;
    void *z_codebook_ptr = z_codebook_buffer->ptr;
    void *z_scale_values_ptr = z_scale_values_buffer->ptr;
    void *z_row_scale_ptr = z_row_scale_buffer == nullptr ? nullptr : z_row_scale_buffer->ptr;
    unsigned long long kernel_z_scale_count = static_cast<unsigned long long>(z_scale_count);
    unsigned long long kernel_z_group_size = static_cast<unsigned long long>(z_group_size);
    unsigned long long kernel_z_row_scale_count =
        static_cast<unsigned long long>(z_row_scale_count);
    void *a_index_ptr = a_index_buffer->ptr;
    void *a_scale_ptr = a_scale_buffer->ptr;
    void *a_codebook_ptr = a_codebook_buffer->ptr;
    void *a_scale_values_ptr = a_scale_values_buffer->ptr;
    void *a_row_scale_ptr = a_row_scale_buffer == nullptr ? nullptr : a_row_scale_buffer->ptr;
    unsigned long long kernel_a_scale_count = static_cast<unsigned long long>(a_scale_count);
    unsigned long long kernel_a_group_size = static_cast<unsigned long long>(a_group_size);
    unsigned long long kernel_a_row_scale_count =
        static_cast<unsigned long long>(a_row_scale_count);
    void *b_index_ptr = b_index_buffer->ptr;
    void *b_scale_ptr = b_scale_buffer->ptr;
    void *b_codebook_ptr = b_codebook_buffer->ptr;
    void *b_scale_values_ptr = b_scale_values_buffer->ptr;
    void *b_row_scale_ptr = b_row_scale_buffer == nullptr ? nullptr : b_row_scale_buffer->ptr;
    unsigned long long kernel_b_scale_count = static_cast<unsigned long long>(b_scale_count);
    unsigned long long kernel_b_group_size = static_cast<unsigned long long>(b_group_size);
    unsigned long long kernel_b_row_scale_count =
        static_cast<unsigned long long>(b_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    void *a_log_ptr = a_log_buffer->ptr;
    void *dt_bias_ptr = dt_bias_buffer->ptr;
    unsigned long long kernel_qkv_rows = static_cast<unsigned long long>(qkv_rows);
    unsigned long long kernel_z_rows = static_cast<unsigned long long>(z_rows);
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *qkv_output_ptr = qkv_output_buffer->ptr;
    void *z_output_ptr = z_output_buffer->ptr;
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *beta_output_ptr = beta_output_buffer->ptr;
    void *kernel_params[] = {
        &qkv_index_ptr,
        &qkv_scale_ptr,
        &qkv_codebook_ptr,
        &qkv_scale_values_ptr,
        &qkv_row_scale_ptr,
        &kernel_qkv_scale_count,
        &kernel_qkv_group_size,
        &qkv_tensor_scale,
        &kernel_qkv_row_scale_count,
        &z_index_ptr,
        &z_scale_ptr,
        &z_codebook_ptr,
        &z_scale_values_ptr,
        &z_row_scale_ptr,
        &kernel_z_scale_count,
        &kernel_z_group_size,
        &z_tensor_scale,
        &kernel_z_row_scale_count,
        &a_index_ptr,
        &a_scale_ptr,
        &a_codebook_ptr,
        &a_scale_values_ptr,
        &a_row_scale_ptr,
        &kernel_a_scale_count,
        &kernel_a_group_size,
        &a_tensor_scale,
        &kernel_a_row_scale_count,
        &b_index_ptr,
        &b_scale_ptr,
        &b_codebook_ptr,
        &b_scale_values_ptr,
        &b_row_scale_ptr,
        &kernel_b_scale_count,
        &kernel_b_group_size,
        &b_tensor_scale,
        &kernel_b_row_scale_count,
        &input_ptr,
        &a_log_ptr,
        &dt_bias_ptr,
        &kernel_qkv_rows,
        &kernel_z_rows,
        &kernel_heads,
        &kernel_cols,
        &qkv_output_ptr,
        &z_output_ptr,
        &gate_output_ptr,
        &beta_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 qkv/z gate/beta";
        }
        return false;
    }
    return true;
}

ullm_status aq4_matvec_f32_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t required_index_bytes,
    size_t groups,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(rows);
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
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status aq4_matvec_batch_f32_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    size_t required_index_bytes,
    size_t groups,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(batch_count * cols);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(batch_count * rows);
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
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            batch_count * cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec batch HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec batch HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_batch_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            batch_count,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            batch_count * rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec batch output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec batch HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status aq4_matvec_add_f32_hip_staging(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t required_index_bytes,
    size_t groups,
    size_t codebook_entries,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_residual(rows);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(rows);
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
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_residual.data(),
            residual_buffer->ptr,
            rows * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec add HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec add HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_add_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            host_residual.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec add output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec add HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipAq4MatvecSiluMulKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_silu_mul_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_silu_mul_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec SiLU-mul HIP kernel" : compile_errors);
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

HipAq4MatvecSiluMulKernelCache &hip_aq4_matvec_silu_mul_kernel_cache() {
    static HipAq4MatvecSiluMulKernelCache cache;
    return cache;
}

bool aq4_matvec_silu_mul_f32_hip_kernel(
    const ullm_runtime_buffer *gate_index_buffer,
    const ullm_runtime_buffer *gate_scale_buffer,
    const ullm_runtime_buffer *gate_codebook_buffer,
    const ullm_runtime_buffer *gate_scale_values_buffer,
    const ullm_runtime_buffer *gate_row_scale_buffer,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    size_t gate_row_scale_count,
    const ullm_runtime_buffer *up_index_buffer,
    const ullm_runtime_buffer *up_scale_buffer,
    const ullm_runtime_buffer *up_codebook_buffer,
    const ullm_runtime_buffer *up_scale_values_buffer,
    const ullm_runtime_buffer *up_row_scale_buffer,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    size_t up_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = gate_index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_silu_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config =
        aq4_matvec_launch_config_for_fused_kernel(
            device_id,
            8u,
            2u,
            "ULLM_AQ4_MATVEC_SILU_MUL_RPB");
    const size_t grid_size =
        (rows + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec SiLU-mul row count exceeds HIP grid limit";
        }
        return false;
    }
    void *gate_index_ptr = gate_index_buffer->ptr;
    void *gate_scale_ptr = gate_scale_buffer->ptr;
    void *gate_codebook_ptr = gate_codebook_buffer->ptr;
    void *gate_scale_values_ptr = gate_scale_values_buffer->ptr;
    void *gate_row_scale_ptr =
        gate_row_scale_buffer == nullptr ? nullptr : gate_row_scale_buffer->ptr;
    unsigned long long kernel_gate_scale_count =
        static_cast<unsigned long long>(gate_scale_count);
    unsigned long long kernel_gate_group_size = static_cast<unsigned long long>(gate_group_size);
    unsigned long long kernel_gate_row_scale_count =
        static_cast<unsigned long long>(gate_row_scale_count);
    void *up_index_ptr = up_index_buffer->ptr;
    void *up_scale_ptr = up_scale_buffer->ptr;
    void *up_codebook_ptr = up_codebook_buffer->ptr;
    void *up_scale_values_ptr = up_scale_values_buffer->ptr;
    void *up_row_scale_ptr = up_row_scale_buffer == nullptr ? nullptr : up_row_scale_buffer->ptr;
    unsigned long long kernel_up_scale_count = static_cast<unsigned long long>(up_scale_count);
    unsigned long long kernel_up_group_size = static_cast<unsigned long long>(up_group_size);
    unsigned long long kernel_up_row_scale_count =
        static_cast<unsigned long long>(up_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &gate_index_ptr,
        &gate_scale_ptr,
        &gate_codebook_ptr,
        &gate_scale_values_ptr,
        &gate_row_scale_ptr,
        &kernel_gate_scale_count,
        &kernel_gate_group_size,
        &gate_tensor_scale,
        &kernel_gate_row_scale_count,
        &up_index_ptr,
        &up_scale_ptr,
        &up_codebook_ptr,
        &up_scale_values_ptr,
        &up_row_scale_ptr,
        &kernel_up_scale_count,
        &kernel_up_group_size,
        &up_tensor_scale,
        &kernel_up_row_scale_count,
        &input_ptr,
        &kernel_rows,
        &kernel_cols,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec SiLU-mul";
        }
        return false;
    }
    return true;
}

ullm_status aq4_matvec_silu_mul_f32_hip_staging(
    const ullm_runtime_buffer *gate_index_buffer,
    const ullm_runtime_buffer *gate_scale_buffer,
    const ullm_runtime_buffer *gate_codebook_buffer,
    const ullm_runtime_buffer *gate_scale_values_buffer,
    const ullm_runtime_buffer *gate_row_scale_buffer,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    size_t gate_row_scale_count,
    size_t gate_required_index_bytes,
    size_t gate_groups,
    size_t gate_codebook_entries,
    const ullm_runtime_buffer *up_index_buffer,
    const ullm_runtime_buffer *up_scale_buffer,
    const ullm_runtime_buffer *up_codebook_buffer,
    const ullm_runtime_buffer *up_scale_values_buffer,
    const ullm_runtime_buffer *up_row_scale_buffer,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    size_t up_row_scale_count,
    size_t up_required_index_bytes,
    size_t up_groups,
    size_t up_codebook_entries,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_gate_indices(gate_required_index_bytes);
    std::vector<std::uint8_t> host_gate_scale_indices(gate_groups);
    std::vector<float> host_gate_codebook(gate_codebook_entries);
    std::vector<float> host_gate_scale_values(gate_scale_count);
    std::vector<float> host_gate_row_scales(gate_row_scale_count);
    std::vector<std::uint8_t> host_up_indices(up_required_index_bytes);
    std::vector<std::uint8_t> host_up_scale_indices(up_groups);
    std::vector<float> host_up_codebook(up_codebook_entries);
    std::vector<float> host_up_scale_values(up_scale_count);
    std::vector<float> host_up_row_scales(up_row_scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_output(rows);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = gate_index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_gate_indices.data(),
            gate_index_buffer->ptr,
            gate_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate_scale_indices.data(),
            gate_scale_buffer->ptr,
            gate_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate_codebook.data(),
            gate_codebook_buffer->ptr,
            gate_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate_scale_values.data(),
            gate_scale_values_buffer->ptr,
            gate_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (gate_row_scale_buffer != nullptr && gate_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_gate_row_scales.data(),
             gate_row_scale_buffer->ptr,
             gate_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_up_indices.data(),
            up_index_buffer->ptr,
            up_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_up_scale_indices.data(),
            up_scale_buffer->ptr,
            up_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_up_codebook.data(),
            up_codebook_buffer->ptr,
            up_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_up_scale_values.data(),
            up_scale_values_buffer->ptr,
            up_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (up_row_scale_buffer != nullptr && up_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_up_row_scales.data(),
             up_row_scale_buffer->ptr,
             up_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec SiLU-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec SiLU-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_silu_mul_f32_host(
            host_gate_indices.data(),
            host_gate_scale_indices.data(),
            host_gate_codebook.data(),
            host_gate_scale_values.data(),
            gate_scale_count,
            gate_group_size,
            gate_tensor_scale,
            gate_row_scale_buffer == nullptr ? nullptr : host_gate_row_scales.data(),
            gate_row_scale_count,
            host_up_indices.data(),
            host_up_scale_indices.data(),
            host_up_codebook.data(),
            host_up_scale_values.data(),
            up_scale_count,
            up_group_size,
            up_tensor_scale,
            up_row_scale_buffer == nullptr ? nullptr : host_up_row_scales.data(),
            up_row_scale_count,
            host_input.data(),
            rows,
            cols,
            host_output.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            rows * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec SiLU-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec SiLU-mul HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipAq4MatvecGateBetaKernelCache {
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
            if (!hiprtc_runtime().compile_aq4_matvec_gate_beta_kernel(arch, &code, &compile_error)) {
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
                    "ullm_aq4_matvec_gate_beta_f32_kernel",
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
            compile_errors.empty() ? "failed to build AQ4 matvec gate/beta HIP kernel" : compile_errors);
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

HipAq4MatvecGateBetaKernelCache &hip_aq4_matvec_gate_beta_kernel_cache() {
    static HipAq4MatvecGateBetaKernelCache cache;
    return cache;
}

bool aq4_matvec_gate_beta_f32_hip_kernel(
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = a_index_buffer->hip_device_id;
    void *function = hip_aq4_matvec_gate_beta_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const Aq4MatvecLaunchConfig launch_config = aq4_matvec_launch_config_for_device(device_id);
    const size_t grid_size =
        (heads + launch_config.rows_per_block - 1) / launch_config.rows_per_block;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 matvec gate/beta head count exceeds HIP grid limit";
        }
        return false;
    }
    void *a_index_ptr = a_index_buffer->ptr;
    void *a_scale_ptr = a_scale_buffer->ptr;
    void *a_codebook_ptr = a_codebook_buffer->ptr;
    void *a_scale_values_ptr = a_scale_values_buffer->ptr;
    void *a_row_scale_ptr = a_row_scale_buffer == nullptr ? nullptr : a_row_scale_buffer->ptr;
    unsigned long long kernel_a_scale_count = static_cast<unsigned long long>(a_scale_count);
    unsigned long long kernel_a_group_size = static_cast<unsigned long long>(a_group_size);
    unsigned long long kernel_a_row_scale_count =
        static_cast<unsigned long long>(a_row_scale_count);
    void *b_index_ptr = b_index_buffer->ptr;
    void *b_scale_ptr = b_scale_buffer->ptr;
    void *b_codebook_ptr = b_codebook_buffer->ptr;
    void *b_scale_values_ptr = b_scale_values_buffer->ptr;
    void *b_row_scale_ptr = b_row_scale_buffer == nullptr ? nullptr : b_row_scale_buffer->ptr;
    unsigned long long kernel_b_scale_count = static_cast<unsigned long long>(b_scale_count);
    unsigned long long kernel_b_group_size = static_cast<unsigned long long>(b_group_size);
    unsigned long long kernel_b_row_scale_count =
        static_cast<unsigned long long>(b_row_scale_count);
    void *input_ptr = input_buffer->ptr;
    void *a_log_ptr = a_log_buffer->ptr;
    void *dt_bias_ptr = dt_bias_buffer->ptr;
    unsigned long long kernel_heads = static_cast<unsigned long long>(heads);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *beta_output_ptr = beta_output_buffer->ptr;
    void *kernel_params[] = {
        &a_index_ptr,
        &a_scale_ptr,
        &a_codebook_ptr,
        &a_scale_values_ptr,
        &a_row_scale_ptr,
        &kernel_a_scale_count,
        &kernel_a_group_size,
        &a_tensor_scale,
        &kernel_a_row_scale_count,
        &b_index_ptr,
        &b_scale_ptr,
        &b_codebook_ptr,
        &b_scale_values_ptr,
        &b_row_scale_ptr,
        &kernel_b_scale_count,
        &kernel_b_group_size,
        &b_tensor_scale,
        &kernel_b_row_scale_count,
        &input_ptr,
        &a_log_ptr,
        &dt_bias_ptr,
        &kernel_heads,
        &kernel_cols,
        &gate_output_ptr,
        &beta_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(grid_size),
            launch_config.block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 matvec gate/beta";
        }
        return false;
    }
    return true;
}

ullm_status aq4_matvec_gate_beta_f32_hip_staging(
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    size_t a_required_index_bytes,
    size_t a_groups,
    size_t a_codebook_entries,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    size_t b_required_index_bytes,
    size_t b_groups,
    size_t b_codebook_entries,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<std::uint8_t> host_a_indices(a_required_index_bytes);
    std::vector<std::uint8_t> host_a_scale_indices(a_groups);
    std::vector<float> host_a_codebook(a_codebook_entries);
    std::vector<float> host_a_scale_values(a_scale_count);
    std::vector<float> host_a_row_scales(a_row_scale_count);
    std::vector<std::uint8_t> host_b_indices(b_required_index_bytes);
    std::vector<std::uint8_t> host_b_scale_indices(b_groups);
    std::vector<float> host_b_codebook(b_codebook_entries);
    std::vector<float> host_b_scale_values(b_scale_count);
    std::vector<float> host_b_row_scales(b_row_scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_a_log(heads);
    std::vector<float> host_dt_bias(heads);
    std::vector<float> host_gate(heads);
    std::vector<float> host_beta(heads);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = a_index_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_a_indices.data(),
            a_index_buffer->ptr,
            a_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_scale_indices.data(),
            a_scale_buffer->ptr,
            a_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_codebook.data(),
            a_codebook_buffer->ptr,
            a_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_scale_values.data(),
            a_scale_values_buffer->ptr,
            a_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (a_row_scale_buffer != nullptr && a_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_a_row_scales.data(),
             a_row_scale_buffer->ptr,
             a_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_b_indices.data(),
            b_index_buffer->ptr,
            b_required_index_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b_scale_indices.data(),
            b_scale_buffer->ptr,
            b_groups,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b_codebook.data(),
            b_codebook_buffer->ptr,
            b_codebook_buffer->bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_b_scale_values.data(),
            b_scale_values_buffer->ptr,
            b_scale_count * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (b_row_scale_buffer != nullptr && b_row_scale_count > 0 &&
         !hip_runtime().copy_async(
             host_b_row_scales.data(),
             b_row_scale_buffer->ptr,
             b_row_scale_count * sizeof(float),
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id)) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            cols * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_a_log.data(),
            a_log_buffer->ptr,
            heads * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_dt_bias.data(),
            dt_bias_buffer->ptr,
            heads * sizeof(float),
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec gate/beta HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec gate/beta HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_gate_beta_f32_host(
            host_a_indices.data(),
            host_a_scale_indices.data(),
            host_a_codebook.data(),
            host_a_scale_values.data(),
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_buffer == nullptr ? nullptr : host_a_row_scales.data(),
            a_row_scale_count,
            host_b_indices.data(),
            host_b_scale_indices.data(),
            host_b_codebook.data(),
            host_b_scale_values.data(),
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_buffer == nullptr ? nullptr : host_b_row_scales.data(),
            b_row_scale_count,
            host_input.data(),
            host_a_log.data(),
            host_dt_bias.data(),
            heads,
            cols,
            host_gate.data(),
            host_beta.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            gate_output_buffer->ptr,
            host_gate.data(),
            heads * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            beta_output_buffer->ptr,
            host_beta.data(),
            heads * sizeof(float),
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec gate/beta output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec gate/beta HIP output staging copy");
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

float bf16_to_f32_host(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void matvec_bf16_f32_host(
    const uint16_t *matrix,
    const float *input,
    size_t rows,
    size_t cols,
    float *output) {
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *row_values = matrix + row * cols;
        float sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            sum += bf16_to_f32_host(row_values[col]) * input[col];
        }
        output[row] = sum;
    }
}

void bf16_row_f32_host(
    const uint16_t *matrix,
    size_t rows,
    size_t cols,
    size_t row_index,
    float *output) {
    if (row_index >= rows) {
        return;
    }
    const uint16_t *row_values = matrix + row_index * cols;
    for (size_t col = 0; col < cols; ++col) {
        output[col] = bf16_to_f32_host(row_values[col]);
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

class HipBf16MatvecKernelCache {
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
            if (!hiprtc_runtime().compile_matvec_bf16_kernel(arch, &code, &compile_error)) {
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
                    "ullm_matvec_bf16_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build BF16 matvec HIP kernel" : compile_errors);
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

HipBf16MatvecKernelCache &hip_bf16_matvec_kernel_cache() {
    static HipBf16MatvecKernelCache cache;
    return cache;
}

class HipBf16RowKernelCache {
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
            if (!hiprtc_runtime().compile_bf16_row_kernel(arch, &code, &compile_error)) {
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
                    "ullm_bf16_row_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build BF16 row HIP kernel" : compile_errors);
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

HipBf16RowKernelCache &hip_bf16_row_kernel_cache() {
    static HipBf16RowKernelCache cache;
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

bool matvec_bf16_f32_hip_kernel(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = matrix_buffer->hip_device_id;
    void *function = hip_bf16_matvec_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 64;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "BF16 matvec row count exceeds HIP grid limit";
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
            *error = "hipModuleLaunchKernel failed for BF16 matvec";
        }
        return false;
    }
    return true;
}

bool bf16_row_f32_hip_kernel(
    const ullm_runtime_buffer *matrix_buffer,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = matrix_buffer->hip_device_id;
    void *function = hip_bf16_row_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_row_index = static_cast<unsigned long long>(row_index);
    void *matrix_ptr = matrix_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &matrix_ptr,
        &kernel_rows,
        &kernel_cols,
        &kernel_row_index,
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
            *error = "hipModuleLaunchKernel failed for BF16 row";
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

ullm_status matvec_bf16_f32_hip_staging(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t required_matrix_bytes,
    size_t required_input_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<uint16_t> host_matrix(required_matrix_bytes / sizeof(uint16_t));
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
        set_error("failed to copy BF16 matvec HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 matvec HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    matvec_bf16_f32_host(host_matrix.data(), host_input.data(), rows, cols, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy BF16 matvec output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 matvec HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status bf16_row_f32_hip_staging(
    const ullm_runtime_buffer *matrix_buffer,
    size_t rows,
    size_t cols,
    size_t row_index,
    size_t required_matrix_bytes,
    size_t required_output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<uint16_t> host_row(cols);
    std::vector<float> host_output(cols);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = matrix_buffer->hip_device_id;
    const size_t row_bytes = cols * sizeof(uint16_t);
    if (row_index >= rows) {
        set_error("BF16 row staging index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t row_offset = row_index * row_bytes;
    if (row_offset > required_matrix_bytes || row_bytes > required_matrix_bytes - row_offset) {
        set_error("BF16 row byte offset overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (!hip_runtime().copy_async(
            host_row.data(),
            static_cast<const std::uint8_t *>(matrix_buffer->ptr) + row_offset,
            row_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy BF16 row HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 row HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    bf16_row_f32_host(host_row.data(), 1, cols, 0, host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            required_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy BF16 row output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize BF16 row HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

void top1_f32_partials_host(
    const float *input,
    size_t elements,
    size_t partial_count,
    float *partial_values,
    uint32_t *partial_indices) {
    constexpr size_t block_size = 256;
    for (size_t block = 0; block < partial_count; ++block) {
        const size_t start = block * block_size;
        const size_t end = std::min(start + block_size, elements);
        float best_value = -std::numeric_limits<float>::max();
        uint32_t best_index = std::numeric_limits<uint32_t>::max();
        for (size_t index = start; index < end; ++index) {
            float value = input[index];
            if (value != value) {
                value = -std::numeric_limits<float>::max();
            }
            const auto token_index = static_cast<uint32_t>(index);
            if (value > best_value || (value == best_value && token_index < best_index)) {
                best_value = value;
                best_index = token_index;
            }
        }
        partial_values[block] = best_value;
        partial_indices[block] = best_index;
    }
}

void top1_pairs_f32_partials_host(
    const float *input_values,
    const uint32_t *input_indices,
    size_t elements,
    size_t partial_count,
    float *partial_values,
    uint32_t *partial_indices) {
    constexpr size_t block_size = 256;
    for (size_t block = 0; block < partial_count; ++block) {
        const size_t start = block * block_size;
        const size_t end = std::min(start + block_size, elements);
        float best_value = -std::numeric_limits<float>::max();
        uint32_t best_index = std::numeric_limits<uint32_t>::max();
        for (size_t index = start; index < end; ++index) {
            float value = input_values[index];
            if (value != value) {
                value = -std::numeric_limits<float>::max();
            }
            const uint32_t token_index = input_indices[index];
            if (value > best_value || (value == best_value && token_index < best_index)) {
                best_value = value;
                best_index = token_index;
            }
        }
        partial_values[block] = best_value;
        partial_indices[block] = best_index;
    }
}

class HipTop1KernelCache {
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
            if (!hiprtc_runtime().compile_top1_kernel(arch, &code, &compile_error)) {
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
                    "ullm_top1_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build top1 HIP kernel" : compile_errors);
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

HipTop1KernelCache &hip_top1_kernel_cache() {
    static HipTop1KernelCache cache;
    return cache;
}

class HipTop1PairsKernelCache {
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
            if (!hiprtc_runtime().compile_top1_pairs_kernel(arch, &code, &compile_error)) {
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
                    "ullm_top1_pairs_f32_kernel",
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
            compile_errors.empty() ? "failed to build top1 pairs HIP kernel" : compile_errors);
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

HipTop1PairsKernelCache &hip_top1_pairs_kernel_cache() {
    static HipTop1PairsKernelCache cache;
    return cache;
}

bool top1_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_top1_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (partial_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "top1 partial count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *input_ptr = input_buffer->ptr;
    void *partial_values_ptr = partial_values_buffer->ptr;
    void *partial_indices_ptr = partial_indices_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &kernel_elements,
        &partial_values_ptr,
        &partial_indices_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(partial_count),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 top1";
        }
        return false;
    }
    return true;
}

bool top1_pairs_f32_hip_kernel(
    const ullm_runtime_buffer *values_buffer,
    const ullm_runtime_buffer *indices_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = values_buffer->hip_device_id;
    void *function = hip_top1_pairs_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (partial_count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "top1 pairs partial count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_elements = static_cast<unsigned long long>(elements);
    void *values_ptr = values_buffer->ptr;
    void *indices_ptr = indices_buffer->ptr;
    void *partial_values_ptr = partial_values_buffer->ptr;
    void *partial_indices_ptr = partial_indices_buffer->ptr;
    void *kernel_params[] = {
        &values_ptr,
        &indices_ptr,
        &kernel_elements,
        &partial_values_ptr,
        &partial_indices_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(partial_count),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 top1 pairs";
        }
        return false;
    }
    return true;
}

ullm_status top1_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    size_t input_bytes,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    size_t partial_values_bytes,
    size_t partial_indices_bytes,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(elements);
    std::vector<float> host_partial_values(partial_count);
    std::vector<uint32_t> host_partial_indices(partial_count);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    top1_f32_partials_host(
        host_input.data(),
        elements,
        partial_count,
        host_partial_values.data(),
        host_partial_indices.data());
    if (!hip_runtime().copy_async(
            partial_values_buffer->ptr,
            host_partial_values.data(),
            partial_values_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            partial_indices_buffer->ptr,
            host_partial_indices.data(),
            partial_indices_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 partial outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status top1_pairs_f32_hip_staging(
    const ullm_runtime_buffer *values_buffer,
    const ullm_runtime_buffer *indices_buffer,
    size_t elements,
    size_t values_bytes,
    size_t indices_bytes,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    size_t partial_values_bytes,
    size_t partial_indices_bytes,
    ullm_runtime_stream *stream) {
    std::vector<float> host_values(elements);
    std::vector<uint32_t> host_indices(elements);
    std::vector<float> host_partial_values(partial_count);
    std::vector<uint32_t> host_partial_indices(partial_count);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = values_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_values.data(),
            values_buffer->ptr,
            values_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_indices.data(),
            indices_buffer->ptr,
            indices_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 pairs HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 pairs HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    top1_pairs_f32_partials_host(
        host_values.data(),
        host_indices.data(),
        elements,
        partial_count,
        host_partial_values.data(),
        host_partial_indices.data());
    if (!hip_runtime().copy_async(
            partial_values_buffer->ptr,
            host_partial_values.data(),
            partial_values_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            partial_indices_buffer->ptr,
            host_partial_indices.data(),
            partial_indices_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 top1 pairs partial outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 top1 pairs HIP output staging copies");
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

void segmented_rmsnorm_f32_host(
    const float *input,
    const float *weight,
    size_t segments,
    size_t segment_size,
    float epsilon,
    float *output) {
    for (size_t segment = 0; segment < segments; ++segment) {
        const size_t base = segment * segment_size;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < segment_size; ++dim) {
            const float value = input[base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(segment_size) + epsilon);
        for (size_t dim = 0; dim < segment_size; ++dim) {
            output[base + dim] = input[base + dim] * inv_rms * weight[dim];
        }
    }
}

void segmented_rmsnorm_silu_mul_f32_host(
    const float *input,
    const float *weight,
    const float *gate,
    size_t segments,
    size_t segment_size,
    float epsilon,
    float *output) {
    for (size_t segment = 0; segment < segments; ++segment) {
        const size_t base = segment * segment_size;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < segment_size; ++dim) {
            const float value = input[base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(segment_size) + epsilon);
        for (size_t dim = 0; dim < segment_size; ++dim) {
            const size_t index = base + dim;
            const float gate_value = gate[index];
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
            const float normed = input[index] * inv_rms * weight[dim];
            output[index] = gate_value * sigmoid * normed;
        }
    }
}

class HipSegmentedRmsNormKernelCache {
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
            if (!hiprtc_runtime().compile_segmented_rmsnorm_kernel(arch, &code, &compile_error)) {
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
                    "ullm_segmented_rmsnorm_f32_kernel",
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
            compile_errors.empty() ? "failed to build segmented RMSNorm HIP kernel" : compile_errors);
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

HipSegmentedRmsNormKernelCache &hip_segmented_rmsnorm_kernel_cache() {
    static HipSegmentedRmsNormKernelCache cache;
    return cache;
}

bool segmented_rmsnorm_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_segmented_rmsnorm_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "segmented RMSNorm segment count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_segments = static_cast<unsigned long long>(segments);
    unsigned long long kernel_segment_size = static_cast<unsigned long long>(segment_size);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &kernel_segments,
        &kernel_segment_size,
        &epsilon,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(segments),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 segmented RMSNorm";
        }
        return false;
    }
    return true;
}

ullm_status segmented_rmsnorm_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    size_t input_output_bytes,
    size_t weight_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(input_output_bytes / sizeof(float));
    std::vector<float> host_weight(weight_bytes / sizeof(float));
    std::vector<float> host_output(input_output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            input_output_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    segmented_rmsnorm_f32_host(
        host_input.data(),
        host_weight.data(),
        segments,
        segment_size,
        epsilon,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            input_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipSegmentedRmsNormSiluMulKernelCache {
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
            if (!hiprtc_runtime().compile_segmented_rmsnorm_silu_mul_kernel(arch, &code, &compile_error)) {
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
                    "ullm_segmented_rmsnorm_silu_mul_f32_kernel",
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
            compile_errors.empty() ? "failed to build segmented RMSNorm SiLU-mul HIP kernel"
                                   : compile_errors);
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

HipSegmentedRmsNormSiluMulKernelCache &hip_segmented_rmsnorm_silu_mul_kernel_cache() {
    static HipSegmentedRmsNormSiluMulKernelCache cache;
    return cache;
}

bool segmented_rmsnorm_silu_mul_f32_hip_kernel(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = input_buffer->hip_device_id;
    void *function = hip_segmented_rmsnorm_silu_mul_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "segmented RMSNorm SiLU-mul segment count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_segments = static_cast<unsigned long long>(segments);
    unsigned long long kernel_segment_size = static_cast<unsigned long long>(segment_size);
    void *input_ptr = input_buffer->ptr;
    void *weight_ptr = weight_buffer->ptr;
    void *gate_ptr = gate_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &input_ptr,
        &weight_ptr,
        &gate_ptr,
        &kernel_segments,
        &kernel_segment_size,
        &epsilon,
        &output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(segments),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for f32 segmented RMSNorm SiLU-mul";
        }
        return false;
    }
    return true;
}

ullm_status segmented_rmsnorm_silu_mul_f32_hip_staging(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    size_t input_output_bytes,
    size_t weight_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_input(input_output_bytes / sizeof(float));
    std::vector<float> host_weight(weight_bytes / sizeof(float));
    std::vector<float> host_gate(input_output_bytes / sizeof(float));
    std::vector<float> host_output(input_output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = input_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            input_output_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_weight.data(),
            weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            input_output_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm SiLU-mul HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm SiLU-mul HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    segmented_rmsnorm_silu_mul_f32_host(
        host_input.data(),
        host_weight.data(),
        host_gate.data(),
        segments,
        segment_size,
        epsilon,
        host_output.data());
    if (!hip_runtime().copy_async(
            output_buffer->ptr,
            host_output.data(),
            input_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy f32 segmented RMSNorm SiLU-mul output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize f32 segmented RMSNorm SiLU-mul HIP output staging copy");
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

void qwen35_split_q_gate_f32_host(
    const float *projected,
    size_t q_heads,
    size_t head_dim,
    float *query_output,
    float *gate_output) {
    for (size_t head = 0; head < q_heads; ++head) {
        const size_t source_base = head * 2 * head_dim;
        const size_t output_base = head * head_dim;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            query_output[output_base + dim] = projected[source_base + dim];
            gate_output[output_base + dim] = projected[source_base + head_dim + dim];
        }
    }
}

void qwen35_qk_norm_rope_f32_host(
    const float *q_projected,
    const float *k_projected,
    const float *q_weight,
    const float *k_weight,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    float *q_gate_output,
    float *q_rope_output,
    float *k_rope_output) {
    const size_t half = rotary_dim / 2;
    const float position = static_cast<float>(position_offset);
    for (size_t head = 0; head < q_heads; ++head) {
        const size_t source_base = head * 2 * head_dim;
        const size_t output_base = head * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = q_projected[source_base + dim];
            q_gate_output[output_base + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first =
                q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second =
                q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * c - second * s;
            q_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            q_rope_output[output_base + dim] =
                q_projected[source_base + dim] * inv_rms * q_weight[dim];
        }
    }
    for (size_t head = 0; head < kv_heads; ++head) {
        const size_t base = head * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = k_projected[base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first = k_projected[base + pair_dim] * inv_rms * k_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second = k_projected[base + second_dim] * inv_rms * k_weight[second_dim];
            k_rope_output[base + pair_dim] = first * c - second * s;
            k_rope_output[base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            k_rope_output[base + dim] = k_projected[base + dim] * inv_rms * k_weight[dim];
        }
    }
}

void qwen35_qk_norm_rope_paged_kv_write_f32_host(
    const float *q_projected,
    const float *k_projected,
    const float *v_projected,
    const float *q_weight,
    const float *k_weight,
    const std::uint32_t *block_table,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    float *q_gate_output,
    float *q_rope_output,
    float *k_cache,
    float *v_cache) {
    const size_t half = rotary_dim / 2;
    const float position = static_cast<float>(position_offset);
    const size_t block_index = cache_position / block_size;
    const size_t block_offset = cache_position - block_index * block_size;
    const size_t physical_timestep =
        static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
    for (size_t head = 0; head < q_heads; ++head) {
        const size_t source_base = head * 2 * head_dim;
        const size_t output_base = head * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = q_projected[source_base + dim];
            q_gate_output[output_base + dim] = q_projected[source_base + head_dim + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first =
                q_projected[source_base + pair_dim] * inv_rms * q_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second =
                q_projected[source_base + second_dim] * inv_rms * q_weight[second_dim];
            q_rope_output[output_base + pair_dim] = first * c - second * s;
            q_rope_output[output_base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            q_rope_output[output_base + dim] =
                q_projected[source_base + dim] * inv_rms * q_weight[dim];
        }
    }
    for (size_t head = 0; head < kv_heads; ++head) {
        const size_t k_source_base = head * head_dim;
        const size_t k_cache_base = (physical_timestep * kv_heads + head) * head_dim;
        float sum_squares = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const float value = k_projected[k_source_base + dim];
            sum_squares += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_squares / static_cast<float>(head_dim) + epsilon);
        for (size_t pair_dim = 0; pair_dim < half; ++pair_dim) {
            const float exponent =
                (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
            const float theta = position / std::pow(rope_base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float first =
                k_projected[k_source_base + pair_dim] * inv_rms * k_weight[pair_dim];
            const size_t second_dim = half + pair_dim;
            const float second =
                k_projected[k_source_base + second_dim] * inv_rms * k_weight[second_dim];
            k_cache[k_cache_base + pair_dim] = first * c - second * s;
            k_cache[k_cache_base + second_dim] = second * c + first * s;
        }
        for (size_t dim = rotary_dim; dim < head_dim; ++dim) {
            k_cache[k_cache_base + dim] =
                k_projected[k_source_base + dim] * inv_rms * k_weight[dim];
        }

        const size_t v_source_base = head * value_dim;
        const size_t v_cache_base = (physical_timestep * kv_heads + head) * value_dim;
        std::copy(
            v_projected + v_source_base,
            v_projected + v_source_base + value_dim,
            v_cache + v_cache_base);
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

class HipQwen35SplitQGateKernelCache {
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
            if (!hiprtc_runtime().compile_qwen35_split_q_gate_kernel(arch, &code, &compile_error)) {
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
                    "ullm_qwen35_split_q_gate_f32_kernel",
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
        append_error(error, compile_errors.empty() ? "failed to build Qwen3.5 q/gate split HIP kernel"
                                                  : compile_errors);
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

HipQwen35SplitQGateKernelCache &hip_qwen35_split_q_gate_kernel_cache() {
    static HipQwen35SplitQGateKernelCache cache;
    return cache;
}

bool qwen35_split_q_gate_f32_hip_kernel(
    const ullm_runtime_buffer *projected_buffer,
    size_t q_heads,
    size_t head_dim,
    ullm_runtime_buffer *query_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = projected_buffer->hip_device_id;
    void *function = hip_qwen35_split_q_gate_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const size_t elements = q_heads * head_dim;
    constexpr unsigned int block_size = 256;
    const size_t grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/gate split element count exceeds HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    void *projected_ptr = projected_buffer->ptr;
    void *query_output_ptr = query_output_buffer->ptr;
    void *gate_output_ptr = gate_output_buffer->ptr;
    void *kernel_params[] = {
        &projected_ptr,
        &kernel_q_heads,
        &kernel_head_dim,
        &query_output_ptr,
        &gate_output_ptr,
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
            *error = "hipModuleLaunchKernel failed for Qwen3.5 q/gate split";
        }
        return false;
    }
    return true;
}

ullm_status qwen35_split_q_gate_f32_hip_staging(
    const ullm_runtime_buffer *projected_buffer,
    size_t q_heads,
    size_t head_dim,
    size_t projected_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *query_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_stream *stream) {
    const size_t projected_elements = q_heads * head_dim * 2;
    const size_t output_elements = q_heads * head_dim;
    std::vector<float> host_projected(projected_elements);
    std::vector<float> host_query(output_elements);
    std::vector<float> host_gate(output_elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = projected_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_projected.data(),
            projected_buffer->ptr,
            projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/gate split HIP input to host staging buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/gate split HIP input staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    qwen35_split_q_gate_f32_host(
        host_projected.data(),
        q_heads,
        head_dim,
        host_query.data(),
        host_gate.data());
    if (!hip_runtime().copy_async(
            query_output_buffer->ptr,
            host_query.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            gate_output_buffer->ptr,
            host_gate.data(),
            output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/gate split HIP outputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/gate split HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipQwen35QkNormRopeKernelCache {
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
            if (!hiprtc_runtime().compile_qwen35_qk_norm_rope_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_qwen35_qk_norm_rope_f32_kernel",
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
        append_error(error, compile_errors.empty()
                                ? "failed to build Qwen3.5 q/k norm RoPE HIP kernel"
                                : compile_errors);
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

HipQwen35QkNormRopeKernelCache &hip_qwen35_qk_norm_rope_kernel_cache() {
    static HipQwen35QkNormRopeKernelCache cache;
    return cache;
}

bool qwen35_qk_norm_rope_f32_hip_kernel(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_projected_buffer->hip_device_id;
    void *function = hip_qwen35_qk_norm_rope_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const size_t segments = q_heads + kv_heads;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/k norm RoPE segment count exceeds HIP grid limit";
        }
        return false;
    }
    constexpr unsigned int block_size = 256;
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_rotary_dim = static_cast<unsigned long long>(rotary_dim);
    unsigned long long kernel_position_offset = static_cast<unsigned long long>(position_offset);
    void *q_projected_ptr = q_projected_buffer->ptr;
    void *k_projected_ptr = k_projected_buffer->ptr;
    void *q_weight_ptr = q_weight_buffer->ptr;
    void *k_weight_ptr = k_weight_buffer->ptr;
    void *q_gate_output_ptr = q_gate_output_buffer->ptr;
    void *q_rope_output_ptr = q_rope_output_buffer->ptr;
    void *k_rope_output_ptr = k_rope_output_buffer->ptr;
    void *kernel_params[] = {
        &q_projected_ptr,
        &k_projected_ptr,
        &q_weight_ptr,
        &k_weight_ptr,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_rotary_dim,
        &kernel_position_offset,
        &rope_base,
        &epsilon,
        &q_gate_output_ptr,
        &q_rope_output_ptr,
        &k_rope_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(segments),
            block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for Qwen3.5 q/k norm RoPE";
        }
        return false;
    }
    return true;
}

ullm_status qwen35_qk_norm_rope_f32_hip_staging(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t q_projected_bytes,
    size_t k_projected_bytes,
    size_t weight_bytes,
    size_t q_output_bytes,
    size_t k_output_bytes,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream) {
    const size_t q_projected_elements = q_heads * head_dim * 2;
    const size_t q_output_elements = q_heads * head_dim;
    const size_t k_output_elements = kv_heads * head_dim;
    std::vector<float> host_q_projected(q_projected_elements);
    std::vector<float> host_k_projected(k_output_elements);
    std::vector<float> host_q_weight(head_dim);
    std::vector<float> host_k_weight(head_dim);
    std::vector<float> host_q_gate(q_output_elements);
    std::vector<float> host_q_rope(q_output_elements);
    std::vector<float> host_k_rope(k_output_elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_projected_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q_projected.data(),
            q_projected_buffer->ptr,
            q_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_projected.data(),
            k_projected_buffer->ptr,
            k_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_q_weight.data(),
            q_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_weight.data(),
            k_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/k norm RoPE HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    qwen35_qk_norm_rope_f32_host(
        host_q_projected.data(),
        host_k_projected.data(),
        host_q_weight.data(),
        host_k_weight.data(),
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        host_q_gate.data(),
        host_q_rope.data(),
        host_k_rope.data());
    if (!hip_runtime().copy_async(
            q_gate_output_buffer->ptr,
            host_q_gate.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_rope_output_buffer->ptr,
            host_q_rope.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            k_rope_output_buffer->ptr,
            host_k_rope.data(),
            k_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy Qwen3.5 q/k norm RoPE HIP outputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE HIP output staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

class HipQwen35QkNormRopePagedKvWriteKernelCache {
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
            if (!hiprtc_runtime().compile_qwen35_qk_norm_rope_paged_kv_write_kernel(
                    arch,
                    &code,
                    &compile_error)) {
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
                    "ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel",
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
        append_error(error, compile_errors.empty()
                                ? "failed to build Qwen3.5 q/k norm RoPE paged KV write HIP kernel"
                                : compile_errors);
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

HipQwen35QkNormRopePagedKvWriteKernelCache
    &hip_qwen35_qk_norm_rope_paged_kv_write_kernel_cache() {
    static HipQwen35QkNormRopePagedKvWriteKernelCache cache;
    return cache;
}

bool qwen35_qk_norm_rope_paged_kv_write_f32_hip_kernel(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *v_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = q_projected_buffer->hip_device_id;
    void *function =
        hip_qwen35_qk_norm_rope_paged_kv_write_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const size_t segments = q_heads + kv_heads + kv_heads;
    if (segments > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "Qwen3.5 q/k norm RoPE paged KV write segment count exceeds HIP grid limit";
        }
        return false;
    }
    constexpr unsigned int launch_block_size = 256;
    unsigned long long kernel_q_heads = static_cast<unsigned long long>(q_heads);
    unsigned long long kernel_kv_heads = static_cast<unsigned long long>(kv_heads);
    unsigned long long kernel_head_dim = static_cast<unsigned long long>(head_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_rotary_dim = static_cast<unsigned long long>(rotary_dim);
    unsigned long long kernel_position_offset = static_cast<unsigned long long>(position_offset);
    unsigned long long kernel_cache_position = static_cast<unsigned long long>(cache_position);
    unsigned long long kernel_block_size = static_cast<unsigned long long>(block_size);
    unsigned long long kernel_cache_blocks = static_cast<unsigned long long>(cache_blocks);
    void *q_projected_ptr = q_projected_buffer->ptr;
    void *k_projected_ptr = k_projected_buffer->ptr;
    void *v_projected_ptr = v_projected_buffer->ptr;
    void *q_weight_ptr = q_weight_buffer->ptr;
    void *k_weight_ptr = k_weight_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *q_gate_output_ptr = q_gate_output_buffer->ptr;
    void *q_rope_output_ptr = q_rope_output_buffer->ptr;
    void *k_cache_ptr = k_cache_buffer->ptr;
    void *v_cache_ptr = v_cache_buffer->ptr;
    void *kernel_params[] = {
        &q_projected_ptr,
        &k_projected_ptr,
        &v_projected_ptr,
        &q_weight_ptr,
        &k_weight_ptr,
        &block_table_ptr,
        &kernel_q_heads,
        &kernel_kv_heads,
        &kernel_head_dim,
        &kernel_value_dim,
        &kernel_rotary_dim,
        &kernel_position_offset,
        &rope_base,
        &epsilon,
        &kernel_cache_position,
        &kernel_block_size,
        &kernel_cache_blocks,
        &q_gate_output_ptr,
        &q_rope_output_ptr,
        &k_cache_ptr,
        &v_cache_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            static_cast<unsigned int>(segments),
            launch_block_size,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for Qwen3.5 q/k norm RoPE paged KV write";
        }
        return false;
    }
    return true;
}

ullm_status qwen35_qk_norm_rope_paged_kv_write_f32_hip_staging(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *v_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    size_t q_projected_bytes,
    size_t k_projected_bytes,
    size_t v_projected_bytes,
    size_t weight_bytes,
    size_t block_table_bytes,
    size_t q_output_bytes,
    size_t k_cache_bytes,
    size_t v_cache_bytes,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    const size_t q_projected_elements = q_heads * head_dim * 2;
    const size_t k_projected_elements = kv_heads * head_dim;
    const size_t v_projected_elements = kv_heads * value_dim;
    const size_t q_output_elements = q_heads * head_dim;
    const size_t weight_elements = head_dim;
    const size_t block_table_entries = block_table_bytes / sizeof(std::uint32_t);
    const size_t k_cache_elements = k_cache_bytes / sizeof(float);
    const size_t v_cache_elements = v_cache_bytes / sizeof(float);
    std::vector<float> host_q_projected(q_projected_elements);
    std::vector<float> host_k_projected(k_projected_elements);
    std::vector<float> host_v_projected(v_projected_elements);
    std::vector<float> host_q_weight(weight_elements);
    std::vector<float> host_k_weight(weight_elements);
    std::vector<std::uint32_t> host_block_table(block_table_entries);
    std::vector<float> host_q_gate(q_output_elements);
    std::vector<float> host_q_rope(q_output_elements);
    std::vector<float> host_k_cache(k_cache_elements);
    std::vector<float> host_v_cache(v_cache_elements);
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_projected_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_q_projected.data(),
            q_projected_buffer->ptr,
            q_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_projected.data(),
            k_projected_buffer->ptr,
            k_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_v_projected.data(),
            v_projected_buffer->ptr,
            v_projected_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_q_weight.data(),
            q_weight_buffer->ptr,
            weight_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_k_weight.data(),
            k_weight_buffer->ptr,
            weight_bytes,
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
        set_error("failed to copy Qwen3.5 q/k norm RoPE paged KV write HIP inputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE paged KV write input copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    qwen35_qk_norm_rope_paged_kv_write_f32_host(
        host_q_projected.data(),
        host_k_projected.data(),
        host_v_projected.data(),
        host_q_weight.data(),
        host_k_weight.data(),
        host_block_table.data(),
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        cache_position,
        block_size,
        host_q_gate.data(),
        host_q_rope.data(),
        host_k_cache.data(),
        host_v_cache.data());
    if (!hip_runtime().copy_async(
            q_gate_output_buffer->ptr,
            host_q_gate.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_rope_output_buffer->ptr,
            host_q_rope.data(),
            q_output_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
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
        set_error("failed to copy Qwen3.5 q/k norm RoPE paged KV write HIP outputs");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize Qwen3.5 q/k norm RoPE paged KV write output copies");
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
    const ullm_runtime_buffer *gate_buffer,
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
    const bool use_head_parallel_kernel =
        head_dim <= launch_block_size && value_dim <= launch_block_size;
    const size_t grid_size = use_head_parallel_kernel
                                 ? q_heads
                                 : (output_elements + launch_block_size - 1) / launch_block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "paged decode attention launch grid exceeds HIP grid limit";
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
    void *gate_ptr = gate_buffer == nullptr ? nullptr : gate_buffer->ptr;
    void *k_ptr = k_cache_buffer->ptr;
    void *v_ptr = v_cache_buffer->ptr;
    void *block_table_ptr = block_table_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {
        &q_ptr,
        &gate_ptr,
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
    const ullm_runtime_buffer *gate_buffer,
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
    size_t gate_bytes,
    size_t k_bytes,
    size_t v_bytes,
    size_t block_table_bytes,
    size_t output_bytes,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_gate(gate_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    std::vector<std::uint32_t> host_block_table(block_table_bytes / sizeof(std::uint32_t));
    std::vector<float> host_output(output_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = q_buffer->hip_device_id;

    bool copied = hip_runtime().copy_async(
            host_q.data(),
            q_buffer->ptr,
            q_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id);
    if (copied && gate_buffer != nullptr) {
        copied = hip_runtime().copy_async(
            host_gate.data(),
            gate_buffer->ptr,
            gate_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id);
    }
    if (!copied ||
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
    if (gate_buffer != nullptr) {
        for (size_t i = 0; i < host_output.size(); ++i) {
            const float gate_value = host_gate[i];
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
            host_output[i] *= sigmoid;
        }
    }
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

void linear_attn_qkv_prepare_f32_host(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    bool qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const size_t q_elements = key_heads * key_dim;
    const size_t k_elements = q_elements;
    const size_t v_elements = value_heads * value_dim;
    const size_t channels = q_elements + k_elements + v_elements;
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t kernel = 0; kernel + 1 < kernel_size; ++kernel) {
            conv_history[kernel * channels + channel] =
                conv_history[(kernel + 1) * channels + channel];
        }
        conv_history[(kernel_size - 1) * channels + channel] = qkv[channel];

        float sum = 0.0f;
        for (size_t kernel = 0; kernel < kernel_size; ++kernel) {
            sum += conv_history[kernel * channels + channel] *
                   conv_weight[channel * kernel_size + kernel];
        }
        const float sigmoid = 1.0f / (1.0f + std::exp(-sum));
        conv_output[channel] = sum * sigmoid;
    }

    for (size_t head = 0; head < key_heads; ++head) {
        const size_t q_source = head * key_dim;
        const size_t k_source = q_elements + head * key_dim;
        const size_t target = head * key_dim;
        float q_square_sum = 0.0f;
        float k_square_sum = 0.0f;
        for (size_t dim = 0; dim < key_dim; ++dim) {
            const float q_value = conv_output[q_source + dim];
            const float k_value = conv_output[k_source + dim];
            q_square_sum += q_value * q_value;
            k_square_sum += k_value * k_value;
        }
        const float q_norm = std::sqrt(q_square_sum + 1.0e-6f);
        const float k_norm = std::sqrt(k_square_sum + 1.0e-6f);
        for (size_t dim = 0; dim < key_dim; ++dim) {
            const float q_value = conv_output[q_source + dim];
            const float k_value = conv_output[k_source + dim];
            q_output[target + dim] = qk_l2_norm ? (q_value / q_norm) * q_scale : q_value * q_scale;
            k_output[target + dim] = qk_l2_norm ? k_value / k_norm : k_value;
        }
    }

    const size_t v_source = q_elements + k_elements;
    std::memcpy(v_output, conv_output + v_source, v_elements * sizeof(float));
}

void linear_attn_qkv_prepare_batch_f32_host(
    const float *qkv,
    const float *conv_weight,
    float *conv_history,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    bool qk_l2_norm,
    float *conv_output,
    float *q_output,
    float *k_output,
    float *v_output) {
    const size_t q_elements = key_heads * key_dim;
    const size_t v_elements = value_heads * value_dim;
    const size_t channels = q_elements * 2 + v_elements;
    for (size_t token = 0; token < sequence_len; ++token) {
        linear_attn_qkv_prepare_f32_host(
            qkv + token * channels,
            conv_weight,
            conv_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            qk_l2_norm,
            conv_output + token * channels,
            q_output + token * q_elements,
            k_output + token * q_elements,
            v_output + token * v_elements);
    }
}

class HipLinearAttnQkvPrepareKernelCache {
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
            if (!hiprtc_runtime().compile_linear_attn_qkv_prepare_kernel(arch, &code, &compile_error)) {
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
                    "ullm_linear_attn_qkv_prepare_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(&compile_errors, "hipModuleGetFunction qkv prepare failed for " + arch);
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
            compile_errors.empty() ? "failed to build linear attention qkv prepare HIP kernel"
                                   : compile_errors);
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

HipLinearAttnQkvPrepareKernelCache &hip_linear_attn_qkv_prepare_kernel_cache() {
    static HipLinearAttnQkvPrepareKernelCache cache;
    return cache;
}

class HipLinearAttnQkvPrepareBatchKernelCache {
public:
    bool functions_for_device(
        int device_id,
        void **prepare_function,
        void **update_history_function,
        std::string *error) {
        if (prepare_function == nullptr || update_history_function == nullptr) {
            append_error(error, "linear attention qkv prepare batch received null function output");
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            *prepare_function = found->second->prepare_function;
            *update_history_function = found->second->update_history_function;
            return true;
        }

        const std::vector<std::string> candidates = hip_arch_candidates(device_id);
        if (candidates.empty()) {
            append_error(error, "unable to infer HIP offload architecture for device");
            return false;
        }

        std::string compile_errors;
        for (const std::string &arch : candidates) {
            std::vector<char> code;
            std::string compile_error;
            if (!hiprtc_runtime().compile_linear_attn_qkv_prepare_batch_kernel(arch, &code, &compile_error)) {
                append_error(&compile_errors, compile_error);
                continue;
            }

            void *module = nullptr;
            if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
                append_error(&compile_errors, "hipModuleLoadData failed for " + arch);
                continue;
            }
            void *prepare = nullptr;
            if (!hip_runtime().module_get_function(
                    &prepare,
                    module,
                    "ullm_linear_attn_qkv_prepare_batch_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(
                    &compile_errors,
                    "hipModuleGetFunction qkv prepare batch failed for " + arch);
                continue;
            }
            void *update_history = nullptr;
            if (!hip_runtime().module_get_function(
                    &update_history,
                    module,
                    "ullm_linear_attn_qkv_prepare_batch_update_history_f32_kernel",
                    device_id)) {
                hip_runtime().module_unload(module, device_id);
                append_error(
                    &compile_errors,
                    "hipModuleGetFunction qkv prepare batch history update failed for " + arch);
                continue;
            }

            auto loaded = std::make_unique<LoadedModule>();
            loaded->module = module;
            loaded->prepare_function = prepare;
            loaded->update_history_function = update_history;
            loaded->arch = arch;
            *prepare_function = loaded->prepare_function;
            *update_history_function = loaded->update_history_function;
            modules_.emplace(device_id, std::move(loaded));
            return true;
        }
        append_error(
            error,
            compile_errors.empty() ? "failed to build linear attention qkv prepare batch HIP kernel"
                                   : compile_errors);
        return false;
    }

private:
    struct LoadedModule {
        void *module = nullptr;
        void *prepare_function = nullptr;
        void *update_history_function = nullptr;
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

HipLinearAttnQkvPrepareBatchKernelCache &hip_linear_attn_qkv_prepare_batch_kernel_cache() {
    static HipLinearAttnQkvPrepareBatchKernelCache cache;
    return cache;
}

bool linear_attn_qkv_prepare_f32_hip_kernel(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    bool qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = qkv_buffer->hip_device_id;
    void *function = hip_linear_attn_qkv_prepare_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t v_elements = value_heads * value_dim;
    const size_t v_grid_size = (v_elements + block_size - 1) / block_size;
    const size_t grid_size = key_heads * 2 + v_grid_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention qkv prepare grid size exceeds HIP limit";
        }
        return false;
    }

    void *qkv_ptr = qkv_buffer->ptr;
    void *conv_weight_ptr = conv_weight_buffer->ptr;
    void *conv_history_ptr = conv_history_buffer->ptr;
    unsigned long long kernel_key_heads = static_cast<unsigned long long>(key_heads);
    unsigned long long kernel_value_heads = static_cast<unsigned long long>(value_heads);
    unsigned long long kernel_key_dim = static_cast<unsigned long long>(key_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_size_arg = static_cast<unsigned long long>(kernel_size);
    int kernel_qk_l2_norm = qk_l2_norm ? 1 : 0;
    void *conv_output_ptr = conv_output_buffer->ptr;
    void *q_output_ptr = q_output_buffer->ptr;
    void *k_output_ptr = k_output_buffer->ptr;
    void *v_output_ptr = v_output_buffer->ptr;
    void *kernel_params[] = {
        &qkv_ptr,
        &conv_weight_ptr,
        &conv_history_ptr,
        &kernel_key_heads,
        &kernel_value_heads,
        &kernel_key_dim,
        &kernel_value_dim,
        &kernel_size_arg,
        &q_scale,
        &kernel_qk_l2_norm,
        &conv_output_ptr,
        &q_output_ptr,
        &k_output_ptr,
        &v_output_ptr,
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
            *error = "hipModuleLaunchKernel failed for linear attention qkv prepare";
        }
        return false;
    }
    return true;
}

bool linear_attn_qkv_prepare_batch_f32_hip_kernel(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    bool qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = qkv_buffer->hip_device_id;
    void *prepare_function = nullptr;
    void *update_history_function = nullptr;
    if (!hip_linear_attn_qkv_prepare_batch_kernel_cache().functions_for_device(
            device_id,
            &prepare_function,
            &update_history_function,
            error)) {
        return false;
    }

    constexpr unsigned int block_size = 256;
    const size_t q_elements = key_heads * key_dim;
    const size_t v_elements = value_heads * value_dim;
    const size_t channels = q_elements * 2 + v_elements;
    const size_t v_grid_size = (v_elements + block_size - 1) / block_size;
    const size_t prepare_grid_x = key_heads * 2 + v_grid_size;
    const size_t update_grid_x = (channels + block_size - 1) / block_size;
    if (prepare_grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        update_grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        sequence_len > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "linear attention qkv prepare batch grid size exceeds HIP limit";
        }
        return false;
    }

    void *qkv_ptr = qkv_buffer->ptr;
    void *conv_weight_ptr = conv_weight_buffer->ptr;
    void *conv_history_ptr = conv_history_buffer->ptr;
    unsigned long long kernel_key_heads = static_cast<unsigned long long>(key_heads);
    unsigned long long kernel_value_heads = static_cast<unsigned long long>(value_heads);
    unsigned long long kernel_key_dim = static_cast<unsigned long long>(key_dim);
    unsigned long long kernel_value_dim = static_cast<unsigned long long>(value_dim);
    unsigned long long kernel_size_arg = static_cast<unsigned long long>(kernel_size);
    unsigned long long kernel_sequence_len = static_cast<unsigned long long>(sequence_len);
    int kernel_qk_l2_norm = qk_l2_norm ? 1 : 0;
    void *conv_output_ptr = conv_output_buffer->ptr;
    void *q_output_ptr = q_output_buffer->ptr;
    void *k_output_ptr = k_output_buffer->ptr;
    void *v_output_ptr = v_output_buffer->ptr;
    void *prepare_params[] = {
        &qkv_ptr,
        &conv_weight_ptr,
        &conv_history_ptr,
        &kernel_key_heads,
        &kernel_value_heads,
        &kernel_key_dim,
        &kernel_value_dim,
        &kernel_size_arg,
        &kernel_sequence_len,
        &q_scale,
        &kernel_qk_l2_norm,
        &conv_output_ptr,
        &q_output_ptr,
        &k_output_ptr,
        &v_output_ptr,
    };
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel_2d(
            prepare_function,
            static_cast<unsigned int>(prepare_grid_x),
            static_cast<unsigned int>(sequence_len),
            block_size,
            prepare_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for linear attention qkv prepare batch";
        }
        return false;
    }

    unsigned long long kernel_channels = static_cast<unsigned long long>(channels);
    void *update_params[] = {
        &qkv_ptr,
        &conv_history_ptr,
        &kernel_sequence_len,
        &kernel_channels,
        &kernel_size_arg,
    };
    if (!hip_runtime().module_launch_kernel(
            update_history_function,
            static_cast<unsigned int>(update_grid_x),
            block_size,
            update_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for linear attention qkv prepare batch history update";
        }
        return false;
    }
    return true;
}

ullm_status linear_attn_qkv_prepare_f32_hip_staging(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    bool qk_l2_norm,
    size_t qkv_bytes,
    size_t conv_history_bytes,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_qkv(qkv_bytes / sizeof(float));
    std::vector<float> host_conv_weight(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_history(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_output(qkv_bytes / sizeof(float));
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = qkv_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_qkv.data(),
            qkv_buffer->ptr,
            qkv_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_weight.data(),
            conv_weight_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_history.data(),
            conv_history_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    linear_attn_qkv_prepare_f32_host(
        host_qkv.data(),
        host_conv_weight.data(),
        host_conv_history.data(),
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        q_scale,
        qk_l2_norm,
        host_conv_output.data(),
        host_q.data(),
        host_k.data(),
        host_v.data());

    if (!hip_runtime().copy_async(
            conv_history_buffer->ptr,
            host_conv_history.data(),
            conv_history_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            conv_output_buffer->ptr,
            host_conv_output.data(),
            qkv_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_output_buffer->ptr,
            host_q.data(),
            q_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            k_output_buffer->ptr,
            host_k.data(),
            k_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            v_output_buffer->ptr,
            host_v.data(),
            v_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status linear_attn_qkv_prepare_batch_f32_hip_staging(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    bool qk_l2_norm,
    size_t qkv_bytes,
    size_t conv_history_bytes,
    size_t q_bytes,
    size_t k_bytes,
    size_t v_bytes,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    std::vector<float> host_qkv(qkv_bytes / sizeof(float));
    std::vector<float> host_conv_weight(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_history(conv_history_bytes / sizeof(float));
    std::vector<float> host_conv_output(qkv_bytes / sizeof(float));
    std::vector<float> host_q(q_bytes / sizeof(float));
    std::vector<float> host_k(k_bytes / sizeof(float));
    std::vector<float> host_v(v_bytes / sizeof(float));
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    const int device_id = qkv_buffer->hip_device_id;

    if (!hip_runtime().copy_async(
            host_qkv.data(),
            qkv_buffer->ptr,
            qkv_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_weight.data(),
            conv_weight_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_conv_history.data(),
            conv_history_buffer->ptr,
            conv_history_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare batch HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare batch HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    linear_attn_qkv_prepare_batch_f32_host(
        host_qkv.data(),
        host_conv_weight.data(),
        host_conv_history.data(),
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        sequence_len,
        q_scale,
        qk_l2_norm,
        host_conv_output.data(),
        host_q.data(),
        host_k.data(),
        host_v.data());

    if (!hip_runtime().copy_async(
            conv_history_buffer->ptr,
            host_conv_history.data(),
            conv_history_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            conv_output_buffer->ptr,
            host_conv_output.data(),
            qkv_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            q_output_buffer->ptr,
            host_q.data(),
            q_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            k_output_buffer->ptr,
            host_k.data(),
            k_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            v_output_buffer->ptr,
            host_v.data(),
            v_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy linear attention qkv prepare batch outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize linear attention qkv prepare batch HIP output staging copy");
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
    unsigned int grid_size = static_cast<unsigned int>(value_heads);
    unsigned int block_size = 1;
    if (sequence_len == 1) {
        if (value_dim != 0 &&
            value_heads > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) / value_dim) {
            if (error != nullptr) {
                *error = "linear attention recurrent fast decode grid exceeds HIP grid limit";
            }
            return false;
        }
        grid_size = static_cast<unsigned int>(value_heads * value_dim);
        const unsigned int default_decode_block = key_dim <= 128 ? 128u : 256u;
        block_size = block_size_from_env(
            "ULLM_LINEAR_ATTN_RECURRENT_DECODE_BLOCK",
            default_decode_block);
    }

    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            grid_size,
            block_size,
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

ullm_status ullm_runtime_aq4_row_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 row received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 row scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 row group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 row rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_index >= rows) {
        set_error("AQ4 row index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 row tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 row buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 row stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 row matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 row byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_output_bytes = cols * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 row index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 row scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 row scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 row scale override buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 row output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 row codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 row requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_row_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                row_scales,
                scale_count,
                group_size,
                tensor_scale,
                row_scale_count,
                rows,
                cols,
                row_index,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    const HipAq4LaunchResult launch_result = aq4_row_f32_hip_kernel(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        row_index,
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

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_ROW_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "AQ4 row HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_output(cols);
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
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            required_scale_value_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr && !hip_runtime().copy_async(
            host_row_scales.data(),
            row_scale_buffer->ptr,
            required_row_scale_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id))) {
        set_error("failed to copy AQ4 row HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 row HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_row_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            row_index,
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
        set_error("failed to copy AQ4 row output to HIP buffer");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 row HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_aq4_matvec_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_matvec_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input,
                row_scales,
                row_scale_count,
                rows,
                cols,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "AQ4 matvec HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_f32_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        input_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        required_index_bytes,
        groups,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_batch_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec batch received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec batch scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec batch group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0 || batch_count == 0) {
        set_error("AQ4 matvec batch rows, cols, and batch count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec batch tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec batch buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec batch stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec batch matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / batch_count ||
        rows > max_size / batch_count ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_elements = cols * batch_count;
    const size_t output_elements = rows * batch_count;
    if (input_elements > max_size / sizeof(float) ||
        output_elements > max_size / sizeof(float)) {
        set_error("AQ4 matvec batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = input_elements * sizeof(float);
    const size_t required_output_bytes = output_elements * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec batch index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec batch scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec batch scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec batch input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec batch row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec batch output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec batch codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec batch requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_matvec_batch_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input,
                row_scales,
                row_scale_count,
                rows,
                cols,
                batch_count,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_batch_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            batch_count,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec batch HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_batch_f32_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        input_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        batch_count,
        required_index_bytes,
        groups,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_top1_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr ||
        partial_values_buffer == nullptr || partial_indices_buffer == nullptr) {
        set_error("AQ4 matvec top1 received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec top1 scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec top1 group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0 || partial_count == 0) {
        set_error("AQ4 matvec top1 rows, cols, and partial_count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        set_error("AQ4 matvec top1 rows exceed uint32 index range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t expected_partials = aq4_matvec_top1_partial_count(rows);
    if (partial_count != expected_partials) {
        set_error("AQ4 matvec top1 partial_count does not match expected row block count");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec top1 tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, partial_values_buffer) ||
        !buffers_share_backend(index_buffer, partial_indices_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec top1 buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(partial_values_buffer, stream) ||
        !stream_matches_buffer(partial_indices_buffer, stream)) {
        set_error("AQ4 matvec top1 stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec top1 matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(uint32_t)) {
        set_error("AQ4 matvec top1 byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    const size_t partial_values_bytes = partial_count * sizeof(float);
    const size_t partial_indices_bytes = partial_count * sizeof(uint32_t);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec top1 index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec top1 scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec top1 scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec top1 input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec top1 row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_values_buffer->bytes < partial_values_bytes) {
        set_error("AQ4 matvec top1 partial values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_indices_buffer->bytes < partial_indices_bytes) {
        set_error("AQ4 matvec top1 partial indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec top1 codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec top1 requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_top1_f32_host(
                static_cast<const std::uint8_t *>(index_buffer->ptr),
                static_cast<const std::uint8_t *>(scale_buffer->ptr),
                static_cast<const float *>(codebook_buffer->ptr),
                static_cast<const float *>(scale_values_buffer->ptr),
                scale_count,
                group_size,
                tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(row_scale_buffer->ptr),
                row_scale_count,
                rows,
                cols,
                partial_count,
                static_cast<float *>(partial_values_buffer->ptr),
                static_cast<uint32_t *>(partial_indices_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_top1_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            partial_values_buffer,
            partial_indices_buffer,
            partial_count,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_TOP1_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec top1 HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_scale_values(scale_count);
    std::vector<float> host_input(cols);
    std::vector<float> host_row_scales(row_scale_count);
    std::vector<float> host_partial_values(partial_count);
    std::vector<uint32_t> host_partial_indices(partial_count);
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
            device_id) ||
        !hip_runtime().copy_async(
            host_scale_values.data(),
            scale_values_buffer->ptr,
            required_scale_value_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            host_input.data(),
            input_buffer->ptr,
            required_input_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
            hip_stream,
            device_id) ||
        (row_scale_buffer != nullptr &&
         !hip_runtime().copy_async(
             host_row_scales.data(),
             row_scale_buffer->ptr,
             required_row_scale_bytes,
             HIP_MEMCPY_DEVICE_TO_HOST,
             hip_stream,
             device_id))) {
        set_error("failed to copy AQ4 matvec top1 HIP inputs to host staging buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec top1 HIP input staging copies");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!aq4_matvec_top1_f32_host(
            host_indices.data(),
            host_scale_indices.data(),
            host_codebook.data(),
            host_scale_values.data(),
            scale_count,
            group_size,
            tensor_scale,
            host_input.data(),
            row_scale_buffer == nullptr ? nullptr : host_row_scales.data(),
            row_scale_count,
            rows,
            cols,
            partial_count,
            host_partial_values.data(),
            host_partial_indices.data())) {
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!hip_runtime().copy_async(
            partial_values_buffer->ptr,
            host_partial_values.data(),
            partial_values_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id) ||
        !hip_runtime().copy_async(
            partial_indices_buffer->ptr,
            host_partial_indices.data(),
            partial_indices_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
            hip_stream,
            device_id)) {
        set_error("failed to copy AQ4 matvec top1 outputs to HIP buffers");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    if (!synchronize_hip_staging(stream, device_id)) {
        set_error("failed to synchronize AQ4 matvec top1 HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    set_error("");
    return ULLM_STATUS_OK;
}

ullm_status ullm_runtime_aq4_matvec_add_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (index_buffer == nullptr || scale_buffer == nullptr || codebook_buffer == nullptr ||
        scale_values_buffer == nullptr || input_buffer == nullptr ||
        residual_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec add received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_count == 0) {
        set_error("AQ4 matvec add scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (group_size == 0) {
        set_error("AQ4 matvec add group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 matvec add rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
        set_error("AQ4 matvec add tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(index_buffer, scale_buffer) ||
        !buffers_share_backend(index_buffer, codebook_buffer) ||
        !buffers_share_backend(index_buffer, scale_values_buffer) ||
        !buffers_share_backend(index_buffer, input_buffer) ||
        !buffers_share_backend(index_buffer, residual_buffer) ||
        !buffers_share_backend(index_buffer, output_buffer) ||
        (row_scale_buffer != nullptr && !buffers_share_backend(index_buffer, row_scale_buffer))) {
        set_error("AQ4 matvec add buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec add stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec add matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t required_index_bytes = elements / 2 + (elements % 2);
    const size_t groups = elements / group_size + (elements % group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float) ||
        scale_count > max_size / sizeof(float) ||
        row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec add byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    const size_t required_scale_value_bytes = scale_count * sizeof(float);
    const size_t required_row_scale_bytes = row_scale_count * sizeof(float);
    if (index_buffer->bytes < required_index_bytes) {
        set_error("AQ4 matvec add index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_buffer->bytes < groups) {
        set_error("AQ4 matvec add scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (scale_values_buffer->bytes < required_scale_value_bytes) {
        set_error("AQ4 matvec add scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec add input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (residual_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec add residual buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_scale_buffer != nullptr && row_scale_buffer->bytes < required_row_scale_bytes) {
        set_error("AQ4 matvec add row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec add output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec add codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t codebook_entries = codebook_buffer->bytes / sizeof(float);
    if (codebook_entries < 16) {
        set_error("AQ4 matvec add requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (index_buffer->backend == BackendKind::Cpu) {
        const auto *indices = static_cast<const std::uint8_t *>(index_buffer->ptr);
        const auto *scale_indices = static_cast<const std::uint8_t *>(scale_buffer->ptr);
        const auto *codebook = static_cast<const float *>(codebook_buffer->ptr);
        const auto *scale_values = static_cast<const float *>(scale_values_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        const auto *residual = static_cast<const float *>(residual_buffer->ptr);
        const auto *row_scales =
            row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(row_scale_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        if (!aq4_matvec_add_f32_host(
                indices,
                scale_indices,
                codebook,
                scale_values,
                scale_count,
                group_size,
                tensor_scale,
                input,
                residual,
                row_scales,
                row_scale_count,
                rows,
                cols,
                output)) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_add_f32_hip_kernel(
            index_buffer,
            scale_buffer,
            codebook_buffer,
            scale_values_buffer,
            input_buffer,
            residual_buffer,
            row_scale_buffer,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec add HIP kernel is unavailable" :
                                       hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_add_f32_hip_staging(
        index_buffer,
        scale_buffer,
        codebook_buffer,
        scale_values_buffer,
        input_buffer,
        residual_buffer,
        row_scale_buffer,
        scale_count,
        group_size,
        tensor_scale,
        row_scale_count,
        rows,
        cols,
        required_index_bytes,
        groups,
        codebook_entries,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_pair_f32(
    const ullm_runtime_buffer *left_index_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    const ullm_runtime_buffer *left_codebook_buffer,
    const ullm_runtime_buffer *left_scale_values_buffer,
    const ullm_runtime_buffer *left_row_scale_buffer,
    size_t left_scale_count,
    size_t left_group_size,
    float left_tensor_scale,
    size_t left_row_scale_count,
    const ullm_runtime_buffer *right_index_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    const ullm_runtime_buffer *right_codebook_buffer,
    const ullm_runtime_buffer *right_scale_values_buffer,
    const ullm_runtime_buffer *right_row_scale_buffer,
    size_t right_scale_count,
    size_t right_group_size,
    float right_tensor_scale,
    size_t right_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream) {
    if (left_index_buffer == nullptr || left_scale_buffer == nullptr ||
        left_codebook_buffer == nullptr || left_scale_values_buffer == nullptr ||
        right_index_buffer == nullptr || right_scale_buffer == nullptr ||
        right_codebook_buffer == nullptr || right_scale_values_buffer == nullptr ||
        input_buffer == nullptr || left_output_buffer == nullptr || right_output_buffer == nullptr) {
        set_error("AQ4 matvec pair received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_scale_count == 0 || right_scale_count == 0) {
        set_error("AQ4 matvec pair scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_group_size == 0 || right_group_size == 0) {
        set_error("AQ4 matvec pair group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_rows == 0 || right_rows == 0 || cols == 0) {
        set_error("AQ4 matvec pair rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(left_tensor_scale) || left_tensor_scale <= 0.0f ||
        !std::isfinite(right_tensor_scale) || right_tensor_scale <= 0.0f) {
        set_error("AQ4 matvec pair tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(left_index_buffer, left_scale_buffer) ||
        !buffers_share_backend(left_index_buffer, left_codebook_buffer) ||
        !buffers_share_backend(left_index_buffer, left_scale_values_buffer) ||
        !buffers_share_backend(left_index_buffer, right_index_buffer) ||
        !buffers_share_backend(left_index_buffer, right_scale_buffer) ||
        !buffers_share_backend(left_index_buffer, right_codebook_buffer) ||
        !buffers_share_backend(left_index_buffer, right_scale_values_buffer) ||
        !buffers_share_backend(left_index_buffer, input_buffer) ||
        !buffers_share_backend(left_index_buffer, left_output_buffer) ||
        !buffers_share_backend(left_index_buffer, right_output_buffer) ||
        (left_row_scale_buffer != nullptr &&
         !buffers_share_backend(left_index_buffer, left_row_scale_buffer)) ||
        (right_row_scale_buffer != nullptr &&
         !buffers_share_backend(left_index_buffer, right_row_scale_buffer))) {
        set_error("AQ4 matvec pair buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(left_output_buffer, stream) ||
        !stream_matches_buffer(right_output_buffer, stream)) {
        set_error("AQ4 matvec pair stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / left_rows || cols > max_size / right_rows ||
        left_rows > max_size - right_rows) {
        set_error("AQ4 matvec pair matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t left_elements = left_rows * cols;
    const size_t right_elements = right_rows * cols;
    const size_t left_required_index_bytes = left_elements / 2 + (left_elements % 2);
    const size_t right_required_index_bytes = right_elements / 2 + (right_elements % 2);
    const size_t left_groups =
        left_elements / left_group_size + (left_elements % left_group_size == 0 ? 0 : 1);
    const size_t right_groups =
        right_elements / right_group_size + (right_elements % right_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        left_rows > max_size / sizeof(float) ||
        right_rows > max_size / sizeof(float) ||
        left_scale_count > max_size / sizeof(float) ||
        right_scale_count > max_size / sizeof(float) ||
        left_row_scale_count > max_size / sizeof(float) ||
        right_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec pair byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t left_required_output_bytes = left_rows * sizeof(float);
    const size_t right_required_output_bytes = right_rows * sizeof(float);
    const size_t left_required_scale_value_bytes = left_scale_count * sizeof(float);
    const size_t right_required_scale_value_bytes = right_scale_count * sizeof(float);
    const size_t left_required_row_scale_bytes = left_row_scale_count * sizeof(float);
    const size_t right_required_row_scale_bytes = right_row_scale_count * sizeof(float);
    if (left_index_buffer->bytes < left_required_index_bytes ||
        right_index_buffer->bytes < right_required_index_bytes) {
        set_error("AQ4 matvec pair index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_scale_buffer->bytes < left_groups || right_scale_buffer->bytes < right_groups) {
        set_error("AQ4 matvec pair scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_scale_values_buffer->bytes < left_required_scale_value_bytes ||
        right_scale_values_buffer->bytes < right_required_scale_value_bytes) {
        set_error("AQ4 matvec pair scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec pair input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((left_row_scale_buffer != nullptr &&
         left_row_scale_buffer->bytes < left_required_row_scale_bytes) ||
        (right_row_scale_buffer != nullptr &&
         right_row_scale_buffer->bytes < right_required_row_scale_bytes)) {
        set_error("AQ4 matvec pair row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_output_buffer->bytes < left_required_output_bytes ||
        right_output_buffer->bytes < right_required_output_bytes) {
        set_error("AQ4 matvec pair output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (left_codebook_buffer->bytes % sizeof(float) != 0 ||
        right_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec pair codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t left_codebook_entries = left_codebook_buffer->bytes / sizeof(float);
    const size_t right_codebook_entries = right_codebook_buffer->bytes / sizeof(float);
    if (left_codebook_entries < 16 || right_codebook_entries < 16) {
        set_error("AQ4 matvec pair requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (left_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(left_index_buffer->ptr),
                static_cast<const std::uint8_t *>(left_scale_buffer->ptr),
                static_cast<const float *>(left_codebook_buffer->ptr),
                static_cast<const float *>(left_scale_values_buffer->ptr),
                left_scale_count,
                left_group_size,
                left_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                left_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(left_row_scale_buffer->ptr),
                left_row_scale_count,
                left_rows,
                cols,
                static_cast<float *>(left_output_buffer->ptr)) ||
            !aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(right_index_buffer->ptr),
                static_cast<const std::uint8_t *>(right_scale_buffer->ptr),
                static_cast<const float *>(right_codebook_buffer->ptr),
                static_cast<const float *>(right_scale_values_buffer->ptr),
                right_scale_count,
                right_group_size,
                right_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                right_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(right_row_scale_buffer->ptr),
                right_row_scale_count,
                right_rows,
                cols,
                static_cast<float *>(right_output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_pair_f32_hip_kernel(
            left_index_buffer,
            left_scale_buffer,
            left_codebook_buffer,
            left_scale_values_buffer,
            left_row_scale_buffer,
            left_scale_count,
            left_group_size,
            left_tensor_scale,
            left_row_scale_count,
            right_index_buffer,
            right_scale_buffer,
            right_codebook_buffer,
            right_scale_values_buffer,
            right_row_scale_buffer,
            right_scale_count,
            right_group_size,
            right_tensor_scale,
            right_row_scale_count,
            input_buffer,
            left_rows,
            right_rows,
            cols,
            left_output_buffer,
            right_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    set_error(
        hip_kernel_error.empty() ? "AQ4 matvec pair HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_aq4_matvec_triple_f32(
    const ullm_runtime_buffer *first_index_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    const ullm_runtime_buffer *first_codebook_buffer,
    const ullm_runtime_buffer *first_scale_values_buffer,
    const ullm_runtime_buffer *first_row_scale_buffer,
    size_t first_scale_count,
    size_t first_group_size,
    float first_tensor_scale,
    size_t first_row_scale_count,
    const ullm_runtime_buffer *second_index_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    const ullm_runtime_buffer *second_codebook_buffer,
    const ullm_runtime_buffer *second_scale_values_buffer,
    const ullm_runtime_buffer *second_row_scale_buffer,
    size_t second_scale_count,
    size_t second_group_size,
    float second_tensor_scale,
    size_t second_row_scale_count,
    const ullm_runtime_buffer *third_index_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    const ullm_runtime_buffer *third_codebook_buffer,
    const ullm_runtime_buffer *third_scale_values_buffer,
    const ullm_runtime_buffer *third_row_scale_buffer,
    size_t third_scale_count,
    size_t third_group_size,
    float third_tensor_scale,
    size_t third_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream) {
    struct Matrix {
        const ullm_runtime_buffer *index = nullptr;
        const ullm_runtime_buffer *scale = nullptr;
        const ullm_runtime_buffer *codebook = nullptr;
        const ullm_runtime_buffer *scale_values = nullptr;
        const ullm_runtime_buffer *row_scale = nullptr;
        size_t scale_count = 0;
        size_t group_size = 0;
        float tensor_scale = 0.0f;
        size_t row_scale_count = 0;
        size_t rows = 0;
        ullm_runtime_buffer *output = nullptr;
        const char *label = "";
    };
    std::array<Matrix, 3> matrices{{
        {first_index_buffer,
         first_scale_buffer,
         first_codebook_buffer,
         first_scale_values_buffer,
         first_row_scale_buffer,
         first_scale_count,
         first_group_size,
         first_tensor_scale,
         first_row_scale_count,
         first_rows,
         first_output_buffer,
         "first"},
        {second_index_buffer,
         second_scale_buffer,
         second_codebook_buffer,
         second_scale_values_buffer,
         second_row_scale_buffer,
         second_scale_count,
         second_group_size,
         second_tensor_scale,
         second_row_scale_count,
         second_rows,
         second_output_buffer,
         "second"},
        {third_index_buffer,
         third_scale_buffer,
         third_codebook_buffer,
         third_scale_values_buffer,
         third_row_scale_buffer,
         third_scale_count,
         third_group_size,
         third_tensor_scale,
         third_row_scale_count,
         third_rows,
         third_output_buffer,
         "third"},
    }};
    if (input_buffer == nullptr) {
        set_error("AQ4 matvec triple received a null input pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    for (const Matrix &matrix : matrices) {
        if (matrix.index == nullptr || matrix.scale == nullptr || matrix.codebook == nullptr ||
            matrix.scale_values == nullptr || matrix.output == nullptr) {
            set_error("AQ4 matvec triple received a null pointer");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.scale_count == 0) {
            set_error("AQ4 matvec triple scale table is empty");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.group_size == 0) {
            set_error("AQ4 matvec triple group size must be greater than zero");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.rows == 0 || cols == 0) {
            set_error("AQ4 matvec triple rows and cols must be greater than zero");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (!std::isfinite(matrix.tensor_scale) || matrix.tensor_scale <= 0.0f) {
            set_error("AQ4 matvec triple tensor scale must be finite and greater than zero");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }

    const ullm_runtime_buffer *base = first_index_buffer;
    for (const Matrix &matrix : matrices) {
        if (!buffers_share_backend(base, matrix.index) ||
            !buffers_share_backend(base, matrix.scale) ||
            !buffers_share_backend(base, matrix.codebook) ||
            !buffers_share_backend(base, matrix.scale_values) ||
            !buffers_share_backend(base, matrix.output) ||
            (matrix.row_scale != nullptr && !buffers_share_backend(base, matrix.row_scale))) {
            set_error("AQ4 matvec triple buffers belong to different backends or devices");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (!stream_matches_buffer(matrix.output, stream)) {
            set_error("AQ4 matvec triple stream belongs to a different backend or device");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!buffers_share_backend(base, input_buffer)) {
        set_error("AQ4 matvec triple buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (first_rows > max_size - second_rows ||
        first_rows + second_rows > max_size - third_rows) {
        set_error("AQ4 matvec triple matrix row count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    std::array<size_t, 3> elements{};
    std::array<size_t, 3> required_index_bytes{};
    std::array<size_t, 3> groups{};
    std::array<size_t, 3> required_output_bytes{};
    std::array<size_t, 3> required_scale_value_bytes{};
    std::array<size_t, 3> required_row_scale_bytes{};
    for (size_t index = 0; index < matrices.size(); ++index) {
        const Matrix &matrix = matrices[index];
        if (cols > max_size / matrix.rows ||
            matrix.rows > max_size / sizeof(float) ||
            matrix.scale_count > max_size / sizeof(float) ||
            matrix.row_scale_count > max_size / sizeof(float)) {
            set_error("AQ4 matvec triple byte size overflows");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        elements[index] = matrix.rows * cols;
        required_index_bytes[index] = elements[index] / 2 + (elements[index] % 2);
        groups[index] =
            elements[index] / matrix.group_size +
            (elements[index] % matrix.group_size == 0 ? 0 : 1);
        required_output_bytes[index] = matrix.rows * sizeof(float);
        required_scale_value_bytes[index] = matrix.scale_count * sizeof(float);
        required_row_scale_bytes[index] = matrix.row_scale_count * sizeof(float);
    }
    if (cols > max_size / sizeof(float)) {
        set_error("AQ4 matvec triple input byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec triple input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < matrices.size(); ++index) {
        const Matrix &matrix = matrices[index];
        if (matrix.index->bytes < required_index_bytes[index]) {
            set_error("AQ4 matvec triple index buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.scale->bytes < groups[index]) {
            set_error("AQ4 matvec triple scale index buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.scale_values->bytes < required_scale_value_bytes[index]) {
            set_error("AQ4 matvec triple scale value buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.row_scale != nullptr &&
            matrix.row_scale->bytes < required_row_scale_bytes[index]) {
            set_error("AQ4 matvec triple row scale buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.output->bytes < required_output_bytes[index]) {
            set_error("AQ4 matvec triple output buffer is too small");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        if (matrix.codebook->bytes % sizeof(float) != 0 ||
            matrix.codebook->bytes / sizeof(float) < 16) {
            set_error("AQ4 matvec triple requires at least 16 f32 codebook entries");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
    }

    if (base->backend == BackendKind::Cpu) {
        for (const Matrix &matrix : matrices) {
            if (!aq4_matvec_f32_host(
                    static_cast<const std::uint8_t *>(matrix.index->ptr),
                    static_cast<const std::uint8_t *>(matrix.scale->ptr),
                    static_cast<const float *>(matrix.codebook->ptr),
                    static_cast<const float *>(matrix.scale_values->ptr),
                    matrix.scale_count,
                    matrix.group_size,
                    matrix.tensor_scale,
                    static_cast<const float *>(input_buffer->ptr),
                    matrix.row_scale == nullptr
                        ? nullptr
                        : static_cast<const float *>(matrix.row_scale->ptr),
                    matrix.row_scale_count,
                    matrix.rows,
                    cols,
                    static_cast<float *>(matrix.output->ptr))) {
                return ULLM_STATUS_INVALID_ARGUMENT;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_triple_f32_hip_kernel(
            first_index_buffer,
            first_scale_buffer,
            first_codebook_buffer,
            first_scale_values_buffer,
            first_row_scale_buffer,
            first_scale_count,
            first_group_size,
            first_tensor_scale,
            first_row_scale_count,
            second_index_buffer,
            second_scale_buffer,
            second_codebook_buffer,
            second_scale_values_buffer,
            second_row_scale_buffer,
            second_scale_count,
            second_group_size,
            second_tensor_scale,
            second_row_scale_count,
            third_index_buffer,
            third_scale_buffer,
            third_codebook_buffer,
            third_scale_values_buffer,
            third_row_scale_buffer,
            third_scale_count,
            third_group_size,
            third_tensor_scale,
            third_row_scale_count,
            input_buffer,
            first_rows,
            second_rows,
            third_rows,
            cols,
            first_output_buffer,
            second_output_buffer,
            third_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    set_error(
        hip_kernel_error.empty() ? "AQ4 matvec triple HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_aq4_matvec_qkv_z_gate_beta_f32(
    const ullm_runtime_buffer *qkv_index_buffer,
    const ullm_runtime_buffer *qkv_scale_buffer,
    const ullm_runtime_buffer *qkv_codebook_buffer,
    const ullm_runtime_buffer *qkv_scale_values_buffer,
    const ullm_runtime_buffer *qkv_row_scale_buffer,
    size_t qkv_scale_count,
    size_t qkv_group_size,
    float qkv_tensor_scale,
    size_t qkv_row_scale_count,
    const ullm_runtime_buffer *z_index_buffer,
    const ullm_runtime_buffer *z_scale_buffer,
    const ullm_runtime_buffer *z_codebook_buffer,
    const ullm_runtime_buffer *z_scale_values_buffer,
    const ullm_runtime_buffer *z_row_scale_buffer,
    size_t z_scale_count,
    size_t z_group_size,
    float z_tensor_scale,
    size_t z_row_scale_count,
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t qkv_rows,
    size_t z_rows,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *qkv_output_buffer,
    ullm_runtime_buffer *z_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    if (qkv_index_buffer == nullptr || qkv_scale_buffer == nullptr ||
        qkv_codebook_buffer == nullptr || qkv_scale_values_buffer == nullptr ||
        z_index_buffer == nullptr || z_scale_buffer == nullptr ||
        z_codebook_buffer == nullptr || z_scale_values_buffer == nullptr ||
        a_index_buffer == nullptr || a_scale_buffer == nullptr ||
        a_codebook_buffer == nullptr || a_scale_values_buffer == nullptr ||
        b_index_buffer == nullptr || b_scale_buffer == nullptr ||
        b_codebook_buffer == nullptr || b_scale_values_buffer == nullptr ||
        input_buffer == nullptr || a_log_buffer == nullptr || dt_bias_buffer == nullptr ||
        qkv_output_buffer == nullptr || z_output_buffer == nullptr ||
        gate_output_buffer == nullptr || beta_output_buffer == nullptr) {
        set_error("AQ4 qkv/z gate/beta received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_scale_count == 0 || z_scale_count == 0 ||
        a_scale_count == 0 || b_scale_count == 0) {
        set_error("AQ4 qkv/z gate/beta scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_group_size == 0 || z_group_size == 0 ||
        a_group_size == 0 || b_group_size == 0) {
        set_error("AQ4 qkv/z gate/beta group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_rows == 0 || z_rows == 0 || heads == 0 || cols == 0) {
        set_error("AQ4 qkv/z gate/beta rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(qkv_tensor_scale) || qkv_tensor_scale <= 0.0f ||
        !std::isfinite(z_tensor_scale) || z_tensor_scale <= 0.0f ||
        !std::isfinite(a_tensor_scale) || a_tensor_scale <= 0.0f ||
        !std::isfinite(b_tensor_scale) || b_tensor_scale <= 0.0f) {
        set_error("AQ4 qkv/z gate/beta tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(qkv_index_buffer, qkv_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, qkv_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, qkv_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_index_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_index_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_index_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_scale_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_codebook_buffer) ||
        !buffers_share_backend(qkv_index_buffer, b_scale_values_buffer) ||
        !buffers_share_backend(qkv_index_buffer, input_buffer) ||
        !buffers_share_backend(qkv_index_buffer, a_log_buffer) ||
        !buffers_share_backend(qkv_index_buffer, dt_bias_buffer) ||
        !buffers_share_backend(qkv_index_buffer, qkv_output_buffer) ||
        !buffers_share_backend(qkv_index_buffer, z_output_buffer) ||
        !buffers_share_backend(qkv_index_buffer, gate_output_buffer) ||
        !buffers_share_backend(qkv_index_buffer, beta_output_buffer) ||
        (qkv_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, qkv_row_scale_buffer)) ||
        (z_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, z_row_scale_buffer)) ||
        (a_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, a_row_scale_buffer)) ||
        (b_row_scale_buffer != nullptr &&
         !buffers_share_backend(qkv_index_buffer, b_row_scale_buffer))) {
        set_error("AQ4 qkv/z gate/beta buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(qkv_output_buffer, stream) ||
        !stream_matches_buffer(z_output_buffer, stream) ||
        !stream_matches_buffer(gate_output_buffer, stream) ||
        !stream_matches_buffer(beta_output_buffer, stream)) {
        set_error("AQ4 qkv/z gate/beta stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / qkv_rows || cols > max_size / z_rows ||
        cols > max_size / heads) {
        set_error("AQ4 qkv/z gate/beta matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_elements = qkv_rows * cols;
    const size_t z_elements = z_rows * cols;
    const size_t gate_elements = heads * cols;
    const size_t qkv_required_index_bytes = qkv_elements / 2 + (qkv_elements % 2);
    const size_t z_required_index_bytes = z_elements / 2 + (z_elements % 2);
    const size_t gate_required_index_bytes = gate_elements / 2 + (gate_elements % 2);
    const size_t qkv_groups =
        qkv_elements / qkv_group_size + (qkv_elements % qkv_group_size == 0 ? 0 : 1);
    const size_t z_groups =
        z_elements / z_group_size + (z_elements % z_group_size == 0 ? 0 : 1);
    const size_t a_groups =
        gate_elements / a_group_size + (gate_elements % a_group_size == 0 ? 0 : 1);
    const size_t b_groups =
        gate_elements / b_group_size + (gate_elements % b_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        qkv_rows > max_size / sizeof(float) ||
        z_rows > max_size / sizeof(float) ||
        heads > max_size / sizeof(float) ||
        qkv_scale_count > max_size / sizeof(float) ||
        z_scale_count > max_size / sizeof(float) ||
        a_scale_count > max_size / sizeof(float) ||
        b_scale_count > max_size / sizeof(float) ||
        qkv_row_scale_count > max_size / sizeof(float) ||
        z_row_scale_count > max_size / sizeof(float) ||
        a_row_scale_count > max_size / sizeof(float) ||
        b_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 qkv/z gate/beta byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t qkv_required_output_bytes = qkv_rows * sizeof(float);
    const size_t z_required_output_bytes = z_rows * sizeof(float);
    const size_t gate_required_output_bytes = heads * sizeof(float);
    const size_t qkv_required_scale_value_bytes = qkv_scale_count * sizeof(float);
    const size_t z_required_scale_value_bytes = z_scale_count * sizeof(float);
    const size_t a_required_scale_value_bytes = a_scale_count * sizeof(float);
    const size_t b_required_scale_value_bytes = b_scale_count * sizeof(float);
    const size_t qkv_required_row_scale_bytes = qkv_row_scale_count * sizeof(float);
    const size_t z_required_row_scale_bytes = z_row_scale_count * sizeof(float);
    const size_t a_required_row_scale_bytes = a_row_scale_count * sizeof(float);
    const size_t b_required_row_scale_bytes = b_row_scale_count * sizeof(float);
    if (qkv_index_buffer->bytes < qkv_required_index_bytes ||
        z_index_buffer->bytes < z_required_index_bytes ||
        a_index_buffer->bytes < gate_required_index_bytes ||
        b_index_buffer->bytes < gate_required_index_bytes) {
        set_error("AQ4 qkv/z gate/beta index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_scale_buffer->bytes < qkv_groups ||
        z_scale_buffer->bytes < z_groups ||
        a_scale_buffer->bytes < a_groups ||
        b_scale_buffer->bytes < b_groups) {
        set_error("AQ4 qkv/z gate/beta scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_scale_values_buffer->bytes < qkv_required_scale_value_bytes ||
        z_scale_values_buffer->bytes < z_required_scale_value_bytes ||
        a_scale_values_buffer->bytes < a_required_scale_value_bytes ||
        b_scale_values_buffer->bytes < b_required_scale_value_bytes) {
        set_error("AQ4 qkv/z gate/beta scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 qkv/z gate/beta input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_log_buffer->bytes < gate_required_output_bytes ||
        dt_bias_buffer->bytes < gate_required_output_bytes) {
        set_error("AQ4 qkv/z gate/beta parameter buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((qkv_row_scale_buffer != nullptr &&
         qkv_row_scale_buffer->bytes < qkv_required_row_scale_bytes) ||
        (z_row_scale_buffer != nullptr &&
         z_row_scale_buffer->bytes < z_required_row_scale_bytes) ||
        (a_row_scale_buffer != nullptr &&
         a_row_scale_buffer->bytes < a_required_row_scale_bytes) ||
        (b_row_scale_buffer != nullptr &&
         b_row_scale_buffer->bytes < b_required_row_scale_bytes)) {
        set_error("AQ4 qkv/z gate/beta row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_output_buffer->bytes < qkv_required_output_bytes ||
        z_output_buffer->bytes < z_required_output_bytes ||
        gate_output_buffer->bytes < gate_required_output_bytes ||
        beta_output_buffer->bytes < gate_required_output_bytes) {
        set_error("AQ4 qkv/z gate/beta output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (qkv_codebook_buffer->bytes % sizeof(float) != 0 ||
        z_codebook_buffer->bytes % sizeof(float) != 0 ||
        a_codebook_buffer->bytes % sizeof(float) != 0 ||
        b_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 qkv/z gate/beta codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_codebook_entries = qkv_codebook_buffer->bytes / sizeof(float);
    const size_t z_codebook_entries = z_codebook_buffer->bytes / sizeof(float);
    const size_t a_codebook_entries = a_codebook_buffer->bytes / sizeof(float);
    const size_t b_codebook_entries = b_codebook_buffer->bytes / sizeof(float);
    if (qkv_codebook_entries < 16 || z_codebook_entries < 16 ||
        a_codebook_entries < 16 || b_codebook_entries < 16) {
        set_error("AQ4 qkv/z gate/beta requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (qkv_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(qkv_index_buffer->ptr),
                static_cast<const std::uint8_t *>(qkv_scale_buffer->ptr),
                static_cast<const float *>(qkv_codebook_buffer->ptr),
                static_cast<const float *>(qkv_scale_values_buffer->ptr),
                qkv_scale_count,
                qkv_group_size,
                qkv_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                qkv_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(qkv_row_scale_buffer->ptr),
                qkv_row_scale_count,
                qkv_rows,
                cols,
                static_cast<float *>(qkv_output_buffer->ptr)) ||
            !aq4_matvec_f32_host(
                static_cast<const std::uint8_t *>(z_index_buffer->ptr),
                static_cast<const std::uint8_t *>(z_scale_buffer->ptr),
                static_cast<const float *>(z_codebook_buffer->ptr),
                static_cast<const float *>(z_scale_values_buffer->ptr),
                z_scale_count,
                z_group_size,
                z_tensor_scale,
                static_cast<const float *>(input_buffer->ptr),
                z_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(z_row_scale_buffer->ptr),
                z_row_scale_count,
                z_rows,
                cols,
                static_cast<float *>(z_output_buffer->ptr)) ||
            !aq4_matvec_gate_beta_f32_host(
                static_cast<const std::uint8_t *>(a_index_buffer->ptr),
                static_cast<const std::uint8_t *>(a_scale_buffer->ptr),
                static_cast<const float *>(a_codebook_buffer->ptr),
                static_cast<const float *>(a_scale_values_buffer->ptr),
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                a_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(a_row_scale_buffer->ptr),
                a_row_scale_count,
                static_cast<const std::uint8_t *>(b_index_buffer->ptr),
                static_cast<const std::uint8_t *>(b_scale_buffer->ptr),
                static_cast<const float *>(b_codebook_buffer->ptr),
                static_cast<const float *>(b_scale_values_buffer->ptr),
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                b_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(b_row_scale_buffer->ptr),
                b_row_scale_count,
                static_cast<const float *>(input_buffer->ptr),
                static_cast<const float *>(a_log_buffer->ptr),
                static_cast<const float *>(dt_bias_buffer->ptr),
                heads,
                cols,
                static_cast<float *>(gate_output_buffer->ptr),
                static_cast<float *>(beta_output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_qkv_z_gate_beta_f32_hip_kernel(
            qkv_index_buffer,
            qkv_scale_buffer,
            qkv_codebook_buffer,
            qkv_scale_values_buffer,
            qkv_row_scale_buffer,
            qkv_scale_count,
            qkv_group_size,
            qkv_tensor_scale,
            qkv_row_scale_count,
            z_index_buffer,
            z_scale_buffer,
            z_codebook_buffer,
            z_scale_values_buffer,
            z_row_scale_buffer,
            z_scale_count,
            z_group_size,
            z_tensor_scale,
            z_row_scale_count,
            a_index_buffer,
            a_scale_buffer,
            a_codebook_buffer,
            a_scale_values_buffer,
            a_row_scale_buffer,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_count,
            b_index_buffer,
            b_scale_buffer,
            b_codebook_buffer,
            b_scale_values_buffer,
            b_row_scale_buffer,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_count,
            input_buffer,
            a_log_buffer,
            dt_bias_buffer,
            qkv_rows,
            z_rows,
            heads,
            cols,
            qkv_output_buffer,
            z_output_buffer,
            gate_output_buffer,
            beta_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    set_error(
        hip_kernel_error.empty() ? "AQ4 qkv/z gate/beta HIP kernel is unavailable" :
                                   hip_kernel_error.c_str());
    return ULLM_STATUS_RUNTIME_ERROR;
}

ullm_status ullm_runtime_aq4_matvec_silu_mul_f32(
    const ullm_runtime_buffer *gate_index_buffer,
    const ullm_runtime_buffer *gate_scale_buffer,
    const ullm_runtime_buffer *gate_codebook_buffer,
    const ullm_runtime_buffer *gate_scale_values_buffer,
    const ullm_runtime_buffer *gate_row_scale_buffer,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    size_t gate_row_scale_count,
    const ullm_runtime_buffer *up_index_buffer,
    const ullm_runtime_buffer *up_scale_buffer,
    const ullm_runtime_buffer *up_codebook_buffer,
    const ullm_runtime_buffer *up_scale_values_buffer,
    const ullm_runtime_buffer *up_row_scale_buffer,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    size_t up_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (gate_index_buffer == nullptr || gate_scale_buffer == nullptr ||
        gate_codebook_buffer == nullptr || gate_scale_values_buffer == nullptr ||
        up_index_buffer == nullptr || up_scale_buffer == nullptr ||
        up_codebook_buffer == nullptr || up_scale_values_buffer == nullptr ||
        input_buffer == nullptr || output_buffer == nullptr) {
        set_error("AQ4 matvec SiLU-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_scale_count == 0 || up_scale_count == 0) {
        set_error("AQ4 matvec SiLU-mul scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_group_size == 0 || up_group_size == 0) {
        set_error("AQ4 matvec SiLU-mul group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("AQ4 matvec SiLU-mul rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(gate_tensor_scale) || gate_tensor_scale <= 0.0f ||
        !std::isfinite(up_tensor_scale) || up_tensor_scale <= 0.0f) {
        set_error("AQ4 matvec SiLU-mul tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(gate_index_buffer, gate_scale_buffer) ||
        !buffers_share_backend(gate_index_buffer, gate_codebook_buffer) ||
        !buffers_share_backend(gate_index_buffer, gate_scale_values_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_index_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_scale_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_codebook_buffer) ||
        !buffers_share_backend(gate_index_buffer, up_scale_values_buffer) ||
        !buffers_share_backend(gate_index_buffer, input_buffer) ||
        !buffers_share_backend(gate_index_buffer, output_buffer) ||
        (gate_row_scale_buffer != nullptr &&
         !buffers_share_backend(gate_index_buffer, gate_row_scale_buffer)) ||
        (up_row_scale_buffer != nullptr &&
         !buffers_share_backend(gate_index_buffer, up_row_scale_buffer))) {
        set_error("AQ4 matvec SiLU-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("AQ4 matvec SiLU-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("AQ4 matvec SiLU-mul matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = rows * cols;
    const size_t gate_required_index_bytes = elements / 2 + (elements % 2);
    const size_t up_required_index_bytes = gate_required_index_bytes;
    const size_t gate_groups =
        elements / gate_group_size + (elements % gate_group_size == 0 ? 0 : 1);
    const size_t up_groups = elements / up_group_size + (elements % up_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float) ||
        gate_scale_count > max_size / sizeof(float) ||
        up_scale_count > max_size / sizeof(float) ||
        gate_row_scale_count > max_size / sizeof(float) ||
        up_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec SiLU-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    const size_t gate_required_scale_value_bytes = gate_scale_count * sizeof(float);
    const size_t up_required_scale_value_bytes = up_scale_count * sizeof(float);
    const size_t gate_required_row_scale_bytes = gate_row_scale_count * sizeof(float);
    const size_t up_required_row_scale_bytes = up_row_scale_count * sizeof(float);
    if (gate_index_buffer->bytes < gate_required_index_bytes ||
        up_index_buffer->bytes < up_required_index_bytes) {
        set_error("AQ4 matvec SiLU-mul index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_scale_buffer->bytes < gate_groups || up_scale_buffer->bytes < up_groups) {
        set_error("AQ4 matvec SiLU-mul scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_scale_values_buffer->bytes < gate_required_scale_value_bytes ||
        up_scale_values_buffer->bytes < up_required_scale_value_bytes) {
        set_error("AQ4 matvec SiLU-mul scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec SiLU-mul input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((gate_row_scale_buffer != nullptr &&
         gate_row_scale_buffer->bytes < gate_required_row_scale_bytes) ||
        (up_row_scale_buffer != nullptr &&
         up_row_scale_buffer->bytes < up_required_row_scale_bytes)) {
        set_error("AQ4 matvec SiLU-mul row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec SiLU-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_codebook_buffer->bytes % sizeof(float) != 0 ||
        up_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec SiLU-mul codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t gate_codebook_entries = gate_codebook_buffer->bytes / sizeof(float);
    const size_t up_codebook_entries = up_codebook_buffer->bytes / sizeof(float);
    if (gate_codebook_entries < 16 || up_codebook_entries < 16) {
        set_error("AQ4 matvec SiLU-mul requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (gate_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_silu_mul_f32_host(
                static_cast<const std::uint8_t *>(gate_index_buffer->ptr),
                static_cast<const std::uint8_t *>(gate_scale_buffer->ptr),
                static_cast<const float *>(gate_codebook_buffer->ptr),
                static_cast<const float *>(gate_scale_values_buffer->ptr),
                gate_scale_count,
                gate_group_size,
                gate_tensor_scale,
                gate_row_scale_buffer == nullptr
                    ? nullptr
                    : static_cast<const float *>(gate_row_scale_buffer->ptr),
                gate_row_scale_count,
                static_cast<const std::uint8_t *>(up_index_buffer->ptr),
                static_cast<const std::uint8_t *>(up_scale_buffer->ptr),
                static_cast<const float *>(up_codebook_buffer->ptr),
                static_cast<const float *>(up_scale_values_buffer->ptr),
                up_scale_count,
                up_group_size,
                up_tensor_scale,
                up_row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(up_row_scale_buffer->ptr),
                up_row_scale_count,
                static_cast<const float *>(input_buffer->ptr),
                rows,
                cols,
                static_cast<float *>(output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_silu_mul_f32_hip_kernel(
            gate_index_buffer,
            gate_scale_buffer,
            gate_codebook_buffer,
            gate_scale_values_buffer,
            gate_row_scale_buffer,
            gate_scale_count,
            gate_group_size,
            gate_tensor_scale,
            gate_row_scale_count,
            up_index_buffer,
            up_scale_buffer,
            up_codebook_buffer,
            up_scale_values_buffer,
            up_row_scale_buffer,
            up_scale_count,
            up_group_size,
            up_tensor_scale,
            up_row_scale_count,
            input_buffer,
            rows,
            cols,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec SiLU-mul HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_silu_mul_f32_hip_staging(
        gate_index_buffer,
        gate_scale_buffer,
        gate_codebook_buffer,
        gate_scale_values_buffer,
        gate_row_scale_buffer,
        gate_scale_count,
        gate_group_size,
        gate_tensor_scale,
        gate_row_scale_count,
        gate_required_index_bytes,
        gate_groups,
        gate_codebook_entries,
        up_index_buffer,
        up_scale_buffer,
        up_codebook_buffer,
        up_scale_values_buffer,
        up_row_scale_buffer,
        up_scale_count,
        up_group_size,
        up_tensor_scale,
        up_row_scale_count,
        up_required_index_bytes,
        up_groups,
        up_codebook_entries,
        input_buffer,
        rows,
        cols,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_aq4_matvec_gate_beta_f32(
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream) {
    if (a_index_buffer == nullptr || a_scale_buffer == nullptr ||
        a_codebook_buffer == nullptr || a_scale_values_buffer == nullptr ||
        b_index_buffer == nullptr || b_scale_buffer == nullptr ||
        b_codebook_buffer == nullptr || b_scale_values_buffer == nullptr ||
        input_buffer == nullptr || a_log_buffer == nullptr || dt_bias_buffer == nullptr ||
        gate_output_buffer == nullptr || beta_output_buffer == nullptr) {
        set_error("AQ4 matvec gate/beta received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_scale_count == 0 || b_scale_count == 0) {
        set_error("AQ4 matvec gate/beta scale table is empty");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_group_size == 0 || b_group_size == 0) {
        set_error("AQ4 matvec gate/beta group size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (heads == 0 || cols == 0) {
        set_error("AQ4 matvec gate/beta heads and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(a_tensor_scale) || a_tensor_scale <= 0.0f ||
        !std::isfinite(b_tensor_scale) || b_tensor_scale <= 0.0f) {
        set_error("AQ4 matvec gate/beta tensor scale must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(a_index_buffer, a_scale_buffer) ||
        !buffers_share_backend(a_index_buffer, a_codebook_buffer) ||
        !buffers_share_backend(a_index_buffer, a_scale_values_buffer) ||
        !buffers_share_backend(a_index_buffer, b_index_buffer) ||
        !buffers_share_backend(a_index_buffer, b_scale_buffer) ||
        !buffers_share_backend(a_index_buffer, b_codebook_buffer) ||
        !buffers_share_backend(a_index_buffer, b_scale_values_buffer) ||
        !buffers_share_backend(a_index_buffer, input_buffer) ||
        !buffers_share_backend(a_index_buffer, a_log_buffer) ||
        !buffers_share_backend(a_index_buffer, dt_bias_buffer) ||
        !buffers_share_backend(a_index_buffer, gate_output_buffer) ||
        !buffers_share_backend(a_index_buffer, beta_output_buffer) ||
        (a_row_scale_buffer != nullptr &&
         !buffers_share_backend(a_index_buffer, a_row_scale_buffer)) ||
        (b_row_scale_buffer != nullptr &&
         !buffers_share_backend(a_index_buffer, b_row_scale_buffer))) {
        set_error("AQ4 matvec gate/beta buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(gate_output_buffer, stream) ||
        !stream_matches_buffer(beta_output_buffer, stream)) {
        set_error("AQ4 matvec gate/beta stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / heads) {
        set_error("AQ4 matvec gate/beta matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = heads * cols;
    const size_t a_required_index_bytes = elements / 2 + (elements % 2);
    const size_t b_required_index_bytes = a_required_index_bytes;
    const size_t a_groups = elements / a_group_size + (elements % a_group_size == 0 ? 0 : 1);
    const size_t b_groups = elements / b_group_size + (elements % b_group_size == 0 ? 0 : 1);
    if (cols > max_size / sizeof(float) ||
        heads > max_size / sizeof(float) ||
        a_scale_count > max_size / sizeof(float) ||
        b_scale_count > max_size / sizeof(float) ||
        a_row_scale_count > max_size / sizeof(float) ||
        b_row_scale_count > max_size / sizeof(float)) {
        set_error("AQ4 matvec gate/beta byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = heads * sizeof(float);
    const size_t a_required_scale_value_bytes = a_scale_count * sizeof(float);
    const size_t b_required_scale_value_bytes = b_scale_count * sizeof(float);
    const size_t a_required_row_scale_bytes = a_row_scale_count * sizeof(float);
    const size_t b_required_row_scale_bytes = b_row_scale_count * sizeof(float);
    if (a_index_buffer->bytes < a_required_index_bytes ||
        b_index_buffer->bytes < b_required_index_bytes) {
        set_error("AQ4 matvec gate/beta index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_scale_buffer->bytes < a_groups || b_scale_buffer->bytes < b_groups) {
        set_error("AQ4 matvec gate/beta scale index buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_scale_values_buffer->bytes < a_required_scale_value_bytes ||
        b_scale_values_buffer->bytes < b_required_scale_value_bytes) {
        set_error("AQ4 matvec gate/beta scale value buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("AQ4 matvec gate/beta input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_log_buffer->bytes < required_output_bytes ||
        dt_bias_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec gate/beta parameter buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if ((a_row_scale_buffer != nullptr &&
         a_row_scale_buffer->bytes < a_required_row_scale_bytes) ||
        (b_row_scale_buffer != nullptr &&
         b_row_scale_buffer->bytes < b_required_row_scale_bytes)) {
        set_error("AQ4 matvec gate/beta row scale buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_output_buffer->bytes < required_output_bytes ||
        beta_output_buffer->bytes < required_output_bytes) {
        set_error("AQ4 matvec gate/beta output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (a_codebook_buffer->bytes % sizeof(float) != 0 ||
        b_codebook_buffer->bytes % sizeof(float) != 0) {
        set_error("AQ4 matvec gate/beta codebook buffer size is not a multiple of f32");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t a_codebook_entries = a_codebook_buffer->bytes / sizeof(float);
    const size_t b_codebook_entries = b_codebook_buffer->bytes / sizeof(float);
    if (a_codebook_entries < 16 || b_codebook_entries < 16) {
        set_error("AQ4 matvec gate/beta requires at least 16 codebook entries");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (a_index_buffer->backend == BackendKind::Cpu) {
        if (!aq4_matvec_gate_beta_f32_host(
                static_cast<const std::uint8_t *>(a_index_buffer->ptr),
                static_cast<const std::uint8_t *>(a_scale_buffer->ptr),
                static_cast<const float *>(a_codebook_buffer->ptr),
                static_cast<const float *>(a_scale_values_buffer->ptr),
                a_scale_count,
                a_group_size,
                a_tensor_scale,
                a_row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(a_row_scale_buffer->ptr),
                a_row_scale_count,
                static_cast<const std::uint8_t *>(b_index_buffer->ptr),
                static_cast<const std::uint8_t *>(b_scale_buffer->ptr),
                static_cast<const float *>(b_codebook_buffer->ptr),
                static_cast<const float *>(b_scale_values_buffer->ptr),
                b_scale_count,
                b_group_size,
                b_tensor_scale,
                b_row_scale_buffer == nullptr ? nullptr : static_cast<const float *>(b_row_scale_buffer->ptr),
                b_row_scale_count,
                static_cast<const float *>(input_buffer->ptr),
                static_cast<const float *>(a_log_buffer->ptr),
                static_cast<const float *>(dt_bias_buffer->ptr),
                heads,
                cols,
                static_cast<float *>(gate_output_buffer->ptr),
                static_cast<float *>(beta_output_buffer->ptr))) {
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (aq4_matvec_gate_beta_f32_hip_kernel(
            a_index_buffer,
            a_scale_buffer,
            a_codebook_buffer,
            a_scale_values_buffer,
            a_row_scale_buffer,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_count,
            b_index_buffer,
            b_scale_buffer,
            b_codebook_buffer,
            b_scale_values_buffer,
            b_row_scale_buffer,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_count,
            input_buffer,
            a_log_buffer,
            dt_bias_buffer,
            heads,
            cols,
            gate_output_buffer,
            beta_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "AQ4 matvec gate/beta HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return aq4_matvec_gate_beta_f32_hip_staging(
        a_index_buffer,
        a_scale_buffer,
        a_codebook_buffer,
        a_scale_values_buffer,
        a_row_scale_buffer,
        a_scale_count,
        a_group_size,
        a_tensor_scale,
        a_row_scale_count,
        a_required_index_bytes,
        a_groups,
        a_codebook_entries,
        b_index_buffer,
        b_scale_buffer,
        b_codebook_buffer,
        b_scale_values_buffer,
        b_row_scale_buffer,
        b_scale_count,
        b_group_size,
        b_tensor_scale,
        b_row_scale_count,
        b_required_index_bytes,
        b_groups,
        b_codebook_entries,
        input_buffer,
        a_log_buffer,
        dt_bias_buffer,
        heads,
        cols,
        gate_output_buffer,
        beta_output_buffer,
        stream);
}

ullm_status ullm_runtime_linear_attn_qkv_prepare_f32(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    int qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    if (qkv_buffer == nullptr || conv_weight_buffer == nullptr ||
        conv_history_buffer == nullptr || conv_output_buffer == nullptr ||
        q_output_buffer == nullptr || k_output_buffer == nullptr || v_output_buffer == nullptr) {
        set_error("linear attention qkv prepare received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 ||
        kernel_size == 0) {
        set_error("linear attention qkv prepare dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(q_scale)) {
        set_error("linear attention qkv prepare q_scale must be finite");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(qkv_buffer, conv_weight_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_history_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_output_buffer) ||
        !buffers_share_backend(qkv_buffer, q_output_buffer) ||
        !buffers_share_backend(qkv_buffer, k_output_buffer) ||
        !buffers_share_backend(qkv_buffer, v_output_buffer)) {
        set_error("linear attention qkv prepare buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(conv_output_buffer, stream) ||
        !stream_matches_buffer(q_output_buffer, stream) ||
        !stream_matches_buffer(k_output_buffer, stream) ||
        !stream_matches_buffer(v_output_buffer, stream)) {
        set_error("linear attention qkv prepare stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (key_heads > max_size / key_dim || value_heads > max_size / value_dim) {
        set_error("linear attention qkv prepare element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = key_heads * key_dim;
    const size_t k_elements = q_elements;
    const size_t v_elements = value_heads * value_dim;
    if (q_elements > max_size - k_elements || q_elements + k_elements > max_size - v_elements) {
        set_error("linear attention qkv prepare channel count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t channels = q_elements + k_elements + v_elements;
    if (channels > max_size / kernel_size ||
        channels > max_size / sizeof(float) ||
        q_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float)) {
        set_error("linear attention qkv prepare byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_bytes = channels * sizeof(float);
    const size_t conv_history_bytes = channels * kernel_size * sizeof(float);
    const size_t q_bytes = q_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    if (qkv_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare qkv buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_weight_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare conv weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_history_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare conv history buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_output_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare conv output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_output_buffer->bytes < q_bytes || k_output_buffer->bytes < k_bytes ||
        v_output_buffer->bytes < v_bytes) {
        set_error("linear attention qkv prepare output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (qkv_buffer->backend == BackendKind::Cpu) {
        linear_attn_qkv_prepare_f32_host(
            static_cast<const float *>(qkv_buffer->ptr),
            static_cast<const float *>(conv_weight_buffer->ptr),
            static_cast<float *>(conv_history_buffer->ptr),
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            qk_l2_norm != 0,
            static_cast<float *>(conv_output_buffer->ptr),
            static_cast<float *>(q_output_buffer->ptr),
            static_cast<float *>(k_output_buffer->ptr),
            static_cast<float *>(v_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_qkv_prepare_f32_hip_kernel(
            qkv_buffer,
            conv_weight_buffer,
            conv_history_buffer,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            qk_l2_norm != 0,
            conv_output_buffer,
            q_output_buffer,
            k_output_buffer,
            v_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "linear attention qkv prepare HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_qkv_prepare_f32_hip_staging(
        qkv_buffer,
        conv_weight_buffer,
        conv_history_buffer,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        q_scale,
        qk_l2_norm != 0,
        qkv_bytes,
        conv_history_bytes,
        q_bytes,
        k_bytes,
        v_bytes,
        conv_output_buffer,
        q_output_buffer,
        k_output_buffer,
        v_output_buffer,
        stream);
}

ullm_status ullm_runtime_linear_attn_qkv_prepare_batch_f32(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
    float q_scale,
    int qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
    ullm_runtime_stream *stream) {
    if (qkv_buffer == nullptr || conv_weight_buffer == nullptr ||
        conv_history_buffer == nullptr || conv_output_buffer == nullptr ||
        q_output_buffer == nullptr || k_output_buffer == nullptr || v_output_buffer == nullptr) {
        set_error("linear attention qkv prepare batch received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 ||
        kernel_size == 0 || sequence_len == 0) {
        set_error("linear attention qkv prepare batch dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(q_scale)) {
        set_error("linear attention qkv prepare batch q_scale must be finite");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(qkv_buffer, conv_weight_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_history_buffer) ||
        !buffers_share_backend(qkv_buffer, conv_output_buffer) ||
        !buffers_share_backend(qkv_buffer, q_output_buffer) ||
        !buffers_share_backend(qkv_buffer, k_output_buffer) ||
        !buffers_share_backend(qkv_buffer, v_output_buffer)) {
        set_error("linear attention qkv prepare batch buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(conv_output_buffer, stream) ||
        !stream_matches_buffer(q_output_buffer, stream) ||
        !stream_matches_buffer(k_output_buffer, stream) ||
        !stream_matches_buffer(v_output_buffer, stream)) {
        set_error("linear attention qkv prepare batch stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (key_heads > max_size / key_dim || value_heads > max_size / value_dim) {
        set_error("linear attention qkv prepare batch element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_elements = key_heads * key_dim;
    const size_t k_elements = q_elements;
    const size_t v_elements = value_heads * value_dim;
    if (q_elements > max_size - k_elements || q_elements + k_elements > max_size - v_elements) {
        set_error("linear attention qkv prepare batch channel count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t channels = q_elements + k_elements + v_elements;
    if (channels > max_size / kernel_size ||
        channels > max_size / sequence_len ||
        q_elements > max_size / sequence_len ||
        k_elements > max_size / sequence_len ||
        v_elements > max_size / sequence_len) {
        set_error("linear attention qkv prepare batch element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_elements = channels * sequence_len;
    const size_t q_output_elements = q_elements * sequence_len;
    const size_t k_output_elements = k_elements * sequence_len;
    const size_t v_output_elements = v_elements * sequence_len;
    if (qkv_elements > max_size / sizeof(float) ||
        channels > max_size / kernel_size ||
        channels * kernel_size > max_size / sizeof(float) ||
        q_output_elements > max_size / sizeof(float) ||
        k_output_elements > max_size / sizeof(float) ||
        v_output_elements > max_size / sizeof(float)) {
        set_error("linear attention qkv prepare batch byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t qkv_bytes = qkv_elements * sizeof(float);
    const size_t conv_history_bytes = channels * kernel_size * sizeof(float);
    const size_t q_bytes = q_output_elements * sizeof(float);
    const size_t k_bytes = k_output_elements * sizeof(float);
    const size_t v_bytes = v_output_elements * sizeof(float);
    if (qkv_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare batch qkv buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_weight_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare batch conv weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_history_buffer->bytes < conv_history_bytes) {
        set_error("linear attention qkv prepare batch conv history buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (conv_output_buffer->bytes < qkv_bytes) {
        set_error("linear attention qkv prepare batch conv output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_output_buffer->bytes < q_bytes || k_output_buffer->bytes < k_bytes ||
        v_output_buffer->bytes < v_bytes) {
        set_error("linear attention qkv prepare batch output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (qkv_buffer->backend == BackendKind::Cpu) {
        linear_attn_qkv_prepare_batch_f32_host(
            static_cast<const float *>(qkv_buffer->ptr),
            static_cast<const float *>(conv_weight_buffer->ptr),
            static_cast<float *>(conv_history_buffer->ptr),
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            sequence_len,
            q_scale,
            qk_l2_norm != 0,
            static_cast<float *>(conv_output_buffer->ptr),
            static_cast<float *>(q_output_buffer->ptr),
            static_cast<float *>(k_output_buffer->ptr),
            static_cast<float *>(v_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (linear_attn_qkv_prepare_batch_f32_hip_kernel(
            qkv_buffer,
            conv_weight_buffer,
            conv_history_buffer,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            sequence_len,
            q_scale,
            qk_l2_norm != 0,
            conv_output_buffer,
            q_output_buffer,
            k_output_buffer,
            v_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "linear attention qkv prepare batch HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return linear_attn_qkv_prepare_batch_f32_hip_staging(
        qkv_buffer,
        conv_weight_buffer,
        conv_history_buffer,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel_size,
        sequence_len,
        q_scale,
        qk_l2_norm != 0,
        qkv_bytes,
        conv_history_bytes,
        q_bytes,
        k_bytes,
        v_bytes,
        conv_output_buffer,
        q_output_buffer,
        k_output_buffer,
        v_output_buffer,
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

ullm_status ullm_runtime_matvec_bf16_f32(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (matrix_buffer == nullptr || input_buffer == nullptr || output_buffer == nullptr) {
        set_error("BF16 matvec received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("BF16 matvec rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(matrix_buffer, input_buffer) ||
        !buffers_share_backend(matrix_buffer, output_buffer)) {
        set_error("BF16 matvec buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("BF16 matvec stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("BF16 matvec matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    if (matrix_elements > max_size / sizeof(uint16_t) ||
        cols > max_size / sizeof(float) ||
        rows > max_size / sizeof(float)) {
        set_error("BF16 matvec byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_matrix_bytes = matrix_elements * sizeof(uint16_t);
    const size_t required_input_bytes = cols * sizeof(float);
    const size_t required_output_bytes = rows * sizeof(float);
    if (matrix_buffer->bytes < required_matrix_bytes) {
        set_error("BF16 matvec matrix buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (input_buffer->bytes < required_input_bytes) {
        set_error("BF16 matvec input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("BF16 matvec output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (matrix_buffer->backend == BackendKind::Cpu) {
        const auto *matrix = static_cast<const uint16_t *>(matrix_buffer->ptr);
        const auto *input = static_cast<const float *>(input_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        matvec_bf16_f32_host(matrix, input, rows, cols, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (matvec_bf16_f32_hip_kernel(
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

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "BF16 matvec HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return matvec_bf16_f32_hip_staging(
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

ullm_status ullm_runtime_bf16_row_f32(
    const ullm_runtime_buffer *matrix_buffer,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (matrix_buffer == nullptr || output_buffer == nullptr) {
        set_error("BF16 row received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        set_error("BF16 row rows and cols must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (row_index >= rows) {
        set_error("BF16 row index is out of range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(matrix_buffer, output_buffer)) {
        set_error("BF16 row buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("BF16 row stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cols > max_size / rows) {
        set_error("BF16 row matrix element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t matrix_elements = rows * cols;
    if (matrix_elements > max_size / sizeof(uint16_t) || cols > max_size / sizeof(float)) {
        set_error("BF16 row byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t required_matrix_bytes = matrix_elements * sizeof(uint16_t);
    const size_t required_output_bytes = cols * sizeof(float);
    if (matrix_buffer->bytes < required_matrix_bytes) {
        set_error("BF16 row matrix buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < required_output_bytes) {
        set_error("BF16 row output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (matrix_buffer->backend == BackendKind::Cpu) {
        const auto *matrix = static_cast<const uint16_t *>(matrix_buffer->ptr);
        auto *output = static_cast<float *>(output_buffer->ptr);
        bf16_row_f32_host(matrix, rows, cols, row_index, output);
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (bf16_row_f32_hip_kernel(
            matrix_buffer,
            rows,
            cols,
            row_index,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_BF16_ROW_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "BF16 row HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return bf16_row_f32_hip_staging(
        matrix_buffer,
        rows,
        cols,
        row_index,
        required_matrix_bytes,
        required_output_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_top1_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || partial_values_buffer == nullptr || partial_indices_buffer == nullptr) {
        set_error("f32 top1 received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0 || partial_count == 0) {
        set_error("f32 top1 elements and partial_count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    constexpr size_t block_size = 256;
    const size_t expected_partials = (elements + block_size - 1) / block_size;
    if (partial_count != expected_partials) {
        set_error("f32 top1 partial_count does not match expected block count");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, partial_values_buffer) ||
        !buffers_share_backend(input_buffer, partial_indices_buffer)) {
        set_error("f32 top1 buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(partial_values_buffer, stream) ||
        !stream_matches_buffer(partial_indices_buffer, stream)) {
        set_error("f32 top1 stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(uint32_t)) {
        set_error("f32 top1 byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        set_error("f32 top1 elements exceed uint32 index range");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_bytes = elements * sizeof(float);
    const size_t partial_values_bytes = partial_count * sizeof(float);
    const size_t partial_indices_bytes = partial_count * sizeof(uint32_t);
    if (input_buffer->bytes < input_bytes) {
        set_error("f32 top1 input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_values_buffer->bytes < partial_values_bytes) {
        set_error("f32 top1 partial values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_indices_buffer->bytes < partial_indices_bytes) {
        set_error("f32 top1 partial indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        top1_f32_partials_host(
            static_cast<const float *>(input_buffer->ptr),
            elements,
            partial_count,
            static_cast<float *>(partial_values_buffer->ptr),
            static_cast<uint32_t *>(partial_indices_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (top1_f32_hip_kernel(
            input_buffer,
            elements,
            partial_values_buffer,
            partial_indices_buffer,
            partial_count,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_TOP1_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 top1 HIP kernel is unavailable" : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return top1_f32_hip_staging(
        input_buffer,
        elements,
        input_bytes,
        partial_values_buffer,
        partial_indices_buffer,
        partial_count,
        partial_values_bytes,
        partial_indices_bytes,
        stream);
}

ullm_status ullm_runtime_top1_pairs_f32(
    const ullm_runtime_buffer *values_buffer,
    const ullm_runtime_buffer *indices_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream) {
    if (values_buffer == nullptr || indices_buffer == nullptr || partial_values_buffer == nullptr ||
        partial_indices_buffer == nullptr) {
        set_error("f32 top1 pairs received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (elements == 0 || partial_count == 0) {
        set_error("f32 top1 pairs elements and partial_count must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    constexpr size_t block_size = 256;
    const size_t expected_partials = (elements + block_size - 1) / block_size;
    if (partial_count != expected_partials) {
        set_error("f32 top1 pairs partial_count does not match expected block count");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(values_buffer, indices_buffer) ||
        !buffers_share_backend(values_buffer, partial_values_buffer) ||
        !buffers_share_backend(values_buffer, partial_indices_buffer)) {
        set_error("f32 top1 pairs buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(partial_values_buffer, stream) ||
        !stream_matches_buffer(partial_indices_buffer, stream)) {
        set_error("f32 top1 pairs stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (elements > max_size / sizeof(float) ||
        elements > max_size / sizeof(uint32_t) ||
        partial_count > max_size / sizeof(float) ||
        partial_count > max_size / sizeof(uint32_t)) {
        set_error("f32 top1 pairs byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t values_bytes = elements * sizeof(float);
    const size_t indices_bytes = elements * sizeof(uint32_t);
    const size_t partial_values_bytes = partial_count * sizeof(float);
    const size_t partial_indices_bytes = partial_count * sizeof(uint32_t);
    if (values_buffer->bytes < values_bytes) {
        set_error("f32 top1 pairs values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (indices_buffer->bytes < indices_bytes) {
        set_error("f32 top1 pairs indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_values_buffer->bytes < partial_values_bytes) {
        set_error("f32 top1 pairs partial values buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (partial_indices_buffer->bytes < partial_indices_bytes) {
        set_error("f32 top1 pairs partial indices buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (values_buffer->backend == BackendKind::Cpu) {
        top1_pairs_f32_partials_host(
            static_cast<const float *>(values_buffer->ptr),
            static_cast<const uint32_t *>(indices_buffer->ptr),
            elements,
            partial_count,
            static_cast<float *>(partial_values_buffer->ptr),
            static_cast<uint32_t *>(partial_indices_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (top1_pairs_f32_hip_kernel(
            values_buffer,
            indices_buffer,
            elements,
            partial_values_buffer,
            partial_indices_buffer,
            partial_count,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_TOP1_PAIRS_KERNEL") != nullptr ||
        std::getenv("ULLM_REQUIRE_HIP_TOP1_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(
            hip_kernel_error.empty() ? "f32 top1 pairs HIP kernel is unavailable"
                                     : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return top1_pairs_f32_hip_staging(
        values_buffer,
        indices_buffer,
        elements,
        values_bytes,
        indices_bytes,
        partial_values_buffer,
        partial_indices_buffer,
        partial_count,
        partial_values_bytes,
        partial_indices_bytes,
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

ullm_status ullm_runtime_segmented_rmsnorm_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || output_buffer == nullptr) {
        set_error("f32 segmented RMSNorm received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (segments == 0 || segment_size == 0) {
        set_error("f32 segmented RMSNorm segments and segment size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("f32 segmented RMSNorm epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 segmented RMSNorm buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 segmented RMSNorm stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (segment_size > max_size / segments ||
        segment_size > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = segments * segment_size;
    if (elements > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm element byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_output_bytes = elements * sizeof(float);
    const size_t weight_bytes = segment_size * sizeof(float);
    if (input_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < weight_bytes) {
        set_error("f32 segmented RMSNorm weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        segmented_rmsnorm_f32_host(
            static_cast<const float *>(input_buffer->ptr),
            static_cast<const float *>(weight_buffer->ptr),
            segments,
            segment_size,
            epsilon,
            static_cast<float *>(output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (segmented_rmsnorm_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            segments,
            segment_size,
            epsilon,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_RMSNORM_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 segmented RMSNorm HIP kernel is unavailable"
                                           : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return segmented_rmsnorm_f32_hip_staging(
        input_buffer,
        weight_buffer,
        segments,
        segment_size,
        epsilon,
        input_output_bytes,
        weight_bytes,
        output_buffer,
        stream);
}

ullm_status ullm_runtime_segmented_rmsnorm_silu_mul_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream) {
    if (input_buffer == nullptr || weight_buffer == nullptr || gate_buffer == nullptr ||
        output_buffer == nullptr) {
        set_error("f32 segmented RMSNorm SiLU-mul received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (segments == 0 || segment_size == 0) {
        set_error("f32 segmented RMSNorm SiLU-mul segments and segment size must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("f32 segmented RMSNorm SiLU-mul epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(input_buffer, weight_buffer) ||
        !buffers_share_backend(input_buffer, gate_buffer) ||
        !buffers_share_backend(input_buffer, output_buffer)) {
        set_error("f32 segmented RMSNorm SiLU-mul buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(output_buffer, stream)) {
        set_error("f32 segmented RMSNorm SiLU-mul stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (segment_size > max_size / segments ||
        segment_size > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm SiLU-mul byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t elements = segments * segment_size;
    if (elements > max_size / sizeof(float)) {
        set_error("f32 segmented RMSNorm SiLU-mul element byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t input_output_bytes = elements * sizeof(float);
    const size_t weight_bytes = segment_size * sizeof(float);
    if (input_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul input buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (weight_buffer->bytes < weight_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul gate buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (output_buffer->bytes < input_output_bytes) {
        set_error("f32 segmented RMSNorm SiLU-mul output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (input_buffer->backend == BackendKind::Cpu) {
        segmented_rmsnorm_silu_mul_f32_host(
            static_cast<const float *>(input_buffer->ptr),
            static_cast<const float *>(weight_buffer->ptr),
            static_cast<const float *>(gate_buffer->ptr),
            segments,
            segment_size,
            epsilon,
            static_cast<float *>(output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (segmented_rmsnorm_silu_mul_f32_hip_kernel(
            input_buffer,
            weight_buffer,
            gate_buffer,
            segments,
            segment_size,
            epsilon,
            output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "f32 segmented RMSNorm SiLU-mul HIP kernel is unavailable"
                                           : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return segmented_rmsnorm_silu_mul_f32_hip_staging(
        input_buffer,
        weight_buffer,
        gate_buffer,
        segments,
        segment_size,
        epsilon,
        input_output_bytes,
        weight_bytes,
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

ullm_status ullm_runtime_qwen35_split_q_gate_f32(
    const ullm_runtime_buffer *projected_buffer,
    size_t q_heads,
    size_t head_dim,
    ullm_runtime_buffer *query_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_stream *stream) {
    if (projected_buffer == nullptr || query_output_buffer == nullptr || gate_output_buffer == nullptr) {
        set_error("Qwen3.5 q/gate split received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_heads == 0 || head_dim == 0) {
        set_error("Qwen3.5 q/gate split q_heads and head_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(projected_buffer, query_output_buffer) ||
        !buffers_share_backend(projected_buffer, gate_output_buffer)) {
        set_error("Qwen3.5 q/gate split buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(query_output_buffer, stream) ||
        !stream_matches_buffer(gate_output_buffer, stream)) {
        set_error("Qwen3.5 q/gate split stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (head_dim > max_size / q_heads) {
        set_error("Qwen3.5 q/gate split element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_elements = q_heads * head_dim;
    if (output_elements > max_size / 2 || output_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/gate split byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t projected_elements = output_elements * 2;
    if (projected_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/gate split projected byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t projected_bytes = projected_elements * sizeof(float);
    if (projected_buffer->bytes < projected_bytes) {
        set_error("Qwen3.5 q/gate split projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (query_output_buffer->bytes < output_bytes) {
        set_error("Qwen3.5 q/gate split query output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_output_buffer->bytes < output_bytes) {
        set_error("Qwen3.5 q/gate split gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (projected_buffer->backend == BackendKind::Cpu) {
        qwen35_split_q_gate_f32_host(
            static_cast<const float *>(projected_buffer->ptr),
            q_heads,
            head_dim,
            static_cast<float *>(query_output_buffer->ptr),
            static_cast<float *>(gate_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (qwen35_split_q_gate_f32_hip_kernel(
            projected_buffer,
            q_heads,
            head_dim,
            query_output_buffer,
            gate_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel = std::getenv("ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty() ? "Qwen3.5 q/gate split HIP kernel is unavailable"
                                           : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return qwen35_split_q_gate_f32_hip_staging(
        projected_buffer,
        q_heads,
        head_dim,
        projected_bytes,
        output_bytes,
        query_output_buffer,
        gate_output_buffer,
        stream);
}

ullm_status ullm_runtime_qwen35_qk_norm_rope_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream) {
    if (q_projected_buffer == nullptr || k_projected_buffer == nullptr ||
        q_weight_buffer == nullptr || k_weight_buffer == nullptr ||
        q_gate_output_buffer == nullptr || q_rope_output_buffer == nullptr ||
        k_rope_output_buffer == nullptr) {
        set_error("Qwen3.5 q/k norm RoPE received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_heads == 0 || kv_heads == 0 || head_dim == 0) {
        set_error("Qwen3.5 q/k norm RoPE heads and head_dim must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        set_error("Qwen3.5 q/k norm RoPE rotary_dim must be even and no greater than head_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(rope_base) || rope_base <= 1.0f) {
        set_error("Qwen3.5 q/k norm RoPE base must be finite and greater than one");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error("Qwen3.5 q/k norm RoPE epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_projected_buffer, k_projected_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_gate_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_rope_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_rope_output_buffer)) {
        set_error("Qwen3.5 q/k norm RoPE buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(q_gate_output_buffer, stream) ||
        !stream_matches_buffer(q_rope_output_buffer, stream) ||
        !stream_matches_buffer(k_rope_output_buffer, stream)) {
        set_error("Qwen3.5 q/k norm RoPE stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (head_dim > max_size / q_heads || head_dim > max_size / kv_heads) {
        set_error("Qwen3.5 q/k norm RoPE element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_output_elements = q_heads * head_dim;
    const size_t k_output_elements = kv_heads * head_dim;
    if (q_output_elements > max_size / 2 ||
        q_output_elements > max_size / sizeof(float) ||
        k_output_elements > max_size / sizeof(float) ||
        head_dim > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_elements = q_output_elements * 2;
    if (q_projected_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE q projected byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_bytes = q_projected_elements * sizeof(float);
    const size_t q_output_bytes = q_output_elements * sizeof(float);
    const size_t k_projected_bytes = k_output_elements * sizeof(float);
    const size_t k_output_bytes = k_output_elements * sizeof(float);
    const size_t weight_bytes = head_dim * sizeof(float);
    if (q_projected_buffer->bytes < q_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_projected_buffer->bytes < k_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE k projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE k weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_gate_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_rope_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE q RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_rope_output_buffer->bytes < k_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE k RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_projected_buffer->backend == BackendKind::Cpu) {
        qwen35_qk_norm_rope_f32_host(
            static_cast<const float *>(q_projected_buffer->ptr),
            static_cast<const float *>(k_projected_buffer->ptr),
            static_cast<const float *>(q_weight_buffer->ptr),
            static_cast<const float *>(k_weight_buffer->ptr),
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            static_cast<float *>(q_gate_output_buffer->ptr),
            static_cast<float *>(q_rope_output_buffer->ptr),
            static_cast<float *>(k_rope_output_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (qwen35_qk_norm_rope_f32_hip_kernel(
            q_projected_buffer,
            k_projected_buffer,
            q_weight_buffer,
            k_weight_buffer,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            q_gate_output_buffer,
            q_rope_output_buffer,
            k_rope_output_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty()
                      ? "Qwen3.5 q/k norm RoPE HIP kernel is unavailable"
                      : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return qwen35_qk_norm_rope_f32_hip_staging(
        q_projected_buffer,
        k_projected_buffer,
        q_weight_buffer,
        k_weight_buffer,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        q_projected_bytes,
        k_projected_bytes,
        weight_bytes,
        q_output_bytes,
        k_output_bytes,
        q_gate_output_buffer,
        q_rope_output_buffer,
        k_rope_output_buffer,
        stream);
}

ullm_status ullm_runtime_qwen35_qk_norm_rope_paged_kv_write_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *v_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream) {
    if (q_projected_buffer == nullptr || k_projected_buffer == nullptr ||
        v_projected_buffer == nullptr || q_weight_buffer == nullptr ||
        k_weight_buffer == nullptr || block_table_buffer == nullptr ||
        q_gate_output_buffer == nullptr || q_rope_output_buffer == nullptr ||
        k_cache_buffer == nullptr || v_cache_buffer == nullptr) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write received a null pointer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 ||
        block_size == 0 || cache_blocks == 0) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write dimensions must be greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write rotary_dim must be even and no greater than head_dim");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(rope_base) || rope_base <= 1.0f) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write base must be finite and greater than one");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write epsilon must be finite and greater than zero");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!buffers_share_backend(q_projected_buffer, k_projected_buffer) ||
        !buffers_share_backend(q_projected_buffer, v_projected_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_weight_buffer) ||
        !buffers_share_backend(q_projected_buffer, block_table_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_gate_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, q_rope_output_buffer) ||
        !buffers_share_backend(q_projected_buffer, k_cache_buffer) ||
        !buffers_share_backend(q_projected_buffer, v_cache_buffer)) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write buffers belong to different backends or devices");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (!stream_matches_buffer(q_gate_output_buffer, stream) ||
        !stream_matches_buffer(q_rope_output_buffer, stream) ||
        !stream_matches_buffer(k_cache_buffer, stream) ||
        !stream_matches_buffer(v_cache_buffer, stream)) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write stream belongs to a different backend or device");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    const size_t max_size = std::numeric_limits<size_t>::max();
    if (cache_blocks > max_size / block_size) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write physical cache size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t physical_tokens = cache_blocks * block_size;
    if (cache_position >= physical_tokens) {
        set_error(
            "Qwen3.5 q/k norm RoPE paged KV write cache_position exceeds physical cache capacity");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t block_index = cache_position / block_size;
    const size_t block_table_entries = block_index + 1;
    if (head_dim > max_size / q_heads || head_dim > max_size / kv_heads ||
        value_dim > max_size / kv_heads) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write token element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_output_elements = q_heads * head_dim;
    const size_t k_elements = kv_heads * head_dim;
    const size_t v_elements = kv_heads * value_dim;
    if (kv_heads > max_size / physical_tokens) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write kv head-cache count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t kv_head_cache = physical_tokens * kv_heads;
    if (head_dim > max_size / kv_head_cache || value_dim > max_size / kv_head_cache) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write cache element count overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t k_cache_elements = kv_head_cache * head_dim;
    const size_t v_cache_elements = kv_head_cache * value_dim;
    if (q_output_elements > max_size / 2 ||
        q_output_elements > max_size / sizeof(float) ||
        k_elements > max_size / sizeof(float) ||
        v_elements > max_size / sizeof(float) ||
        head_dim > max_size / sizeof(float) ||
        block_table_entries > max_size / sizeof(std::uint32_t) ||
        k_cache_elements > max_size / sizeof(float) ||
        v_cache_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_elements = q_output_elements * 2;
    if (q_projected_elements > max_size / sizeof(float)) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q projected byte size overflows");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    const size_t q_projected_bytes = q_projected_elements * sizeof(float);
    const size_t k_projected_bytes = k_elements * sizeof(float);
    const size_t v_projected_bytes = v_elements * sizeof(float);
    const size_t weight_bytes = head_dim * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);
    const size_t q_output_bytes = q_output_elements * sizeof(float);
    const size_t k_cache_bytes = k_cache_elements * sizeof(float);
    const size_t v_cache_bytes = v_cache_elements * sizeof(float);
    if (q_projected_buffer->bytes < q_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_projected_buffer->bytes < k_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write k projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_projected_buffer->bytes < v_projected_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write v projected buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_weight_buffer->bytes < weight_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write k weight buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (block_table_buffer->bytes < block_table_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write block table buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_gate_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q gate output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (q_rope_output_buffer->bytes < q_output_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write q RoPE output buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (k_cache_buffer->bytes < k_cache_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write k cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (v_cache_buffer->bytes < v_cache_bytes) {
        set_error("Qwen3.5 q/k norm RoPE paged KV write v cache buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }

    if (q_projected_buffer->backend == BackendKind::Cpu) {
        const auto *block_table = static_cast<const std::uint32_t *>(block_table_buffer->ptr);
        if (static_cast<size_t>(block_table[block_index]) >= cache_blocks) {
            set_error(
                "Qwen3.5 q/k norm RoPE paged KV write block table contains an out-of-range block id");
            return ULLM_STATUS_INVALID_ARGUMENT;
        }
        qwen35_qk_norm_rope_paged_kv_write_f32_host(
            static_cast<const float *>(q_projected_buffer->ptr),
            static_cast<const float *>(k_projected_buffer->ptr),
            static_cast<const float *>(v_projected_buffer->ptr),
            static_cast<const float *>(q_weight_buffer->ptr),
            static_cast<const float *>(k_weight_buffer->ptr),
            block_table,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            cache_position,
            block_size,
            static_cast<float *>(q_gate_output_buffer->ptr),
            static_cast<float *>(q_rope_output_buffer->ptr),
            static_cast<float *>(k_cache_buffer->ptr),
            static_cast<float *>(v_cache_buffer->ptr));
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (qwen35_qk_norm_rope_paged_kv_write_f32_hip_kernel(
            q_projected_buffer,
            k_projected_buffer,
            v_projected_buffer,
            q_weight_buffer,
            k_weight_buffer,
            block_table_buffer,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            cache_position,
            block_size,
            cache_blocks,
            q_gate_output_buffer,
            q_rope_output_buffer,
            k_cache_buffer,
            v_cache_buffer,
            stream,
            &hip_kernel_error)) {
        set_error("");
        return ULLM_STATUS_OK;
    }

    const bool require_hip_kernel =
        std::getenv("ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL") != nullptr;
    if (require_hip_kernel) {
        set_error(hip_kernel_error.empty()
                      ? "Qwen3.5 q/k norm RoPE paged KV write HIP kernel is unavailable"
                      : hip_kernel_error.c_str());
        return ULLM_STATUS_RUNTIME_ERROR;
    }
    return qwen35_qk_norm_rope_paged_kv_write_f32_hip_staging(
        q_projected_buffer,
        k_projected_buffer,
        v_projected_buffer,
        q_weight_buffer,
        k_weight_buffer,
        block_table_buffer,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        epsilon,
        cache_position,
        block_size,
        q_projected_bytes,
        k_projected_bytes,
        v_projected_bytes,
        weight_bytes,
        block_table_bytes,
        q_output_bytes,
        k_cache_bytes,
        v_cache_bytes,
        q_gate_output_buffer,
        q_rope_output_buffer,
        k_cache_buffer,
        v_cache_buffer,
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

ullm_status paged_decode_attn_f32_impl(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
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
    if ((gate_buffer != nullptr && !buffers_share_backend(q_buffer, gate_buffer)) ||
        !buffers_share_backend(q_buffer, k_cache_buffer) ||
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
    const size_t gate_bytes = gate_buffer == nullptr ? 0 : output_elements * sizeof(float);
    const size_t k_bytes = k_elements * sizeof(float);
    const size_t v_bytes = v_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t block_table_bytes = block_table_entries * sizeof(std::uint32_t);
    if (q_buffer->bytes < q_bytes) {
        set_error("f32 paged decode attention q buffer is too small");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    if (gate_buffer != nullptr && gate_buffer->bytes < gate_bytes) {
        set_error("f32 paged decode attention gate buffer is too small");
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
        const auto *gate =
            gate_buffer == nullptr ? nullptr : static_cast<const float *>(gate_buffer->ptr);
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
        if (gate != nullptr) {
            for (size_t i = 0; i < output_elements; ++i) {
                const float gate_value = gate[i];
                const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
                output[i] *= sigmoid;
            }
        }
        set_error("");
        return ULLM_STATUS_OK;
    }

    std::string hip_kernel_error;
    if (paged_decode_attn_f32_hip_kernel(
            q_buffer,
            gate_buffer,
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
        gate_buffer,
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
        gate_bytes,
        k_bytes,
        v_bytes,
        block_table_bytes,
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
    return paged_decode_attn_f32_impl(
        q_buffer,
        nullptr,
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
        stream);
}

ullm_status ullm_runtime_paged_decode_attn_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
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
    if (gate_buffer == nullptr) {
        set_error("f32 paged decode attention sigmoid gate received a null gate buffer");
        return ULLM_STATUS_INVALID_ARGUMENT;
    }
    return paged_decode_attn_f32_impl(
        q_buffer,
        gate_buffer,
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
