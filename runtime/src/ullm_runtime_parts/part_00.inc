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

#if defined(ULLM_HAVE_HIP_RUNTIME_API)
#include <hip/hip_runtime_api.h>
#include <hip/hip_deprecated.h>
#endif

namespace {

constexpr unsigned int kAq4MatvecTop1RowsPerBlock = 8u;
constexpr unsigned int kRocwmmaCachedPrefixValueGroupWidth = 64u;
// The tiled AQ4 GEMM is intentionally conservative in its first production slice.
// Keep these values in one place so the host classifier, launcher, and HIPRTC source
// cannot silently drift apart.
constexpr unsigned int kAq4TiledBm = 8u;
constexpr unsigned int kAq4TiledBn = 32u;
constexpr unsigned int kAq4TiledBk = 128u;
constexpr unsigned int kAq4TiledThreads = 256u;
static_assert(kAq4TiledBm * kAq4TiledBn == kAq4TiledThreads, "AQ4 tiled block must have one thread per output");
static_assert(kAq4TiledBk % 16u == 0u, "AQ4 tiled K tile must preserve group16 geometry");

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

bool rocwmma_cached_prefix_value_group_width_is_valid(unsigned long value) {
    return value == 16ul || value == 32ul || value == 64ul || value == 128ul || value == 256ul;
}

unsigned int rocwmma_cached_prefix_default_value_group_width(size_t new_tokens) {
    return new_tokens < 64u ? 16u : kRocwmmaCachedPrefixValueGroupWidth;
}

unsigned int rocwmma_cached_prefix_value_group_width(size_t new_tokens) {
    const char *raw = std::getenv("ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH");
    if (raw == nullptr || raw[0] == '\0') {
        return rocwmma_cached_prefix_default_value_group_width(new_tokens);
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || !rocwmma_cached_prefix_value_group_width_is_valid(value)) {
        return rocwmma_cached_prefix_default_value_group_width(new_tokens);
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

    std::string device_arch_name(int device_id) {
        load_once();
#if defined(ULLM_HAVE_HIP_RUNTIME_API)
        if (hip_get_device_properties_r0600_ != nullptr) {
            hipDeviceProp_t property{};
            if (hip_get_device_properties_r0600_(&property, device_id) == 0) {
                property.gcnArchName[sizeof(property.gcnArchName) - 1] = '\0';
                return std::string(property.gcnArchName);
            }
        }
        if (hip_get_device_properties_r0000_ != nullptr) {
            hipDeviceProp_tR0000 property{};
            if (hip_get_device_properties_r0000_(&property, device_id) == 0) {
                property.gcnArchName[sizeof(property.gcnArchName) - 1] = '\0';
                return std::string(property.gcnArchName);
            }
        }
#else
        (void)device_id;
#endif
        return "";
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

    bool memset_async(void *dst, int value, size_t bytes, void *stream, int device_id) {
        load_once();
        if (bytes == 0) {
            return true;
        }
        if (hip_memset_async_ == nullptr || !set_device(device_id)) {
            return false;
        }
        return hip_memset_async_(dst, value, bytes, stream) == 0;
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
        int device_id,
        unsigned int dynamic_shared_bytes = 0) {
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
                   dynamic_shared_bytes,
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
#if defined(ULLM_HAVE_HIP_RUNTIME_API)
    using hip_get_device_properties_r0600_fn = int (*)(hipDeviceProp_t *, int);
    using hip_get_device_properties_r0000_fn = int (*)(hipDeviceProp_tR0000 *, int);
#endif
    using hip_set_device_fn = int (*)(int);
    using hip_malloc_fn = int (*)(void **, size_t);
    using hip_free_fn = int (*)(void *);
    using hip_stream_create_fn = int (*)(void **);
    using hip_stream_destroy_fn = int (*)(void *);
    using hip_stream_synchronize_fn = int (*)(void *);
    using hip_device_synchronize_fn = int (*)();
    using hip_memcpy_async_fn = int (*)(void *, const void *, size_t, int, void *);
    using hip_memset_async_fn = int (*)(void *, int, size_t, void *);
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
#if defined(ULLM_HAVE_HIP_RUNTIME_API)
            hip_get_device_properties_r0600_ =
                reinterpret_cast<hip_get_device_properties_r0600_fn>(
                    dlsym(handle_, "hipGetDevicePropertiesR0600"));
            hip_get_device_properties_r0000_ =
                reinterpret_cast<hip_get_device_properties_r0000_fn>(
                    dlsym(handle_, "hipGetDevicePropertiesR0000"));
#endif
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
            hip_memset_async_ =
                reinterpret_cast<hip_memset_async_fn>(dlsym(handle_, "hipMemsetAsync"));
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
#if defined(ULLM_HAVE_HIP_RUNTIME_API)
    hip_get_device_properties_r0600_fn hip_get_device_properties_r0600_ = nullptr;
    hip_get_device_properties_r0000_fn hip_get_device_properties_r0000_ = nullptr;
#endif
    hip_set_device_fn hip_set_device_ = nullptr;
    hip_malloc_fn hip_malloc_ = nullptr;
    hip_free_fn hip_free_ = nullptr;
    hip_stream_create_fn hip_stream_create_ = nullptr;
    hip_stream_destroy_fn hip_stream_destroy_ = nullptr;
    hip_stream_synchronize_fn hip_stream_synchronize_ = nullptr;
    hip_device_synchronize_fn hip_device_synchronize_ = nullptr;
    hip_memcpy_async_fn hip_memcpy_async_ = nullptr;
    hip_memset_async_fn hip_memset_async_ = nullptr;
    hip_module_load_data_fn hip_module_load_data_ = nullptr;
    hip_module_get_function_fn hip_module_get_function_ = nullptr;
    hip_module_launch_kernel_fn hip_module_launch_kernel_ = nullptr;
    hip_module_unload_fn hip_module_unload_ = nullptr;
};

class HipRtcRuntime {
public:
    bool compile_wmma_fp8_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            wmma_fp8_probe_kernel_source(),
            "ullm_wmma_fp8_probe.hip",
            code,
            error);
    }

    bool compile_wmma_fp8_qk_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            wmma_fp8_qk_probe_kernel_source(),
            "ullm_wmma_fp8_qk_probe.hip",
            code,
            error);
    }

    bool compile_rocwmma_fp8_qk_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            rocwmma_fp8_qk_probe_kernel_source(),
            "ullm_rocwmma_fp8_qk_probe.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_rocwmma_fp8_attn_probe_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            rocwmma_fp8_attn_probe_kernel_source(),
            "ullm_rocwmma_fp8_attn_probe.hip",
            code,
            error,
            rocwmma_include_options());
    }

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

    bool compile_aq4_matvec_tiled_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_tiled_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_tiled_f32.hip",
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

    bool compile_qwen35_qk_norm_rope_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            qwen35_qk_norm_rope_batch_kernel_source(),
            "ullm_qwen35_qk_norm_rope_batch_f32.hip",
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

    bool compile_causal_attn_f32_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_flash2_f32_kernel_source(),
            "ullm_causal_attn_f32_flash2.hip",
            code,
            error);
    }

    bool compile_causal_attn_batch_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_batch_kernel_source(),
            "ullm_causal_attn_batch_f32.hip",
            code,
            error);
    }

    bool compile_causal_attn_batch_f32_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            causal_attn_batch_flash2_f32_kernel_source(),
            "ullm_causal_attn_batch_f32_flash2.hip",
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

    bool compile_cached_prefix_attn_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_attn_kernel_source(),
            "ullm_cached_prefix_attn_f32.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_fp8_e4m3_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = cached_prefix_attn_fp8_e4m3_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_cached_prefix_attn_fp8_e4m3.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_f32_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_flash2_f32_kernel_source(),
            "ullm_cached_prefix_attn_f32_flash2.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_fp8_e4m3_flash2_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_flash2_fp8_e4m3_kernel_source(),
            "ullm_cached_prefix_attn_fp8_e4m3_flash2.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_fp8_e4m3_rocwmma_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_rocwmma_fp8_e4m3_kernel_source(),
            "ullm_cached_prefix_attn_fp8_e4m3_rocwmma.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_sq_fp8_matvec_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_f32.hip",
            code,
            error);
    }

    bool compile_sq_fp8_matvec_batch_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_batch_f32.hip",
            code,
            error);
    }

    bool compile_sq_fp8_matvec_pair_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_pair_f32.hip",
            code,
            error);
    }

    bool compile_sq_fp8_matvec_triple_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq_fp8_matvec_kernel_source(),
            "ullm_sq_fp8_matvec_triple_f32.hip",
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

    bool compile_paged_chunk_kernel(const std::string &arch, std::vector<char> *code, std::string *error) {
        return compile_kernel(
            arch,
            paged_chunk_kernel_source(),
            "ullm_paged_chunk_f32.hip",
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
        std::string *error,
        const std::vector<std::string> &extra_options = {}) {
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

        std::vector<std::string> option_storage;
        option_storage.reserve(3u + extra_options.size());
        option_storage.push_back("--offload-arch=" + arch);
        option_storage.push_back("--std=c++17");
        option_storage.push_back("-O3");
        option_storage.insert(option_storage.end(), extra_options.begin(), extra_options.end());
        std::vector<const char *> options;
        options.reserve(option_storage.size());
        for (const std::string &option : option_storage) {
            options.push_back(option.c_str());
        }
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

    static std::vector<std::string> rocwmma_include_options() {
        std::vector<std::string> options;
        const char *rocm_path = std::getenv("ROCM_PATH");
        if (rocm_path != nullptr && rocm_path[0] != '\0') {
            options.push_back(std::string("-I") + rocm_path + "/include");
        }
        options.push_back("-I/opt/rocm/include");
        options.push_back("-I/opt/rocm-7.2.1/include");
        return options;
    }

#include "../ullm_runtime_hiprtc_sources.inc"
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

std::vector<std::string> hip_arch_candidates(int device_id);

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
    copy_cstr(
        info->gcn_arch_name,
        sizeof(info->gcn_arch_name),
        hip_runtime().device_arch_name(hip_index));
    info->flags = static_cast<uint32_t>(hip_runtime().runtime_version());
}

enum class BackendKind : uint32_t {
    Cpu = 0,
    Hip = 1,
};

constexpr int HIP_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int HIP_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int HIP_MEMCPY_DEVICE_TO_DEVICE = 3;

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

bool aq4_scale_indices_host_are_valid(
    const std::uint8_t *scale_indices,
    size_t groups,
    size_t scale_count) {
    for (size_t group = 0; group < groups; ++group) {
        if (static_cast<size_t>(scale_indices[group]) >= scale_count) {
            set_error("AQ4 matvec batch scale index is out of range");
            return false;
        }
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

class HipWmmaFp8ProbeKernelCache {
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
            if (!hiprtc_runtime().compile_wmma_fp8_probe_kernel(arch, &code, &compile_error)) {
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
                    "ullm_wmma_fp8_probe_kernel",
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
            compile_errors.empty() ? "failed to build WMMA FP8 probe HIP kernel" :
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

HipWmmaFp8ProbeKernelCache &hip_wmma_fp8_probe_kernel_cache() {
    static HipWmmaFp8ProbeKernelCache cache;
    return cache;
}

bool wmma_fp8_probe_hip_kernel(
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    void *function = hip_wmma_fp8_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for WMMA FP8 probe";
        }
        return false;
    }
    return true;
}

class HipWmmaFp8QkProbeKernelCache {
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
            if (!hiprtc_runtime().compile_wmma_fp8_qk_probe_kernel(arch, &code, &compile_error)) {
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
                    "ullm_wmma_fp8_qk_probe_kernel",
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
            compile_errors.empty() ? "failed to build WMMA FP8 QK probe HIP kernel" :
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

HipWmmaFp8QkProbeKernelCache &hip_wmma_fp8_qk_probe_kernel_cache() {
    static HipWmmaFp8QkProbeKernelCache cache;
    return cache;
}

bool wmma_fp8_qk_probe_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    if (major != 12) {
        if (error != nullptr) {
            *error = "WMMA FP8 QK probe requires RDNA4/gfx12 HIP device";
        }
        return false;
    }

    void *function = hip_wmma_fp8_qk_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const void *q_ptr = q_buffer->ptr;
    const void *k_ptr = k_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&q_ptr, &k_ptr, &output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for WMMA FP8 QK probe";
        }
        return false;
    }
    return true;
}

class HipRocwmmaFp8QkProbeKernelCache {
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
            if (!hiprtc_runtime().compile_rocwmma_fp8_qk_probe_kernel(
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
                    "ullm_rocwmma_fp8_qk_probe_kernel",
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
            compile_errors.empty() ? "failed to build rocWMMA FP8 QK probe HIP kernel" :
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

HipRocwmmaFp8QkProbeKernelCache &hip_rocwmma_fp8_qk_probe_kernel_cache() {
    static HipRocwmmaFp8QkProbeKernelCache cache;
    return cache;
}

bool rocwmma_fp8_qk_probe_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    if (major != 12) {
        if (error != nullptr) {
            *error = "rocWMMA FP8 QK probe requires RDNA4/gfx12 HIP device";
        }
        return false;
    }

    void *function = hip_rocwmma_fp8_qk_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const void *q_ptr = q_buffer->ptr;
    const void *k_ptr = k_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&q_ptr, &k_ptr, &output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for rocWMMA FP8 QK probe";
        }
        return false;
    }
    return true;
}

class HipRocwmmaFp8AttnProbeKernelCache {
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
            if (!hiprtc_runtime().compile_rocwmma_fp8_attn_probe_kernel(
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
                    "ullm_rocwmma_fp8_attn_probe_kernel",
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
            compile_errors.empty() ? "failed to build rocWMMA FP8 attention probe HIP kernel" :
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

HipRocwmmaFp8AttnProbeKernelCache &hip_rocwmma_fp8_attn_probe_kernel_cache() {
    static HipRocwmmaFp8AttnProbeKernelCache cache;
    return cache;
}

bool rocwmma_fp8_attn_probe_hip_kernel(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    std::string *error) {
    const int device_id = output_buffer->hip_device_id;
    int major = 0;
    int minor = 0;
    hip_runtime().device_compute_capability(device_id, &major, &minor);
    if (major != 12) {
        if (error != nullptr) {
            *error = "rocWMMA FP8 attention probe requires RDNA4/gfx12 HIP device";
        }
        return false;
    }

    void *function = hip_rocwmma_fp8_attn_probe_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }

    const void *q_ptr = q_buffer->ptr;
    const void *k_ptr = k_buffer->ptr;
    const void *v_ptr = v_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
    void *kernel_params[] = {&q_ptr, &k_ptr, &v_ptr, &output_ptr};
    void *hip_stream = stream == nullptr ? nullptr : stream->stream;
    if (!hip_runtime().module_launch_kernel(
            function,
            1u,
            32u,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for rocWMMA FP8 attention probe";
        }
        return false;
    }
    return true;
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

class HipAq4MatvecTiledKernelCache {
public:
    void *function_for_device(int device_id, std::string *error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = modules_.find(device_id);
        if (found != modules_.end()) {
            return found->second->function;
        }
        auto failed = failures_.find(device_id);
        if (failed != failures_.end()) {
            append_error(error, failed->second);
            return nullptr;
        }

        // This cache is deliberately exact-architecture: gfx1200 and other gfx12 variants
        // must stay on the legacy kernel until their own measured tile is admitted.
        if (hip_runtime().device_arch_name(device_id) != "gfx1201") {
            append_error(error, "AQ4 tiled GEMM requires exact gfx1201 architecture");
            return nullptr;
        }

        std::vector<char> code;
        std::string compile_error;
        if (!hiprtc_runtime().compile_aq4_matvec_tiled_kernel(
                "gfx1201",
                &code,
                &compile_error)) {
            const std::string message = compile_error.empty() ?
                "failed to build AQ4 tiled GEMM HIP kernel" : compile_error;
            failures_.emplace(device_id, message);
            append_error(error, message);
            return nullptr;
        }
        void *module = nullptr;
        if (!hip_runtime().module_load_data(&module, code.data(), device_id)) {
            const std::string message = "hipModuleLoadData failed for gfx1201 AQ4 tiled GEMM";
            failures_.emplace(device_id, message);
            append_error(error, message);
            return nullptr;
        }
        void *function = nullptr;
        if (!hip_runtime().module_get_function(
                &function,
                module,
                "ullm_aq4_gemm_tiled_f32_kernel",
                device_id)) {
            hip_runtime().module_unload(module, device_id);
            const std::string message = "hipModuleGetFunction failed for AQ4 tiled GEMM";
            failures_.emplace(device_id, message);
            append_error(error, message);
            return nullptr;
        }

        auto loaded = std::make_unique<LoadedModule>();
        loaded->module = module;
        loaded->function = function;
        loaded->arch = "gfx1201";
        void *result = loaded->function;
        modules_.emplace(device_id, std::move(loaded));
        return result;
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
    std::unordered_map<int, std::string> failures_;
};

HipAq4MatvecTiledKernelCache &hip_aq4_matvec_tiled_kernel_cache() {
    static HipAq4MatvecTiledKernelCache cache;
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

enum class Aq4MatvecBatchDispatchKind : uint32_t {
    Legacy = 0u,
    Tiled = 1u,
};

bool aq4_tiled_experimental_enabled() {
    const char *raw = std::getenv("ULLM_EXPERIMENTAL_HIP_AQ4_TILED_GEMM");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
}

bool aq4_matvec_batch_tiled_shape_is_supported(
    const std::string &arch,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    if (!aq4_tiled_experimental_enabled()) {
        return false;
    }
    // Keep dispatch fail-closed. The initial tile handles row/K multiples exactly while
    // guarding batch tails in-kernel; ragged rows or K remain on the established kernel.
    if (arch != "gfx1201" || group_size != 16u || batch_count < kAq4TiledBm || rows == 0u ||
        cols == 0u || rows % kAq4TiledBn != 0u || cols % kAq4TiledBk != 0u) {
        return false;
    }
    if (rows > std::numeric_limits<size_t>::max() - (kAq4TiledBn - 1u)) {
        return false;
    }
    if (cols > std::numeric_limits<size_t>::max() - (kAq4TiledBk - 1u)) {
        return false;
    }
    if (batch_count > std::numeric_limits<size_t>::max() - (kAq4TiledBm - 1u)) {
        return false;
    }
    return true;
}

Aq4MatvecBatchDispatchKind aq4_matvec_batch_dispatch_kind_for_device(
    int device_id,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count) {
    if (device_id < 0) {
        return Aq4MatvecBatchDispatchKind::Legacy;
    }
    return aq4_matvec_batch_tiled_shape_is_supported(
               hip_runtime().device_arch_name(device_id),
               group_size,
               rows,
               cols,
               batch_count) ?
        Aq4MatvecBatchDispatchKind::Tiled :
        Aq4MatvecBatchDispatchKind::Legacy;
}

bool aq4_matvec_batch_f32_tiled_hip_kernel(
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
    void *function = hip_aq4_matvec_tiled_kernel_cache().function_for_device(device_id, error);
    if (function == nullptr) {
        return false;
    }
    const size_t grid_x = (rows + kAq4TiledBn - 1u) / kAq4TiledBn;
    const size_t grid_y = (batch_count + kAq4TiledBm - 1u) / kAq4TiledBm;
    if (grid_x == 0u || grid_y == 0u ||
        grid_x > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        grid_y > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        if (error != nullptr) {
            *error = "AQ4 tiled GEMM dimensions exceed HIP grid limit";
        }
        return false;
    }
    unsigned long long kernel_scale_count = static_cast<unsigned long long>(scale_count);
    unsigned long long kernel_group_size = static_cast<unsigned long long>(group_size);
    unsigned long long kernel_rows = static_cast<unsigned long long>(rows);
    unsigned long long kernel_cols = static_cast<unsigned long long>(cols);
    unsigned long long kernel_batch_count = static_cast<unsigned long long>(batch_count);
    unsigned long long kernel_row_scale_count = static_cast<unsigned long long>(row_scale_count);
    void *index_ptr = index_buffer->ptr;
    void *scale_ptr = scale_buffer->ptr;
    void *codebook_ptr = codebook_buffer->ptr;
    void *scale_values_ptr = scale_values_buffer->ptr;
    void *input_ptr = input_buffer->ptr;
    void *row_scale_ptr = row_scale_buffer == nullptr ? nullptr : row_scale_buffer->ptr;
    void *output_ptr = output_buffer->ptr;
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
            static_cast<unsigned int>(grid_y),
            kAq4TiledThreads,
            kernel_params,
            hip_stream,
            device_id)) {
        if (error != nullptr) {
            *error = "hipModuleLaunchKernel failed for AQ4 tiled GEMM";
        }
        return false;
    }
    return true;
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

#include "../kernels/sq8_0/sq8_0_matvec_runtime.inc"

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

void qwen35_qk_norm_rope_batch_f32_host(
    const float *q_projected,
    const float *k_projected,
    const float *q_weight,
    const float *k_weight,
    size_t q_heads,
    size_t kv_heads,
    size_t sequence_len,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    float *q_gate_output,
    float *q_rope_output,
    float *k_rope_output) {
    const size_t q_projected_stride = q_heads * head_dim * 2;
    const size_t q_output_stride = q_heads * head_dim;
    const size_t k_stride = kv_heads * head_dim;
    for (size_t timestep = 0; timestep < sequence_len; ++timestep) {
        qwen35_qk_norm_rope_f32_host(
            q_projected + timestep * q_projected_stride,
            k_projected + timestep * k_stride,
            q_weight,
            k_weight,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset + timestep,
            rope_base,
            epsilon,
            q_gate_output + timestep * q_output_stride,
            q_rope_output + timestep * q_output_stride,
            k_rope_output + timestep * k_stride);
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

void cached_prefix_attn_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t token_index = 0; token_index < new_tokens; ++token_index) {
        const size_t cache_len = cached_prefix_len + token_index + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (token_index * q_heads + q_head) * head_dim;
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

            const size_t output_base = (token_index * q_heads + q_head) * value_dim;
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
}

float fp8_e4m3_to_f32_unscaled(uint8_t value) {
    const unsigned int raw = static_cast<unsigned int>(value);
    const unsigned int sign = raw >> 7;
    const unsigned int exponent = (raw >> 3) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    float magnitude = 0.0f;
    if (exponent == 0u) {
        magnitude = static_cast<float>(mantissa) * 0.001953125f;
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f,
                               static_cast<int>(exponent) - 7);
    }
    return sign == 0u ? magnitude : -magnitude;
}

void wmma_fp8_qk_probe_host(
    const uint8_t *q,
    const uint8_t *k,
    float *output) {
    constexpr size_t tile = 16;
    for (size_t row = 0; row < tile; ++row) {
        for (size_t col = 0; col < tile; ++col) {
            float sum = 0.0f;
            for (size_t dim = 0; dim < tile; ++dim) {
                const float q_value = fp8_e4m3_to_f32_unscaled(q[row * tile + dim]);
                const float k_value = fp8_e4m3_to_f32_unscaled(k[col * tile + dim]);
                sum += q_value * k_value;
            }
            output[row * tile + col] = sum;
        }
    }
}

void rocwmma_fp8_attn_probe_host(
    const uint8_t *q,
    const uint8_t *k,
    const float *v,
    float *output) {
    constexpr size_t q_tokens = 16;
    constexpr size_t kv_tokens = 32;
    constexpr size_t head_dim = 16;
    for (size_t row = 0; row < q_tokens; ++row) {
        float scores[kv_tokens];
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t token = 0; token < kv_tokens; ++token) {
            float score = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                const float q_value = fp8_e4m3_to_f32_unscaled(q[row * head_dim + dim]);
                const float k_value = fp8_e4m3_to_f32_unscaled(k[token * head_dim + dim]);
                score += q_value * k_value;
            }
            scores[token] = score;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0f;
        for (size_t value = 0; value < head_dim; ++value) {
            output[row * head_dim + value] = 0.0f;
        }
        for (size_t token = 0; token < kv_tokens; ++token) {
            const float weight = std::exp(scores[token] - max_score);
            denominator += weight;
            for (size_t value = 0; value < head_dim; ++value) {
                output[row * head_dim + value] += weight * v[token * head_dim + value];
            }
        }
        for (size_t value = 0; value < head_dim; ++value) {
            output[row * head_dim + value] /= denominator;
        }
    }
}

void cached_prefix_attn_fp8_e4m3_host(
    const float *q,
    const uint8_t *k_cache,
    const uint8_t *v_cache,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t token_index = 0; token_index < new_tokens; ++token_index) {
        const size_t cache_len = cached_prefix_len + token_index + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (token_index * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                score *= softmax_scale;
                max_score = std::max(max_score, score);
            }

            float denominator = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                denominator += std::exp(score * softmax_scale - max_score);
            }

            const size_t output_base = (token_index * q_heads + q_head) * value_dim;
            for (size_t value = 0; value < value_dim; ++value) {
                float weighted = 0.0f;
                for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                    const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    float score = 0.0f;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        score += q[q_base + dim] *
                                 (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                    }
                    const float weight = std::exp(score * softmax_scale - max_score);
                    const size_t v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted +=
                        weight * (fp8_e4m3_to_f32_unscaled(v_cache[v_index]) * v_scale);
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
}

void cached_prefix_attn_fp8_e4m3_rocwmma_host(
    const uint8_t *q,
    const uint8_t *k_cache,
    const uint8_t *v_cache,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t token_index = 0; token_index < new_tokens; ++token_index) {
        const size_t cache_len = cached_prefix_len + token_index + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (token_index * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += (fp8_e4m3_to_f32_unscaled(q[q_base + dim]) * q_scale) *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                score *= softmax_scale;
                max_score = std::max(max_score, score);
            }

            float denominator = 0.0f;
            for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += (fp8_e4m3_to_f32_unscaled(q[q_base + dim]) * q_scale) *
                             (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                }
                denominator += std::exp(score * softmax_scale - max_score);
            }

            const size_t output_base = (token_index * q_heads + q_head) * value_dim;
            for (size_t value = 0; value < value_dim; ++value) {
                float weighted = 0.0f;
                for (size_t source_timestep = 0; source_timestep < cache_len; ++source_timestep) {
                    const size_t k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    float score = 0.0f;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        score += (fp8_e4m3_to_f32_unscaled(q[q_base + dim]) * q_scale) *
                                 (fp8_e4m3_to_f32_unscaled(k_cache[k_base + dim]) * k_scale);
                    }
                    const float weight = std::exp(score * softmax_scale - max_score);
                    const size_t v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted +=
                        weight * (fp8_e4m3_to_f32_unscaled(v_cache[v_index]) * v_scale);
                }
                output[output_base + value] = weighted / denominator;
            }
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

void paged_kv_write_chunk_f32_host(
    const float *k,
    const float *v,
    const std::uint32_t *block_table,
    size_t cache_start,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float *k_cache,
    float *v_cache) {
    (void)cache_blocks;
    for (size_t row = 0; row < m; ++row) {
        const size_t logical_position = cache_start + row;
        const size_t block_index = logical_position / block_size;
        const size_t block_offset = logical_position - block_index * block_size;
        const size_t physical_timestep =
            static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
        for (size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
            const size_t k_src = (row * kv_heads + kv_head) * head_dim;
            const size_t k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
            std::copy(k + k_src, k + k_src + head_dim, k_cache + k_dst);
            const size_t v_src = (row * kv_heads + kv_head) * value_dim;
            const size_t v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
            std::copy(v + v_src, v + v_src + value_dim, v_cache + v_dst);
        }
    }
}

void paged_causal_gqa_chunk_f32_host(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const std::uint32_t *block_table,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float *output) {
    const size_t q_per_kv = q_heads / kv_heads;
    for (size_t row = 0; row < m; ++row) {
        const size_t cache_len = cached_prefix_len + row + 1;
        for (size_t q_head = 0; q_head < q_heads; ++q_head) {
            const size_t kv_head = q_head / q_per_kv;
            const size_t q_base = (row * q_heads + q_head) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t source = 0; source < cache_len; ++source) {
                const size_t block_index = source / block_size;
                const size_t block_offset = source - block_index * block_size;
                const size_t physical =
                    static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
                const size_t k_base = (physical * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                max_score = std::max(max_score, score * softmax_scale);
            }
            float denominator = 0.0f;
            const size_t output_base = (row * q_heads + q_head) * value_dim;
            std::vector<float> weighted(value_dim, 0.0f);
            for (size_t source = 0; source < cache_len; ++source) {
                const size_t block_index = source / block_size;
                const size_t block_offset = source - block_index * block_size;
                const size_t physical =
                    static_cast<size_t>(block_table[block_index]) * block_size + block_offset;
                const size_t k_base = (physical * kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += q[q_base + dim] * k_cache[k_base + dim];
                }
                const float weight = std::exp(score * softmax_scale - max_score);
                denominator += weight;
                const size_t v_base = (physical * kv_heads + kv_head) * value_dim;
                for (size_t value = 0; value < value_dim; ++value) {
                    weighted[value] += weight * v_cache[v_base + value];
                }
            }
            for (size_t value = 0; value < value_dim; ++value) {
                output[output_base + value] = weighted[value] / denominator;
            }
        }
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
