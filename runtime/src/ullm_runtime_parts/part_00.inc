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
#include <type_traits>
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
constexpr unsigned int kAq4RegisterBm4 = 4u;
constexpr unsigned int kAq4RegisterBm8 = 8u;
constexpr unsigned int kAq4RegisterGroup8 = 8u;
constexpr unsigned int kAq4RegisterGroup16 = 16u;
constexpr unsigned int kAq4RegisterRowsPerBlock = 32u;
constexpr unsigned int kAq4RegisterLanesPerRow = 8u;
constexpr unsigned int kAq4RegisterThreads = 256u;
static_assert(
    kAq4RegisterRowsPerBlock * kAq4RegisterLanesPerRow == kAq4RegisterThreads,
    "AQ4 register tile must map eight lanes to each row");
// One WMMA workgroup contains eight wave32s, each producing a 16x16 column slice of a
// group16/M=128 output tile with 16 rows and 128 batch columns.
constexpr unsigned int kAq4WmmaPrototypeBm = 128u;
constexpr unsigned int kAq4WmmaPrototypeBn = 16u;
constexpr unsigned int kAq4WmmaPrototypeBk = 32u;
constexpr unsigned int kAq4WmmaPrototypeV3Bk = 64u;
constexpr unsigned int kAq4WmmaPrototypeThreads = 256u;
static_assert(
    (kAq4WmmaPrototypeBm / 16u) * 32u == kAq4WmmaPrototypeThreads,
    "AQ4 WMMA prototype requires eight wave32s");
static_assert(
    kAq4WmmaPrototypeBk == 32u,
    "AQ4 WMMA prototype Wide-K tile must contain two K=16 WMMA operations");
static_assert(
    kAq4WmmaPrototypeV3Bk == 2u * kAq4WmmaPrototypeBk,
    "AQ4 WMMA prototype v3 stage must contain two Wide-K microtiles");

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
        if (kind < 1 || kind > 3) {
            return false;
        }
        if (bytes == 0) {
            return true;
        }
        if (!set_device(device_id)) {
            return false;
        }
        const auto hip_dst = static_cast<hip_device_pointer>(dst);
        const auto hip_src = static_cast<hip_device_pointer>(const_cast<void *>(src));
        const auto hip_stream = static_cast<hip_stream_handle>(stream);
        switch (kind) {
        case 1:
            return hip_memcpy_htod_async_ != nullptr &&
                hip_memcpy_htod_async_(hip_dst, src, bytes, hip_stream) == 0;
        case 2:
            return hip_memcpy_dtoh_async_ != nullptr &&
                hip_memcpy_dtoh_async_(dst, hip_src, bytes, hip_stream) == 0;
        case 3:
            return hip_memcpy_dtod_async_ != nullptr &&
                hip_memcpy_dtod_async_(hip_dst, hip_src, bytes, hip_stream) == 0;
        default:
            return false;
        }
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
#if defined(ULLM_HAVE_HIP_RUNTIME_API)
    using hip_status_code = hipError_t;
    using hip_device_pointer = hipDeviceptr_t;
    using hip_stream_handle = hipStream_t;
#else
    using hip_status_code = int;
    using hip_device_pointer = void *;
    using hip_stream_handle = void *;
#endif
    using hip_memcpy_htod_async_fn =
        hip_status_code (*)(hip_device_pointer, const void *, size_t, hip_stream_handle);
    using hip_memcpy_dtoh_async_fn =
        hip_status_code (*)(void *, hip_device_pointer, size_t, hip_stream_handle);
    using hip_memcpy_dtod_async_fn =
        hip_status_code (*)(hip_device_pointer, hip_device_pointer, size_t, hip_stream_handle);
#if defined(ULLM_HAVE_HIP_RUNTIME_API)
    static_assert(std::is_same_v<hip_memcpy_htod_async_fn, decltype(&hipMemcpyHtoDAsync)>);
    static_assert(std::is_same_v<hip_memcpy_dtoh_async_fn, decltype(&hipMemcpyDtoHAsync)>);
    static_assert(std::is_same_v<hip_memcpy_dtod_async_fn, decltype(&hipMemcpyDtoDAsync)>);
#endif
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
            hip_memcpy_htod_async_ = reinterpret_cast<hip_memcpy_htod_async_fn>(
                dlsym(handle_, "hipMemcpyHtoDAsync"));
            hip_memcpy_dtoh_async_ = reinterpret_cast<hip_memcpy_dtoh_async_fn>(
                dlsym(handle_, "hipMemcpyDtoHAsync"));
            hip_memcpy_dtod_async_ = reinterpret_cast<hip_memcpy_dtod_async_fn>(
                dlsym(handle_, "hipMemcpyDtoDAsync"));
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
    hip_memcpy_htod_async_fn hip_memcpy_htod_async_ = nullptr;
    hip_memcpy_dtoh_async_fn hip_memcpy_dtoh_async_ = nullptr;
    hip_memcpy_dtod_async_fn hip_memcpy_dtod_async_ = nullptr;
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

    bool compile_aq4_matvec_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_shuffle_prototype_f32.hip",
            code,
            error);
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

    bool compile_aq4_matvec_register_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_register_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_register_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_wmma_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wmma_prototype_kernel_source();
        if (source.empty()) {
            append_error(error, "AQ4 WMMA production direct-output source transformation failed");
            return false;
        }
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_wmma_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_ragged_m_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wmma_ragged_m_prototype_kernel_source();
        if (source.empty()) {
            append_error(error, "AQ4 WMMA ragged-M prototype source transformation failed");
            return false;
        }
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_wmma_ragged_m_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_prototype_v3_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            aq4_matvec_wmma_prototype_v3_kernel_source(),
            "ullm_aq4_gemm_wmma_prototype_v3_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_group8_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            aq4_matvec_wmma_group8_prototype_kernel_source(),
            "ullm_aq4_gemm_wmma_group8_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
    }

    bool compile_aq4_matvec_wmma_group8_ragged_m_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_wmma_group8_ragged_m_prototype_kernel_source();
        if (source.empty()) {
            append_error(error, "AQ4 group8 ragged-M WMMA prototype source transformation failed");
            return false;
        }
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_gemm_wmma_group8_ragged_m_prototype_f32.hip",
            code,
            error,
            rocwmma_include_options());
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

    bool compile_aq4_matvec_add_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_add_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_add_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_add_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_add_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_add_shuffle_prototype_f32.hip",
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

    bool compile_aq4_matvec_triple_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_triple_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_triple_shuffle_prototype_f32.hip",
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

    bool compile_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source =
            aq4_matvec_qkv_z_gate_beta_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source =
            aq4_matvec_qkv_z_gate_beta_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_f32.hip",
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

    bool compile_aq4_matvec_silu_mul_wide_load_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_silu_mul_wide_load_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_silu_mul_wide_load_prototype_f32.hip",
            code,
            error);
    }

    bool compile_aq4_matvec_silu_mul_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = aq4_matvec_silu_mul_shuffle_prototype_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_aq4_matvec_silu_mul_shuffle_prototype_f32.hip",
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
        const std::string source = rmsnorm_kernel_source_for_arch(arch);
        return compile_kernel(arch, source.c_str(), "ullm_rmsnorm_f32.hip", code, error);
    }

    bool compile_rmsnorm_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = rmsnorm_shuffle_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_rmsnorm_shuffle_prototype_f32.hip",
            code,
            error);
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
        const std::string source = segmented_rmsnorm_silu_mul_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_segmented_rmsnorm_silu_mul_f32.hip",
            code,
            error);
    }

    bool compile_segmented_rmsnorm_silu_mul_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = segmented_rmsnorm_silu_mul_shuffle_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_segmented_rmsnorm_silu_mul_shuffle_prototype_f32.hip",
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

    bool compile_cached_prefix_attn_f32_flash2_staged_wave32_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source =
            cached_prefix_flash2_f32_staged_wave32_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_sq8_0_cached_prefix_attn_f32_flash2_staged_wave32_prototype.hip",
            code,
            error);
    }

    bool compile_cached_prefix_attn_f32_flash2_gqa_grouped_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            cached_prefix_flash2_f32_gqa_grouped_serial_kernel_source(),
            "ullm_cached_prefix_attn_f32_flash2_gqa_grouped.hip",
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

    bool compile_sq8_1_matvec_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            sq8_1_matvec_kernel_source(),
            "ullm_sq8_1_matvec_f32.hip",
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

    bool compile_paged_causal_gqa_chunk_wmma_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            paged_causal_gqa_chunk_wmma_kernel_source(),
            "ullm_paged_causal_gqa_chunk_wmma_f32.hip",
            code,
            error,
            rocwmma_include_options());
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
        const std::string source = linear_attn_qkv_prepare_kernel_source_for_arch(arch);
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_linear_attn_qkv_prepare_f32.hip",
            code,
            error);
    }

    bool compile_linear_attn_qkv_prepare_shuffle_prototype_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        const std::string source = linear_attn_qkv_prepare_shuffle_prototype_kernel_source();
        return compile_kernel(
            arch,
            source.c_str(),
            "ullm_linear_attn_qkv_prepare_shuffle_prototype_f32.hip",
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

    bool compile_linear_attn_recurrent_generic_kernel(
        const std::string &arch,
        std::vector<char> *code,
        std::string *error) {
        return compile_kernel(
            arch,
            linear_attn_recurrent_generic_kernel_source(),
            "ullm_linear_attn_recurrent_generic_f32.hip",
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

#include "../parts/probe_kernels.inc"
#include "../parts/aq4_kernels.inc"
#include "../parts/primitives_kernels_part_a.inc"
