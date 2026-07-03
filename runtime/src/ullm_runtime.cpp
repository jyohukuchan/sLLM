// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "ullm_runtime.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
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
};

HipRuntime &hip_runtime() {
    static HipRuntime runtime;
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

    std::vector<std::uint8_t> host_indices(required_index_bytes);
    std::vector<std::uint8_t> host_scale_indices(groups);
    std::vector<float> host_codebook(codebook_entries);
    std::vector<float> host_output(elements);
    ullm_runtime_stream *copy_stream = stream;
    void *hip_stream = copy_stream == nullptr ? nullptr : copy_stream->stream;
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
    if (!synchronize_hip_staging(copy_stream, device_id)) {
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
    if (!synchronize_hip_staging(copy_stream, device_id)) {
        set_error("failed to synchronize AQ4 HIP output staging copy");
        return ULLM_STATUS_RUNTIME_ERROR;
    }

    set_error("");
    return ULLM_STATUS_OK;
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
