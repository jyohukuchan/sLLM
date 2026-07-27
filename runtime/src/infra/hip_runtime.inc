// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

