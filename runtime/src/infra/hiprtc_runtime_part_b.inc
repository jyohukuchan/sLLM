// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
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

