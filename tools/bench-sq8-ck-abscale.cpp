#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/type_convert.hpp"

namespace {

#define HIP_CHECK(call)                                                                      \
    do {                                                                                     \
        const hipError_t err__ = (call);                                                     \
        if (err__ != hipSuccess) {                                                           \
            throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(err__)); \
        }                                                                                    \
    } while (false)

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

struct Options {
    int device = 0;
    std::size_t m = 8;
    std::size_t n = 5120;
    std::size_t k = 5120;
    int warmups = 5;
    int repeats = 20;
};

std::string json_string(std::string_view value);

class ExitError : public std::runtime_error {
  public:
    ExitError(int code, std::string message) : std::runtime_error(std::move(message)), code_(code) {}

    int code() const { return code_; }

  private:
    int code_;
};

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes == 0) {
            throw std::runtime_error("zero-byte device allocation requested");
        }
        HIP_CHECK(hipMalloc(&ptr_, bytes));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            (void)hipFree(ptr_);
        }
    }

    void* get() { return ptr_; }
    const void* get() const { return ptr_; }
    std::size_t bytes() const { return bytes_; }

  private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

class HipStream {
  public:
    HipStream() { HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking)); }

    HipStream(const HipStream&) = delete;
    HipStream& operator=(const HipStream&) = delete;

    ~HipStream() {
        if (stream_ != nullptr) {
            (void)hipStreamDestroy(stream_);
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
            (void)hipEventDestroy(event_);
        }
    }

    hipEvent_t get() const { return event_; }

  private:
    hipEvent_t event_ = nullptr;
};

struct CandidateGroup {
    std::string name;
    std::vector<std::unique_ptr<DeviceOp>> instances;
};

struct CandidateMeasurement {
    std::string group;
    std::string instance;
    bool supported = false;
    bool runnable = false;
    bool correct = false;
    double max_abs_diff = 0.0;
    float p50_ms = 0.0f;
    std::string error;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--device N] [--m N] [--n N] [--k N]"
              << " [--warmups N] [--repeats N]\n";
    std::exit(2);
}

std::size_t parse_size(std::string_view text, const char* option) {
    if (text.empty() || text.front() == '-') {
        throw ExitError(2, std::string(option) + " must be a non-negative integer");
    }
    std::string owned(text);
    char* end = nullptr;
    const unsigned long long value = std::strtoull(owned.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') {
        throw ExitError(2, std::string("invalid integer for ") + option);
    }
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw ExitError(2, std::string(option) + " is too large");
    }
    return static_cast<std::size_t>(value);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto need_value = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                throw ExitError(2, std::string("missing value for ") + std::string(arg));
            }
            return std::string_view(argv[++i]);
        };

        if (arg == "--device") {
            const std::size_t value = parse_size(need_value(), "--device");
            if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw ExitError(2, "--device is too large");
            }
            options.device = static_cast<int>(value);
        } else if (arg == "--m") {
            options.m = parse_size(need_value(), "--m");
        } else if (arg == "--n") {
            options.n = parse_size(need_value(), "--n");
        } else if (arg == "--k") {
            options.k = parse_size(need_value(), "--k");
        } else if (arg == "--warmups") {
            const std::size_t value = parse_size(need_value(), "--warmups");
            if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw ExitError(2, "--warmups is too large");
            }
            options.warmups = static_cast<int>(value);
        } else if (arg == "--repeats") {
            const std::size_t value = parse_size(need_value(), "--repeats");
            if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw ExitError(2, "--repeats is too large");
            }
            options.repeats = static_cast<int>(value);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            throw ExitError(2, std::string("unknown option: ") + std::string(arg));
        }
    }

    if (options.m == 0 || options.n == 0 || options.k == 0) {
        throw ExitError(2, "M, N, and K must be positive");
    }
    if (options.repeats <= 0) {
        throw ExitError(2, "--repeats must be positive");
    }

    const auto ck_max = static_cast<std::size_t>(std::numeric_limits<ck::index_t>::max());
    if (options.m > ck_max || options.n > ck_max || options.k > ck_max) {
        throw ExitError(2, "M, N, or K exceeds ck::index_t");
    }
    return options;
}

void maybe_reexec_with_selected_device(int argc, char** argv, const Options& options) {
    if (std::getenv("ULLM_CK_ABSCALE_ISOLATED") != nullptr) {
        return;
    }

    const std::string requested_device = std::to_string(options.device);
    std::string selected_device = requested_device;
    const char* original_visibility = std::getenv("HIP_VISIBLE_DEVICES");
    std::vector<std::string> selectors;
    if (original_visibility != nullptr) {
        std::string_view remaining(original_visibility);
        while (true) {
            const std::size_t comma = remaining.find(',');
            std::string_view selector = remaining.substr(0, comma);
            while (!selector.empty() && selector.front() == ' ') {
                selector.remove_prefix(1);
            }
            while (!selector.empty() && selector.back() == ' ') {
                selector.remove_suffix(1);
            }
            if (!selector.empty()) {
                selectors.emplace_back(selector);
            }
            if (comma == std::string_view::npos) {
                break;
            }
            remaining.remove_prefix(comma + 1);
        }
        if (static_cast<std::size_t>(options.device) >= selectors.size()) {
            throw ExitError(2, "--device is outside the existing HIP_VISIBLE_DEVICES list");
        }
        selected_device = selectors[static_cast<std::size_t>(options.device)];
    }

    const auto set_environment = [](const char* name, const std::string& value) {
        if (setenv(name, value.c_str(), 1) != 0) {
            throw std::runtime_error(std::string("setenv failed for ") + name + ": " +
                                     std::strerror(errno));
        }
    };
    set_environment("ULLM_CK_ABSCALE_ORIGINAL_VISIBILITY",
                    original_visibility != nullptr ? original_visibility : "");
    set_environment("ULLM_CK_ABSCALE_SELECTED_TOKEN", selected_device);
    set_environment("ULLM_CK_ABSCALE_REQUESTED_DEVICE", requested_device);
    set_environment("ULLM_CK_ABSCALE_ISOLATED", "1");

    if (selectors.size() == 1 && options.device == 0) {
        return;
    }
    set_environment("HIP_VISIBLE_DEVICES", selected_device);

    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        arguments.emplace_back(argv[i]);
    }
    bool replaced = false;
    for (std::size_t i = 0; i + 1 < arguments.size(); ++i) {
        if (arguments[i] == "--device") {
            arguments[i + 1] = "0";
            replaced = true;
            break;
        }
    }
    if (!replaced) {
        arguments.emplace_back("--device");
        arguments.emplace_back("0");
    }

    std::vector<char*> exec_argv;
    exec_argv.reserve(arguments.size() + 1);
    for (auto& argument : arguments) {
        exec_argv.push_back(argument.data());
    }
    exec_argv.push_back(nullptr);
    execv("/proc/self/exe", exec_argv.data());
    throw std::runtime_error(std::string("execv failed: ") + std::strerror(errno));
}

int requested_device_for_report(const Options& options) {
    const char* requested = std::getenv("ULLM_CK_ABSCALE_REQUESTED_DEVICE");
    if (requested == nullptr) {
        return options.device;
    }
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(requested, &end, 10);
    if (errno != 0 || end == requested || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid ULLM_CK_ABSCALE_REQUESTED_DEVICE");
    }
    return static_cast<int>(value);
}

std::string selected_device_token_for_report(const Options& options) {
    const char* selected = std::getenv("ULLM_CK_ABSCALE_SELECTED_TOKEN");
    return selected != nullptr ? selected : std::to_string(options.device);
}

std::string original_visibility_json() {
    const char* original = std::getenv("ULLM_CK_ABSCALE_ORIGINAL_VISIBILITY");
    if (original == nullptr || *original == '\0') {
        return "null";
    }
    return json_string(original);
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, std::string_view label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw ExitError(2, std::string(label) + " size overflows size_t");
    }
    return lhs * rhs;
}

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

std::string normalized_arch(const hipDeviceProp_t& props) {
    std::string arch(props.gcnArchName);
    const auto feature_separator = arch.find(':');
    if (feature_separator != std::string::npos) {
        arch.resize(feature_separator);
    }
    return arch;
}

std::string json_string(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(ch) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    out << '"';
    return out.str();
}

float p50(std::vector<float> samples) {
    if (samples.empty()) {
        throw std::runtime_error("cannot compute p50 of empty samples");
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t mid = samples.size() / 2;
    if ((samples.size() & 1U) != 0) {
        return samples[mid];
    }
    return 0.5f * (samples[mid - 1] + samples[mid]);
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
    DeviceOp& op,
    const DeviceBuffer& a,
    const DeviceBuffer& b,
    DeviceBuffer& c,
    const DeviceBuffer& a_scale,
    const DeviceBuffer& b_scale,
    const Options& options) {
    const auto m = static_cast<ck::index_t>(options.m);
    const auto n = static_cast<ck::index_t>(options.n);
    const auto k = static_cast<ck::index_t>(options.k);

    // B is logically column-major [K, N] with stride K. Its byte offset is
    // n * K + k, exactly matching canonical SQ8 weight bytes stored row-major [N, K].
    return op.MakeArgumentPointer(a.get(),
                                  b.get(),
                                  std::array<const void*, 0>{},
                                  c.get(),
                                  m,
                                  n,
                                  k,
                                  k,
                                  k,
                                  std::array<ck::index_t, 0>{},
                                  n,
                                  a_scale.get(),
                                  b_scale.get(),
                                  PassThrough{},
                                  PassThrough{},
                                  PassThrough{});
}

CandidateMeasurement measure_candidate(
    const std::string& group,
    DeviceOp& op,
    ck::tensor_operation::device::BaseArgument& argument,
    DeviceBuffer& output,
    const Options& options,
    hipStream_t stream,
    float expected) {
    CandidateMeasurement measurement;
    measurement.group = group;
    measurement.instance = op.GetTypeString();
    measurement.supported = true;

    try {
        auto invoker = op.MakeInvokerPointer();
        StreamConfig stream_config;
        stream_config.stream_id_ = stream;
        stream_config.time_kernel_ = false;
        stream_config.log_level_ = 0;

        HIP_CHECK(hipMemsetAsync(output.get(), 0, output.bytes(), stream));
        for (int i = 0; i < options.warmups; ++i) {
            (void)invoker->Run(&argument, stream_config);
            HIP_CHECK(hipGetLastError());
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        HipEvent start;
        HipEvent stop;
        std::vector<float> samples;
        samples.reserve(static_cast<std::size_t>(options.repeats));
        for (int i = 0; i < options.repeats; ++i) {
            HIP_CHECK(hipEventRecord(start.get(), stream));
            (void)invoker->Run(&argument, stream_config);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipEventRecord(stop.get(), stream));
            HIP_CHECK(hipEventSynchronize(stop.get()));
            float elapsed_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
            samples.push_back(elapsed_ms);
        }

        std::vector<ck::bhalf_t> host_output(checked_mul(options.m, options.n, "output"));
        HIP_CHECK(hipMemcpyAsync(host_output.data(),
                                 output.get(),
                                 output.bytes(),
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        double max_abs_diff = 0.0;
        bool finite = true;
        for (const ck::bhalf_t value : host_output) {
            const float actual = ck::type_convert<float>(value);
            if (!std::isfinite(actual)) {
                finite = false;
                break;
            }
            max_abs_diff = std::max(max_abs_diff,
                                    std::abs(static_cast<double>(actual) -
                                             static_cast<double>(expected)));
        }

        measurement.runnable = true;
        measurement.max_abs_diff = finite ? max_abs_diff : std::numeric_limits<double>::infinity();
        measurement.correct = finite && max_abs_diff == 0.0;
        measurement.p50_ms = p50(std::move(samples));
    } catch (const std::exception& error) {
        measurement.error = error.what();
    }
    return measurement;
}

void print_error_json(std::string_view error) {
    std::cout << "{\"schema_version\":\"ullm.sq8.ck_abscale_probe.v1\","
              << "\"status\":\"error\",\"fallback\":\"not_used\",\"error\":"
              << json_string(error) << "}\n";
}

void print_no_support_json(const Options& options,
                           const hipDeviceProp_t& props,
                           std::string_view arch,
                           const std::vector<CandidateGroup>& groups,
                           std::size_t candidate_count,
                           std::string_view error) {
    std::cout << "{\n"
              << "  \"schema_version\": \"ullm.sq8.ck_abscale_probe.v1\",\n"
              << "  \"status\": \"error\",\n"
              << "  \"fallback\": \"not_used\",\n"
              << "  \"error\": " << json_string(error) << ",\n"
              << "  \"device\": {\"hip_device\": " << requested_device_for_report(options)
              << ", \"requested_hip_device\": " << requested_device_for_report(options)
              << ", \"visible_hip_device\": " << options.device
              << ", \"original_hip_visible_devices\": " << original_visibility_json()
              << ", \"selected_device_token\": "
              << json_string(selected_device_token_for_report(options)) << ", \"name\": "
              << json_string(props.name) << ", \"arch\": " << json_string(arch) << "},\n"
              << "  \"shape\": {\"m\": " << options.m << ", \"n\": " << options.n
              << ", \"k\": " << options.k << "},\n"
              << "  \"candidate_count\": " << candidate_count << ",\n"
              << "  \"supported_count\": 0,\n"
              << "  \"candidate_groups\": [\n";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        std::cout << "    {\"name\": " << json_string(groups[i].name)
                  << ", \"instance_count\": " << groups[i].instances.size()
                  << ", \"supported_count\": 0}"
                  << (i + 1 == groups.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n"
              << "  \"selected_instance\": null,\n"
              << "  \"correctness\": null,\n"
              << "  \"timing\": null\n"
              << "}\n";
}

int run(const Options& options) {
    HIP_CHECK(hipSetDevice(options.device));
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, options.device));
    const std::string arch = normalized_arch(props);
    if (arch != "gfx1201") {
        throw ExitError(4, "SQ8 CK ABScale probe requires gfx1201; selected device is " + arch);
    }

    const std::size_t a_elements = checked_mul(options.m, options.k, "A");
    const std::size_t b_elements = checked_mul(options.n, options.k, "B");
    const std::size_t c_elements = checked_mul(options.m, options.n, "C");
    const std::size_t k_blocks = ceil_div(options.k, 128);
    const std::size_t n_blocks = ceil_div(options.n, 128);
    const std::size_t a_scale_elements = checked_mul(options.m, k_blocks, "A scale");
    const std::size_t b_scale_elements = checked_mul(n_blocks, k_blocks, "B scale");

    const ck::f8_t one_f8 = ck::type_convert<ck::f8_t>(1.0f);
    std::vector<ck::f8_t> host_a(a_elements, one_f8);
    std::vector<ck::f8_t> host_b(b_elements, one_f8);
    std::vector<float> host_a_scale(a_scale_elements, 1.0f);
    std::vector<float> host_b_scale(b_scale_elements, 1.0f);

    DeviceBuffer device_a(checked_mul(a_elements, sizeof(ck::f8_t), "A bytes"));
    DeviceBuffer device_b(checked_mul(b_elements, sizeof(ck::f8_t), "B bytes"));
    DeviceBuffer device_c(checked_mul(c_elements, sizeof(ck::bhalf_t), "C bytes"));
    DeviceBuffer device_a_scale(checked_mul(a_scale_elements, sizeof(float), "A scale bytes"));
    DeviceBuffer device_b_scale(checked_mul(b_scale_elements, sizeof(float), "B scale bytes"));

    HIP_CHECK(hipMemcpy(device_a.get(), host_a.data(), device_a.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b.get(), host_b.data(), device_b.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_a_scale.get(),
                        host_a_scale.data(),
                        device_a_scale.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b_scale.get(),
                        host_b_scale.data(),
                        device_b_scale.bytes(),
                        hipMemcpyHostToDevice));

    auto groups = make_candidate_groups();
    std::size_t candidate_count = 0;
    std::size_t supported_count = 0;
    std::vector<std::size_t> group_supported_counts(groups.size(), 0);
    std::vector<CandidateMeasurement> measurements;
    HipStream stream;

    const ck::bhalf_t expected_bf16 =
        ck::type_convert<ck::bhalf_t>(static_cast<float>(options.k));
    const float expected = ck::type_convert<float>(expected_bf16);

    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto& group = groups[group_index];
        candidate_count += group.instances.size();
        for (auto& instance : group.instances) {
            auto argument = make_argument(*instance,
                                          device_a,
                                          device_b,
                                          device_c,
                                          device_a_scale,
                                          device_b_scale,
                                          options);
            if (!instance->IsSupportedArgument(argument.get())) {
                continue;
            }
            ++supported_count;
            ++group_supported_counts[group_index];
            measurements.push_back(measure_candidate(group.name,
                                                     *instance,
                                                     *argument,
                                                     device_c,
                                                     options,
                                                     stream.get(),
                                                     expected));
        }
    }

    if (supported_count == 0) {
        print_no_support_json(options,
                              props,
                              arch,
                              groups,
                              candidate_count,
                              "CK reported no supported ABScale instance for the requested shape");
        return 5;
    }

    const CandidateMeasurement* selected = nullptr;
    for (const auto& measurement : measurements) {
        if (!measurement.runnable || !measurement.correct) {
            continue;
        }
        if (selected == nullptr || measurement.p50_ms < selected->p50_ms) {
            selected = &measurement;
        }
    }
    if (selected == nullptr) {
        throw ExitError(6, "no supported CK ABScale instance completed the all-ones check");
    }

    std::cout << std::setprecision(9);
    std::cout << "{\n"
              << "  \"schema_version\": \"ullm.sq8.ck_abscale_probe.v1\",\n"
              << "  \"status\": \"passed\",\n"
              << "  \"fallback\": \"not_used\",\n"
              << "  \"device\": {\"hip_device\": " << requested_device_for_report(options)
              << ", \"requested_hip_device\": " << requested_device_for_report(options)
              << ", \"visible_hip_device\": " << options.device
              << ", \"original_hip_visible_devices\": " << original_visibility_json()
              << ", \"selected_device_token\": "
              << json_string(selected_device_token_for_report(options))
              << ", \"name\": " << json_string(props.name) << ", \"arch\": "
              << json_string(arch) << "},\n"
              << "  \"shape\": {\"m\": " << options.m << ", \"n\": " << options.n
              << ", \"k\": " << options.k << "},\n"
              << "  \"contract\": {\"a_type\": \"fp8_e4m3_ocp\", "
                 "\"b_type\": \"fp8_e4m3_ocp\", \"output_type\": \"bf16\", "
                 "\"a_layout\": \"row_major_mk\", \"b_layout\": \"column_major_kn\", "
                 "\"b_storage\": \"canonical_row_major_nk_bytes\", "
                 "\"scale_block\": {\"m\": 1, \"n\": 128, \"k\": 128}},\n"
              << "  \"candidate_count\": " << candidate_count << ",\n"
              << "  \"supported_count\": " << supported_count << ",\n"
              << "  \"candidate_groups\": [\n";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        std::cout << "    {\"name\": " << json_string(groups[i].name)
                  << ", \"instance_count\": " << groups[i].instances.size()
                  << ", \"supported_count\": " << group_supported_counts[i] << "}"
                  << (i + 1 == groups.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n"
              << "  \"candidate_measurements\": [\n";
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        const auto& measurement = measurements[i];
        std::cout << "    {\"group\": " << json_string(measurement.group)
                  << ", \"instance\": " << json_string(measurement.instance)
                  << ", \"supported\": true, \"runnable\": "
                  << (measurement.runnable ? "true" : "false") << ", \"correct\": "
                  << (measurement.correct ? "true" : "false");
        if (measurement.runnable) {
            std::cout << ", \"max_abs_diff\": ";
            if (std::isfinite(measurement.max_abs_diff)) {
                std::cout << measurement.max_abs_diff;
            } else {
                std::cout << "null";
            }
            std::cout << ", \"kernel_p50_ms\": " << measurement.p50_ms;
        }
        if (!measurement.error.empty()) {
            std::cout << ", \"error\": " << json_string(measurement.error);
        }
        std::cout << "}" << (i + 1 == measurements.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n"
              << "  \"selected_instance\": {\"group\": " << json_string(selected->group)
              << ", \"instance\": " << json_string(selected->instance) << "},\n"
              << "  \"correctness\": {\"case\": \"all_ones\", "
                 "\"expected_value_bf16\": "
              << expected << ", \"max_abs_diff\": " << selected->max_abs_diff
              << ", \"passed\": true},\n"
              << "  \"timing\": {\"scope\": \"kernel_only\", "
                 "\"source\": \"hip_event\", \"warmups\": "
              << options.warmups << ", \"repeats\": " << options.repeats
              << ", \"p50_ms\": " << selected->p50_ms << "}\n"
              << "}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        maybe_reexec_with_selected_device(argc, argv, options);
        return run(options);
    } catch (const ExitError& error) {
        print_error_json(error.what());
        return error.code();
    } catch (const std::exception& error) {
        print_error_json(error.what());
        return 3;
    }
}
