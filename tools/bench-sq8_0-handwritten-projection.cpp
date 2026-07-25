// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Isolated SQ8_0 M=1 projection feasibility probe.  This is deliberately
// linked directly to the private helper symbols rather than the public runtime
// ABI so it cannot affect the existing dispatcher or serving default.

#include <hip/hip_runtime.h>

#include <array>
#include <bit>
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sq8_ck_gfx1201.h"
#include "sq8_handwritten_gfx1201.h"

namespace {

constexpr double kTheoreticalHbmGbPerSecond = 640.0;
constexpr std::size_t kScaleBlock = 128u;

struct Options {
    std::filesystem::path output;
    std::string mode;
    int device = 0;
    int warmups = 8;
    int repeats = 64;
};

struct Shape {
    const char* family;
    std::size_t n;
    std::size_t k;
    std::size_t per_layer_calls;
    uint32_t ck_implementation;
    const char* ck_instance;
};

constexpr std::array<Shape, 4> kShapes = {{
    {"q_o", 5120u, 5120u, 2u, 1u, "Default 16x128x128"},
    {"k_v", 1024u, 5120u, 2u, 1u, "Default 16x128x128"},
    {"gate_up", 17408u, 5120u, 2u, 2u, "KPadding 16x128x256"},
    {"down", 5120u, 17408u, 1u, 4u, "Default 16x128x256"},
}};

[[noreturn]] void fail(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

void hip_check(hipError_t status, std::string_view operation) {
    if (status != hipSuccess) {
        std::ostringstream message;
        message << operation << " failed: " << hipGetErrorString(status) << " ("
                << static_cast<int>(status) << ')';
        throw std::runtime_error(message.str());
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes == 0u) {
            fail("zero-byte device allocation requested");
        }
        hip_check(hipMalloc(&pointer_, bytes), "hipMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            const hipError_t status = hipFree(pointer_);
            if (status != hipSuccess) {
                std::cerr << "hipFree during cleanup failed: " << hipGetErrorString(status) << '\n';
            }
        }
    }

    void* get() { return pointer_; }
    const void* get() const { return pointer_; }
    std::size_t bytes() const { return bytes_; }

  private:
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0u;
};

class Stream {
  public:
    Stream() { hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() {
        if (stream_ != nullptr) {
            const hipError_t status = hipStreamDestroy(stream_);
            if (status != hipSuccess) {
                std::cerr << "hipStreamDestroy during cleanup failed: " << hipGetErrorString(status)
                          << '\n';
            }
        }
    }
    hipStream_t get() const { return stream_; }

  private:
    hipStream_t stream_ = nullptr;
};

class Event {
  public:
    Event() { hip_check(hipEventCreateWithFlags(&event_, hipEventDefault), "hipEventCreate"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (event_ != nullptr) {
            const hipError_t status = hipEventDestroy(event_);
            if (status != hipSuccess) {
                std::cerr << "hipEventDestroy during cleanup failed: " << hipGetErrorString(status)
                          << '\n';
            }
        }
    }
    hipEvent_t get() const { return event_; }

  private:
    hipEvent_t event_ = nullptr;
};

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, std::string_view label) {
    if (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        fail(std::string(label) + " overflows");
    }
    return lhs * rhs;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto next = [&]() -> std::string_view {
            if (++index >= argc) {
                fail(std::string(argument) + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--output") {
            options.output = std::filesystem::path(next());
        } else if (argument == "--mode") {
            options.mode = next();
        } else if (argument == "--device") {
            options.device = std::stoi(std::string(next()));
        } else if (argument == "--warmups") {
            options.warmups = std::stoi(std::string(next()));
        } else if (argument == "--repeats") {
            options.repeats = std::stoi(std::string(next()));
        } else {
            fail("unknown argument: " + std::string(argument));
        }
    }
    if (options.output.empty() ||
        (options.mode != "gate" && options.mode != "baseline" && options.mode != "measure") ||
        options.device != 0 || options.warmups < 0 || options.repeats <= 0) {
        fail(
            "usage: --output PATH --mode gate|baseline|measure [--device 0] [--warmups N] [--repeats N]");
    }
    return options;
}

void validate_visible_gfx1201(int device) {
    const char* visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (visible == nullptr || visible[0] == '\0' || std::strchr(visible, ',') != nullptr) {
        fail("probe requires exactly one HIP_VISIBLE_DEVICES token");
    }
    int count = 0;
    hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count != 1 || device != 0) {
        fail("probe requires exactly one visible device at internal ordinal zero");
    }
    hip_check(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    const bool is_gfx1201 = std::strncmp(properties.gcnArchName, "gfx1201", 7u) == 0 &&
                            (properties.gcnArchName[7] == '\0' ||
                             properties.gcnArchName[7] == ':');
    if (!is_gfx1201 || properties.major != 12 || properties.minor != 0) {
        fail(std::string("probe requires gfx1201 compute 12.0, selected ") + properties.gcnArchName);
    }
}

float f32_from_bf16_bits(uint16_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16u);
}

std::vector<unsigned char> canonical_payload(std::size_t count, uint32_t seed) {
    // Cover every finite OCP E4M3FN raw code.  The two NaN encodings 0x7f
    // and 0xff are intentionally excluded because serving payloads reject
    // non-finite values before projection.
    std::vector<unsigned char> result(count);
    for (std::size_t index = 0u; index < result.size(); ++index) {
        unsigned int code = static_cast<unsigned int>((index * 73u + seed) % 254u);
        if (code >= 0x7fu) {
            ++code;
        }
        result[index] = static_cast<unsigned char>(code);
    }
    return result;
}

std::vector<float> canonical_bf16_origin_scales(std::size_t count, uint32_t seed) {
    constexpr std::array<uint16_t, 8> values = {
        0x3b80u, 0x3c00u, 0x3c80u, 0x3d00u, 0x3d80u, 0x3e00u, 0x3e80u, 0x3f00u,
    };
    std::vector<float> result(count);
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] = f32_from_bf16_bits(values[(index * 13u + seed) % values.size()]);
    }
    return result;
}

std::vector<float> runtime_activation_scales(std::size_t count, uint32_t seed) {
    // The existing M=1 CK quantizer writes F32 [1,128] activation scales,
    // rather than artifact BF16 scales. Keep mantissas deliberately varied so
    // this smoke differential cannot accidentally test only BF16 values.
    constexpr std::array<uint32_t, 16> values = {
        0x390a3d71u, 0x3991c47bu, 0x3a2b851fu, 0x3ab9d036u,
        0x3b4c6a7fu, 0x3bd75b61u, 0x3c183f92u, 0x3c6f1a01u,
        0x3ca44b18u, 0x3ced91b4u, 0x3d31c47bu, 0x3d70a3d7u,
        0x3da5e354u, 0x3df28f5cu, 0x3e21e3a8u, 0x3e5b6db7u,
    };
    std::vector<float> result(count);
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] = std::bit_cast<float>(values[(index * 11u + seed) % values.size()]);
    }
    return result;
}

bool launch_ck(const Shape& shape,
               const DeviceBuffer& activation,
               const DeviceBuffer& activation_scales,
               const DeviceBuffer& weight,
               const DeviceBuffer& weight_scales,
               DeviceBuffer& workspace,
               DeviceBuffer& output,
               hipStream_t stream,
               std::string* error) {
    std::array<char, 512> helper_error{};
    const int result = ullm_sq8_ck_gfx1201_projection(activation.get(),
                                                        activation_scales.get(),
                                                        weight.get(),
                                                        weight_scales.get(),
                                                        1u,
                                                        shape.n,
                                                        shape.k,
                                                        workspace.get(),
                                                        output.get(),
                                                        stream,
                                                        0,
                                                        shape.ck_implementation,
                                                        helper_error.data(),
                                                        helper_error.size());
    if (result == 0 && error != nullptr) {
        *error = helper_error.data();
    }
    return result != 0;
}

bool launch_handwritten(const Shape& shape,
                        const DeviceBuffer& activation,
                        const DeviceBuffer& activation_scales,
                        const DeviceBuffer& weight,
                        const DeviceBuffer& weight_scales,
                        DeviceBuffer& output,
                        hipStream_t stream,
                        std::string* error) {
    std::array<char, 512> helper_error{};
    const int result = ullm_sq8_handwritten_gfx1201_m1_wmma_projection(
        activation.get(),
        activation_scales.get(),
        weight.get(),
        weight_scales.get(),
        shape.n,
        shape.k,
        output.get(),
        stream,
        0,
        helper_error.data(),
        helper_error.size());
    if (result == 0 && error != nullptr) {
        *error = helper_error.data();
    }
    return result != 0;
}

template <typename Launch>
double measure_microseconds(Launch&& launch, int repeats, Stream& stream) {
    Event start;
    Event finish;
    hip_check(hipEventRecord(start.get(), stream.get()), "hipEventRecord(start)");
    for (int repeat = 0; repeat < repeats; ++repeat) {
        launch();
    }
    hip_check(hipEventRecord(finish.get(), stream.get()), "hipEventRecord(finish)");
    hip_check(hipEventSynchronize(finish.get()), "hipEventSynchronize");
    float elapsed_ms = 0.0f;
    hip_check(hipEventElapsedTime(&elapsed_ms, start.get(), finish.get()), "hipEventElapsedTime");
    return static_cast<double>(elapsed_ms) * 1000.0 / static_cast<double>(repeats);
}

struct Correctness {
    bool finite = true;
    std::size_t bitwise_mismatches = 0u;
    std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();
    double max_abs = 0.0;
    double max_rel = 0.0;
};

Correctness compare(const std::vector<float>& ck, const std::vector<float>& handwritten) {
    if (ck.size() != handwritten.size()) {
        fail("comparison size mismatch");
    }
    Correctness result;
    for (std::size_t index = 0u; index < ck.size(); ++index) {
        const float left = ck[index];
        const float right = handwritten[index];
        if (!std::isfinite(left) || !std::isfinite(right)) {
            result.finite = false;
        }
        if (std::bit_cast<uint32_t>(left) != std::bit_cast<uint32_t>(right)) {
            ++result.bitwise_mismatches;
            result.first_mismatch = std::min(result.first_mismatch, index);
        }
        const double absolute = std::abs(static_cast<double>(left) - static_cast<double>(right));
        result.max_abs = std::max(result.max_abs, absolute);
        const double denominator = std::max(std::abs(static_cast<double>(left)), 1.0e-30);
        result.max_rel = std::max(result.max_rel, absolute / denominator);
    }
    return result;
}

struct Timing {
    std::optional<double> ck_us;
    std::optional<double> handwritten_us;
};

struct HandwrittenResources {
    uint32_t vgpr_per_thread = 0u;
    std::size_t static_lds_bytes = 0u;
    std::size_t local_bytes_per_thread = 0u;
    int threads_per_block = 0;
    int active_blocks_per_cu = 0;
};

struct ShapeResult {
    Shape shape;
    Correctness correctness;
    Timing timing;
    uint64_t canonical_input_bytes = 0u;
    uint64_t ck_route_bytes = 0u;
    uint64_t handwritten_route_bytes = 0u;
};

uint64_t as_u64(std::size_t value, std::string_view label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<uint64_t>::max())) {
        fail(std::string(label) + " does not fit u64");
    }
    return static_cast<uint64_t>(value);
}

ShapeResult run_shape(const Shape& shape, const Options& options, Stream& stream) {
    const std::size_t activation_scale_count = shape.k / kScaleBlock;
    const std::size_t weight_elements = checked_mul(shape.n, shape.k, "weight elements");
    const std::size_t weight_scale_count =
        checked_mul(shape.n / kScaleBlock, shape.k / kScaleBlock, "weight scale count");

    std::vector<unsigned char> activation_host = canonical_payload(shape.k, 3u);
    std::vector<float> activation_scales_host = runtime_activation_scales(activation_scale_count, 5u);
    std::vector<unsigned char> weight_host = canonical_payload(weight_elements, 11u);
    std::vector<float> weight_scales_host = canonical_bf16_origin_scales(weight_scale_count, 19u);

    DeviceBuffer activation(shape.k);
    DeviceBuffer activation_scales(checked_mul(activation_scale_count, sizeof(float), "activation scales"));
    DeviceBuffer weight(weight_elements);
    DeviceBuffer weight_scales(checked_mul(weight_scale_count, sizeof(float), "weight scales"));
    DeviceBuffer workspace(checked_mul(shape.n, sizeof(uint16_t), "CK workspace"));
    DeviceBuffer ck_output(checked_mul(shape.n, sizeof(float), "CK output"));
    DeviceBuffer handwritten_output(checked_mul(shape.n, sizeof(float), "handwritten output"));

    hip_check(hipMemcpyAsync(activation.get(),
                             activation_host.data(),
                             activation_host.size(),
                             hipMemcpyHostToDevice,
                             stream.get()),
              "copy activation");
    hip_check(hipMemcpyAsync(activation_scales.get(),
                             activation_scales_host.data(),
                             activation_scales.bytes(),
                             hipMemcpyHostToDevice,
                             stream.get()),
              "copy activation scales");
    hip_check(hipMemcpyAsync(weight.get(),
                             weight_host.data(),
                             weight_host.size(),
                             hipMemcpyHostToDevice,
                             stream.get()),
              "copy weight");
    hip_check(hipMemcpyAsync(weight_scales.get(),
                             weight_scales_host.data(),
                             weight_scales.bytes(),
                             hipMemcpyHostToDevice,
                             stream.get()),
              "copy weight scales");
    hip_check(hipStreamSynchronize(stream.get()), "synchronize uploads");

    auto ck_launch = [&]() {
        std::string error;
        if (!launch_ck(shape,
                       activation,
                       activation_scales,
                       weight,
                       weight_scales,
                       workspace,
                       ck_output,
                       stream.get(),
                       &error)) {
            fail("CK launch failed: " + error);
        }
    };
    auto handwritten_launch = [&]() {
        std::string error;
        if (!launch_handwritten(shape,
                                activation,
                                activation_scales,
                                weight,
                                weight_scales,
                                handwritten_output,
                                stream.get(),
                                &error)) {
            fail("handwritten launch failed: " + error);
        }
    };

    ck_launch();
    handwritten_launch();
    hip_check(hipStreamSynchronize(stream.get()), "synchronize correctness launches");

    std::vector<float> ck_host(shape.n);
    std::vector<float> handwritten_host(shape.n);
    hip_check(hipMemcpyAsync(ck_host.data(),
                             ck_output.get(),
                             ck_output.bytes(),
                             hipMemcpyDeviceToHost,
                             stream.get()),
              "copy CK output");
    hip_check(hipMemcpyAsync(handwritten_host.data(),
                             handwritten_output.get(),
                             handwritten_output.bytes(),
                             hipMemcpyDeviceToHost,
                             stream.get()),
              "copy handwritten output");
    hip_check(hipStreamSynchronize(stream.get()), "synchronize correctness readback");

    ShapeResult result{
        shape,
        compare(ck_host, handwritten_host),
        {},
        as_u64(shape.k + activation_scales.bytes() + weight.bytes() + weight_scales.bytes() +
                   handwritten_output.bytes(),
               "canonical route bytes"),
        as_u64(shape.k + activation_scales.bytes() + weight.bytes() + weight_scales.bytes() +
                   workspace.bytes() + ck_output.bytes(),
               "CK route bytes"),
        as_u64(shape.k + activation_scales.bytes() + weight.bytes() + weight_scales.bytes() +
                   handwritten_output.bytes(),
               "handwritten route bytes"),
    };

    if (options.mode == "baseline") {
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            ck_launch();
        }
        hip_check(hipStreamSynchronize(stream.get()), "synchronize CK baseline warmups");
        result.timing.ck_us = measure_microseconds(ck_launch, options.repeats, stream);
    } else if (options.mode == "measure" && result.correctness.finite &&
               result.correctness.bitwise_mismatches == 0u) {
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            ck_launch();
            handwritten_launch();
        }
        hip_check(hipStreamSynchronize(stream.get()), "synchronize warmups");
        result.timing.ck_us = measure_microseconds(ck_launch, options.repeats, stream);
        result.timing.handwritten_us =
            measure_microseconds(handwritten_launch, options.repeats, stream);
    }
    return result;
}

void write_number_or_null(std::ostream& output, const std::optional<double>& value) {
    if (value.has_value()) {
        output << std::setprecision(12) << *value;
    } else {
        output << "null";
    }
}

double gb_per_second(uint64_t bytes, double microseconds) {
    return static_cast<double>(bytes) / microseconds / 1000.0;
}

void write_json(const Options& options,
                const hipDeviceProp_t& properties,
                const std::vector<ShapeResult>& results,
                const HandwrittenResources* resources) {
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail("failed to open result path " + options.output.string());
    }
    const bool numerical_pass = std::all_of(results.begin(), results.end(), [](const ShapeResult& value) {
        return value.correctness.finite && value.correctness.bitwise_mismatches == 0u;
    });
    output << "{\n"
           << "  \"schema_version\": \"ullm.sq8_0.handwritten_projection_component.v1\",\n"
           << "  \"mode\": \"" << options.mode << "\",\n"
           << "  \"numeric_gate\": {\n"
           << "    \"criterion\": \"finite and bitwise-identical F32 after the CK BF16 workspace boundary for all four actual M=1 projection shapes\",\n"
           << "    \"passed\": " << (numerical_pass ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"timing_policy\": \"baseline times CK only; measure times the handwritten candidate only after every component numerical comparison passes\",\n"
           << "  \"semantics\": {\n"
           << "    \"activation_payload\": \"OCP E4M3FN raw bytes\",\n"
           << "    \"weight_payload\": \"OCP E4M3FN raw bytes\",\n"
           << "    \"activation_scale\": \"runtime F32 [1,128] scale emitted by the resident CK quantizer\",\n"
           << "    \"weight_scale\": \"canonical [128,128] BF16 scale expanded to F32 by the resident loader\"\n"
           << "  },\n"
           << "  \"device\": {\n"
           << "    \"hip_visible_device\": \"" << std::getenv("HIP_VISIBLE_DEVICES") << "\",\n"
           << "    \"gcn_arch_name\": \"" << properties.gcnArchName << "\",\n"
           << "    \"compute_major\": " << properties.major << ",\n"
           << "    \"compute_minor\": " << properties.minor << "\n"
           << "  },\n"
           << "  \"theoretical_hbm_gb_s\": " << kTheoreticalHbmGbPerSecond << ",\n";
    if (resources != nullptr) {
        output << "  \"handwritten_hip_resource_query\": {\n"
               << "    \"vgpr_per_thread\": " << resources->vgpr_per_thread << ",\n"
               << "    \"static_lds_bytes\": " << resources->static_lds_bytes << ",\n"
               << "    \"local_bytes_per_thread\": " << resources->local_bytes_per_thread << ",\n"
               << "    \"threads_per_block\": " << resources->threads_per_block << ",\n"
               << "    \"active_blocks_per_cu\": " << resources->active_blocks_per_cu << "\n"
               << "  },\n";
    }
    output << "  \"shapes\": [\n";
    for (std::size_t index = 0u; index < results.size(); ++index) {
        const ShapeResult& result = results[index];
        output << "    {\n"
               << "      \"family\": \"" << result.shape.family << "\",\n"
               << "      \"m\": 1,\n"
               << "      \"n\": " << result.shape.n << ",\n"
               << "      \"k\": " << result.shape.k << ",\n"
               << "      \"per_layer_calls\": " << result.shape.per_layer_calls << ",\n"
               << "      \"ck_instance\": \"" << result.shape.ck_instance << "\",\n"
               << "      \"correctness\": {\n"
               << "        \"finite\": " << (result.correctness.finite ? "true" : "false") << ",\n"
               << "        \"bitwise_mismatches\": " << result.correctness.bitwise_mismatches << ",\n"
               << "        \"first_mismatch\": ";
        if (result.correctness.first_mismatch == std::numeric_limits<std::size_t>::max()) {
            output << "null";
        } else {
            output << result.correctness.first_mismatch;
        }
        output << ",\n"
               << "        \"max_abs\": " << std::setprecision(12) << result.correctness.max_abs << ",\n"
               << "        \"max_rel\": " << std::setprecision(12) << result.correctness.max_rel << "\n"
               << "      },\n"
               << "      \"logical_route_bytes\": {\n"
               << "        \"canonical_input_plus_f32_output\": " << result.canonical_input_bytes << ",\n"
               << "        \"ck_including_bf16_workspace_and_f32_output\": " << result.ck_route_bytes << ",\n"
               << "        \"handwritten_f32_output\": " << result.handwritten_route_bytes << "\n"
               << "      },\n"
               << "      \"timing\": {\n"
               << "        \"ck_us\": ";
        write_number_or_null(output, result.timing.ck_us);
        output << ",\n        \"handwritten_us\": ";
        write_number_or_null(output, result.timing.handwritten_us);
        output << ",\n        \"ck_logical_gb_s\": ";
        if (result.timing.ck_us) {
            output << std::setprecision(12)
                   << gb_per_second(result.ck_route_bytes, *result.timing.ck_us);
        } else {
            output << "null";
        }
        output << ",\n        \"handwritten_logical_gb_s\": ";
        if (result.timing.handwritten_us) {
            output << std::setprecision(12)
                   << gb_per_second(result.handwritten_route_bytes, *result.timing.handwritten_us);
        } else {
            output << "null";
        }
        output << ",\n        \"ck_theoretical_hbm_ratio\": ";
        if (result.timing.ck_us) {
            output << std::setprecision(12)
                   << gb_per_second(result.ck_route_bytes, *result.timing.ck_us) /
                          kTheoreticalHbmGbPerSecond;
        } else {
            output << "null";
        }
        output << ",\n        \"handwritten_theoretical_hbm_ratio\": ";
        if (result.timing.handwritten_us) {
            output << std::setprecision(12)
                   << gb_per_second(result.handwritten_route_bytes, *result.timing.handwritten_us) /
                          kTheoreticalHbmGbPerSecond;
        } else {
            output << "null";
        }
        output << ",\n        \"ck_ns_per_output_element\": ";
        if (result.timing.ck_us) {
            output << std::setprecision(12)
                   << *result.timing.ck_us * 1000.0 / static_cast<double>(result.shape.n);
        } else {
            output << "null";
        }
        output << ",\n        \"handwritten_ns_per_output_element\": ";
        if (result.timing.handwritten_us) {
            output << std::setprecision(12)
                   << *result.timing.handwritten_us * 1000.0 / static_cast<double>(result.shape.n);
        } else {
            output << "null";
        }
        output << "\n      }\n    }" << (index + 1u == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    if (!output) {
        fail("failed to write result JSON");
    }
}

int run(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    validate_visible_gfx1201(options.device);
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, options.device), "hipGetDeviceProperties");
    Stream stream;
    std::vector<ShapeResult> results;
    results.reserve(kShapes.size());
    for (const Shape& shape : kShapes) {
        results.push_back(run_shape(shape, options, stream));
    }

    // Query after measurements, so resource introspection cannot perturb event timing.
    HandwrittenResources resources{};
    std::array<char, 512> helper_error{};
    if (!ullm_sq8_handwritten_gfx1201_m1_wmma_resources(options.device,
                                                         &resources.vgpr_per_thread,
                                                         &resources.static_lds_bytes,
                                                         &resources.local_bytes_per_thread,
                                                         &resources.threads_per_block,
                                                         &resources.active_blocks_per_cu,
                                                         helper_error.data(),
                                                         helper_error.size())) {
        fail(std::string("handwritten resource query failed: ") + helper_error.data());
    }
    write_json(options, properties, results, &resources);
    const bool pass = std::all_of(results.begin(), results.end(), [](const ShapeResult& value) {
        return value.correctness.finite && value.correctness.bitwise_mismatches == 0u;
    });
    if (!pass && options.mode != "baseline") {
        return 2;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "bench-sq8_0-handwritten-projection: " << error.what() << '\n';
        return 1;
    }
}
