// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Direct CK shape-admission probe for the proposed wide SQ8 prefill widths.
// It deliberately bypasses the Rust/C++ measured-M dispatch gates and calls
// the existing gfx1201 CK helper with the same four Qwen3-14B projection
// shapes and implementation choices.  It is not a performance benchmark.

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "sq8_ck_gfx1201.h"

namespace {

constexpr std::size_t kScaleBlock = 128;
constexpr unsigned int kDefault128 = 1;
constexpr unsigned int kKPadding256 = 2;
constexpr unsigned int kDefault256x128 = 3;
constexpr unsigned int kDefault128x256 = 4;

struct Shape {
    std::string_view name;
    std::size_t n;
    std::size_t k;
};

constexpr std::array<Shape, 4> kShapes = {{
    {"q_or_o", 5120, 5120},
    {"k_or_v", 1024, 5120},
    {"gate_or_up", 17408, 5120},
    {"down", 5120, 17408},
}};

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        if (bytes == 0) {
            throw std::runtime_error("zero-byte device allocation");
        }
        check(hipMalloc(&pointer_, bytes), "hipMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)hipFree(pointer_);
        }
    }

    void* get() const { return pointer_; }

  private:
    static void check(hipError_t status, std::string_view operation) {
        if (status != hipSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
        }
    }

    void* pointer_ = nullptr;
};

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, std::string_view label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::runtime_error(std::string(label) + " overflow");
    }
    return lhs * rhs;
}

void check(hipError_t status, std::string_view operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

std::string escape_json(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (char character : text) {
        if (character == '"' || character == '\\') {
            result.push_back('\\');
        }
        if (character == '\n') {
            result += "\\n";
        } else if (character == '\r') {
            result += "\\r";
        } else if (character != '\n') {
            result.push_back(character);
        }
    }
    return result;
}

unsigned int implementation_for(std::size_t m, const Shape& shape) {
    if ((shape.n == 5120 && shape.k == 5120) || (shape.n == 1024 && shape.k == 5120)) {
        return kDefault128;
    }
    if (shape.n == 17408 && shape.k == 5120) {
        return m == 128 ? kDefault256x128 : kKPadding256;
    }
    if (shape.n == 5120 && shape.k == 17408) {
        return m == 128 ? kDefault128 : kDefault128x256;
    }
    throw std::runtime_error("unknown Qwen3 projection shape");
}

void zero(DeviceBuffer& buffer, std::size_t bytes, hipStream_t stream, std::string_view name) {
    check(hipMemsetAsync(buffer.get(), 0, bytes, stream), std::string("hipMemsetAsync ") + std::string(name));
}

bool run_shape(std::size_t m, const Shape& shape, hipStream_t stream) {
    const std::size_t activation_elements = checked_mul(m, shape.k, "activation elements");
    const std::size_t activation_scale_elements = checked_mul(m, shape.k / kScaleBlock, "activation scale elements");
    const std::size_t weight_elements = checked_mul(shape.n, shape.k, "weight elements");
    const std::size_t weight_scale_elements = checked_mul(shape.n / kScaleBlock, shape.k / kScaleBlock, "weight scale elements");
    const std::size_t output_elements = checked_mul(m, shape.n, "output elements");
    const std::size_t input_bytes = checked_mul(activation_elements, sizeof(float), "input bytes");
    const std::size_t activation_scale_bytes = checked_mul(activation_scale_elements, sizeof(float), "activation scale bytes");
    const std::size_t weight_scale_bytes = checked_mul(weight_scale_elements, sizeof(float), "weight scale bytes");
    const std::size_t workspace_bytes = checked_mul(output_elements, sizeof(std::uint16_t), "workspace bytes");
    const std::size_t output_bytes = checked_mul(output_elements, sizeof(float), "output bytes");

    DeviceBuffer input(input_bytes);
    DeviceBuffer activation(activation_elements);
    DeviceBuffer activation_scales(activation_scale_bytes);
    DeviceBuffer weight(weight_elements);
    DeviceBuffer weight_scales(weight_scale_bytes);
    DeviceBuffer workspace(workspace_bytes);
    DeviceBuffer output(output_bytes);
    zero(input, input_bytes, stream, "input");
    zero(weight, weight_elements, stream, "weight");
    zero(weight_scales, weight_scale_bytes, stream, "weight_scales");

    std::array<char, 512> error{};
    const int quantized = ullm_sq8_ck_gfx1201_quantize_activation(
        input.get(),
        activation.get(),
        activation_scales.get(),
        m,
        shape.k,
        stream,
        0,
        error.data(),
        error.size());
    const unsigned int implementation = implementation_for(m, shape);
    int projected = 0;
    if (quantized != 0) {
        error.fill('\0');
        projected = ullm_sq8_ck_gfx1201_projection(
            activation.get(),
            activation_scales.get(),
            weight.get(),
            weight_scales.get(),
            m,
            shape.n,
            shape.k,
            workspace.get(),
            output.get(),
            stream,
            0,
            implementation,
            error.data(),
            error.size());
    }
    if (quantized != 0 && projected != 0) {
        check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    }
    std::cout << "{\"m\":" << m << ",\"shape\":\"" << shape.name
              << "\",\"n\":" << shape.n << ",\"k\":" << shape.k
              << ",\"implementation\":" << implementation
              << ",\"quantize_ok\":" << (quantized != 0 ? "true" : "false")
              << ",\"projection_ok\":" << (projected != 0 ? "true" : "false")
              << ",\"error\":\"" << escape_json(error.data()) << "\"}\n";
    return quantized != 0 && projected != 0;
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
        if (device_count != 1) {
            throw std::runtime_error("probe requires exactly one HIP-visible device");
        }
        check(hipSetDevice(0), "hipSetDevice");
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
        if (std::strncmp(properties.gcnArchName, "gfx1201", 7) != 0) {
            throw std::runtime_error(std::string("probe requires gfx1201, got ") + properties.gcnArchName);
        }
        hipStream_t stream = nullptr;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        bool passed = true;
        for (const std::size_t m : {128u, 256u, 512u, 1024u, 2048u, 4096u}) {
            for (const Shape& shape : kShapes) {
                try {
                    passed = run_shape(m, shape, stream) && passed;
                } catch (const std::exception& error) {
                    std::cout << "{\"m\":" << m << ",\"shape\":\"" << shape.name
                              << "\",\"quantize_ok\":false,\"projection_ok\":false,\"error\":\""
                              << escape_json(error.what()) << "\"}\n";
                    passed = false;
                }
            }
        }
        check(hipStreamDestroy(stream), "hipStreamDestroy");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "wide_m_ck_shape_probe: " << error.what() << '\n';
        return 2;
    }
}
