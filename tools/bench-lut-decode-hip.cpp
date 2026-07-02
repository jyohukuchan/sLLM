#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

#define HIP_CHECK(call)                                                                          \
    do {                                                                                         \
        hipError_t err__ = (call);                                                               \
        if (err__ != hipSuccess) {                                                               \
            throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(err__));     \
        }                                                                                        \
    } while (false)

struct Options {
    std::size_t values = 64ull * 1024ull * 1024ull;
    int repeats = 20;
    int warmups = 5;
    int device = 0;
    int block_size = 256;
};

struct Metric {
    std::string target;
    int bits = 0;
    std::string mode;
    std::size_t values = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t table_bytes = 0;
    float best_ms = 0.0f;
    float median_ms = 0.0f;
    double best_ns_per_value = 0.0;
    double median_ns_per_value = 0.0;
    double best_gvalues_per_s = 0.0;
    double best_gbytes_per_s = 0.0;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--values N] [--repeats N] [--warmups N] [--device N] [--block-size N]\n";
    std::exit(2);
}

std::size_t parse_size(std::string_view text) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(std::string(text).c_str(), &end, 10);
    if (end == nullptr || *end != '\0') {
        throw std::runtime_error("invalid integer");
    }
    return static_cast<std::size_t>(value);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        auto need_value = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            ++i;
            return std::string_view(argv[i]);
        };
        if (arg == "--values") {
            options.values = parse_size(need_value());
        } else if (arg == "--repeats") {
            options.repeats = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--warmups") {
            options.warmups = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--device") {
            options.device = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--block-size") {
            options.block_size = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            usage(argv[0]);
        }
    }
    if (options.values == 0 || options.repeats <= 0 || options.warmups < 0 || options.block_size <= 0) {
        usage(argv[0]);
    }
    return options;
}

std::uint32_t lcg_next(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

std::vector<std::uint16_t> make_indices(std::size_t values, int bits) {
    const std::uint32_t mask = (1u << bits) - 1u;
    std::vector<std::uint16_t> indices(values);
    std::uint32_t state = 0x87654321u ^ static_cast<std::uint32_t>(bits * 0x9e3779b9u);
    for (std::size_t i = 0; i < values; ++i) {
        indices[i] = static_cast<std::uint16_t>(lcg_next(state) & mask);
    }
    return indices;
}

std::vector<std::uint8_t> pack_indices(const std::vector<std::uint16_t>& indices, int bits) {
    const std::size_t out_bytes = (indices.size() * static_cast<std::size_t>(bits) + 7) / 8;
    std::vector<std::uint8_t> packed(out_bytes + 2, 0);
    std::size_t bit_pos = 0;
    for (std::uint16_t index : indices) {
        std::uint32_t value = index;
        for (int b = 0; b < bits; ++b) {
            if ((value >> b) & 1u) {
                packed[bit_pos >> 3] |= static_cast<std::uint8_t>(1u << (bit_pos & 7));
            }
            ++bit_pos;
        }
    }
    return packed;
}

template <typename T>
std::vector<T> make_lut(int bits) {
    const std::size_t entries = 1ull << bits;
    std::vector<T> lut(entries);
    for (std::size_t i = 0; i < entries; ++i) {
        if constexpr (sizeof(T) == 1) {
            lut[i] = static_cast<T>((i * 37u + 11u) & 0xffu);
        } else {
            lut[i] = static_cast<T>((i * 131u + 0x3c00u) & 0xffffu);
        }
    }
    return lut;
}

template <typename T>
__global__ void store_kernel(T* output, std::size_t values) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < values) {
        output[i] = static_cast<T>(i);
    }
}

template <typename T>
__global__ void aligned_lut_kernel(const std::uint16_t* indices, const T* lut, T* output, std::size_t values) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < values) {
        output[i] = lut[indices[i]];
    }
}

template <typename T>
__global__ void packed_lut_kernel(
    const std::uint8_t* packed,
    const T* lut,
    T* output,
    std::size_t values,
    int bits
) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < values) {
        const std::size_t bit_pos = i * static_cast<std::size_t>(bits);
        const std::size_t byte_pos = bit_pos >> 3;
        const int shift = static_cast<int>(bit_pos & 7);
        const std::uint32_t word = static_cast<std::uint32_t>(packed[byte_pos])
            | (static_cast<std::uint32_t>(packed[byte_pos + 1]) << 8)
            | (static_cast<std::uint32_t>(packed[byte_pos + 2]) << 16);
        const std::uint32_t mask = (1u << bits) - 1u;
        const std::uint32_t index = (word >> shift) & mask;
        output[i] = lut[index];
    }
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Launch>
Metric time_kernel(
    const Options& options,
    std::string target,
    int bits,
    std::string mode,
    std::size_t input_bytes,
    std::size_t output_bytes,
    std::size_t table_bytes,
    Launch&& launch
) {
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < options.warmups; ++i) {
        launch();
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start{};
    hipEvent_t stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> samples;
    for (int i = 0; i < options.repeats; ++i) {
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    const float best_ms = *std::min_element(samples.begin(), samples.end());
    const float med_ms = median(samples);
    const double best_ns = static_cast<double>(best_ms) * 1.0e6;
    const double med_ns = static_cast<double>(med_ms) * 1.0e6;
    Metric metric;
    metric.target = std::move(target);
    metric.bits = bits;
    metric.mode = std::move(mode);
    metric.values = options.values;
    metric.input_bytes = input_bytes;
    metric.output_bytes = output_bytes;
    metric.table_bytes = table_bytes;
    metric.best_ms = best_ms;
    metric.median_ms = med_ms;
    metric.best_ns_per_value = best_ns / static_cast<double>(options.values);
    metric.median_ns_per_value = med_ns / static_cast<double>(options.values);
    metric.best_gvalues_per_s = static_cast<double>(options.values) / best_ns;
    metric.best_gbytes_per_s = static_cast<double>(input_bytes + output_bytes) / best_ns;
    return metric;
}

template <typename T>
void bench_target(const Options& options, std::vector<Metric>& metrics, const std::string& target) {
    const int block = options.block_size;
    const int grid = static_cast<int>((options.values + static_cast<std::size_t>(block) - 1) / block);
    T* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_output, options.values * sizeof(T)));

    metrics.push_back(time_kernel(
        options,
        target,
        0,
        "store_only",
        0,
        options.values * sizeof(T),
        0,
        [&]() { store_kernel<T><<<grid, block>>>(d_output, options.values); HIP_CHECK(hipGetLastError()); }
    ));

    const int max_bits = sizeof(T) == 1 ? 7 : 12;
    for (int bits = 1; bits <= max_bits; ++bits) {
        const auto h_indices = make_indices(options.values, bits);
        const auto h_packed = pack_indices(h_indices, bits);
        const auto h_lut = make_lut<T>(bits);

        std::uint16_t* d_indices = nullptr;
        std::uint8_t* d_packed = nullptr;
        T* d_lut = nullptr;
        HIP_CHECK(hipMalloc(&d_indices, h_indices.size() * sizeof(h_indices[0])));
        HIP_CHECK(hipMalloc(&d_packed, h_packed.size()));
        HIP_CHECK(hipMalloc(&d_lut, h_lut.size() * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(h_indices[0]), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_packed, h_packed.data(), h_packed.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_lut, h_lut.data(), h_lut.size() * sizeof(T), hipMemcpyHostToDevice));

        metrics.push_back(time_kernel(
            options,
            target,
            bits,
            "aligned_index_lut",
            h_indices.size() * sizeof(h_indices[0]),
            options.values * sizeof(T),
            h_lut.size() * sizeof(T),
            [&]() { aligned_lut_kernel<T><<<grid, block>>>(d_indices, d_lut, d_output, options.values); HIP_CHECK(hipGetLastError()); }
        ));
        metrics.push_back(time_kernel(
            options,
            target,
            bits,
            "packed_index_lut",
            h_packed.size() - 2,
            options.values * sizeof(T),
            h_lut.size() * sizeof(T),
            [&]() { packed_lut_kernel<T><<<grid, block>>>(d_packed, d_lut, d_output, options.values, bits); HIP_CHECK(hipGetLastError()); }
        ));

        HIP_CHECK(hipFree(d_indices));
        HIP_CHECK(hipFree(d_packed));
        HIP_CHECK(hipFree(d_lut));
    }
    HIP_CHECK(hipFree(d_output));
}

void print_json(const Options& options, const hipDeviceProp_t& props, const std::vector<Metric>& metrics) {
    std::cout << "{\n";
    std::cout << "  \"schema_version\": \"lut-decode-hip-benchmark-v0.1\",\n";
    std::cout << "  \"values\": " << options.values << ",\n";
    std::cout << "  \"repeats\": " << options.repeats << ",\n";
    std::cout << "  \"warmups\": " << options.warmups << ",\n";
    std::cout << "  \"device\": " << options.device << ",\n";
    std::cout << "  \"device_name\": \"" << props.name << "\",\n";
    std::cout << "  \"gcn_arch_name\": \"" << props.gcnArchName << "\",\n";
    std::cout << "  \"block_size\": " << options.block_size << ",\n";
    std::cout << "  \"notes\": [\n";
    std::cout << "    \"LUT is a normal global-memory array and relies on hardware caches; no explicit register/shared-memory placement is used.\",\n";
    std::cout << "    \"FP8 and FP16 outputs are represented as opaque uint8/uint16 payloads; this measures LUT decode and store cost, not arithmetic conversion.\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"metrics\": [\n";
    std::cout << std::fixed << std::setprecision(9);
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        const Metric& m = metrics[i];
        std::cout << "    {\n";
        std::cout << "      \"target\": \"" << m.target << "\",\n";
        std::cout << "      \"bits\": " << m.bits << ",\n";
        std::cout << "      \"mode\": \"" << m.mode << "\",\n";
        std::cout << "      \"values\": " << m.values << ",\n";
        std::cout << "      \"input_bytes\": " << m.input_bytes << ",\n";
        std::cout << "      \"output_bytes\": " << m.output_bytes << ",\n";
        std::cout << "      \"table_bytes\": " << m.table_bytes << ",\n";
        std::cout << "      \"best_ms\": " << m.best_ms << ",\n";
        std::cout << "      \"median_ms\": " << m.median_ms << ",\n";
        std::cout << "      \"best_ns_per_value\": " << m.best_ns_per_value << ",\n";
        std::cout << "      \"median_ns_per_value\": " << m.median_ns_per_value << ",\n";
        std::cout << "      \"best_gvalues_per_s\": " << m.best_gvalues_per_s << ",\n";
        std::cout << "      \"best_gbytes_per_s\": " << m.best_gbytes_per_s << "\n";
        std::cout << "    }" << (i + 1 == metrics.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        int device_count = 0;
        HIP_CHECK(hipGetDeviceCount(&device_count));
        if (options.device < 0 || options.device >= device_count) {
            throw std::runtime_error("invalid HIP device index");
        }
        HIP_CHECK(hipSetDevice(options.device));
        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, options.device));

        std::vector<Metric> metrics;
        bench_target<std::uint8_t>(options, metrics, "fp8_payload_u8");
        bench_target<std::uint16_t>(options, metrics, "fp16_payload_u16");
        print_json(options, props, metrics);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
