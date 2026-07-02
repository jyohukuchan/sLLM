#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

volatile std::uint64_t g_sink = 0;

struct Options {
    std::size_t values = 16ull * 1024ull * 1024ull;
    int repeats = 7;
    int warmups = 2;
};

struct Metric {
    std::string target;
    int bits = 0;
    std::string mode;
    std::size_t values = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t table_bytes = 0;
    double best_ns_per_value = 0.0;
    double median_ns_per_value = 0.0;
    double best_gvalues_per_s = 0.0;
    double best_gbytes_per_s = 0.0;
    std::uint64_t checksum = 0;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--values N] [--repeats N] [--warmups N]\n";
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
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            ++i;
            (void)name;
            return std::string_view(argv[i]);
        };
        if (arg == "--values") {
            options.values = parse_size(need_value("--values"));
        } else if (arg == "--repeats") {
            options.repeats = static_cast<int>(parse_size(need_value("--repeats")));
        } else if (arg == "--warmups") {
            options.warmups = static_cast<int>(parse_size(need_value("--warmups")));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            usage(argv[0]);
        }
    }
    if (options.values == 0 || options.repeats <= 0 || options.warmups < 0) {
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
    std::uint32_t state = 0x12345678u ^ static_cast<std::uint32_t>(bits * 0x9e3779b9u);
    for (std::size_t i = 0; i < values; ++i) {
        indices[i] = static_cast<std::uint16_t>(lcg_next(state) & mask);
    }
    return indices;
}

std::vector<std::uint8_t> pack_indices(const std::vector<std::uint16_t>& indices, int bits) {
    const std::size_t out_bytes = (indices.size() * static_cast<std::size_t>(bits) + 7) / 8;
    std::vector<std::uint8_t> packed(out_bytes, 0);
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
            // FP8 payload stand-in: an opaque 8-bit value loaded through a LUT.
            lut[i] = static_cast<T>((i * 37u + 11u) & 0xffu);
        } else {
            // FP16 payload stand-in: an opaque IEEE half/bfloat-like 16-bit payload.
            lut[i] = static_cast<T>((i * 131u + 0x3c00u) & 0xffffu);
        }
    }
    return lut;
}

template <typename T>
std::uint64_t checksum_output(const std::vector<T>& output) {
    std::uint64_t sum = 1469598103934665603ull;
    for (T value : output) {
        sum ^= static_cast<std::uint64_t>(value);
        sum *= 1099511628211ull;
    }
    return sum;
}

template <typename F>
double time_once(F&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename T>
void bench_store(const Options& options, std::vector<T>& output, std::vector<Metric>& metrics, std::string target) {
    std::vector<double> samples;
    auto fn = [&]() {
        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] = static_cast<T>(i);
        }
    };
    for (int i = 0; i < options.warmups; ++i) {
        fn();
    }
    for (int i = 0; i < options.repeats; ++i) {
        samples.push_back(time_once(fn));
    }
    g_sink ^= checksum_output(output);
    const double best_ns = *std::min_element(samples.begin(), samples.end());
    const double med_ns = median(samples);
    Metric metric;
    metric.target = std::move(target);
    metric.bits = 0;
    metric.mode = "store_only";
    metric.values = output.size();
    metric.output_bytes = output.size() * sizeof(T);
    metric.best_ns_per_value = best_ns / static_cast<double>(output.size());
    metric.median_ns_per_value = med_ns / static_cast<double>(output.size());
    metric.best_gvalues_per_s = static_cast<double>(output.size()) / best_ns;
    metric.best_gbytes_per_s = static_cast<double>(metric.output_bytes) / best_ns;
    metric.checksum = g_sink;
    metrics.push_back(metric);
}

template <typename T>
void bench_aligned_lut(
    const Options& options,
    int bits,
    const std::vector<std::uint16_t>& indices,
    const std::vector<T>& lut,
    std::vector<T>& output,
    std::vector<Metric>& metrics,
    std::string target
) {
    std::vector<double> samples;
    auto fn = [&]() {
        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] = lut[indices[i]];
        }
    };
    for (int i = 0; i < options.warmups; ++i) {
        fn();
    }
    for (int i = 0; i < options.repeats; ++i) {
        samples.push_back(time_once(fn));
    }
    g_sink ^= checksum_output(output);
    const double best_ns = *std::min_element(samples.begin(), samples.end());
    const double med_ns = median(samples);
    Metric metric;
    metric.target = std::move(target);
    metric.bits = bits;
    metric.mode = "aligned_index_lut";
    metric.values = output.size();
    metric.input_bytes = indices.size() * sizeof(indices[0]);
    metric.output_bytes = output.size() * sizeof(T);
    metric.table_bytes = lut.size() * sizeof(T);
    metric.best_ns_per_value = best_ns / static_cast<double>(output.size());
    metric.median_ns_per_value = med_ns / static_cast<double>(output.size());
    metric.best_gvalues_per_s = static_cast<double>(output.size()) / best_ns;
    metric.best_gbytes_per_s = static_cast<double>(metric.input_bytes + metric.output_bytes) / best_ns;
    metric.checksum = g_sink;
    metrics.push_back(metric);
}

template <typename T>
void bench_packed_lut(
    const Options& options,
    int bits,
    const std::vector<std::uint8_t>& packed,
    const std::vector<T>& lut,
    std::vector<T>& output,
    std::vector<Metric>& metrics,
    std::string target
) {
    const std::uint64_t mask = (1ull << bits) - 1ull;
    std::vector<double> samples;
    auto fn = [&]() {
        std::uint64_t acc = 0;
        int acc_bits = 0;
        std::size_t in_pos = 0;
        for (std::size_t i = 0; i < output.size(); ++i) {
            while (acc_bits < bits) {
                acc |= static_cast<std::uint64_t>(packed[in_pos++]) << acc_bits;
                acc_bits += 8;
            }
            const std::uint32_t index = static_cast<std::uint32_t>(acc & mask);
            acc >>= bits;
            acc_bits -= bits;
            output[i] = lut[index];
        }
    };
    for (int i = 0; i < options.warmups; ++i) {
        fn();
    }
    for (int i = 0; i < options.repeats; ++i) {
        samples.push_back(time_once(fn));
    }
    g_sink ^= checksum_output(output);
    const double best_ns = *std::min_element(samples.begin(), samples.end());
    const double med_ns = median(samples);
    Metric metric;
    metric.target = std::move(target);
    metric.bits = bits;
    metric.mode = "packed_index_lut";
    metric.values = output.size();
    metric.input_bytes = packed.size();
    metric.output_bytes = output.size() * sizeof(T);
    metric.table_bytes = lut.size() * sizeof(T);
    metric.best_ns_per_value = best_ns / static_cast<double>(output.size());
    metric.median_ns_per_value = med_ns / static_cast<double>(output.size());
    metric.best_gvalues_per_s = static_cast<double>(output.size()) / best_ns;
    metric.best_gbytes_per_s = static_cast<double>(metric.input_bytes + metric.output_bytes) / best_ns;
    metric.checksum = g_sink;
    metrics.push_back(metric);
}

void print_json(const Options& options, const std::vector<Metric>& metrics) {
    std::cout << "{\n";
    std::cout << "  \"schema_version\": \"lut-decode-benchmark-v0.1\",\n";
    std::cout << "  \"values\": " << options.values << ",\n";
    std::cout << "  \"repeats\": " << options.repeats << ",\n";
    std::cout << "  \"warmups\": " << options.warmups << ",\n";
    std::cout << "  \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n";
    std::cout << "  \"notes\": [\n";
    std::cout << "    \"FP8 and FP16 outputs are represented as opaque uint8/uint16 payloads; this measures LUT decode and store cost, not arithmetic conversion.\",\n";
    std::cout << "    \"packed_index_lut includes scalar bit unpack from a dense little-endian bitstream. aligned_index_lut isolates LUT load plus output store with uint16 indices.\"\n";
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
        std::cout << "      \"best_ns_per_value\": " << m.best_ns_per_value << ",\n";
        std::cout << "      \"median_ns_per_value\": " << m.median_ns_per_value << ",\n";
        std::cout << "      \"best_gvalues_per_s\": " << m.best_gvalues_per_s << ",\n";
        std::cout << "      \"best_gbytes_per_s\": " << m.best_gbytes_per_s << ",\n";
        std::cout << "      \"checksum\": " << m.checksum << "\n";
        std::cout << "    }" << (i + 1 == metrics.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"sink\": " << g_sink << "\n";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::vector<Metric> metrics;

        {
            std::vector<std::uint8_t> output(options.values);
            bench_store(options, output, metrics, "fp8_payload_u8");
            for (int bits = 1; bits <= 7; ++bits) {
                const auto indices = make_indices(options.values, bits);
                const auto packed = pack_indices(indices, bits);
                const auto lut = make_lut<std::uint8_t>(bits);
                bench_aligned_lut(options, bits, indices, lut, output, metrics, "fp8_payload_u8");
                bench_packed_lut(options, bits, packed, lut, output, metrics, "fp8_payload_u8");
            }
        }

        {
            std::vector<std::uint16_t> output(options.values);
            bench_store(options, output, metrics, "fp16_payload_u16");
            for (int bits = 1; bits <= 12; ++bits) {
                const auto indices = make_indices(options.values, bits);
                const auto packed = pack_indices(indices, bits);
                const auto lut = make_lut<std::uint16_t>(bits);
                bench_aligned_lut(options, bits, indices, lut, output, metrics, "fp16_payload_u16");
                bench_packed_lut(options, bits, packed, lut, output, metrics, "fp16_payload_u16");
            }
        }

        print_json(options, metrics);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
