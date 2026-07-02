#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

#define HIP_CHECK(call)                                                                      \
    do {                                                                                     \
        hipError_t err__ = (call);                                                           \
        if (err__ != hipSuccess) {                                                           \
            throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(err__)); \
        }                                                                                    \
    } while (false)

struct Options {
    int device = 0;
    std::size_t m = 1;
    std::size_t n = 4096;
    std::size_t k = 4096;
    int repeats = 20;
    int warmups = 5;
    int block_size = 256;
};

struct Metric {
    std::string target;
    int bits = 0;
    std::size_t lut_bytes = 0;
    std::size_t expanded_weight_bytes = 0;
    std::size_t packed_weight_bytes = 0;
    float baseline_best_ms = 0.0f;
    float fused_best_ms = 0.0f;
    float baseline_median_ms = 0.0f;
    float fused_median_ms = 0.0f;
    double baseline_ns_per_weight_value = 0.0;
    double fused_ns_per_weight_value = 0.0;
    double overhead_ns_per_weight_value = 0.0;
    double fused_over_baseline = 0.0;
    double baseline_gflops = 0.0;
    double fused_gflops = 0.0;
    float max_abs_diff = 0.0f;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--device N] [--m N] [--n N] [--k N]"
              << " [--repeats N] [--warmups N] [--block-size N]\n";
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
        if (arg == "--device") {
            options.device = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--m") {
            options.m = parse_size(need_value());
        } else if (arg == "--n") {
            options.n = parse_size(need_value());
        } else if (arg == "--k") {
            options.k = parse_size(need_value());
        } else if (arg == "--repeats") {
            options.repeats = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--warmups") {
            options.warmups = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--block-size") {
            options.block_size = static_cast<int>(parse_size(need_value()));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            usage(argv[0]);
        }
    }
    if (options.m == 0 || options.n == 0 || options.k == 0 || options.repeats <= 0 || options.warmups < 0) {
        usage(argv[0]);
    }
    if ((options.block_size & (options.block_size - 1)) != 0 || options.block_size <= 0) {
        throw std::runtime_error("--block-size must be a positive power of two");
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
    std::uint32_t state = 0xa5a5a5a5u ^ static_cast<std::uint32_t>(bits * 0x9e3779b9u);
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

std::vector<__half> make_input(std::size_t values) {
    std::vector<__half> input(values);
    std::uint32_t state = 0x12345678u;
    for (std::size_t i = 0; i < values; ++i) {
        const float v = (static_cast<int>(lcg_next(state) & 0x3ffu) - 512) * (1.0f / 512.0f);
        input[i] = __float2half(v);
    }
    return input;
}

std::vector<__half> make_half_lut(int bits) {
    const std::size_t entries = 1ull << bits;
    std::vector<__half> lut(entries);
    if (entries == 1) {
        lut[0] = __float2half(0.0f);
        return lut;
    }
    for (std::size_t i = 0; i < entries; ++i) {
        const float centered = (static_cast<float>(i) / static_cast<float>(entries - 1)) * 2.0f - 1.0f;
        lut[i] = __float2half(centered);
    }
    return lut;
}

std::vector<std::uint8_t> make_fp8_lut(int bits) {
    const std::size_t entries = 1ull << bits;
    std::vector<std::uint8_t> lut(entries);
    for (std::size_t i = 0; i < entries; ++i) {
        const std::uint8_t sign = (i & 1u) ? 0x80u : 0x00u;
        const std::uint8_t exp = static_cast<std::uint8_t>(4u + ((i * 5u) & 0x7u));
        const std::uint8_t mant = static_cast<std::uint8_t>((i * 3u) & 0x7u);
        lut[i] = static_cast<std::uint8_t>(sign | (exp << 3) | mant);
    }
    return lut;
}

std::vector<__half> expand_half_weights(const std::vector<std::uint16_t>& indices, const std::vector<__half>& lut) {
    std::vector<__half> weights(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        weights[i] = lut[indices[i]];
    }
    return weights;
}

std::vector<std::uint8_t> expand_fp8_weights(
    const std::vector<std::uint16_t>& indices,
    const std::vector<std::uint8_t>& lut
) {
    std::vector<std::uint8_t> weights(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        weights[i] = lut[indices[i]];
    }
    return weights;
}

__device__ float fp8_e4m3_to_float(std::uint8_t x) {
    const float sign = (x & 0x80u) ? -1.0f : 1.0f;
    const int exp = (x >> 3) & 0x0f;
    const int mant = x & 0x07;
    if (exp == 0) {
        return sign * ldexpf(static_cast<float>(mant) / 8.0f, -6);
    }
    if (exp == 0x0f) {
        return sign * 448.0f;
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

template <typename Weight>
__device__ float weight_to_float(Weight value);

template <>
__device__ float weight_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ float weight_to_float<std::uint8_t>(std::uint8_t value) {
    return fp8_e4m3_to_float(value);
}

template <typename Weight>
__global__ void expanded_dot_kernel(
    const __half* x,
    const Weight* weights,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    extern __shared__ float scratch[];
    const std::size_t col = blockIdx.x;
    const std::size_t row = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (std::size_t kk = tid; kk < k; kk += blockDim.x) {
        const float xv = __half2float(x[row * k + kk]);
        const float wv = weight_to_float(weights[col * k + kk]);
        acc = fmaf(xv, wv, acc);
    }
    scratch[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row * n + col] = scratch[0];
    }
}

template <typename Weight>
__global__ void fused_lut_dot_kernel(
    const __half* x,
    const std::uint8_t* packed,
    const Weight* lut,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    int bits
) {
    extern __shared__ float scratch[];
    const std::size_t col = blockIdx.x;
    const std::size_t row = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    const std::uint32_t mask = (1u << bits) - 1u;
    float acc = 0.0f;
    for (std::size_t kk = tid; kk < k; kk += blockDim.x) {
        const std::size_t weight_index = col * k + kk;
        const std::size_t bit_pos = weight_index * static_cast<std::size_t>(bits);
        const std::size_t byte_pos = bit_pos >> 3;
        const int shift = static_cast<int>(bit_pos & 7);
        const std::uint32_t word = static_cast<std::uint32_t>(packed[byte_pos])
            | (static_cast<std::uint32_t>(packed[byte_pos + 1]) << 8)
            | (static_cast<std::uint32_t>(packed[byte_pos + 2]) << 16);
        const std::uint32_t index = (word >> shift) & mask;
        const float xv = __half2float(x[row * k + kk]);
        const float wv = weight_to_float(lut[index]);
        acc = fmaf(xv, wv, acc);
    }
    scratch[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row * n + col] = scratch[0];
    }
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Launch>
std::pair<float, float> time_launch(const Options& options, Launch&& launch) {
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
    return {*std::min_element(samples.begin(), samples.end()), median(samples)};
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

template <typename Weight>
Metric bench_one(
    const Options& options,
    const std::string& target,
    int bits,
    const std::vector<__half>& h_x,
    const std::vector<std::uint16_t>& h_indices,
    const std::vector<std::uint8_t>& h_packed,
    const std::vector<Weight>& h_lut,
    const std::vector<Weight>& h_weights
) {
    const std::size_t x_bytes = h_x.size() * sizeof(h_x[0]);
    const std::size_t out_values = options.m * options.n;
    const std::size_t out_bytes = out_values * sizeof(float);
    const std::size_t expanded_bytes = h_weights.size() * sizeof(Weight);
    const std::size_t packed_bytes = (h_indices.size() * static_cast<std::size_t>(bits) + 7) / 8;
    const std::size_t lut_bytes = h_lut.size() * sizeof(Weight);

    __half* d_x = nullptr;
    Weight* d_weights = nullptr;
    std::uint8_t* d_packed = nullptr;
    Weight* d_lut = nullptr;
    float* d_out_base = nullptr;
    float* d_out_fused = nullptr;
    HIP_CHECK(hipMalloc(&d_x, x_bytes));
    HIP_CHECK(hipMalloc(&d_weights, expanded_bytes));
    HIP_CHECK(hipMalloc(&d_packed, h_packed.size()));
    HIP_CHECK(hipMalloc(&d_lut, lut_bytes));
    HIP_CHECK(hipMalloc(&d_out_base, out_bytes));
    HIP_CHECK(hipMalloc(&d_out_fused, out_bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_weights.data(), expanded_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_packed, h_packed.data(), h_packed.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_lut, h_lut.data(), lut_bytes, hipMemcpyHostToDevice));

    const dim3 grid(static_cast<unsigned int>(options.n), static_cast<unsigned int>(options.m));
    const int block = options.block_size;
    const std::size_t shared_bytes = static_cast<std::size_t>(block) * sizeof(float);

    auto launch_base = [&]() {
        expanded_dot_kernel<Weight><<<grid, block, shared_bytes>>>(
            d_x,
            d_weights,
            d_out_base,
            options.m,
            options.n,
            options.k
        );
        HIP_CHECK(hipGetLastError());
    };
    auto launch_fused = [&]() {
        fused_lut_dot_kernel<Weight><<<grid, block, shared_bytes>>>(
            d_x,
            d_packed,
            d_lut,
            d_out_fused,
            options.m,
            options.n,
            options.k,
            bits
        );
        HIP_CHECK(hipGetLastError());
    };

    const auto [base_best, base_median] = time_launch(options, launch_base);
    const auto [fused_best, fused_median] = time_launch(options, launch_fused);
    launch_base();
    launch_fused();
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> h_out_base(out_values);
    std::vector<float> h_out_fused(out_values);
    HIP_CHECK(hipMemcpy(h_out_base.data(), d_out_base, out_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_out_fused.data(), d_out_fused, out_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_packed));
    HIP_CHECK(hipFree(d_lut));
    HIP_CHECK(hipFree(d_out_base));
    HIP_CHECK(hipFree(d_out_fused));

    const double weight_values = static_cast<double>(options.n * options.k);
    const double flops = 2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) * static_cast<double>(options.k);
    Metric metric;
    metric.target = target;
    metric.bits = bits;
    metric.lut_bytes = lut_bytes;
    metric.expanded_weight_bytes = expanded_bytes;
    metric.packed_weight_bytes = packed_bytes;
    metric.baseline_best_ms = base_best;
    metric.fused_best_ms = fused_best;
    metric.baseline_median_ms = base_median;
    metric.fused_median_ms = fused_median;
    metric.baseline_ns_per_weight_value = static_cast<double>(base_best) * 1.0e6 / weight_values;
    metric.fused_ns_per_weight_value = static_cast<double>(fused_best) * 1.0e6 / weight_values;
    metric.overhead_ns_per_weight_value = metric.fused_ns_per_weight_value - metric.baseline_ns_per_weight_value;
    metric.fused_over_baseline = static_cast<double>(fused_best) / static_cast<double>(base_best);
    metric.baseline_gflops = flops / (static_cast<double>(base_best) * 1.0e6);
    metric.fused_gflops = flops / (static_cast<double>(fused_best) * 1.0e6);
    metric.max_abs_diff = max_abs_diff(h_out_base, h_out_fused);
    return metric;
}

void print_json(const Options& options, const hipDeviceProp_t& props, const std::vector<Metric>& metrics) {
    std::cout << "{\n";
    std::cout << "  \"schema_version\": \"fused-lut-matmul-hip-benchmark-v0.1\",\n";
    std::cout << "  \"device\": " << options.device << ",\n";
    std::cout << "  \"device_name\": \"" << props.name << "\",\n";
    std::cout << "  \"gcn_arch_name\": \"" << props.gcnArchName << "\",\n";
    std::cout << "  \"m\": " << options.m << ",\n";
    std::cout << "  \"n\": " << options.n << ",\n";
    std::cout << "  \"k\": " << options.k << ",\n";
    std::cout << "  \"repeats\": " << options.repeats << ",\n";
    std::cout << "  \"warmups\": " << options.warmups << ",\n";
    std::cout << "  \"block_size\": " << options.block_size << ",\n";
    std::cout << "  \"notes\": [\n";
    std::cout << "    \"Baseline reads expanded FP8/FP16 payload weights and performs the same dot kernel.\",\n";
    std::cout << "    \"Fused path reads packed codebook-index, loads the payload from a normal global-memory LUT, and immediately uses it in the dot kernel.\",\n";
    std::cout << "    \"This benchmark uses scalar dot/reduction kernels, not MFMA/Tensor-core GEMM. It measures fused LUT overhead under the same kernel structure, not final optimized inference throughput.\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"metrics\": [\n";
    std::cout << std::fixed << std::setprecision(9);
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        const Metric& m = metrics[i];
        std::cout << "    {\n";
        std::cout << "      \"target\": \"" << m.target << "\",\n";
        std::cout << "      \"bits\": " << m.bits << ",\n";
        std::cout << "      \"lut_bytes\": " << m.lut_bytes << ",\n";
        std::cout << "      \"expanded_weight_bytes\": " << m.expanded_weight_bytes << ",\n";
        std::cout << "      \"packed_weight_bytes\": " << m.packed_weight_bytes << ",\n";
        std::cout << "      \"baseline_best_ms\": " << m.baseline_best_ms << ",\n";
        std::cout << "      \"fused_best_ms\": " << m.fused_best_ms << ",\n";
        std::cout << "      \"baseline_median_ms\": " << m.baseline_median_ms << ",\n";
        std::cout << "      \"fused_median_ms\": " << m.fused_median_ms << ",\n";
        std::cout << "      \"baseline_ns_per_weight_value\": " << m.baseline_ns_per_weight_value << ",\n";
        std::cout << "      \"fused_ns_per_weight_value\": " << m.fused_ns_per_weight_value << ",\n";
        std::cout << "      \"overhead_ns_per_weight_value\": " << m.overhead_ns_per_weight_value << ",\n";
        std::cout << "      \"fused_over_baseline\": " << m.fused_over_baseline << ",\n";
        std::cout << "      \"baseline_gflops\": " << m.baseline_gflops << ",\n";
        std::cout << "      \"fused_gflops\": " << m.fused_gflops << ",\n";
        std::cout << "      \"max_abs_diff\": " << m.max_abs_diff << "\n";
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

        const std::size_t weight_values = options.n * options.k;
        const auto h_x = make_input(options.m * options.k);
        std::vector<Metric> metrics;

        for (int bits = 1; bits <= 7; ++bits) {
            const auto h_indices = make_indices(weight_values, bits);
            const auto h_packed = pack_indices(h_indices, bits);
            const auto h_lut = make_fp8_lut(bits);
            const auto h_weights = expand_fp8_weights(h_indices, h_lut);
            metrics.push_back(bench_one<std::uint8_t>(
                options,
                "fp8_payload_e4m3_u8",
                bits,
                h_x,
                h_indices,
                h_packed,
                h_lut,
                h_weights
            ));
        }

        for (int bits = 1; bits <= 12; ++bits) {
            const auto h_indices = make_indices(weight_values, bits);
            const auto h_packed = pack_indices(h_indices, bits);
            const auto h_lut = make_half_lut(bits);
            const auto h_weights = expand_half_weights(h_indices, h_lut);
            metrics.push_back(bench_one<__half>(
                options,
                "fp16_payload_half",
                bits,
                h_x,
                h_indices,
                h_packed,
                h_lut,
                h_weights
            ));
        }

        print_json(options, props, metrics);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
