// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// CPU counterpart of hw-microbench-rdna4-cdna3.hip.cpp.  It is intentionally
// standalone: compile it with a host C++ compiler and run it on a rented VM.
// Timed regions contain only the specified kernel.  Allocation/initialisation,
// warmups, checksum consumption, and JSON emission are outside those regions.

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    size_t workset_mib = 512;  // Per STREAM vector, deliberately above the VM L3.
    unsigned warmups = 3;
    unsigned repeats = 7;
    unsigned compute_warmups = 2;
    unsigned compute_repeats = 5;
    uint64_t compute_iterations = 200000000ull;
    std::string mode = "all";
};

double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::array<double, 3> loadavg() {
    std::array<double, 3> result{};
    (void)getloadavg(result.data(), static_cast<int>(result.size()));
    return result;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void json_string(std::ostream& os, std::string_view s) {
    os << '"';
    for (const char c : s) {
        if (c == '"' || c == '\\') os << '\\';
        os << c;
    }
    os << '"';
}

void emit_sample(std::string_view name, unsigned threads, unsigned sample, double seconds,
                 uint64_t bytes, uint64_t flops, const std::array<double, 3>& before,
                 const std::array<double, 3>& after) {
    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema\":\"ullm.cpu-microbench.v1\",\"kind\":\"sample\",\"name\":";
    json_string(std::cout, name);
    std::cout << ",\"threads\":" << threads << ",\"sample\":" << sample
              << ",\"seconds\":" << seconds << ",\"bytes\":" << bytes
              << ",\"flops\":" << flops << ",\"loadavg_before\":[" << before[0] << ',' << before[1] << ',' << before[2]
              << "],\"loadavg_after\":[" << after[0] << ',' << after[1] << ',' << after[2] << "]}\n";
}

void emit_summary(std::string_view name, unsigned threads, const std::vector<double>& samples,
                  uint64_t bytes, uint64_t flops, unsigned warmups) {
    const double elapsed = median(samples);
    std::cout << std::fixed << std::setprecision(3)
              << "{\"schema\":\"ullm.cpu-microbench.v1\",\"kind\":\"summary\",\"name\":";
    json_string(std::cout, name);
    std::cout << ",\"threads\":" << threads << ",\"statistic\":\"median\",\"warmups\":" << warmups
              << ",\"repeats\":" << samples.size() << ",\"seconds\":" << elapsed;
    if (bytes) std::cout << ",\"bandwidth_GBps\":" << static_cast<double>(bytes) / elapsed / 1e9;
    if (flops) std::cout << ",\"TFLOPS\":" << static_cast<double>(flops) / elapsed / 1e12;
    std::cout << ",\"bytes_timed_region\":" << bytes << ",\"flops_timed_region\":" << flops
              << ",\"timed_region\":\"one complete kernel invocation; allocation, initialization, warmups, checksum, and output excluded\"}\n";
}

enum class StreamOp { read, copy, triad };

double stream_once(StreamOp op, const float* a, const float* b, float* c, size_t n, unsigned threads) {
    double checksum = 0.0;
    const double begin = now_seconds();
#pragma omp parallel for schedule(static) num_threads(threads) reduction(+ : checksum)
    for (size_t i = 0; i < n; ++i) {
        if (op == StreamOp::read) checksum += a[i];
        else if (op == StreamOp::copy) c[i] = a[i];
        else c[i] = a[i] + 1.2345f * b[i];
    }
    const double end = now_seconds();
    // Makes the read loop and stores observable without contaminating timing.
    static volatile double observed = 0.0;
    if (op == StreamOp::read) observed += checksum;
    else observed += c[n / 2];
    return end - begin;
}

void run_stream(const Options& o, unsigned threads, StreamOp op, std::string_view name) {
    const size_t n = o.workset_mib * 1024ull * 1024ull / sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f);
    const uint64_t bytes = (op == StreamOp::read ? 1ull : op == StreamOp::copy ? 2ull : 3ull) * n * sizeof(float);
    for (unsigned i = 0; i < o.warmups; ++i) (void)stream_once(op, a.data(), b.data(), c.data(), n, threads);
    std::vector<double> samples;
    samples.reserve(o.repeats);
    for (unsigned i = 0; i < o.repeats; ++i) {
        const auto before = loadavg();
        const double seconds = stream_once(op, a.data(), b.data(), c.data(), n, threads);
        const auto after = loadavg();
        samples.push_back(seconds);
        emit_sample(name, threads, i, seconds, bytes, 0, before, after);
    }
    emit_summary(name, threads, samples, bytes, 0, o.warmups);
}

__attribute__((target("avx512f,fma")))
float fp32_fma_loop(uint64_t iterations) {
    const __m512 x = _mm512_set1_ps(1.00001f);
    const __m512 y = _mm512_set1_ps(0.99999f);
    __m512 a0 = _mm512_set1_ps(0.1f), a1 = a0, a2 = a0, a3 = a0;
    __m512 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (uint64_t i = 0; i < iterations; ++i) {
        a0 = _mm512_fmadd_ps(x, y, a0); a1 = _mm512_fmadd_ps(x, y, a1);
        a2 = _mm512_fmadd_ps(x, y, a2); a3 = _mm512_fmadd_ps(x, y, a3);
        a4 = _mm512_fmadd_ps(x, y, a4); a5 = _mm512_fmadd_ps(x, y, a5);
        a6 = _mm512_fmadd_ps(x, y, a6); a7 = _mm512_fmadd_ps(x, y, a7);
    }
    const __m512 sum01 = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    const __m512 sum45 = _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7));
    alignas(64) float result[16];
    _mm512_store_ps(result, _mm512_add_ps(sum01, sum45));
    return std::accumulate(std::begin(result), std::end(result), 0.0f);
}

__attribute__((target("avx512f,avx512bf16")))
float bf16_dp_loop(uint64_t iterations) {
    const __m512bh x = (__m512bh)_mm512_set1_epi16(0x3f80);  // bf16 1.0
    const __m512bh y = (__m512bh)_mm512_set1_epi16(0x3f80);
    __m512 a0 = _mm512_set1_ps(0.1f), a1 = a0, a2 = a0, a3 = a0;
    __m512 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (uint64_t i = 0; i < iterations; ++i) {
        a0 = _mm512_dpbf16_ps(a0, x, y); a1 = _mm512_dpbf16_ps(a1, x, y);
        a2 = _mm512_dpbf16_ps(a2, x, y); a3 = _mm512_dpbf16_ps(a3, x, y);
        a4 = _mm512_dpbf16_ps(a4, x, y); a5 = _mm512_dpbf16_ps(a5, x, y);
        a6 = _mm512_dpbf16_ps(a6, x, y); a7 = _mm512_dpbf16_ps(a7, x, y);
    }
    const __m512 sum01 = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    const __m512 sum45 = _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7));
    alignas(64) float result[16];
    _mm512_store_ps(result, _mm512_add_ps(sum01, sum45));
    return std::accumulate(std::begin(result), std::end(result), 0.0f);
}

using ComputeFn = float (*)(uint64_t);

double compute_once(ComputeFn fn, uint64_t iterations, unsigned threads) {
    float observed = 0.0f;
    const double begin = now_seconds();
#pragma omp parallel num_threads(threads) reduction(+ : observed)
    observed += fn(iterations);
    const double end = now_seconds();
    static volatile float sink = 0.0f;
    sink += observed;
    return end - begin;
}

void run_compute(const Options& o, unsigned threads, std::string_view name, ComputeFn fn, uint64_t flops_per_iteration) {
    for (unsigned i = 0; i < o.compute_warmups; ++i) (void)compute_once(fn, o.compute_iterations, threads);
    std::vector<double> samples;
    samples.reserve(o.compute_repeats);
    const uint64_t flops = flops_per_iteration * o.compute_iterations * threads;
    for (unsigned i = 0; i < o.compute_repeats; ++i) {
        const auto before = loadavg();
        const double seconds = compute_once(fn, o.compute_iterations, threads);
        const auto after = loadavg();
        samples.push_back(seconds);
        emit_sample(name, threads, i, seconds, 0, flops, before, after);
    }
    emit_summary(name, threads, samples, 0, flops, o.compute_warmups);
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* flag) -> const char* {
            if (arg != flag || i + 1 == argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (arg == "--workset-mib") o.workset_mib = std::stoull(value("--workset-mib"));
        else if (arg == "--warmups") o.warmups = std::stoul(value("--warmups"));
        else if (arg == "--repeats") o.repeats = std::stoul(value("--repeats"));
        else if (arg == "--compute-warmups") o.compute_warmups = std::stoul(value("--compute-warmups"));
        else if (arg == "--compute-repeats") o.compute_repeats = std::stoul(value("--compute-repeats"));
        else if (arg == "--compute-iterations") o.compute_iterations = std::stoull(value("--compute-iterations"));
        else if (arg == "--mode") o.mode = value("--mode");
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (o.workset_mib < 64 || o.warmups == 0 || o.repeats == 0 || o.compute_repeats == 0 || o.compute_iterations == 0)
        throw std::runtime_error("invalid benchmark options");
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        const std::array<unsigned, 4> thread_counts{1, 4, 8, 13};
        std::cout << "{\"schema\":\"ullm.cpu-microbench.v1\",\"kind\":\"configuration\",\"workset_mib_per_vector\":"
                  << o.workset_mib << ",\"stream_warmups\":" << o.warmups << ",\"stream_repeats\":" << o.repeats
                  << ",\"compute_warmups\":" << o.compute_warmups << ",\"compute_repeats\":" << o.compute_repeats
                  << ",\"compute_iterations_per_thread\":" << o.compute_iterations << "}\n";
        for (const unsigned threads : thread_counts) {
            if (o.mode == "all" || o.mode == "stream") {
                run_stream(o, threads, StreamOp::read, "stream_read");
                run_stream(o, threads, StreamOp::copy, "stream_copy");
                run_stream(o, threads, StreamOp::triad, "stream_triad");
            }
            if (o.mode == "all" || o.mode == "compute") {
                // Eight independent vector instructions per loop.  FP32 FMA is
                // 8 * 16 lanes * 2 FLOP = 256 FLOP/iteration; dpbf16 consumes
                // two BF16 products per FP32 lane, so it is 512 FLOP/iteration.
                run_compute(o, threads, "avx512_fp32_fma", fp32_fma_loop, 256);
                run_compute(o, threads, "avx512_bf16_dp", bf16_dp_loop, 512);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "cpu-microbench error: " << e.what() << '\n';
        return 1;
    }
}
