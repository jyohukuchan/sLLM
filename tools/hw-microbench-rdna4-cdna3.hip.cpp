// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Standalone RDNA4/CDNA3 hardware microbenchmark.  It deliberately does not
// link uLLM runtime code: a rental host can build and execute it independently.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr unsigned kTile = 16;
constexpr unsigned kDefaultWarmups = 5;
constexpr unsigned kDefaultRepeats = 11;
constexpr unsigned kDefaultInner = 10;
constexpr size_t kDefaultWorksetMiB = 256; // Per STREAM vector: beyond both GPUs' cache hierarchies.

struct Options {
    int device = 0;
    unsigned warmups = kDefaultWarmups;
    unsigned repeats = kDefaultRepeats;
    unsigned inner = kDefaultInner;
    size_t workset_mib = kDefaultWorksetMiB;
    std::string mode = "all";
    std::string output;
    double memory_peak_gbps = 0.0;
    double bf16_peak_tflops = 0.0;
    double fp8_peak_tflops = 0.0;
};

void check(hipError_t status, const char* call) {
    if (status != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(status));
}
#define HIP_CHECK(call) check((call), #call)

template <typename T> class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) : count_(count) { HIP_CHECK(hipMalloc(&ptr_, count * sizeof(T))); }
    ~DeviceBuffer() { if (ptr_) hipFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
    size_t count() const { return count_; }
  private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

class EventTimer {
  public:
    EventTimer() { HIP_CHECK(hipEventCreate(&start_)); HIP_CHECK(hipEventCreate(&stop_)); }
    ~EventTimer() { hipEventDestroy(start_); hipEventDestroy(stop_); }
    template <typename F> double measure(F&& f) {
        HIP_CHECK(hipEventRecord(start_)); f(); HIP_CHECK(hipEventRecord(stop_));
        HIP_CHECK(hipEventSynchronize(stop_)); float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms) / 1000.0;
    }
  private:
    hipEvent_t start_{}; hipEvent_t stop_{};
};

template <typename F> double median_seconds(unsigned warmups, unsigned repeats, F&& f) {
    for (unsigned i = 0; i < warmups; ++i) f();
    std::vector<double> samples;
    samples.reserve(repeats);
    for (unsigned i = 0; i < repeats; ++i) samples.push_back(f());
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

__global__ void stream_read_kernel(const float* __restrict__ input, float* __restrict__ partial, size_t n) {
    float sum = 0.0f;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) sum += input[i];
    __shared__ float reduce[256]; reduce[threadIdx.x] = sum; __syncthreads();
    for (unsigned offset = 128; offset; offset >>= 1) { if (threadIdx.x < offset) reduce[threadIdx.x] += reduce[threadIdx.x + offset]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(partial, reduce[0]);
}
__global__ void stream_copy_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) output[i] = input[i];
}
__global__ void stream_triad_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) c[i] = a[i] + 1.2345f * b[i];
}

// A and C are row-major; B is physically column-major. One CTA is exactly
// one hardware wave, so rocWMMA owns the target-dependent accumulator layout.
__global__ void bf16_gemm_kernel(const uint16_t* a, const uint16_t* b_col, float* c,
                                 unsigned m, unsigned n, unsigned k) {
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx942__)
    using namespace rocwmma;
    using A = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using B = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using C = fragment<accumulator, 16, 16, 16, float32_t, row_major>;
    const unsigned row = blockIdx.y * kTile, col = blockIdx.x * kTile;
    if (row + kTile > m || col + kTile > n) return;
    C acc; fill_fragment(acc, 0.0f);
    for (unsigned kk = 0; kk < k; kk += 16) {
        A af; B bf;
        load_matrix_sync(af, reinterpret_cast<const bfloat16_t*>(a + static_cast<size_t>(row) * k + kk), k);
        load_matrix_sync(bf, reinterpret_cast<const bfloat16_t*>(b_col + static_cast<size_t>(col) * k + kk), k);
        mma_sync(acc, af, bf, acc);
    }
    store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc, n, mem_row_major);
#else
    (void)a; (void)b_col; (void)c; (void)m; (void)n; (void)k;
#endif
}

__global__ void fp8_gemm_kernel(const uint8_t* a, const uint8_t* b_col, float* c,
                                unsigned m, unsigned n, unsigned k) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1200__) || defined(__gfx1201__)
    using namespace rocwmma;
    using A = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
    using B = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
    using C = fragment<accumulator, 16, 16, 16, float32_t, row_major>;
    constexpr unsigned ktile = 16; constexpr float scale = 1.0f;
#elif defined(__gfx942__)
    using namespace rocwmma;
    using A = fragment<matrix_a, 16, 16, 32, float8_fnuz_t, row_major>;
    using B = fragment<matrix_b, 16, 16, 32, float8_fnuz_t, col_major>;
    using C = fragment<accumulator, 16, 16, 32, float32_t, row_major>;
    // The raw bytes encode OCP E4M3FN. gfx942 reads them as FNUZ, whose
    // values are half. Both operands are converted, so compensate by x4.
    constexpr unsigned ktile = 32; constexpr float scale = 4.0f;
#else
    (void)a; (void)b_col; (void)c; (void)m; (void)n; (void)k; return;
#endif
    const unsigned row = blockIdx.y * kTile, col = blockIdx.x * kTile;
    if (row + kTile > m || col + kTile > n) return;
    C acc; fill_fragment(acc, 0.0f);
    for (unsigned kk = 0; kk < k; kk += ktile) {
        A af; B bf;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1200__) || defined(__gfx1201__)
        load_matrix_sync(af, reinterpret_cast<const float8_t*>(a + static_cast<size_t>(row) * k + kk), k);
        load_matrix_sync(bf, reinterpret_cast<const float8_t*>(b_col + static_cast<size_t>(col) * k + kk), k);
#elif defined(__gfx942__)
        load_matrix_sync(af, reinterpret_cast<const float8_fnuz_t*>(a + static_cast<size_t>(row) * k + kk), k);
        load_matrix_sync(bf, reinterpret_cast<const float8_fnuz_t*>(b_col + static_cast<size_t>(col) * k + kk), k);
#endif
        mma_sync(acc, af, bf, acc);
    }
    if constexpr (scale != 1.0f) for (unsigned i = 0; i < C::num_elements; ++i) acc[i] *= scale;
    store_matrix_sync(c + static_cast<size_t>(row) * n + col, acc, n, mem_row_major);
}

float bf16_to_f32(uint16_t x) { return std::bit_cast<float>(static_cast<uint32_t>(x) << 16); }
uint16_t f32_to_bf16(float x) { uint32_t u = std::bit_cast<uint32_t>(x); return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
float ocp_e4m3_to_f32(uint8_t x) {
    if ((x & 0x7f) == 0) return (x & 0x80) ? -0.0f : 0.0f;
    unsigned e = (x >> 3) & 15, f = x & 7; float sign = (x & 0x80) ? -1.0f : 1.0f;
    return e == 0 ? sign * std::ldexp(static_cast<float>(f), -9) : sign * std::ldexp(1.0f + f / 8.0f, static_cast<int>(e) - 7);
}

std::string arch_name(const hipDeviceProp_t& p) { return std::string(p.gcnArchName); }
void require_supported_arch(const hipDeviceProp_t& p) {
    const std::string arch = arch_name(p);
    if (arch.rfind("gfx1201", 0) != 0 && arch.rfind("gfx942", 0) != 0)
        throw std::runtime_error("this benchmark requires exact gfx1201 or gfx942; selected " + arch);
}
void json_string(std::ostream& os, std::string_view s) { os << '"'; for (char c : s) { if (c == '"' || c == '\\') os << '\\'; os << c; } os << '"'; }
void emit_metric(std::ostream& os, std::string_view kind, std::string_view name, double value,
                 std::string_view unit, double peak, unsigned warmups, unsigned repeats, unsigned inner,
                 std::string_view arch, std::string_view extra = "") {
    os << "{\"schema\":\"ullm.hw-microbench.v1\",\"kind\":"; json_string(os, kind);
    os << ",\"name\":"; json_string(os, name); os << ",\"value\":" << std::fixed << std::setprecision(3) << value;
    os << ",\"unit\":"; json_string(os, unit); os << ",\"peak\":" << peak;
    os << ",\"peak_percent\":" << (peak > 0 ? value * 100.0 / peak : 0.0);
    os << ",\"statistic\":\"median\",\"warmups\":" << warmups << ",\"repeats\":" << repeats << ",\"inner_iterations\":" << inner;
    os << ",\"arch\":"; json_string(os, arch); if (!extra.empty()) os << ',' << extra; os << "}\n";
}

void validate_gemm(const hipDeviceProp_t& p) {
    constexpr unsigned m = 16, n = 16, k = 32;
    std::vector<uint16_t> a_bf(m * k), b_bf(n * k);
    std::vector<uint8_t> a_fp(m * k), b_fp(n * k);
    constexpr std::array<uint8_t, 8> fp_values{0x28, 0x30, 0x38, 0x3a, 0xa8, 0xb0, 0xb8, 0xba};
    for (size_t i = 0; i < a_bf.size(); ++i) { a_bf[i] = f32_to_bf16((static_cast<int>(i % 9) - 4) * 0.125f); a_fp[i] = fp_values[i % fp_values.size()]; }
    for (size_t i = 0; i < b_bf.size(); ++i) { b_bf[i] = f32_to_bf16((static_cast<int>((i * 3) % 11) - 5) * 0.125f); b_fp[i] = fp_values[(i * 5) % fp_values.size()]; }
    DeviceBuffer<uint16_t> da(a_bf.size()), db(b_bf.size()); DeviceBuffer<uint8_t> dfa(a_fp.size()), dfb(b_fp.size()); DeviceBuffer<float> dc(m * n);
    HIP_CHECK(hipMemcpy(da.get(), a_bf.data(), a_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db.get(), b_bf.data(), b_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dfa.get(), a_fp.data(), a_fp.size(), hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(dfb.get(), b_fp.data(), b_fp.size(), hipMemcpyHostToDevice));
    dim3 grid(1, 1), block(p.warpSize);
    hipLaunchKernelGGL(bf16_gemm_kernel, grid, block, 0, 0, da.get(), db.get(), dc.get(), m, n, k); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(m * n); HIP_CHECK(hipMemcpy(got.data(), dc.get(), got.size() * sizeof(float), hipMemcpyDeviceToHost));
    double bf_max = 0;
    for (unsigned r = 0; r < m; ++r) for (unsigned c = 0; c < n; ++c) { float ref = 0; for (unsigned q = 0; q < k; ++q) ref += bf16_to_f32(a_bf[r*k+q]) * bf16_to_f32(b_bf[c*k+q]); bf_max = std::max(bf_max, std::abs(static_cast<double>(ref - got[r*n+c]))); }
    if (bf_max > 2e-4) throw std::runtime_error("BF16 CPU reference mismatch max_abs=" + std::to_string(bf_max));
    hipLaunchKernelGGL(fp8_gemm_kernel, grid, block, 0, 0, dfa.get(), dfb.get(), dc.get(), m, n, k); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got.data(), dc.get(), got.size() * sizeof(float), hipMemcpyDeviceToHost));
    double fp_max = 0;
    for (unsigned r = 0; r < m; ++r) for (unsigned c = 0; c < n; ++c) { float ref = 0; for (unsigned q = 0; q < k; ++q) ref += ocp_e4m3_to_f32(a_fp[r*k+q]) * ocp_e4m3_to_f32(b_fp[c*k+q]); fp_max = std::max(fp_max, std::abs(static_cast<double>(ref - got[r*n+c]))); }
    if (fp_max > 2e-3) throw std::runtime_error("FP8 OCP/FNUZ CPU reference mismatch max_abs=" + std::to_string(fp_max));
    std::cout << "{\"schema\":\"ullm.hw-microbench.v1\",\"kind\":\"validation\",\"bf16_cpu_max_abs\":" << bf_max << ",\"fp8_ocp_fnuz_cpu_max_abs\":" << fp_max << ",\"fp8_gfx942_two_operand_scale\":4,\"status\":\"pass\"}\n";
}

void run_bandwidth(const Options& o, const hipDeviceProp_t& p, std::ostream& out) {
    const size_t n = o.workset_mib * 1024ull * 1024ull / sizeof(float); if (n < 1024) throw std::runtime_error("workset too small");
    DeviceBuffer<float> a(n), b(n), c(n), partial(1); std::vector<float> host(n, 1.0f);
    HIP_CHECK(hipMemcpy(a.get(), host.data(), n*sizeof(float), hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(b.get(), host.data(), n*sizeof(float), hipMemcpyHostToDevice));
    dim3 block(256), grid(std::min<size_t>(65535, (n + 255) / 256)); EventTimer timer;
    const std::string arch = arch_name(p);
    auto sample = [&](auto launch) { return median_seconds(o.warmups, o.repeats, [&] { return timer.measure([&] { for (unsigned j=0;j<o.inner;++j) launch(); }); }); };
    double read_s = sample([&] { HIP_CHECK(hipMemset(partial.get(), 0, sizeof(float))); hipLaunchKernelGGL(stream_read_kernel, grid, block, 0, 0, a.get(), partial.get(), n); HIP_CHECK(hipGetLastError()); });
    emit_metric(out, "bandwidth", "read", (double(n)*sizeof(float)*o.inner/read_s)/1e9, "GB/s", o.memory_peak_gbps, o.warmups,o.repeats,o.inner,arch, "\"bytes_per_iteration\":4");
    double copy_s = sample([&] { hipLaunchKernelGGL(stream_copy_kernel, grid, block, 0, 0, a.get(), c.get(), n); HIP_CHECK(hipGetLastError()); });
    emit_metric(out, "bandwidth", "copy", (double(n)*sizeof(float)*2*o.inner/copy_s)/1e9, "GB/s", o.memory_peak_gbps, o.warmups,o.repeats,o.inner,arch, "\"bytes_per_iteration\":8");
    double triad_s = sample([&] { hipLaunchKernelGGL(stream_triad_kernel, grid, block, 0, 0, a.get(), b.get(), c.get(), n); HIP_CHECK(hipGetLastError()); });
    emit_metric(out, "bandwidth", "triad", (double(n)*sizeof(float)*3*o.inner/triad_s)/1e9, "GB/s", o.memory_peak_gbps, o.warmups,o.repeats,o.inner,arch, "\"bytes_per_iteration\":12");
}

template <bool FP8> void run_gemm_case(const Options& o, const hipDeviceProp_t& p, std::ostream& out, std::string_view name, unsigned m, unsigned n, unsigned k) {
    const unsigned align = FP8 && arch_name(p).rfind("gfx942",0)==0 ? 32 : 16;
    if (m % 16 || n % 16 || k % align) throw std::runtime_error("invalid GEMM shape alignment");
    const size_t ae = size_t(m)*k, be = size_t(n)*k, ce = size_t(m)*n;
    DeviceBuffer<float> c(ce); std::vector<uint8_t> init8;
    dim3 block(p.warpSize), grid(n/16,m/16); EventTimer timer;
    if constexpr (FP8) {
        init8.resize(ae + be); constexpr std::array<uint8_t,8> vals{0x28,0x30,0x38,0x3a,0xa8,0xb0,0xb8,0xba}; for(size_t i=0;i<init8.size();++i) init8[i]=vals[i%vals.size()];
        DeviceBuffer<uint8_t> a(ae), b(be); HIP_CHECK(hipMemcpy(a.get(),init8.data(),ae,hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(b.get(),init8.data()+ae,be,hipMemcpyHostToDevice));
        double s=median_seconds(o.warmups,o.repeats,[&]{return timer.measure([&]{for(unsigned j=0;j<o.inner;++j){hipLaunchKernelGGL(fp8_gemm_kernel,grid,block,0,0,a.get(),b.get(),c.get(),m,n,k);HIP_CHECK(hipGetLastError());}});});
        emit_metric(out,"gemm","fp8_"+std::string(name),(2.0*m*n*k*o.inner/s)/1e12,"TFLOPS",o.fp8_peak_tflops,o.warmups,o.repeats,o.inner,arch_name(p),"\"m\":"+std::to_string(m)+",\"n\":"+std::to_string(n)+",\"k\":"+std::to_string(k)+",\"flops_per_iteration\":"+std::to_string(2ull*m*n*k)+",\"fp8_input\":\"OCP E4M3FN; gfx942 raw-to-FNUZ x4 compensated\"");
    } else {
        std::vector<uint16_t> init(ae+be); for(size_t i=0;i<init.size();++i) init[i]=f32_to_bf16((int(i%11)-5)*0.125f);
        DeviceBuffer<uint16_t> a(ae),b(be); HIP_CHECK(hipMemcpy(a.get(),init.data(),ae*2,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(b.get(),init.data()+ae,be*2,hipMemcpyHostToDevice));
        double s=median_seconds(o.warmups,o.repeats,[&]{return timer.measure([&]{for(unsigned j=0;j<o.inner;++j){hipLaunchKernelGGL(bf16_gemm_kernel,grid,block,0,0,a.get(),b.get(),c.get(),m,n,k);HIP_CHECK(hipGetLastError());}});});
        emit_metric(out,"gemm","bf16_"+std::string(name),(2.0*m*n*k*o.inner/s)/1e12,"TFLOPS",o.bf16_peak_tflops,o.warmups,o.repeats,o.inner,arch_name(p),"\"m\":"+std::to_string(m)+",\"n\":"+std::to_string(n)+",\"k\":"+std::to_string(k)+",\"flops_per_iteration\":"+std::to_string(2ull*m*n*k));
    }
}

void run_gemm(const Options& o, const hipDeviceProp_t& p, std::ostream& out) {
    // Small, medium, a square compute-intense case, and a real Qwen3-14B projection.
    for (const auto& s : std::array<std::array<unsigned,3>,4>{{{256,256,256},{1024,1024,1024},{4096,4096,4096},{256,5120,5120}}}) {
        std::string name = (s[0]==256 && s[1]==5120) ? "qwen3_14b_hidden" : std::to_string(s[0])+"x"+std::to_string(s[1])+"x"+std::to_string(s[2]);
        run_gemm_case<false>(o,p,out,name,s[0],s[1],s[2]); run_gemm_case<true>(o,p,out,name,s[0],s[1],s[2]);
    }
}

Options parse(int argc, char** argv) {
    Options o; auto value=[&](int& i){if(++i>=argc)throw std::runtime_error("missing option value");return std::string(argv[i]);};
    for(int i=1;i<argc;++i){std::string a=argv[i]; if(a=="--device")o.device=std::stoi(value(i)); else if(a=="--warmups")o.warmups=std::stoul(value(i)); else if(a=="--repeats")o.repeats=std::stoul(value(i)); else if(a=="--inner")o.inner=std::stoul(value(i)); else if(a=="--workset-mib")o.workset_mib=std::stoull(value(i)); else if(a=="--mode")o.mode=value(i); else if(a=="--output")o.output=value(i); else if(a=="--memory-peak-gbps")o.memory_peak_gbps=std::stod(value(i)); else if(a=="--bf16-peak-tflops")o.bf16_peak_tflops=std::stod(value(i)); else if(a=="--fp8-peak-tflops")o.fp8_peak_tflops=std::stod(value(i)); else if(a=="--help") { std::cout<<"usage: hw-microbench-rdna4-cdna3 [--mode all|validate|bandwidth|gemm] [--output FILE] [--memory-peak-gbps N] [--bf16-peak-tflops N] [--fp8-peak-tflops N]\n"; std::exit(0); } else throw std::runtime_error("unknown option: "+a); }
    if(!o.warmups||!o.repeats||!o.inner)throw std::runtime_error("warmups, repeats and inner must be positive"); return o;
}
}

int main(int argc, char** argv) try {
    Options o=parse(argc,argv); HIP_CHECK(hipSetDevice(o.device)); hipDeviceProp_t p{}; HIP_CHECK(hipGetDeviceProperties(&p,o.device)); require_supported_arch(p);
    std::ofstream file; std::ostream* out=&std::cout; if(!o.output.empty()){file.open(o.output);if(!file)throw std::runtime_error("cannot open output");out=&file;}
    *out << "{\"schema\":\"ullm.hw-microbench.v1\",\"kind\":\"environment\",\"arch\":"; json_string(*out,arch_name(p)); *out << ",\"device\":";json_string(*out,p.name);*out<<",\"wavefront\":"<<p.warpSize<<",\"timing\":\"HIP events around only repeated kernel launches; median\",\"fp8_contract\":\"OCP E4M3FN values; gfx942 retains raw bytes as FNUZ and compensates both operands by x4\"}\n";
    if(o.mode=="all"||o.mode=="validate") validate_gemm(p); if(o.mode=="all"||o.mode=="bandwidth")run_bandwidth(o,p,*out); if(o.mode=="all"||o.mode=="gemm")run_gemm(o,p,*out);
    if(o.mode!="all"&&o.mode!="validate"&&o.mode!="bandwidth"&&o.mode!="gemm")throw std::runtime_error("invalid mode"); return 0;
} catch(const std::exception& e) { std::cerr<<"hw-microbench failed: "<<e.what()<<'\n'; return 1; }
