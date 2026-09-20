// WU2: actual production quantizer and selected hipBLASLt/software control.
#include "../src/matmul_kernel_internal.hpp"
#include "phase87_wu2_fused.hpp"
#include "phase87_wu2_gemv.hpp"
#include "phase87_wu2_lt_control.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

// Third bounded variant: the C2 dot4 GEMV without per-block quantization,
// to separate conversion/accumulation cost from fusion cost.
__global__ void phase87_wu2_dot4_gemv(const uint8_t *a, const float *as,
                                      const uint8_t *w, const float *ws,
                                      uint16_t *out, uint64_t m, uint64_t k,
                                      uint64_t n) {
  unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  uint64_t row = blockIdx.y, col = uint64_t(blockIdx.x) * 8 + wave;
  if (row >= m || col >= n)
    return;
  float sums[4] = {};
  for (uint64_t j = uint64_t(lane) * 16; j < k; j += 512) {
    uint32_t av[4], wv[4];
    const auto *ap = a + row * k + j;
    const auto *wp = w + col * k + j;
    if (j + 16 <= k &&
        ((reinterpret_cast<uintptr_t>(ap) | reinterpret_cast<uintptr_t>(wp)) &
         15) == 0) {
      const uint4 aa = *reinterpret_cast<const uint4 *>(ap);
      using Packed4 = uint32_t __attribute__((ext_vector_type(4)));
      const Packed4 ww =
          __builtin_nontemporal_load(reinterpret_cast<const Packed4 *>(wp));
      av[0] = aa.x;
      av[1] = aa.y;
      av[2] = aa.z;
      av[3] = aa.w;
      wv[0] = ww[0];
      wv[1] = ww[1];
      wv[2] = ww[2];
      wv[3] = ww[3];
    } else {
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) {
        av[i] =
            sllm_phase87_wu2::load_dword_or_bytes(a + row * k, j + i * 4, k);
        wv[i] =
            sllm_phase87_wu2::load_dword_or_bytes(w + col * k, j + i * 4, k);
      }
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
      sums[i] = phase87_wu2::phase87_wu2_dot4<true>(av[i], wv[i], sums[i]);
  }
  float acc = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  for (unsigned off = 16; off; off >>= 1)
    acc += __shfl_down(acc, off, 32);
  if (lane == 0)
    out[row * n + col] =
        sllm_phase87_wu2::f32_to_bf16_rne(acc * as[row] * ws[col]);
}
__global__ void phase87_wu2_decode_check(float *out) {
  unsigned c = threadIdx.x;
  uint32_t packed = 0;
  for (unsigned b = 0; b < 4; ++b)
    packed |= ((c + 64 * b) & 255) << (8 * b);
  float values[4];
  sllm_phase87_wu2::decode_e4m3fnx4(packed, values);
  for (unsigned b = 0; b < 4; ++b)
    out[c * 4 + b] = values[b];
}
namespace {
void need(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
void hipcheck(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
size_t live = 0;
bool free_ok = true;
template <class T> struct GPU {
  T *p = nullptr;
  size_t count;
  explicit GPU(size_t n) : count(n) {
    hipcheck(hipMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)));
    ++live;
  }
  ~GPU() {
    if (p) {
      if (hipFree(p) == hipSuccess)
        --live;
      else
        free_ok = false;
    }
  }
  void put(const std::vector<T> &v) {
    need(v.size() == count, "upload shape");
    hipcheck(hipMemcpy(p, v.data(), count * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> get() {
    std::vector<T> x(count);
    hipcheck(hipMemcpy(x.data(), p, count * sizeof(T), hipMemcpyDeviceToHost));
    return x;
  }
};
uint16_t bf16(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  b += 0x7fff + ((b >> 16) & 1);
  return uint16_t(b >> 16);
}
float f32(uint16_t x) {
  uint32_t b = uint32_t(x) << 16;
  float r;
  std::memcpy(&r, &b, 4);
  return r;
}
float decode(uint8_t x) {
  unsigned e = (x >> 3) & 15, f = x & 7;
  if (e == 15 && f == 7)
    return NAN;
  float v =
      e ? std::ldexp(float(8 + f), int(e) - 10) : std::ldexp(float(f), -9);
  return (x & 128) ? -v : v;
}
uint8_t encode(float x) {
  bool neg = std::signbit(x);
  float a = std::abs(x), best = INFINITY;
  uint8_t q = 0;
  for (unsigned c = 0; c <= 126; ++c) {
    float d = std::abs(a - decode(uint8_t(c)));
    if (d < best || (d == best && !(c & 1))) {
      best = d;
      q = uint8_t(c);
    }
  }
  return uint8_t(q | (neg ? 128 : 0));
}
__host__ __device__ uint8_t weight_code(uint64_t col, uint64_t k, int pattern) {
  if (pattern == 1)
    return uint8_t(0x38 | ((k & 1) ? 0x80 : 0));
  uint32_t z = uint32_t(col) * 1664525U + uint32_t(k) * 1013904223U;
  z ^= z >> 13;
  return uint8_t((0x20 + (z % 32)) | ((z % 13 == 0) ? 0x80 : 0));
}
__global__ void fill_weight(uint8_t *p, uint64_t bytes, uint64_t k, uint64_t n,
                            int pattern) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < bytes) {
    uint64_t local = i % (k * n);
    p[i] = weight_code(local / k, local % k, pattern);
  }
}
struct Buffers {
  uint64_t m, k, n, copies;
  bool r9700;
  GPU<uint16_t> a, out;
  GPU<uint8_t> aq, w, diag;
  GPU<float> as, ws;
  std::unique_ptr<phase87_wu2::LtControl> lt;
  Buffers(uint64_t M, uint64_t K, uint64_t N, uint64_t C, bool r)
      : m(M), k(K), n(N), copies(C), r9700(r), a(M * K), out(M * N + 32),
        aq(M * K), w(C * K * N), diag(M * (K + 4)), as(M), ws(N) {}
  void quant() {
    hipcheck(sllm_matmul_kernel::launch_fp8_quantize(a.p, aq.p, as.p, m, k,
                                                     false, nullptr));
  }
  void control(unsigned copy) {
    if (r9700) {
      need(bool(lt), "control unavailable");
      need(lt->launch(aq.p, w.p + copy * k * n, out.p, nullptr) ==
               HIPBLAS_STATUS_SUCCESS,
           "Lt launch");
      return;
    }
    using namespace sllm_matmul_kernel;
    auto v = select_fp8_outer_variant(m, k, n, "gfx1030");
    hipError_t status;
#define CALL(fn)                                                               \
  status = fn(aq.p, as.p, w.p + copy * k * n, ws.p, out.p, m, k, n, nullptr)
    if (v == KernelVariant::Fp8OuterDecodeGfx1030LdsLutWave4Col32) {
      CALL(launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32);
    } else if (v == KernelVariant::Fp8OuterDecodeGfx1030Dword8Wave4Col32) {
      CALL(launch_fp8_outer_decode_gfx1030_dword8_wave4col32);
    } else if (v == KernelVariant::Fp8OuterDecodeGfx1030FusedM2_4) {
      CALL(launch_fp8_outer_decode_gfx1030_fused_m2_4);
    } else if (v == KernelVariant::Fp8OuterDecodeGfx1030Half2Wave4Col32) {
      CALL(launch_fp8_outer_decode_gfx1030_half2_wave4col32);
    } else if (v == KernelVariant::Fp8OuterPrefillGfx1030Half2_64x64) {
      CALL(launch_fp8_outer_prefill_gfx1030_half2_64x64);
    } else if (v == KernelVariant::Fp8OuterPrefillGfx1030Half2_128x64) {
      CALL(launch_fp8_outer_prefill_gfx1030_half2_128x64);
    } else if (v ==
               KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave4Col32) {
      CALL(launch_fp8_outer_decode_gfx1030_activation_shared_wave4col32);
    } else if (v ==
               KernelVariant::Fp8OuterDecodeGfx1030ActivationSharedWave8Col64) {
      CALL(launch_fp8_outer_decode_gfx1030_activation_shared_wave8col64);
    } else if (v == KernelVariant::Fp8OuterPrefillTiled16) {
      CALL(launch_fp8_outer_prefill_tiled16);
    } else if (v == KernelVariant::Fp8Emulation) {
      CALL(launch_fp8_emulation);
    } else
      throw std::runtime_error("unmapped software control");
#undef CALL
    hipcheck(status);
  }
  void gemv(unsigned copy) {
#define C1(rows)                                                               \
  hipLaunchKernelGGL(                                                          \
      sllm_phase87_wu2::sllm_phase87_wu2_fp8_gemv_c1_m##rows##_v1,             \
      dim3((n + 15) / 16), dim3(128), 0, nullptr, aq.p, as.p,                  \
      w.p + copy * k * n, ws.p, out.p, m, k, n)
    if (m == 1) {
      C1(1);
    } else if (m == 2) {
      C1(2);
    } else {
      C1(3);
    }
#undef C1
    hipcheck(hipGetLastError());
  }
  void dot4(unsigned copy) {
#if defined(SLLM_PHASE87_WU2_PRODUCTION)
    hipcheck(sllm_matmul_kernel::launch_fp8_outer_gfx1201_dot4(
        aq.p, as.p, w.p + copy * k * n, ws.p, out.p, m, k, n, nullptr));
#else
    hipLaunchKernelGGL(phase87_wu2_dot4_gemv, dim3((n + 7) / 8, m), dim3(256),
                       0, nullptr, aq.p, as.p, w.p + copy * k * n, ws.p, out.p,
                       m, k, n);
    hipcheck(hipGetLastError());
#endif
  }
  void fused(unsigned copy, bool diagnostic = false) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(phase87_wu2::phase87_wu2_fused_gemv<true>),
        dim3((n + 7) / 8, m), dim3(256), size_t(k), nullptr, a.p,
        w.p + copy * k * n, ws.p, out.p, m, k, n, diagnostic ? diag.p : nullptr,
        k + 4);
    hipcheck(hipGetLastError());
  }
  void launch(int v, unsigned copy) {
    if (v == 2)
      fused(copy);
    else {
      quant();
      if (v == 0)
        control(copy);
      else if (v == 1)
        gemv(copy);
      else
        dot4(copy);
    }
  }
};
struct Timing {
  float quant, dot, total;
};
Timing timed(Buffers &b, int v, unsigned copy) {
  hipEvent_t a, c, d;
  hipcheck(hipEventCreate(&a));
  hipcheck(hipEventCreate(&c));
  hipcheck(hipEventCreate(&d));
  hipcheck(hipEventRecord(a));
  if (v != 2)
    b.quant();
  hipcheck(hipEventRecord(c));
  if (v == 0)
    b.control(copy);
  else if (v == 1)
    b.gemv(copy);
  else if (v == 2)
    b.fused(copy);
  else
    b.dot4(copy);
  hipcheck(hipEventRecord(d));
  hipcheck(hipEventSynchronize(d));
  Timing t;
  hipcheck(hipEventElapsedTime(&t.quant, a, c));
  hipcheck(hipEventElapsedTime(&t.dot, c, d));
  hipcheck(hipEventElapsedTime(&t.total, a, d));
  hipcheck(hipEventDestroy(a));
  hipcheck(hipEventDestroy(c));
  hipcheck(hipEventDestroy(d));
  if (v == 2)
    t.quant = 0;
  return t;
}
void warm(Buffers &b, int v) {
  auto start = std::chrono::steady_clock::now();
  unsigned copy = 0;
  do {
    for (int i = 0; i < 32; ++i)
      b.launch(v, copy++ % b.copies);
    hipcheck(hipDeviceSynchronize());
  } while (std::chrono::steady_clock::now() - start <
           std::chrono::milliseconds(300));
}
float median(std::vector<float> x) {
  std::sort(x.begin(), x.end());
  return x[x.size() / 2];
}
void arr(const char *key, const std::vector<float> &x) {
  std::cout << ",\"" << key << "\":[";
  for (size_t i = 0; i < x.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << x[i];
  }
  std::cout << ']';
}
} // namespace
int main(int argc, char **argv) {
  try {
    uint64_t m = 1, k = 5120, n = 10240;
    int pattern = 0;
    bool bench = true;
    std::string target = "gfx1201";
    for (int i = 1; i < argc; ++i) {
      need(i + 1 < argc, "argument value");
      std::string key = argv[i], v = argv[++i];
      if (key == "--m")
        m = std::stoull(v);
      else if (key == "--k")
        k = std::stoull(v);
      else if (key == "--n")
        n = std::stoull(v);
      else if (key == "--pattern")
        pattern = std::stoi(v);
      else if (key == "--bench")
        bench = std::stoi(v) != 0;
      else if (key == "--target")
        target = v;
      else
        throw std::runtime_error("unknown argument");
    }
    need(m >= 1 && m <= 3 && k >= 1 && k <= 17408 && n >= 1, "shape");
    hipDeviceProp_t device{};
    hipcheck(hipGetDeviceProperties(&device, 0));
    need(target == device.gcnArchName, "wrong target");
    const uint64_t copies =
        bench ? std::max<uint64_t>(1, ((UINT64_C(512) << 20) + k * n - 1) /
                                          (k * n))
              : 1;
    {
      GPU<float> codec(1024);
      hipLaunchKernelGGL(phase87_wu2_decode_check, dim3(1), dim3(256), 0,
                         nullptr, codec.p);
      hipcheck(hipDeviceSynchronize());
      auto values = codec.get();
      for (unsigned c = 0; c < 256; ++c)
        for (unsigned b = 0; b < 4; ++b) {
          float ref = decode(uint8_t(c + 64 * b));
          float got = values[c * 4 + b];
          need(std::isnan(ref) ? std::isnan(got)
                               : std::memcmp(&ref, &got, 4) == 0,
               "packed native ingress oracle");
        }
    }
    need(pattern >= 0 && pattern <= 2, "pattern");
    bool all_ok = true;
    {
      Buffers b(m, k, n, copies, target == "gfx1201");
      std::vector<uint16_t> a(m * k);
      std::vector<float> scales(n);
      for (uint64_t i = 0; i < m * k; ++i) {
        float v = pattern == 2   ? 0.0F
                  : pattern == 1 ? 1.0F
                                 : float(16 + (i * 17) % 49) / 64.0F;
        if (!pattern && i % 11 == 0)
          v = -v;
        a[i] = bf16(v);
      }
      for (uint64_t c = 0; c < n; ++c)
        scales[c] = pattern == 1 ? 1.0F : float(8 + c % 7) / 256.0F;
      b.a.put(a);
      b.ws.put(scales);
      hipLaunchKernelGGL(fill_weight, dim3((b.w.count + 255) / 256), dim3(256),
                         0, nullptr, b.w.p, b.w.count, k, n, pattern);
      hipcheck(hipGetLastError());
      hipcheck(hipDeviceSynchronize());
      std::vector<uint8_t> expected_a(m * k);
      std::vector<float> expected_scale(m);
      for (uint64_t row = 0; row < m; ++row) {
        float amax = 0;
        for (uint64_t j = 0; j < k; ++j)
          amax = std::max(amax, std::abs(f32(a[row * k + j])));
        float s = amax == 0 ? 1.0F : amax / 448.0F;
        expected_scale[row] = s;
        for (uint64_t j = 0; j < k; ++j)
          expected_a[row * k + j] = encode(f32(a[row * k + j]) / s);
      }
      b.quant();
      hipcheck(hipDeviceSynchronize());
      need(b.aq.get() == expected_a && b.as.get() == expected_scale,
           "production quantizer vs independent oracle");
      bool control_available = true;
      if (b.r9700) {
        try {
          b.lt =
              std::make_unique<phase87_wu2::LtControl>(m, k, n, b.ws.p, b.as.p);
        } catch (const std::exception &e) {
          if (bench)
            throw;
          control_available = false;
          std::cerr << "control unsupported boundary: " << e.what() << '\n';
        }
      }
      std::cout << std::defaultfloat << std::setprecision(9);
      std::cout << "{\"kind\":\"identity\",\"target\":\"" << target
                << "\",\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n
                << ",\"pattern\":" << pattern
                << ",\"packed_decode_codes\":1024,\"weight_pool_bytes\":"
                << b.w.count << ",\"copies\":" << copies
                << ",\"control_available\":"
                << (control_available ? "true" : "false");
#if defined(SLLM_PHASE87_WU2_PRODUCTION)
      std::cout << ",\"production_dot4\":true";
#else
      std::cout << ",\"production_dot4\":false";
#endif
      if (b.lt)
        std::cout << ",\"rank\":" << b.lt->rank
                  << ",\"algorithm_index\":" << b.lt->index;
      std::cout << "}\n";
      std::set<uint64_t> columns;
      if (n <= 257) {
        for (uint64_t c = 0; c < n; ++c)
          columns.insert(c);
      } else {
        for (uint64_t c = 0; c < 64; ++c)
          columns.insert(c * (n - 1) / 63);
        for (uint64_t c : n > 0 ? std::vector<uint64_t>{0, 1, 3, 7, 15, 16, 31,
                                                        32, n - 2, n - 1}
                                : std::vector<uint64_t>{})
          if (c < n)
            columns.insert(c);
      }
      std::vector<float> expected;
      for (uint64_t row = 0; row < m; ++row)
        for (auto c : columns) {
          float sum = 0;
          for (uint64_t j = 0; j < k; ++j)
            sum += decode(expected_a[row * k + j]) *
                   decode(weight_code(c, j, pattern));
          expected.push_back(sum * expected_scale[row] * scales[c]);
        }
      for (int v = control_available ? 0 : 1; v < 4; ++v) {
        hipcheck(hipMemset(b.out.p, 0x5a, b.out.count * 2));
        b.launch(v, 0);
        hipcheck(hipDeviceSynchronize());
        auto first = b.out.get();
        b.launch(v, 0);
        hipcheck(hipDeviceSynchronize());
        auto second = b.out.get();
        bool ok = first == second, finite = true, guard = true;
        for (uint64_t i = 0; i < m * n; ++i)
          finite &= std::isfinite(f32(first[i]));
        for (size_t i = m * n; i < first.size(); ++i)
          guard &= first[i] == 0x5a5a;
        uint32_t ulp = 0;
        float maxabs = 0;
        size_t index = 0;
        for (uint64_t row = 0; row < m; ++row)
          for (auto c : columns) {
            float ref = expected[index++], actual = f32(first[row * n + c]);
            uint16_t ref_b = bf16(ref);
            float abs = std::abs(actual - f32(ref_b));
            maxabs = std::max(maxabs, abs);
            uint32_t u = first[row * n + c] >= ref_b
                             ? first[row * n + c] - ref_b
                             : ref_b - first[row * n + c];
            if (actual == 0 && ref == 0)
              u = 0;
            ulp = std::max(ulp, u);
            ok &= u <= 4;
          }
        bool quant_match = true;
        if (v == 2) {
          hipcheck(hipMemset(b.diag.p, 0x5a, b.diag.count));
          b.fused(0, true);
          hipcheck(hipDeviceSynchronize());
          auto q = b.diag.get();
          for (uint64_t row = 0; row < m; ++row) {
            for (uint64_t j = 0; j < k; ++j)
              quant_match &= q[row * (k + 4) + j] == expected_a[row * k + j];
            float s;
            std::memcpy(&s, q.data() + row * (k + 4) + k, 4);
            quant_match &= std::memcmp(&s, &expected_scale[row], 4) == 0;
          }
        }
        ok &= finite && guard && quant_match;
        all_ok &= ok;
        std::cout << "{\"kind\":\"oracle\",\"variant\":" << v << ",\"state\":\""
                  << (ok ? "PASS" : "FAIL") << "\",\"max_ulp\":" << ulp
                  << ",\"max_abs\":" << maxabs
                  << ",\"oracle_outputs\":" << expected.size()
                  << ",\"finite\":" << (finite ? "true" : "false")
                  << ",\"repeat\":" << (first == second ? "true" : "false")
                  << ",\"guard\":" << (guard ? "true" : "false")
                  << ",\"quantizer_bitwise\":"
                  << (quant_match ? "true" : "false") << "}\n";
      }
      if (all_ok && bench) {
        for (int candidate = 1; candidate <= 3; ++candidate) {
          std::vector<float> bq, bd, bt, cq, cd, ct;
          for (unsigned round = 0; round < 3; ++round)
            for (unsigned position = 0; position < 2; ++position) {
              int v = ((round % 2 == 0) == (position == 0)) ? 0 : candidate;
              warm(b, v);
              for (unsigned sample = 0; sample < 9; ++sample) {
                auto t = timed(
                    b, v,
                    ((round * 9 + sample) * std::max<uint64_t>(1, copies / 9)) %
                        copies);
                (v ? cq : bq).push_back(t.quant);
                (v ? cd : bd).push_back(t.dot);
                (v ? ct : bt).push_back(t.total);
              }
            }
          std::cout << std::defaultfloat << std::setprecision(9);
          std::cout << "{\"kind\":\"performance\",\"variant\":" << candidate
                    << ",\"warmup_ms\":300,\"samples\":27,\"order\":\"AB-BA-"
                       "AB\",\"control_quant_ms\":"
                    << median(bq) << ",\"control_dot_ms\":" << median(bd)
                    << ",\"control_total_ms\":" << median(bt)
                    << ",\"candidate_quant_ms\":" << median(cq)
                    << ",\"candidate_dot_ms\":" << median(cd)
                    << ",\"candidate_total_ms\":" << median(ct);
          arr("control_quant_samples_ms", bq);
          arr("control_dot_samples_ms", bd);
          arr("control_total_samples_ms", bt);
          arr("candidate_quant_samples_ms", cq);
          arr("candidate_dot_samples_ms", cd);
          arr("candidate_total_samples_ms", ct);
          std::cout << "}\n";
        }
      }
    }
    need(live == 0 && free_ok, "cleanup");
    std::cout
        << "{\"kind\":\"cleanup\",\"state\":\"PASS\",\"live_allocations\":0}\n";
    return all_ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
