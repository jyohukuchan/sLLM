// WU1 operator benchmark. Existing project Phase83 fixture/FP32 oracle reused.
#define SLLM_PUBLIC_RUNTIME_HOST_TEST 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "../src/causal_attention_kernel.hip.cpp"
#pragma clang diagnostic pop
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sllm_causal_attention_kernel {
namespace {
#include "phase87_wu1_c1.hpp"
#include "phase87_wu1_c2.hpp"
#include "phase87_wu1_c3.hpp"

// C3 merge reads each partition directly; larger splits do not exceed LDS.
template <unsigned Splits>
__global__ void phase87_wu1_merge(const float *w, uint16_t *out, unsigned M) {
  const unsigned head = blockIdx.x, d = threadIdx.x;
  if (head >= M * 24 || d >= 256)
    return;
  const uint64_t base = uint64_t(head) * Splits * 258;
  float maximum = -INFINITY;
  for (unsigned s = 0; s < Splits; ++s)
    maximum = fmaxf(maximum, w[base + s * 258]);
  float denom = 0, acc = 0;
  for (unsigned s = 0; s < Splits; ++s) {
    float scale = expf(w[base + s * 258] - maximum);
    denom += w[base + s * 258 + 1] * scale;
    acc += w[base + s * 258 + 2 + d] * scale;
  }
  out[uint64_t(head) * 256 + d] = f32_to_bf16_rne(acc / denom);
}
} // namespace
} // namespace sllm_causal_attention_kernel
namespace {
constexpr uint32_t kQueryHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kBlocksPerRow = kHeadDim / 32U;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr uint64_t kQueryValues = static_cast<uint64_t>(kQueryHeads) * kHeadDim;

[[maybe_unused]] bool hip_ok(const hipError_t status,
                             const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << ": " << hipGetErrorString(status) << '\n';
  return false;
}

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  uint32_t rounded = upper;
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++rounded;
  }
  return static_cast<uint16_t>(rounded);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint8_t e4m3fn_encode(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F) {
    return sign;
  }
  if (!std::isfinite(magnitude) || magnitude >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  uint32_t bits = 0U;
  std::memcpy(&bits, &magnitude, sizeof(bits));
  const uint32_t rounded =
      bits + UINT32_C(0x0007ffff) + ((bits >> 20U) & UINT32_C(1));
  const uint32_t exponent = ((rounded >> 23U) & UINT32_C(0xff)) - 120U;
  const uint32_t code = (exponent << 3U) | ((rounded >> 20U) & UINT32_C(7));
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>(std::min(code, UINT32_C(0x7e))));
}

float e4m3fn_decode(const uint8_t value) {
  const float sign = (value & UINT8_C(0x80)) != 0U ? -1.0F : 1.0F;
  const uint32_t magnitude = value & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT32_C(7);
  if (exponent == 0U) {
    return sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  }
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

uint32_t bf16_ulp_distance(const uint16_t lhs, const uint16_t rhs) {
  const auto ordered = [](const uint16_t value) {
    const uint32_t magnitude = value & UINT16_C(0x7fff);
    return (value & UINT16_C(0x8000)) != 0U ? UINT32_C(0x8000) - magnitude
                                            : UINT32_C(0x8000) + value;
  };
  const uint32_t lhs_ordered = ordered(lhs);
  const uint32_t rhs_ordered = ordered(rhs);
  return lhs_ordered >= rhs_ordered ? lhs_ordered - rhs_ordered
                                    : rhs_ordered - lhs_ordered;
}

struct Fixture final {
  uint32_t query_count = 1U;
  std::vector<uint16_t> query;
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> key_scales;
  std::vector<uint8_t> value_scales;
};

Fixture make_fixture(const uint64_t committed_length,
                     const uint32_t query_count) {
  Fixture fixture;
  fixture.query_count = query_count;
  fixture.query.resize(static_cast<std::size_t>(query_count * kQueryValues));
  const uint64_t kv_rows = committed_length * kKvHeads;
  fixture.key.resize(static_cast<std::size_t>(kv_rows * kHeadDim));
  fixture.value.resize(static_cast<std::size_t>(kv_rows * kHeadDim));
  fixture.key_scales.resize(static_cast<std::size_t>(kv_rows * kBlocksPerRow));
  fixture.value_scales.resize(
      static_cast<std::size_t>(kv_rows * kBlocksPerRow));

  for (uint32_t query_index = 0U; query_index != query_count; ++query_index) {
    for (uint32_t head = 0U; head != kQueryHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float source = 0.026F + 0.0021F * static_cast<float>(query_index) +
                       0.0017F * static_cast<float>(head % 9U) +
                       0.00031F * static_cast<float>(dimension % 23U);
        if (((query_index + head * 3U + dimension) % 11U) == 0U) {
          source = -source;
        }
        fixture.query[static_cast<std::size_t>(
            (query_index * kQueryHeads + head) * kHeadDim + dimension)] =
            f32_to_bf16(source);
      }
    }
  }

  for (uint64_t row = 0U; row != kv_rows; ++row) {
    const uint32_t kv_head = static_cast<uint32_t>(row % kKvHeads);
    const uint64_t token = row / kKvHeads;
    for (uint32_t block = 0U; block != kBlocksPerRow; ++block) {
      const uint8_t key_scale =
          static_cast<uint8_t>(125U + ((token + kv_head + block) & 3U));
      const uint8_t value_scale =
          static_cast<uint8_t>(126U + ((2U * token + kv_head + block) & 3U));
      fixture
          .key_scales[static_cast<std::size_t>(row * kBlocksPerRow + block)] =
          key_scale;
      fixture
          .value_scales[static_cast<std::size_t>(row * kBlocksPerRow + block)] =
          value_scale;
      const float key_scale_value =
          std::ldexp(1.0F, static_cast<int>(key_scale) - 127);
      const float value_scale_value =
          std::ldexp(1.0F, static_cast<int>(value_scale) - 127);
      for (uint32_t lane = 0U; lane != 32U; ++lane) {
        const uint32_t dimension = block * 32U + lane;
        float key_source = 0.19F + 0.013F * static_cast<float>(block) +
                           0.007F * static_cast<float>(kv_head) +
                           0.0007F * static_cast<float>(token % 17U) +
                           0.0011F * static_cast<float>(lane % 19U);
        float value_source = 0.31F + 0.021F * static_cast<float>(block) +
                             0.009F * static_cast<float>(kv_head) +
                             0.0009F * static_cast<float>(token % 13U) +
                             0.0013F * static_cast<float>(lane % 17U);
        if (((token + kv_head + dimension) % 13U) == 0U) {
          key_source = -key_source;
        }
        if (((2U * token + kv_head + dimension) % 17U) == 0U) {
          value_source = -value_source;
        }
        const std::size_t index =
            static_cast<std::size_t>(row * kHeadDim + dimension);
        fixture.key[index] = e4m3fn_encode(key_source / key_scale_value);
        fixture.value[index] = e4m3fn_encode(value_source / value_scale_value);
      }
    }
  }
  return fixture;
}

float decoded(const std::vector<uint8_t> &values,
              const std::vector<uint8_t> &scales, const uint64_t row,
              const uint32_t dimension) {
  const uint8_t scale =
      scales[static_cast<std::size_t>(row * kBlocksPerRow + dimension / 32U)];
  return e4m3fn_decode(
             values[static_cast<std::size_t>(row * kHeadDim + dimension)]) *
         std::ldexp(1.0F, static_cast<int>(scale) - 127);
}

std::vector<uint16_t> oracle(const Fixture &fixture,
                             const uint64_t committed_length,
                             const uint32_t query_count) {
  std::vector<uint16_t> expected(
      static_cast<std::size_t>(query_count * kQueryValues));
  const uint64_t start_position = committed_length - query_count;
  for (uint32_t query_index = 0U; query_index != query_count; ++query_index) {
    const uint64_t row_length = start_position + query_index + 1U;
    std::vector<float> scores(static_cast<std::size_t>(row_length));
    for (uint32_t query_head = 0U; query_head != kQueryHeads; ++query_head) {
      const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
      float maximum = -std::numeric_limits<float>::infinity();
      for (uint64_t token = 0U; token != row_length; ++token) {
        const uint64_t row = token * kKvHeads + kv_head;
        float score = 0.0F;
        for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
          score += bf16_to_f32(fixture.query[static_cast<std::size_t>(
                       (query_index * kQueryHeads + query_head) * kHeadDim +
                       dimension)]) *
                   decoded(fixture.key, fixture.key_scales, row, dimension);
        }
        score *= kAttentionScale;
        scores[static_cast<std::size_t>(token)] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float accumulated = 0.0F;
        for (uint64_t token = 0U; token != row_length; ++token) {
          const uint64_t row = token * kKvHeads + kv_head;
          accumulated +=
              (scores[static_cast<std::size_t>(token)] / denominator) *
              decoded(fixture.value, fixture.value_scales, row, dimension);
        }
        expected[static_cast<std::size_t>(
            (query_index * kQueryHeads + query_head) * kHeadDim + dimension)] =
            f32_to_bf16(accumulated);
      }
    }
  }
  return expected;
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
bool cleanup_ok = true;
size_t live_allocations = 0;
template <class T> struct Gpu {
  T *ptr = nullptr;
  size_t count;
  explicit Gpu(size_t n) : count(n) {
    check(hipMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
    ++live_allocations;
  }
  ~Gpu() {
    if (ptr) {
      const auto status = hipFree(ptr);
      if (status == hipSuccess)
        --live_allocations;
      else
        cleanup_ok = false;
    }
  }
  void put(const std::vector<T> &v) {
    require(v.size() == count, "upload shape");
    check(hipMemcpy(ptr, v.data(), count * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> get() {
    std::vector<T> v(count);
    check(hipMemcpy(v.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};
struct WU1Buffers {
  Gpu<uint16_t> q, out;
  Gpu<uint8_t> k, v, ks, vs;
  Gpu<float> work;
  uint64_t length;
  unsigned m;
  WU1Buffers(const Fixture &f, uint64_t l, unsigned count)
      : q(f.query.size()), out(f.query.size() + 32), k(f.key.size()),
        v(f.value.size()), ks(f.key_scales.size()), vs(f.value_scales.size()),
        work(size_t(count) * 24 * 128 * 258 + 64), length(l), m(count) {
    q.put(f.query);
    k.put(f.key);
    v.put(f.value);
    ks.put(f.key_scales);
    vs.put(f.value_scales);
    check(hipMemset(out.ptr, 0x5a, out.count * sizeof(uint16_t)));
    check(hipMemset(work.ptr, 0x5a, work.count * sizeof(float)));
  }
};
template <unsigned Splits> void baseline_stage1(WU1Buffers &b) {
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          sllm_causal_attention_kernel::
              causal_attention_decode_wave_split_staged_stage1_kernel<true, 6,
                                                                      Splits>),
      dim3(b.m * 24 * Splits), dim3(32), 0, nullptr, b.q.ptr, b.k.ptr, b.v.ptr,
      b.ks.ptr, b.vs.ptr, nullptr, nullptr, b.work.ptr, b.m, b.length - b.m,
      24U, 4U, 256U, 1.0F, 1.0F);
}
void stage1(WU1Buffers &b, int variant) {
  switch (variant) {
  case 0:
    baseline_stage1<32>(b);
    break;
  case 1:
    hipLaunchKernelGGL(sllm_causal_attention_kernel::phase87_wu1_c1_stage1,
                       dim3(b.m * 4 * 32), dim3(192), 0, nullptr, b.q.ptr,
                       b.k.ptr, b.v.ptr, b.ks.ptr, b.vs.ptr, b.work.ptr, b.m,
                       b.length - b.m);
    break;
  case 2:
    hipLaunchKernelGGL(sllm_causal_attention_kernel::phase87_wu1_c2_stage1,
                       dim3(b.m * 4 * 32), dim3(192), 0, nullptr, b.q.ptr,
                       b.k.ptr, b.v.ptr, b.ks.ptr, b.vs.ptr, b.work.ptr, b.m,
                       b.length - b.m);
    break;
  case 3:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::phase87_wu1_c3_stage1<true, 6, 64>),
        dim3(b.m * 24 * 64), dim3(32), 0, nullptr, b.q.ptr, b.k.ptr, b.v.ptr,
        b.ks.ptr, b.vs.ptr, nullptr, nullptr, b.work.ptr, b.m, b.length - b.m,
        24U, 4U, 256U, 1.0F, 1.0F);
    break;
  case 4:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::phase87_wu1_c3_stage1<true, 6, 128>),
        dim3(b.m * 24 * 128), dim3(32), 0, nullptr, b.q.ptr, b.k.ptr, b.v.ptr,
        b.ks.ptr, b.vs.ptr, nullptr, nullptr, b.work.ptr, b.m, b.length - b.m,
        24U, 4U, 256U, 1.0F, 1.0F);
    break;
  default:
    throw std::runtime_error("variant");
  }
  check(hipGetLastError());
}
void stage2(WU1Buffers &b, int variant) {
  if (variant <= 2) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_wave_split_staged_stage2_kernel<32>),
        dim3(b.m * 24), dim3(256), 0, nullptr, b.work.ptr, b.out.ptr, b.m, 24U,
        4U, 256U);
  } else if (variant == 3) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sllm_causal_attention_kernel::phase87_wu1_merge<64>),
        dim3(b.m * 24), dim3(256), 0, nullptr, b.work.ptr, b.out.ptr, b.m);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sllm_causal_attention_kernel::phase87_wu1_merge<128>),
        dim3(b.m * 24), dim3(256), 0, nullptr, b.work.ptr, b.out.ptr, b.m);
  }
  check(hipGetLastError());
}
void launch(WU1Buffers &b, int variant) {
  stage1(b, variant);
  stage2(b, variant);
}
void warm(WU1Buffers &b, int variant) {
  const auto begin = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 32; ++i)
      launch(b, variant);
    check(hipDeviceSynchronize());
  } while (std::chrono::steady_clock::now() - begin <
           std::chrono::milliseconds(300));
}
struct Timed {
  float stage1, stage2, total;
};
Timed measure(WU1Buffers &b, int variant) {
  hipEvent_t a, c, d;
  check(hipEventCreate(&a));
  check(hipEventCreate(&c));
  check(hipEventCreate(&d));
  check(hipEventRecord(a));
  stage1(b, variant);
  check(hipEventRecord(c));
  stage2(b, variant);
  check(hipEventRecord(d));
  check(hipEventSynchronize(d));
  Timed t{};
  check(hipEventElapsedTime(&t.stage1, a, c));
  check(hipEventElapsedTime(&t.stage2, c, d));
  check(hipEventElapsedTime(&t.total, a, d));
  check(hipEventDestroy(a));
  check(hipEventDestroy(c));
  check(hipEventDestroy(d));
  return t;
}
struct Errors {
  uint32_t ulp = 0;
  float abs = 0, relative = 0;
  bool repeat = true, guard = true, finite = true;
};
Errors validate(WU1Buffers &b, int variant,
                const std::vector<uint16_t> &expected) {
  check(hipMemset(b.out.ptr, 0x5a, b.out.count * sizeof(uint16_t)));
  check(hipMemset(b.work.ptr, 0x5a, b.work.count * sizeof(float)));
  launch(b, variant);
  check(hipDeviceSynchronize());
  auto first = b.out.get();
  launch(b, variant);
  check(hipDeviceSynchronize());
  auto second = b.out.get();
  Errors e;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float actual = bf16_to_f32(first[i]), ref = bf16_to_f32(expected[i]);
    e.finite &= std::isfinite(actual);
    e.ulp = std::max(e.ulp, bf16_ulp_distance(first[i], expected[i]));
    e.abs = std::max(e.abs, std::abs(actual - ref));
    e.relative = std::max(e.relative, std::abs(actual - ref) /
                                          std::max(std::abs(ref), 1.0e-6F));
  }
  e.repeat = first == second;
  for (size_t i = expected.size(); i < first.size(); ++i)
    e.guard &= first[i] == 0x5a5a;
  auto workspace = b.work.get();
  for (size_t i = size_t(b.m) * 24 *
                  (variant == 3   ? 64
                   : variant == 4 ? 128
                                  : 32) *
                  258;
       i < workspace.size(); ++i) {
    uint32_t raw;
    std::memcpy(&raw, &workspace[i], 4);
    e.guard &= raw == 0x5a5a5a5a;
  }
  return e;
}
double med(std::vector<float> a) {
  std::sort(a.begin(), a.end());
  return a[a.size() / 2];
}
void causal_fixture(Fixture &f, uint64_t length, unsigned m) {
  std::fill(f.query.begin(), f.query.end(), f32_to_bf16(0.25F));
  std::fill(f.key.begin(), f.key.end(), 0);
  std::fill(f.value.begin(), f.value.end(), 0);
  std::fill(f.key_scales.begin(), f.key_scales.end(), 127);
  std::fill(f.value_scales.begin(), f.value_scales.end(), 127);
  for (uint64_t p = length - m; p < length; ++p)
    for (unsigned h = 0; h < 4; ++h)
      for (unsigned d = 0; d < 256; ++d) {
        auto index = (p * 4 + h) * 256 + d;
        f.key[index] = e4m3fn_encode(8.0F);
        f.value[index] = e4m3fn_encode(0.5F + float(p - (length - m)) * 0.5F +
                                       float(h) * 0.125F);
      }
}
} // namespace

int main(int argc, char **argv) {
  try {
    uint64_t length = 8193;
    unsigned m = 1;
    int pattern = 0;
    bool bench = true;
    std::string expected_target;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      require(i + 1 < argc, "argument value");
      std::string val = argv[++i];
      if (arg == "--length")
        length = std::stoull(val);
      else if (arg == "--m")
        m = std::stoul(val);
      else if (arg == "--pattern")
        pattern = std::stoi(val);
      else if (arg == "--bench")
        bench = std::stoi(val) != 0;
      else if (arg == "--target")
        expected_target = val;
      else
        throw std::runtime_error("unknown argument");
    }
    require(length >= m && m >= 1 && m <= 3, "shape");
    hipDeviceProp_t device{};
    check(hipGetDeviceProperties(&device, 0));
    require(expected_target == device.gcnArchName, "wrong GPU target");
    require(pattern >= 0 && pattern <= 2, "pattern");
    Fixture fixture = make_fixture(length, m);
    if (pattern == 1)
      causal_fixture(fixture, length, m);
    if (pattern == 2) {
      for (auto &q : fixture.query)
        q = f32_to_bf16(bf16_to_f32(q) * 8.0F);
      for (auto &scale : fixture.key_scales)
        scale += 4;
    }
    const auto expected = oracle(fixture, length, m);
    bool pass = true;
    {
      WU1Buffers b(fixture, length, m);
      std::vector<uint16_t> control_output;
      Errors errors[5];
      for (int v = 0; v < 5; ++v) {
        errors[v] = validate(b, v, expected);
        auto e = errors[v];
        bool ok = e.finite && e.repeat && e.guard && e.ulp <= 4 &&
                  e.abs <= 0.03125F && e.relative <= 0.04F;
        pass &= ok;
        auto observed = b.out.get();
        if (v == 0)
          control_output = observed;
        const bool control_bitwise = observed == control_output;
        std::cout << "{\"kind\":\"oracle\",\"target\":\"" << expected_target
                  << "\",\"length\":" << length << ",\"m\":" << m
                  << ",\"pattern\":" << pattern << ",\"variant\":" << v
                  << ",\"max_ulp\":" << e.ulp << ",\"max_abs\":" << e.abs
                  << ",\"max_relative\":" << e.relative
                  << ",\"repeat\":" << (e.repeat ? "true" : "false")
                  << ",\"control_bitwise\":"
                  << (control_bitwise ? "true" : "false")
                  << ",\"guard\":" << (e.guard ? "true" : "false")
                  << ",\"finite\":" << (e.finite ? "true" : "false")
                  << ",\"state\":\"" << (ok ? "PASS" : "FAIL") << "\"}\n";
      }
      if (pass && bench)
        for (int candidate = 1; candidate < 5; ++candidate) {
          std::vector<float> base1, cand1, base_total, cand_total;
          for (int round = 0; round < 3; ++round) {
            for (int pos = 0; pos < 2; ++pos) {
              const int v = ((round % 2 == 0) == (pos == 0)) ? 0 : candidate;
              warm(b, v);
              for (int sample = 0; sample < 9; ++sample) {
                auto t = measure(b, v);
                (v == 0 ? base1 : cand1).push_back(t.stage1);
                (v == 0 ? base_total : cand_total).push_back(t.total);
              }
            }
          }
          std::cout << "{\"kind\":\"performance\",\"target\":\""
                    << expected_target << "\",\"length\":" << length
                    << ",\"m\":" << m << ",\"pattern\":" << pattern
                    << ",\"variant\":" << candidate
                    << ",\"control_stage1_ms\":" << med(base1)
                    << ",\"candidate_stage1_ms\":" << med(cand1)
                    << ",\"control_total_ms\":" << med(base_total)
                    << ",\"candidate_total_ms\":" << med(cand_total)
                    << ",\"samples_per_variant\":27,\"warmup_ms\":300,"
                       "\"order\":\"AB-BA-AB\"";
          auto samples = [](const char *name,
                            const std::vector<float> &values) {
            std::cout << ",\"" << name << "\":[";
            for (size_t i = 0; i < values.size(); ++i) {
              if (i)
                std::cout << ',';
              std::cout << values[i];
            }
            std::cout << ']';
          };
          samples("control_stage1_samples_ms", base1);
          samples("candidate_stage1_samples_ms", cand1);
          samples("control_total_samples_ms", base_total);
          samples("candidate_total_samples_ms", cand_total);
          std::cout << "}\n";
        }
    }
    check(hipDeviceSynchronize());
    require(cleanup_ok && live_allocations == 0, "cleanup failed");
    std::cout
        << "{\"kind\":\"cleanup\",\"state\":\"PASS\",\"live_allocations\":0}\n";
    return pass ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
