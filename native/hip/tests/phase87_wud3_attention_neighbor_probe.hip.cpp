// Phase 87 WU-D3: measure the real staged attention -> production NVFP4
// sequence while keeping candidate bodies behind an explicit hook.
//
// This probe owns a small BF16/MXFP8 attention fixture and an independent
// FP32 oracle.  It calls the current production stage-1/stage-2 kernels
// directly so that HIP events can separate attention from the following
// production NVFP4 calls.  Candidate bodies are intentionally not enabled
// until WU-D2 selects the relevant persistence window.
//
// Direct build (one exact target per binary):
//   amdclang++ -O3 -ffp-contract=off -std=c++17 -Wall -Wextra -Werror \
//     -x hip --offload-arch=<target> -I include -I native/hip/src \
//     -I native/lowp/include
//     native/hip/tests/phase87_wud3_attention_neighbor_probe.hip.cpp \
//     -L<lowp-build-dir> -lsllm_lowp -L/opt/rocm/lib -lamdhip64 \
//     -o phase87-wud3-<target>
//
// The first line printed by the executable identifies the requested NVFP4
// shape, attention context/M, candidate and repeat count before any GPU
// allocation.  Default execution is control-only and uses both NVFP4 shapes.

#define SLLM_PUBLIC_RUNTIME_HOST_TEST 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "../src/causal_attention_kernel.hip.cpp"
#pragma clang diagnostic pop
#include "../../lowp/include/lowp/detail/lowp_kernel_internal.hpp"
#include "phase87_wud3_candidates.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kScaleBlock = 32U;
constexpr uint32_t kScaleBytesPerRow = kHeadDim / kScaleBlock;
constexpr uint32_t kGqaRatio = kQHeads / kKvHeads;
constexpr uint32_t kMaxM = 3U;
constexpr uint32_t kLongSplits = 128U;
constexpr uint32_t kShortSplits = 32U;
constexpr uint32_t kAttentionGuard = 32U;
constexpr uint32_t kWorkspaceGuard = 64U;
constexpr uint32_t kNvOutputGuard = 32U;
constexpr uint32_t kNvWeightPool = 4U;
constexpr uint32_t kNvOracleColumns = 37U;
constexpr uint32_t kDefaultWarmupMs = 300U;
constexpr uint32_t kDefaultRounds = 3U;
constexpr uint32_t kDefaultSamples = 9U;
constexpr uint32_t kDefaultNvRepeats = 1U;
constexpr float kAttentionScale = 1.0F / 16.0F;

struct NvShape final {
  uint64_t k;
  uint64_t n;
  const char *name;
};

constexpr std::array<NvShape, 2> kNvShapes = {
    NvShape{5120U, 17408U, "wide-k5120-n17408"},
    NvShape{17408U, 5120U, "down-k17408-n5120"},
};

struct Options final {
  std::string target;
  std::string shape = "both";
  std::string candidate = "control";
  uint64_t length = 8256U;
  uint32_t m = 1U;
  uint32_t pattern = 0U;
  uint32_t nv_repeats = kDefaultNvRepeats;
  uint32_t warmup_ms = kDefaultWarmupMs;
  uint32_t rounds = kDefaultRounds;
  uint32_t samples = kDefaultSamples;
};

struct AttentionFixture final {
  uint32_t m = 1U;
  std::vector<uint16_t> query;
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> key_scales;
  std::vector<uint8_t> value_scales;
};

struct NvHostFixture final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weights;
  std::vector<uint8_t> weight_scales;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess)
    fail(std::string(operation) + ": " + hipGetErrorString(status));
}

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint32_t bf16_ulp_distance(const uint16_t lhs, const uint16_t rhs) {
  const auto ordered = [](const uint16_t value) {
    const uint32_t magnitude = value & UINT16_C(0x7fff);
    return (value & UINT16_C(0x8000)) != 0U ? UINT32_C(0x8000) - magnitude
                                            : UINT32_C(0x8000) + value;
  };
  const uint32_t a = ordered(lhs);
  const uint32_t b = ordered(rhs);
  return a >= b ? a - b : b - a;
}

uint8_t e4m3fn_encode(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F)
    return sign;
  if (!std::isfinite(magnitude) || magnitude >= 448.0F)
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
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
  if (exponent == 0U)
    return sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

AttentionFixture make_attention_fixture(const uint64_t length,
                                        const uint32_t m) {
  AttentionFixture fixture;
  fixture.m = m;
  fixture.query.resize(static_cast<size_t>(m) * kQHeads * kHeadDim);
  const uint64_t rows = length * kKvHeads;
  fixture.key.resize(static_cast<size_t>(rows) * kHeadDim);
  fixture.value.resize(static_cast<size_t>(rows) * kHeadDim);
  fixture.key_scales.resize(static_cast<size_t>(rows) * kScaleBytesPerRow);
  fixture.value_scales.resize(static_cast<size_t>(rows) * kScaleBytesPerRow);

  for (uint32_t query = 0U; query < m; ++query) {
    for (uint32_t head = 0U; head < kQHeads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        float source = 0.026F + 0.0021F * static_cast<float>(query) +
                       0.0017F * static_cast<float>(head % 9U) +
                       0.00031F * static_cast<float>(dimension % 23U);
        if (((query + head * 3U + dimension) % 11U) == 0U)
          source = -source;
        fixture.query[(static_cast<size_t>(query) * kQHeads + head) * kHeadDim +
                      dimension] = f32_to_bf16(source);
      }
    }
  }
  for (uint64_t row = 0U; row < rows; ++row) {
    const uint32_t kv_head = static_cast<uint32_t>(row % kKvHeads);
    const uint64_t token = row / kKvHeads;
    for (uint32_t block = 0U; block < kScaleBytesPerRow; ++block) {
      const uint8_t key_scale =
          static_cast<uint8_t>(125U + ((token + kv_head + block) & 3U));
      const uint8_t value_scale =
          static_cast<uint8_t>(126U + ((2U * token + kv_head + block) & 3U));
      fixture.key_scales[static_cast<size_t>(row * kScaleBytesPerRow + block)] =
          key_scale;
      fixture
          .value_scales[static_cast<size_t>(row * kScaleBytesPerRow + block)] =
          value_scale;
      const float key_scale_value =
          std::ldexp(1.0F, static_cast<int>(key_scale) - 127);
      const float value_scale_value =
          std::ldexp(1.0F, static_cast<int>(value_scale) - 127);
      for (uint32_t lane = 0U; lane < kScaleBlock; ++lane) {
        const uint32_t dimension = block * kScaleBlock + lane;
        float key_source = 0.19F + 0.013F * static_cast<float>(block) +
                           0.007F * static_cast<float>(kv_head) +
                           0.0007F * static_cast<float>(token % 17U) +
                           0.0011F * static_cast<float>(lane % 19U);
        float value_source = 0.31F + 0.021F * static_cast<float>(block) +
                             0.009F * static_cast<float>(kv_head) +
                             0.0009F * static_cast<float>(token % 13U) +
                             0.0013F * static_cast<float>(lane % 17U);
        if (((token + kv_head + dimension) % 13U) == 0U)
          key_source = -key_source;
        if (((2U * token + kv_head + dimension) % 17U) == 0U)
          value_source = -value_source;
        const size_t index = static_cast<size_t>(row * kHeadDim + dimension);
        fixture.key[index] = e4m3fn_encode(key_source / key_scale_value);
        fixture.value[index] = e4m3fn_encode(value_source / value_scale_value);
      }
    }
  }
  return fixture;
}

void apply_attention_pattern(AttentionFixture *const fixture,
                             const uint32_t pattern) {
  if (pattern == 0U)
    return;
  if (pattern == 1U) {
    std::fill(fixture->query.begin(), fixture->query.end(), f32_to_bf16(0.25F));
    std::fill(fixture->key.begin(), fixture->key.end(), 0U);
    std::fill(fixture->value.begin(), fixture->value.end(), 0U);
    std::fill(fixture->key_scales.begin(), fixture->key_scales.end(), 127U);
    std::fill(fixture->value_scales.begin(), fixture->value_scales.end(), 127U);
    const uint64_t start =
        static_cast<uint64_t>(fixture->key.size() / (kKvHeads * kHeadDim)) -
        fixture->m;
    const uint64_t length =
        static_cast<uint64_t>(fixture->key.size() / (kKvHeads * kHeadDim));
    for (uint64_t token = start; token < length; ++token) {
      for (uint32_t head = 0U; head < kKvHeads; ++head) {
        for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
          const size_t index = static_cast<size_t>(
              (token * kKvHeads + head) * kHeadDim + dimension);
          fixture->key[index] = e4m3fn_encode(8.0F);
          fixture->value[index] =
              e4m3fn_encode(0.5F + static_cast<float>(token - start) * 0.5F +
                            static_cast<float>(head) * 0.125F);
        }
      }
    }
    return;
  }
  if (pattern == 2U) {
    for (uint16_t &query : fixture->query)
      query = f32_to_bf16(bf16_to_f32(query) * 8.0F);
    for (uint8_t &scale : fixture->key_scales)
      scale = static_cast<uint8_t>(scale + 4U);
    return;
  }
  fail("pattern must be 0, 1, or 2");
}

float decoded_attention(const std::vector<uint8_t> &values,
                        const std::vector<uint8_t> &scales, const uint64_t row,
                        const uint32_t dimension) {
  const uint8_t scale = scales[static_cast<size_t>(row * kScaleBytesPerRow +
                                                   dimension / kScaleBlock)];
  return e4m3fn_decode(
             values[static_cast<size_t>(row * kHeadDim + dimension)]) *
         std::ldexp(1.0F, static_cast<int>(scale) - 127);
}

std::vector<uint16_t> attention_oracle(const AttentionFixture &fixture,
                                       const uint64_t length) {
  const uint32_t m = fixture.m;
  std::vector<uint16_t> expected(static_cast<size_t>(m) * kQHeads * kHeadDim);
  const uint64_t start_position = length - m;
  for (uint32_t query = 0U; query < m; ++query) {
    const uint64_t row_length = start_position + query + 1U;
    std::vector<float> scores(static_cast<size_t>(row_length));
    for (uint32_t query_head = 0U; query_head < kQHeads; ++query_head) {
      const uint32_t kv_head = query_head / kGqaRatio;
      float maximum = -std::numeric_limits<float>::infinity();
      for (uint64_t token = 0U; token < row_length; ++token) {
        const uint64_t row = token * kKvHeads + kv_head;
        float score = 0.0F;
        for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension)
          score +=
              bf16_to_f32(fixture.query[(static_cast<size_t>(query) * kQHeads +
                                         query_head) *
                                            kHeadDim +
                                        dimension]) *
              decoded_attention(fixture.key, fixture.key_scales, row,
                                dimension);
        score *= kAttentionScale;
        scores[static_cast<size_t>(token)] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        float accumulated = 0.0F;
        for (uint64_t token = 0U; token < row_length; ++token) {
          const uint64_t row = token * kKvHeads + kv_head;
          accumulated += scores[static_cast<size_t>(token)] / denominator *
                         decoded_attention(fixture.value, fixture.value_scales,
                                           row, dimension);
        }
        expected[(static_cast<size_t>(query) * kQHeads + query_head) *
                     kHeadDim +
                 dimension] = f32_to_bf16(accumulated);
      }
    }
  }
  return expected;
}

uint8_t nvfp4_code(const uint64_t index, const uint64_t salt) {
  return static_cast<uint8_t>((index * 13U + salt * 7U + 3U) & 0x0fU);
}

float e2m1_decode(const uint8_t code) {
  constexpr std::array<float, 8> values = {0.0F, 0.5F, 1.0F, 1.5F,
                                           2.0F, 3.0F, 4.0F, 6.0F};
  const float value = values[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

NvHostFixture make_nv_fixture(const NvShape shape) {
  NvHostFixture fixture;
  const uint64_t blocks = shape.k / 16U;
  fixture.activation.resize(static_cast<size_t>(shape.k / 2U));
  fixture.activation_scales.resize(static_cast<size_t>(blocks));
  fixture.weights.resize(static_cast<size_t>(shape.n * shape.k / 2U));
  fixture.weight_scales.resize(static_cast<size_t>(shape.n * blocks));
  for (uint64_t inner = 0U; inner < shape.k; ++inner) {
    const uint8_t code = nvfp4_code(inner, 1U);
    uint8_t &packed = fixture.activation[static_cast<size_t>(inner / 2U)];
    if ((inner & 1U) == 0U)
      packed = code;
    else
      packed |= static_cast<uint8_t>(code << 4U);
  }
  for (uint64_t block = 0U; block < blocks; ++block)
    fixture.activation_scales[static_cast<size_t>(block)] =
        static_cast<uint8_t>((block & 1U) == 0U ? 0x38U : 0x40U);
  const uint64_t row_bytes = shape.k / 2U;
  for (uint64_t column = 0U; column < shape.n; ++column) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t code = nvfp4_code(inner, column + 11U);
      uint8_t &packed =
          fixture.weights[static_cast<size_t>(column * row_bytes + inner / 2U)];
      if ((inner & 1U) == 0U)
        packed = code;
      else
        packed |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      fixture.weight_scales[static_cast<size_t>(column * blocks + block)] =
          static_cast<uint8_t>(((block + column) & 1U) != 0U ? 0x40U : 0x38U);
  }
  return fixture;
}

std::vector<uint16_t> nv_oracle_prefix(const NvShape shape,
                                       const NvHostFixture &fixture) {
  const uint64_t blocks = shape.k / 16U;
  const uint64_t row_bytes = shape.k / 2U;
  const uint32_t columns =
      static_cast<uint32_t>(std::min<uint64_t>(shape.n, kNvOracleColumns));
  std::vector<uint16_t> expected(columns);
  for (uint32_t column = 0U; column < columns; ++column) {
    float accumulator = 0.0F;
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t a_pair =
          fixture.activation[static_cast<size_t>(inner / 2U)];
      const uint8_t w_pair = fixture.weights[static_cast<size_t>(
          static_cast<uint64_t>(column) * row_bytes + inner / 2U)];
      const uint8_t a_code = (inner & 1U) == 0U ? a_pair & 0x0fU : a_pair >> 4U;
      const uint8_t w_code = (inner & 1U) == 0U ? w_pair & 0x0fU : w_pair >> 4U;
      accumulator +=
          e2m1_decode(a_code) *
          e4m3fn_decode(fixture.activation_scales[inner / 16U]) *
          e2m1_decode(w_code) *
          e4m3fn_decode(
              fixture.weight_scales[static_cast<uint64_t>(column) * blocks +
                                    inner / 16U]);
    }
    expected[column] = f32_to_bf16(accumulator * 0.75F * 1.125F);
  }
  return expected;
}

template <typename T> struct DeviceBuffer final {
  T *ptr = nullptr;
  size_t count = 0U;

  explicit DeviceBuffer(const size_t elements) : count(elements) {
    check(hipMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)),
          "hipMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : ptr(std::exchange(other.ptr, nullptr)),
        count(std::exchange(other.count, 0U)) {}
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (ptr != nullptr)
        (void)hipFree(ptr);
      ptr = std::exchange(other.ptr, nullptr);
      count = std::exchange(other.count, 0U);
    }
    return *this;
  }
  ~DeviceBuffer() {
    if (ptr != nullptr)
      (void)hipFree(ptr);
  }
  void upload(const std::vector<T> &host) {
    if (host.size() != count)
      fail("device upload shape mismatch");
    check(hipMemcpy(ptr, host.data(), count * sizeof(T), hipMemcpyHostToDevice),
          "hipMemcpy H2D");
  }
  std::vector<T> download() const {
    std::vector<T> host(count);
    check(hipMemcpy(host.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost),
          "hipMemcpy D2H");
    return host;
  }
};

struct AttentionBuffers final {
  DeviceBuffer<uint16_t> query;
  DeviceBuffer<uint16_t> output;
  DeviceBuffer<uint8_t> key;
  DeviceBuffer<uint8_t> value;
  DeviceBuffer<uint8_t> key_scales;
  DeviceBuffer<uint8_t> value_scales;
  DeviceBuffer<float> workspace;
  uint64_t length;
  uint32_t m;
  uint32_t splits;
  bool gqa;
  hipStream_t stream = nullptr;

  AttentionBuffers(const AttentionFixture &fixture, const uint64_t length_in,
                   const std::string_view target)
      : query(fixture.query.size()),
        output(fixture.query.size() + kAttentionGuard), key(fixture.key.size()),
        value(fixture.value.size()), key_scales(fixture.key_scales.size()),
        value_scales(fixture.value_scales.size()),
        workspace(static_cast<size_t>(fixture.m) * kQHeads * kLongSplits *
                      (kHeadDim + 2U) +
                  kWorkspaceGuard),
        length(length_in), m(fixture.m),
        splits(length_in >= 8192U && fixture.m <= kMaxM ? kLongSplits
                                                        : kShortSplits),
        gqa(target == "gfx1030") {
    query.upload(fixture.query);
    key.upload(fixture.key);
    value.upload(fixture.value);
    key_scales.upload(fixture.key_scales);
    value_scales.upload(fixture.value_scales);
    check(hipStreamCreate(&stream), "hipStreamCreate attention");
    clear_guards();
  }
  AttentionBuffers(const AttentionBuffers &) = delete;
  AttentionBuffers &operator=(const AttentionBuffers &) = delete;
  ~AttentionBuffers() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
  void clear_guards() {
    check(hipMemsetAsync(output.ptr, 0x5a, output.count * sizeof(uint16_t),
                         stream),
          "clear attention output");
    check(hipMemsetAsync(workspace.ptr, 0x5a, workspace.count * sizeof(float),
                         stream),
          "clear attention workspace");
    check(hipStreamSynchronize(stream), "clear attention synchronize");
  }
  size_t used_workspace() const {
    return static_cast<size_t>(m) * kQHeads * splits * (kHeadDim + 2U);
  }
};

struct NvBuffers final {
  DeviceBuffer<uint8_t> activation;
  DeviceBuffer<uint8_t> activation_scales;
  std::vector<DeviceBuffer<uint8_t>> weights;
  std::vector<DeviceBuffer<uint8_t>> weight_scales;
  DeviceBuffer<float> weight_tensor_scale;
  DeviceBuffer<float> input_tensor_scale;
  DeviceBuffer<uint16_t> output;
  hipStream_t stream = nullptr;

  explicit NvBuffers(const NvShape shape, const NvHostFixture &fixture)
      : activation(fixture.activation.size()),
        activation_scales(fixture.activation_scales.size()),
        weight_tensor_scale(1U), input_tensor_scale(1U),
        output(static_cast<size_t>(shape.n) + kNvOutputGuard) {
    activation.upload(fixture.activation);
    activation_scales.upload(fixture.activation_scales);
    weights.reserve(kNvWeightPool);
    weight_scales.reserve(kNvWeightPool);
    for (uint32_t slot = 0U; slot < kNvWeightPool; ++slot) {
      weights.emplace_back(fixture.weights.size());
      weight_scales.emplace_back(fixture.weight_scales.size());
      weights.back().upload(fixture.weights);
      weight_scales.back().upload(fixture.weight_scales);
    }
    const float weight_scale = 0.75F;
    const float input_scale = 1.125F;
    std::vector<float> weight_scale_host = {weight_scale};
    std::vector<float> input_scale_host = {input_scale};
    weight_tensor_scale.upload(weight_scale_host);
    input_tensor_scale.upload(input_scale_host);
    check(hipStreamCreate(&stream), "hipStreamCreate NVFP4");
    check(hipMemsetAsync(output.ptr, 0x5a, output.count * sizeof(uint16_t),
                         stream),
          "clear NVFP4 output");
    check(hipStreamSynchronize(stream), "clear NVFP4 synchronize");
  }
  NvBuffers(const NvBuffers &) = delete;
  NvBuffers &operator=(const NvBuffers &) = delete;
  ~NvBuffers() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
};

hipError_t launch_control_attention_stage1(AttentionBuffers &buffers) {
  const uint64_t start = buffers.length - buffers.m;
  const dim3 grid = buffers.gqa ? dim3(buffers.m * kKvHeads * buffers.splits)
                                : dim3(buffers.m * kQHeads * buffers.splits);
  if (buffers.splits == kLongSplits) {
    if (buffers.gqa) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_gqa6_staged32_split_stage1_kernel<
                      kLongSplits>),
          grid, dim3(192U), 0U, buffers.stream, buffers.query.ptr,
          buffers.key.ptr, buffers.value.ptr, buffers.key_scales.ptr,
          buffers.value_scales.ptr, buffers.workspace.ptr, buffers.m, start);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_wave_split_staged_stage1_kernel<
                      true, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, kLongSplits>),
          grid, dim3(32U), 0U, buffers.stream, buffers.query.ptr,
          buffers.key.ptr, buffers.value.ptr, buffers.key_scales.ptr,
          buffers.value_scales.ptr, nullptr, nullptr, buffers.workspace.ptr,
          buffers.m, start, kQHeads, kKvHeads, kHeadDim, 1.0F, 1.0F);
    }
  } else if (buffers.gqa) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_gqa6_staged32_stage1_kernel),
        grid, dim3(192U), 0U, buffers.stream, buffers.query.ptr,
        buffers.key.ptr, buffers.value.ptr, buffers.key_scales.ptr,
        buffers.value_scales.ptr, buffers.workspace.ptr, buffers.m, start);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_wave_split_staged_stage1_kernel<
                    true, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, kShortSplits>),
        grid, dim3(32U), 0U, buffers.stream, buffers.query.ptr, buffers.key.ptr,
        buffers.value.ptr, buffers.key_scales.ptr, buffers.value_scales.ptr,
        nullptr, nullptr, buffers.workspace.ptr, buffers.m, start, kQHeads,
        kKvHeads, kHeadDim, 1.0F, 1.0F);
  }
  return hipGetLastError();
}

hipError_t launch_control_attention_stage2(AttentionBuffers &buffers) {
  if (buffers.splits == kLongSplits) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_gqa6_staged32_split_merge_kernel<
                    kLongSplits>),
        dim3(buffers.m * kQHeads), dim3(256U), 0U, buffers.stream,
        buffers.workspace.ptr, buffers.output.ptr, buffers.m);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_wave_split_staged_stage2_kernel<
                    kShortSplits>),
        dim3(buffers.m * kQHeads), dim3(256U), 0U, buffers.stream,
        buffers.workspace.ptr, buffers.output.ptr, buffers.m, kQHeads, kKvHeads,
        kHeadDim);
  }
  return hipGetLastError();
}

template <uint32_t kSplits, uint32_t kKeyTile, bool kRemapBlocks,
          bool kLookahead>
hipError_t launch_gqa_candidate_stage1(AttentionBuffers &buffers) {
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(sllm_causal_attention_kernel::phase87_wud3_detail::stage1<
                      kSplits, kKeyTile, kRemapBlocks, kLookahead>),
      dim3(buffers.m * kKvHeads * kSplits), dim3(192U), 0U, buffers.stream,
      buffers.query.ptr, buffers.key.ptr, buffers.value.ptr,
      buffers.key_scales.ptr, buffers.value_scales.ptr, buffers.workspace.ptr,
      buffers.m, buffers.length - buffers.m);
  return hipGetLastError();
}

hipError_t launch_attention_stage1(AttentionBuffers &buffers,
                                   const Phase87Wud3Candidate candidate) {
  if (!phase87_wud3_candidate_implemented(candidate))
    return hipErrorNotSupported;
  // gfx1201 deliberately remains on the production regular-wave provider for
  // C1/C3.  C2 does not force a new GQA provider there either.
  if (!buffers.gqa || candidate == Phase87Wud3Candidate::Control)
    return launch_control_attention_stage1(buffers);
  if (buffers.splits == kLongSplits) {
    switch (candidate) {
    case Phase87Wud3Candidate::C1Tile32:
      return launch_gqa_candidate_stage1<kLongSplits, 32U, false, false>(
          buffers);
    case Phase87Wud3Candidate::C2BlockRemap:
      return launch_gqa_candidate_stage1<kLongSplits, 8U, true, false>(buffers);
    case Phase87Wud3Candidate::C3Prefetch:
      return launch_gqa_candidate_stage1<kLongSplits, 8U, false, true>(buffers);
    default:
      break;
    }
  } else {
    switch (candidate) {
    case Phase87Wud3Candidate::C1Tile32:
      return launch_gqa_candidate_stage1<kShortSplits, 32U, false, false>(
          buffers);
    case Phase87Wud3Candidate::C2BlockRemap:
      return launch_gqa_candidate_stage1<kShortSplits, 8U, true, false>(
          buffers);
    case Phase87Wud3Candidate::C3Prefetch:
      return launch_gqa_candidate_stage1<kShortSplits, 8U, false, true>(
          buffers);
    default:
      break;
    }
  }
  return hipErrorInvalidValue;
}

hipError_t launch_attention_stage2(AttentionBuffers &buffers,
                                   const Phase87Wud3Candidate candidate) {
  if (!phase87_wud3_candidate_implemented(candidate))
    return hipErrorNotSupported;
  return launch_control_attention_stage2(buffers);
}

void launch_nvfp4(const NvShape shape, NvBuffers &buffers, const uint32_t slot,
                  const hipStream_t stream) {
  check(sllm_matmul_kernel::launch_nvfp4_w4a4(
            buffers.activation.ptr, buffers.activation_scales.ptr,
            buffers.weights[slot % kNvWeightPool].ptr,
            buffers.weight_scales[slot % kNvWeightPool].ptr,
            buffers.weight_tensor_scale.ptr, buffers.input_tensor_scale.ptr,
            buffers.output.ptr, 1U, shape.k, shape.n,
            sllm_matmul_kernel::KernelVariant::Nvfp4W4A4DecodeScaleLut, stream),
        "launch production NVFP4 W4A4 scale-LUT");
}

struct EventSet final {
  std::vector<hipEvent_t> events;
  explicit EventSet(const size_t count) : events(count, nullptr) {
    for (hipEvent_t &event : events)
      check(hipEventCreate(&event), "hipEventCreate");
  }
  EventSet(const EventSet &) = delete;
  EventSet &operator=(const EventSet &) = delete;
  ~EventSet() {
    for (hipEvent_t event : events)
      if (event != nullptr)
        (void)hipEventDestroy(event);
  }
};

struct Timed final {
  float stage1_ms = 0.0F;
  float stage2_ms = 0.0F;
  float attention_ms = 0.0F;
  std::vector<float> nv_ms;
  float nv_total_ms = 0.0F;
  float sequence_ms = 0.0F;
};

Timed measure_sequence(const NvShape shape, AttentionBuffers &attention,
                       NvBuffers &nv, const Phase87Wud3Candidate candidate,
                       const uint32_t repeats) {
  EventSet events(static_cast<size_t>(repeats) + 4U);
  const hipEvent_t start = events.events[0];
  const hipEvent_t after_stage1 = events.events[1];
  const hipEvent_t after_attention = events.events[2];
  check(hipEventRecord(start, attention.stream), "event sequence start");
  check(launch_attention_stage1(attention, candidate),
        "launch attention stage1");
  check(hipEventRecord(after_stage1, attention.stream), "event after stage1");
  check(launch_attention_stage2(attention, candidate),
        "launch attention stage2");
  check(hipEventRecord(after_attention, attention.stream),
        "event after attention");
  // The production NVFP4 call must be immediately downstream of attention on
  // the same stream.  No host synchronization is inserted between calls.
  for (uint32_t repeat = 0U; repeat < repeats; ++repeat) {
    launch_nvfp4(shape, nv, repeat, attention.stream);
    check(hipEventRecord(events.events[3U + repeat], attention.stream),
          "event after NVFP4");
  }
  check(hipEventSynchronize(events.events[3U + repeats - 1U]),
        "sequence synchronize");

  Timed timed;
  timed.nv_ms.resize(repeats);
  check(hipEventElapsedTime(&timed.stage1_ms, start, after_stage1),
        "elapsed attention stage1");
  check(hipEventElapsedTime(&timed.stage2_ms, after_stage1, after_attention),
        "elapsed attention stage2");
  timed.attention_ms = timed.stage1_ms + timed.stage2_ms;
  for (uint32_t repeat = 0U; repeat < repeats; ++repeat) {
    const hipEvent_t previous =
        repeat == 0U ? after_attention : events.events[2U + repeat];
    check(hipEventElapsedTime(&timed.nv_ms[repeat], previous,
                              events.events[3U + repeat]),
          "elapsed NVFP4 call");
  }
  timed.nv_total_ms =
      std::accumulate(timed.nv_ms.begin(), timed.nv_ms.end(), 0.0F);
  check(hipEventElapsedTime(&timed.sequence_ms, start,
                            events.events[3U + repeats - 1U]),
        "elapsed attention plus NVFP4");
  return timed;
}

void launch_sequence(const NvShape shape, AttentionBuffers &attention,
                     NvBuffers &nv, const Phase87Wud3Candidate candidate,
                     const uint32_t repeats) {
  check(launch_attention_stage1(attention, candidate), "warm attention stage1");
  check(launch_attention_stage2(attention, candidate), "warm attention stage2");
  for (uint32_t repeat = 0U; repeat < repeats; ++repeat)
    launch_nvfp4(shape, nv, repeat, attention.stream);
}

void warm_sequence(const NvShape shape, AttentionBuffers &attention,
                   NvBuffers &nv, const Phase87Wud3Candidate candidate,
                   const uint32_t repeats, const uint32_t warmup_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(warmup_ms);
  do {
    for (uint32_t iteration = 0U; iteration < 32U; ++iteration)
      launch_sequence(shape, attention, nv, candidate, repeats);
    check(hipStreamSynchronize(attention.stream), "warm attention synchronize");
    check(hipStreamSynchronize(nv.stream), "warm NVFP4 synchronize");
  } while (std::chrono::steady_clock::now() < deadline);
}

struct AttentionErrors final {
  uint32_t max_ulp = 0U;
  float max_abs = 0.0F;
  float max_relative = 0.0F;
  bool finite = true;
  bool repeat = true;
  bool output_guard = true;
  bool workspace_guard = true;
  std::vector<uint16_t> output;
};

AttentionErrors validate_attention(AttentionBuffers &buffers,
                                   const Phase87Wud3Candidate candidate,
                                   const std::vector<uint16_t> &expected) {
  buffers.clear_guards();
  check(launch_attention_stage1(buffers, candidate),
        "validate attention stage1");
  check(launch_attention_stage2(buffers, candidate),
        "validate attention stage2");
  check(hipStreamSynchronize(buffers.stream), "validate attention sync");
  const std::vector<uint16_t> first = buffers.output.download();
  check(launch_attention_stage1(buffers, candidate), "repeat attention stage1");
  check(launch_attention_stage2(buffers, candidate), "repeat attention stage2");
  check(hipStreamSynchronize(buffers.stream), "repeat attention sync");
  const std::vector<uint16_t> second = buffers.output.download();
  AttentionErrors errors;
  errors.output = first;
  errors.repeat = first == second;
  for (size_t index = 0U; index < expected.size(); ++index) {
    const float actual = bf16_to_f32(first[index]);
    const float reference = bf16_to_f32(expected[index]);
    errors.finite &= std::isfinite(actual);
    errors.max_ulp = std::max(errors.max_ulp,
                              bf16_ulp_distance(first[index], expected[index]));
    errors.max_abs = std::max(errors.max_abs, std::fabs(actual - reference));
    errors.max_relative = std::max(errors.max_relative,
                                   std::fabs(actual - reference) /
                                       std::max(std::fabs(reference), 1.0e-6F));
  }
  for (size_t index = expected.size(); index < first.size(); ++index)
    errors.output_guard &= first[index] == UINT16_C(0x5a5a);
  const std::vector<float> workspace = buffers.workspace.download();
  for (size_t index = buffers.used_workspace(); index < workspace.size();
       ++index) {
    uint32_t raw = 0U;
    std::memcpy(&raw, &workspace[index], sizeof(raw));
    errors.workspace_guard &= raw == UINT32_C(0x5a5a5a5a);
  }
  return errors;
}

struct AttentionComparison final {
  uint32_t max_ulp = 0U;
  float max_abs = 0.0F;
  bool bitwise = true;
};

AttentionComparison
compare_attention_outputs(const std::vector<uint16_t> &control,
                          const std::vector<uint16_t> &candidate) {
  if (control.size() != candidate.size())
    fail("attention output comparison shape mismatch");
  AttentionComparison comparison;
  for (size_t index = 0U; index < control.size(); ++index) {
    comparison.bitwise &= control[index] == candidate[index];
    comparison.max_ulp =
        std::max(comparison.max_ulp,
                 bf16_ulp_distance(control[index], candidate[index]));
    comparison.max_abs =
        std::max(comparison.max_abs, std::fabs(bf16_to_f32(control[index]) -
                                               bf16_to_f32(candidate[index])));
  }
  return comparison;
}

struct NvErrors final {
  uint32_t max_ulp = 0U;
  float max_abs = 0.0F;
  bool finite = true;
  bool repeat = true;
  bool output_guard = true;
};

NvErrors validate_nv(const NvShape shape, NvBuffers &buffers,
                     const std::vector<uint16_t> &expected) {
  check(hipMemsetAsync(buffers.output.ptr, 0x5a,
                       buffers.output.count * sizeof(uint16_t), buffers.stream),
        "clear NVFP4 validation output");
  launch_nvfp4(shape, buffers, 0U, buffers.stream);
  check(hipStreamSynchronize(buffers.stream), "validate NVFP4 sync");
  const std::vector<uint16_t> first = buffers.output.download();
  launch_nvfp4(shape, buffers, 1U, buffers.stream);
  check(hipStreamSynchronize(buffers.stream), "repeat NVFP4 sync");
  const std::vector<uint16_t> second = buffers.output.download();
  NvErrors errors;
  errors.repeat = first == second;
  for (size_t index = 0U; index < expected.size(); ++index) {
    const float actual = bf16_to_f32(first[index]);
    const float reference = bf16_to_f32(expected[index]);
    errors.finite &= std::isfinite(actual);
    errors.max_ulp = std::max(errors.max_ulp,
                              bf16_ulp_distance(first[index], expected[index]));
    errors.max_abs = std::max(errors.max_abs, std::fabs(actual - reference));
  }
  for (size_t index = 0U; index < static_cast<size_t>(shape.n); ++index)
    errors.finite &= std::isfinite(bf16_to_f32(first[index]));
  for (size_t index = static_cast<size_t>(shape.n); index < first.size();
       ++index)
    errors.output_guard &= first[index] == UINT16_C(0x5a5a);
  return errors;
}

template <typename T> double median(std::vector<T> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  return static_cast<double>(values[values.size() / 2U]);
}

void print_nv_samples(const char *const prefix,
                      const std::vector<std::vector<float>> &samples) {
  std::printf(",\"%s\":[", prefix);
  for (size_t row = 0U; row < samples.size(); ++row) {
    if (row != 0U)
      std::printf(",");
    std::printf("[");
    for (size_t index = 0U; index < samples[row].size(); ++index) {
      if (index != 0U)
        std::printf(",");
      std::printf("%.9g", samples[row][index]);
    }
    std::printf("]");
  }
  std::printf("]");
}

Phase87Wud3Candidate parse_candidate(const std::string_view value) {
  const std::array<std::pair<std::string_view, Phase87Wud3Candidate>, 5> names =
      {{
          {"control", Phase87Wud3Candidate::Control},
          {"c1-tile32", Phase87Wud3Candidate::C1Tile32},
          {"c1-tile64", Phase87Wud3Candidate::C1Tile64},
          {"c2-block-remap", Phase87Wud3Candidate::C2BlockRemap},
          {"c3-prefetch", Phase87Wud3Candidate::C3Prefetch},
      }};
  for (const auto &[name, candidate] : names)
    if (name == value)
      return candidate;
  fail("unknown --candidate: " + std::string(value));
}

void parse_options(const int argc, char **const argv, Options *const options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto value = [&](const std::string_view prefix) -> std::string {
      if (argument.substr(0U, prefix.size()) != prefix)
        return {};
      return std::string(argument.substr(prefix.size()));
    };
    if (argument == "--help") {
      std::printf(
          "options: --target=<gfx1030|gfx1201> --shape=<wide|down|both> "
          "--candidate=<control|c1-tile32|c1-tile64|c2-block-remap|c3-prefetch>"
          " "
          "--length=<N> --m=<1..3> --pattern=<0..2> --nv-repeats=<N> "
          "--warmup-ms=<N> --rounds=<N> --samples=<N>\n");
      std::exit(EXIT_SUCCESS);
    }
    if (const std::string target = value("--target="); !target.empty()) {
      options->target = target;
      continue;
    }
    if (const std::string shape = value("--shape="); !shape.empty()) {
      options->shape = shape;
      continue;
    }
    if (const std::string candidate = value("--candidate=");
        !candidate.empty()) {
      options->candidate = candidate;
      continue;
    }
    if (const std::string length = value("--length="); !length.empty()) {
      options->length = std::stoull(length);
      continue;
    }
    if (const std::string m = value("--m="); !m.empty()) {
      options->m = static_cast<uint32_t>(std::stoul(m));
      continue;
    }
    if (const std::string pattern = value("--pattern="); !pattern.empty()) {
      options->pattern = static_cast<uint32_t>(std::stoul(pattern));
      continue;
    }
    if (const std::string repeats = value("--nv-repeats="); !repeats.empty()) {
      options->nv_repeats = static_cast<uint32_t>(std::stoul(repeats));
      continue;
    }
    if (const std::string warmup = value("--warmup-ms="); !warmup.empty()) {
      options->warmup_ms = static_cast<uint32_t>(std::stoul(warmup));
      continue;
    }
    if (const std::string rounds = value("--rounds="); !rounds.empty()) {
      options->rounds = static_cast<uint32_t>(std::stoul(rounds));
      continue;
    }
    if (const std::string samples = value("--samples="); !samples.empty()) {
      options->samples = static_cast<uint32_t>(std::stoul(samples));
      continue;
    }
    fail("unknown argument: " + std::string(argument));
  }
  if (options->target != "gfx1030" && options->target != "gfx1201")
    fail("--target must be gfx1030 or gfx1201");
  if (options->shape != "wide" && options->shape != "down" &&
      options->shape != "both")
    fail("--shape must be wide, down, or both");
  if (options->length < options->m || options->m == 0U || options->m > kMaxM)
    fail("--length/--m shape is invalid");
  if (options->length < 1024U)
    fail("--length must be at least 1024 for the current staged provider");
  if (options->nv_repeats == 0U || options->rounds == 0U ||
      options->samples == 0U)
    fail("repeat, rounds, and samples must be nonzero");
}

int run(const Options &options) {
  const Phase87Wud3Candidate candidate = parse_candidate(options.candidate);
  // This is intentionally before hipGetDeviceProperties and before every
  // hipMalloc: shape confusion must be visible even if the requested GPU is
  // busy or the candidate is not yet enabled.
  std::printf("probe=phase87_wud3_attention_neighbor requested_target=%s "
              "requested_shape=%s length=%llu m=%u pattern=%u candidate=%s "
              "nv_repeats=%u warmup_ms=%u rounds=%u samples=%u "
              "nv_activation=synthetic_fixed\n",
              options.target.c_str(), options.shape.c_str(),
              static_cast<unsigned long long>(options.length), options.m,
              options.pattern, phase87_wud3_candidate_name(candidate),
              options.nv_repeats, options.warmup_ms, options.rounds,
              options.samples);
  if (!phase87_wud3_candidate_preflight_safe(candidate)) {
    std::printf("candidate_status=UNAVAILABLE reason=tile64-lds-preflight "
                "compiled_tile32_lds_bytes=65536 tile64_lds_bytes=131072\n");
    return EXIT_SUCCESS;
  }
  if (!phase87_wud3_candidate_implemented(candidate)) {
    std::printf("candidate_status=UNAVAILABLE reason=awaiting-wu-d2\n");
    return EXIT_SUCCESS;
  }

  hipDeviceProp_t device{};
  check(hipGetDeviceProperties(&device, 0), "hipGetDeviceProperties");
  if (options.target != device.gcnArchName)
    fail("requested target does not match current GPU: " +
         std::string(device.gcnArchName));
  std::printf("resolved_target=%s gqa=%s long_split=%s\n", device.gcnArchName,
              options.target == "gfx1030" ? "true" : "false",
              options.length >= 8192U && options.m <= kMaxM ? "true" : "false");
  const bool candidate_falls_back_to_control =
      options.target != "gfx1030" && candidate != Phase87Wud3Candidate::Control;
  std::printf("candidate_effective=%s\n",
              candidate_falls_back_to_control
                  ? "control-fallback"
                  : phase87_wud3_candidate_name(candidate));
  const uint32_t active_splits = options.length >= 8192U && options.m <= kMaxM
                                     ? kLongSplits
                                     : kShortSplits;
  const uint32_t stage1_grid =
      options.m * (options.target == "gfx1030" ? kKvHeads : kQHeads) *
      active_splits;
  const uint32_t stage1_block = options.target == "gfx1030" ? 192U : 32U;
  std::printf("control_dispatch stage1_grid=%u stage1_block=%u "
              "stage2_grid=%u stage2_block=256 splits=%u\n",
              stage1_grid, stage1_block, options.m * kQHeads, active_splits);

  AttentionFixture attention_fixture =
      make_attention_fixture(options.length, options.m);
  apply_attention_pattern(&attention_fixture, options.pattern);
  const std::vector<uint16_t> attention_expected =
      attention_oracle(attention_fixture, options.length);
  AttentionBuffers attention(attention_fixture, options.length, options.target);

  const size_t first_shape = options.shape == "down" ? 1U : 0U;
  const size_t last_shape = options.shape == "wide" ? 1U : 2U;
  bool all_ok = true;
  for (size_t shape_index = first_shape; shape_index < last_shape;
       ++shape_index) {
    const NvShape shape = kNvShapes[shape_index];
    const NvHostFixture nv_fixture = make_nv_fixture(shape);
    const std::vector<uint16_t> nv_expected =
        nv_oracle_prefix(shape, nv_fixture);
    NvBuffers nv(shape, nv_fixture);

    const AttentionErrors control_attention_errors = validate_attention(
        attention, Phase87Wud3Candidate::Control, attention_expected);
    const AttentionErrors candidate_attention_errors =
        candidate == Phase87Wud3Candidate::Control
            ? control_attention_errors
            : validate_attention(attention, candidate, attention_expected);
    const AttentionComparison attention_comparison = compare_attention_outputs(
        control_attention_errors.output, candidate_attention_errors.output);
    const bool control_attention_pass =
        control_attention_errors.finite && control_attention_errors.repeat &&
        control_attention_errors.output_guard &&
        control_attention_errors.workspace_guard &&
        control_attention_errors.max_ulp <= 4U &&
        control_attention_errors.max_abs <= 0.03125F &&
        control_attention_errors.max_relative <= 0.04F;
    const bool candidate_attention_pass =
        candidate_attention_errors.finite &&
        candidate_attention_errors.repeat &&
        candidate_attention_errors.output_guard &&
        candidate_attention_errors.workspace_guard &&
        candidate_attention_errors.max_ulp <= 4U &&
        candidate_attention_errors.max_abs <= 0.03125F &&
        candidate_attention_errors.max_relative <= 0.04F;
    const bool attention_pass =
        control_attention_pass && candidate_attention_pass;
    const NvErrors nv_errors = validate_nv(shape, nv, nv_expected);
    const bool nv_pass = nv_errors.finite && nv_errors.repeat &&
                         nv_errors.output_guard && nv_errors.max_ulp <= 4U;
    all_ok = all_ok && attention_pass && nv_pass;
    std::printf(
        "{\"kind\":\"attention_oracle\",\"shape\":\"%s\","
        "\"target\":\"%s\",\"length\":%llu,\"m\":%u,"
        "\"max_ulp\":%u,\"max_abs\":%.9g,\"max_relative\":%.9g,"
        "\"control_max_ulp\":%u,\"candidate_max_ulp\":%u,"
        "\"candidate_vs_control_bitwise\":%s,"
        "\"candidate_vs_control_max_ulp\":%u,"
        "\"candidate_vs_control_max_abs\":%.9g,"
        "\"control_finite\":%s,\"candidate_finite\":%s,"
        "\"control_repeat\":%s,\"candidate_repeat\":%s,"
        "\"control_output_guard\":%s,\"candidate_output_guard\":%s,"
        "\"control_workspace_guard\":%s,\"candidate_workspace_guard\":%s,"
        "\"state\":\"%s\"}\n",
        shape.name, options.target.c_str(),
        static_cast<unsigned long long>(options.length), options.m,
        candidate_attention_errors.max_ulp, candidate_attention_errors.max_abs,
        candidate_attention_errors.max_relative,
        control_attention_errors.max_ulp, candidate_attention_errors.max_ulp,
        attention_comparison.bitwise ? "true" : "false",
        attention_comparison.max_ulp, attention_comparison.max_abs,
        control_attention_errors.finite ? "true" : "false",
        candidate_attention_errors.finite ? "true" : "false",
        control_attention_errors.repeat ? "true" : "false",
        candidate_attention_errors.repeat ? "true" : "false",
        control_attention_errors.output_guard ? "true" : "false",
        candidate_attention_errors.output_guard ? "true" : "false",
        control_attention_errors.workspace_guard ? "true" : "false",
        candidate_attention_errors.workspace_guard ? "true" : "false",
        attention_pass ? "PASS" : "FAIL");
    std::printf("{\"kind\":\"nv_oracle\",\"shape\":\"%s\","
                "\"target\":\"%s\",\"columns\":%u,\"max_ulp\":%u,"
                "\"max_abs\":%.9g,\"finite\":%s,\"repeat\":%s,"
                "\"output_guard\":%s,\"state\":\"%s\"}\n",
                shape.name, options.target.c_str(),
                static_cast<unsigned>(nv_expected.size()), nv_errors.max_ulp,
                nv_errors.max_abs, nv_errors.finite ? "true" : "false",
                nv_errors.repeat ? "true" : "false",
                nv_errors.output_guard ? "true" : "false",
                nv_pass ? "PASS" : "FAIL");

    std::vector<float> control_stage1;
    std::vector<float> candidate_stage1;
    std::vector<float> control_stage2;
    std::vector<float> candidate_stage2;
    std::vector<float> control_attention;
    std::vector<float> candidate_attention;
    std::vector<float> control_nv_total;
    std::vector<float> candidate_nv_total;
    std::vector<float> control_sequence;
    std::vector<float> candidate_sequence;
    std::vector<std::vector<float>> control_nv_calls;
    std::vector<std::vector<float>> candidate_nv_calls;
    for (uint32_t round = 0U; round < options.rounds; ++round) {
      for (uint32_t position = 0U; position < 4U; ++position) {
        // ABBA: control, candidate, candidate, control.  With control-only
        // this still exercises the exact same stream and event protocol.
        const Phase87Wud3Candidate selected =
            (position == 0U || position == 3U) ? Phase87Wud3Candidate::Control
                                               : candidate;
        warm_sequence(shape, attention, nv, selected, options.nv_repeats,
                      options.warmup_ms);
        for (uint32_t sample = 0U; sample < options.samples; ++sample) {
          const Timed timed = measure_sequence(shape, attention, nv, selected,
                                               options.nv_repeats);
          const bool is_control = selected == Phase87Wud3Candidate::Control;
          (is_control ? control_stage1 : candidate_stage1)
              .push_back(timed.stage1_ms);
          (is_control ? control_stage2 : candidate_stage2)
              .push_back(timed.stage2_ms);
          (is_control ? control_attention : candidate_attention)
              .push_back(timed.attention_ms);
          (is_control ? control_nv_total : candidate_nv_total)
              .push_back(timed.nv_total_ms);
          (is_control ? control_sequence : candidate_sequence)
              .push_back(timed.sequence_ms);
          (is_control ? control_nv_calls : candidate_nv_calls)
              .push_back(timed.nv_ms);
        }
      }
    }
    const bool candidate_samples_present = !candidate_sequence.empty();
    std::printf("{\"kind\":\"performance\",\"shape\":\"%s\","
                "\"target\":\"%s\",\"candidate\":\"%s\","
                "\"candidate_effective\":\"%s\","
                "\"nv_repeats\":%u,\"warmup_ms\":%u,\"rounds\":%u,"
                "\"samples_per_cell\":%u,\"order\":\"ABBA\","
                "\"control_stage1_ms\":%.9g,\"control_stage2_ms\":%.9g,"
                "\"control_attention_ms\":%.9g,\"control_nv_total_ms\":%.9g,"
                "\"control_sequence_ms\":%.9g,\"candidate_samples_present\":%s",
                shape.name, options.target.c_str(),
                phase87_wud3_candidate_name(candidate),
                candidate_falls_back_to_control
                    ? "control-fallback"
                    : phase87_wud3_candidate_name(candidate),
                options.nv_repeats, options.warmup_ms, options.rounds,
                options.samples, median(control_stage1), median(control_stage2),
                median(control_attention), median(control_nv_total),
                median(control_sequence),
                candidate_samples_present ? "true" : "false");
    print_nv_samples("control_nv_calls_ms", control_nv_calls);
    if (candidate_samples_present) {
      std::printf(",\"candidate_stage1_ms\":%.9g,\"candidate_stage2_ms\":%.9g,"
                  "\"candidate_attention_ms\":%.9g,"
                  "\"candidate_nv_total_ms\":%.9g,"
                  "\"candidate_sequence_ms\":%.9g",
                  median(candidate_stage1), median(candidate_stage2),
                  median(candidate_attention), median(candidate_nv_total),
                  median(candidate_sequence));
      print_nv_samples("candidate_nv_calls_ms", candidate_nv_calls);
    }
    std::printf("}\n");
  }
  check(hipDeviceSynchronize(), "final device synchronize");
  std::printf("{\"kind\":\"cleanup\",\"state\":\"PASS\"}\n");
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(const int argc, char **const argv) {
  try {
    Options options;
    parse_options(argc, argv, &options);
    return run(options);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87_wud3_attention_neighbor_probe: %s\n",
                 error.what());
    return EXIT_FAILURE;
  }
}
