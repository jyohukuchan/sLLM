// Phase 87 WU-P1: compare the actual Qwen3.8 MXFP8-E4 GQA6 attention
// providers with probe-only 128-token paged counterparts. Production source
// is included as the control; this file is not part of the runtime library.
#define SLLM_PUBLIC_RUNTIME_HOST_TEST 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "../src/causal_attention_kernel.hip.cpp"
#pragma clang diagnostic pop

namespace sllm_causal_attention_kernel {
namespace {
#include "phase87_wup1_paged_decode.hpp"
#include "phase87_wup1_paged_prefill.hpp"
} // namespace
} // namespace sllm_causal_attention_kernel

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kScaleBlock = 32U;
constexpr uint32_t kScalesPerRow = kHeadDim / kScaleBlock;
constexpr uint32_t kPageTokens = 128U;
constexpr uint32_t kOutputGuard = 32U;
constexpr uint32_t kWorkspaceGuard = 64U;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check(const hipError_t status, const char *operation) {
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

double e4m3fn_decode(const uint8_t value) {
  const double sign = (value & UINT8_C(0x80)) != 0U ? -1.0 : 1.0;
  const uint32_t magnitude = value & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT32_C(7);
  if (exponent == 0U)
    return sign * static_cast<double>(mantissa) * std::ldexp(1.0, -9);
  return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                           static_cast<int>(exponent) - 7);
}

struct Fixture final {
  uint64_t length;
  uint32_t m;
  std::vector<uint16_t> query;
  std::vector<uint8_t> key, value, key_scales, value_scales;
};

Fixture make_fixture(const uint64_t length, const uint32_t m) {
  if (m == 0U || m > length)
    fail("invalid fixture dimensions");
  Fixture fixture{};
  fixture.length = length;
  fixture.m = m;
  fixture.query.resize(static_cast<size_t>(m) * kQHeads * kHeadDim);
  const size_t kv_rows = static_cast<size_t>(length) * kKvHeads;
  fixture.key.resize(kv_rows * kHeadDim);
  fixture.value.resize(kv_rows * kHeadDim);
  fixture.key_scales.resize(kv_rows * kScalesPerRow);
  fixture.value_scales.resize(kv_rows * kScalesPerRow);
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
  for (size_t row = 0U; row < kv_rows; ++row) {
    const uint32_t kv_head = static_cast<uint32_t>(row % kKvHeads);
    const uint64_t token = row / kKvHeads;
    for (uint32_t block = 0U; block < kScalesPerRow; ++block) {
      const uint8_t key_scale =
          static_cast<uint8_t>(125U + ((token + kv_head + block) & 3U));
      const uint8_t value_scale =
          static_cast<uint8_t>(126U + ((2U * token + kv_head + block) & 3U));
      fixture.key_scales[row * kScalesPerRow + block] = key_scale;
      fixture.value_scales[row * kScalesPerRow + block] = value_scale;
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
        const size_t index = row * kHeadDim + dimension;
        fixture.key[index] = e4m3fn_encode(key_source / key_scale_value);
        fixture.value[index] = e4m3fn_encode(value_source / value_scale_value);
      }
    }
  }
  return fixture;
}

struct PagedFixture final {
  std::vector<uint32_t> table;
  std::vector<uint8_t> key, value, key_scales, value_scales;
};

PagedFixture page_fixture(const Fixture &fixture) {
  PagedFixture paged;
  const uint64_t pages = (fixture.length + kPageTokens - 1U) / kPageTokens;
  paged.table.resize(static_cast<size_t>(pages));
  const uint64_t physical_tokens = pages * kPageTokens;
  const size_t rows = static_cast<size_t>(physical_tokens) * kKvHeads;
  paged.key.resize(rows * kHeadDim, UINT8_C(0xa5));
  paged.value.resize(rows * kHeadDim, UINT8_C(0xa5));
  paged.key_scales.resize(rows * kScalesPerRow, UINT8_C(0xa5));
  paged.value_scales.resize(rows * kScalesPerRow, UINT8_C(0xa5));
  for (uint64_t page = 0U; page < pages; ++page)
    paged.table[static_cast<size_t>(page)] =
        static_cast<uint32_t>(pages - page - 1U);
  for (uint64_t token = 0U; token < fixture.length; ++token) {
    const uint64_t physical_token =
        static_cast<uint64_t>(
            paged.table[static_cast<size_t>(token / kPageTokens)]) *
            kPageTokens +
        token % kPageTokens;
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      const size_t src = (static_cast<size_t>(token) * kKvHeads + head);
      const size_t dst =
          (static_cast<size_t>(physical_token) * kKvHeads + head);
      std::memcpy(paged.key.data() + dst * kHeadDim,
                  fixture.key.data() + src * kHeadDim, kHeadDim);
      std::memcpy(paged.value.data() + dst * kHeadDim,
                  fixture.value.data() + src * kHeadDim, kHeadDim);
      std::memcpy(paged.key_scales.data() + dst * kScalesPerRow,
                  fixture.key_scales.data() + src * kScalesPerRow,
                  kScalesPerRow);
      std::memcpy(paged.value_scales.data() + dst * kScalesPerRow,
                  fixture.value_scales.data() + src * kScalesPerRow,
                  kScalesPerRow);
    }
  }
  if (pages > 1U && paged.table[0] == 0U)
    fail("page table must be non-identity");
  return paged;
}

bool cleanup_ok = true;
size_t live_allocations = 0U;
template <class T> struct DeviceBuffer final {
  T *ptr = nullptr;
  size_t count;
  explicit DeviceBuffer(const size_t n) : count(n) {
    check(hipMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)),
          "hipMalloc");
    ++live_allocations;
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      if (hipFree(ptr) == hipSuccess)
        --live_allocations;
      else
        cleanup_ok = false;
    }
  }
  void upload(const std::vector<T> &values) {
    if (values.size() != count)
      fail("upload shape mismatch");
    check(
        hipMemcpy(ptr, values.data(), count * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  }
  std::vector<T> download() const {
    std::vector<T> values(count);
    check(
        hipMemcpy(values.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost),
        "download");
    return values;
  }
};

struct DeviceFixture final {
  DeviceBuffer<uint16_t> query, output_a, output_b;
  DeviceBuffer<uint8_t> key, value, key_scales, value_scales;
  DeviceBuffer<uint8_t> paged_key, paged_value, paged_key_scales,
      paged_value_scales;
  DeviceBuffer<uint32_t> table;
  DeviceBuffer<float> workspace_a, workspace_b;
  hipStream_t stream = nullptr;
  uint64_t length;
  uint32_t m;
  uint32_t splits;
  size_t output_elements;
  size_t workspace_elements;

  DeviceFixture(const Fixture &fixture, const PagedFixture &paged,
                const uint32_t selected_splits)
      : query(fixture.query.size()),
        output_a(fixture.query.size() + kOutputGuard),
        output_b(fixture.query.size() + kOutputGuard), key(fixture.key.size()),
        value(fixture.value.size()), key_scales(fixture.key_scales.size()),
        value_scales(fixture.value_scales.size()), paged_key(paged.key.size()),
        paged_value(paged.value.size()),
        paged_key_scales(paged.key_scales.size()),
        paged_value_scales(paged.value_scales.size()),
        table(paged.table.size()),
        workspace_a(static_cast<size_t>(fixture.m) * kQHeads * selected_splits *
                        (kHeadDim + 2U) +
                    kWorkspaceGuard),
        workspace_b(static_cast<size_t>(fixture.m) * kQHeads * selected_splits *
                        (kHeadDim + 2U) +
                    kWorkspaceGuard),
        length(fixture.length), m(fixture.m), splits(selected_splits),
        output_elements(fixture.query.size()),
        workspace_elements(static_cast<size_t>(fixture.m) * kQHeads *
                           selected_splits * (kHeadDim + 2U)) {
    query.upload(fixture.query);
    key.upload(fixture.key);
    value.upload(fixture.value);
    key_scales.upload(fixture.key_scales);
    value_scales.upload(fixture.value_scales);
    paged_key.upload(paged.key);
    paged_value.upload(paged.value);
    paged_key_scales.upload(paged.key_scales);
    paged_value_scales.upload(paged.value_scales);
    table.upload(paged.table);
    check(hipStreamCreate(&stream), "hipStreamCreate");
    reset();
  }
  ~DeviceFixture() {
    if (stream != nullptr && hipStreamDestroy(stream) != hipSuccess)
      cleanup_ok = false;
  }
  void reset() {
    check(hipMemsetAsync(output_a.ptr, 0x5a, output_a.count * sizeof(uint16_t),
                         stream),
          "clear output A");
    check(hipMemsetAsync(output_b.ptr, 0x5a, output_b.count * sizeof(uint16_t),
                         stream),
          "clear output B");
    check(hipMemsetAsync(workspace_a.ptr, 0x5a,
                         workspace_a.count * sizeof(float), stream),
          "clear workspace A");
    check(hipMemsetAsync(workspace_b.ptr, 0x5a,
                         workspace_b.count * sizeof(float), stream),
          "clear workspace B");
    check(hipStreamSynchronize(stream), "reset synchronize");
  }
};

double decoded(const std::vector<uint8_t> &values,
               const std::vector<uint8_t> &scales, const uint64_t row,
               const uint32_t dimension) {
  const uint8_t scale = scales[static_cast<size_t>(row) * kScalesPerRow +
                               dimension / kScaleBlock];
  return e4m3fn_decode(
             values[static_cast<size_t>(row) * kHeadDim + dimension]) *
         std::ldexp(1.0, static_cast<int>(scale) - 127);
}

// The independent FP64 oracle samples causal positions, GQA groups, scale
// boundaries, and tail dimensions. Full output is compared bitwise A versus B.
double oracle_max_abs(const Fixture &fixture,
                      const std::vector<uint16_t> &output) {
  const std::array<std::pair<uint32_t, uint32_t>, 3> pairs = {
      std::pair<uint32_t, uint32_t>{0U, 0U},
      {fixture.m / 2U, 11U},
      {fixture.m - 1U, kQHeads - 1U}};
  constexpr std::array<uint32_t, 6> dimensions = {0U,   31U,  32U,
                                                  127U, 128U, 255U};
  double max_abs = 0.0;
  for (const auto [row, query_head] : pairs) {
    const uint32_t kv_head = query_head / (kQHeads / kKvHeads);
    const uint64_t visible = fixture.length - fixture.m + row + 1U;
    std::vector<double> scores(static_cast<size_t>(visible));
    double maximum = -std::numeric_limits<double>::infinity();
    for (uint64_t token = 0U; token < visible; ++token) {
      const uint64_t kv_row = token * kKvHeads + kv_head;
      double score = 0.0;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const size_t query_index =
            (static_cast<size_t>(row) * kQHeads + query_head) * kHeadDim +
            dimension;
        score += static_cast<double>(bf16_to_f32(fixture.query[query_index])) *
                 decoded(fixture.key, fixture.key_scales, kv_row, dimension);
      }
      score /= 16.0;
      scores[static_cast<size_t>(token)] = score;
      maximum = std::max(maximum, score);
    }
    double denominator = 0.0;
    for (double &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (const uint32_t dimension : dimensions) {
      double numerator = 0.0;
      for (uint64_t token = 0U; token < visible; ++token) {
        const uint64_t kv_row = token * kKvHeads + kv_head;
        numerator +=
            scores[static_cast<size_t>(token)] *
            decoded(fixture.value, fixture.value_scales, kv_row, dimension);
      }
      const size_t index =
          (static_cast<size_t>(row) * kQHeads + query_head) * kHeadDim +
          dimension;
      const double actual = static_cast<double>(bf16_to_f32(output[index]));
      max_abs = std::max(max_abs, std::abs(actual - numerator / denominator));
    }
  }
  return max_abs;
}

bool output_guard_intact(const std::vector<uint16_t> &output,
                         const size_t elements) {
  return std::all_of(output.begin() + static_cast<ptrdiff_t>(elements),
                     output.end(),
                     [](const uint16_t value) { return value == 0x5a5aU; });
}

bool workspace_guard_intact(const std::vector<float> &workspace,
                            const size_t elements) {
  const auto *const bytes =
      reinterpret_cast<const uint8_t *>(workspace.data() + elements);
  const size_t remaining = (workspace.size() - elements) * sizeof(float);
  for (size_t index = 0U; index < remaining; ++index) {
    if (bytes[index] != UINT8_C(0x5a))
      return false;
  }
  return true;
}

enum class Kind { Decode, Prefill };

void launch_decode_stage1(const DeviceFixture &b, const bool paged,
                          const bool gqa_shared) {
  const uint64_t start = b.length - b.m;
  if (paged) {
    check(sllm_causal_attention_kernel::launch_wup1_paged_decode_stage1(
              b.query.ptr, b.paged_key.ptr, b.paged_value.ptr,
              b.paged_key_scales.ptr, b.paged_value_scales.ptr, b.table.ptr,
              b.workspace_b.ptr, b.m, start, b.splits, gqa_shared, b.stream),
          "paged decode stage1");
    return;
  }
  const dim3 grid(b.m * (gqa_shared ? kKvHeads : kQHeads) * b.splits);
  if (b.splits == 128U) {
    if (gqa_shared) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_gqa6_staged32_split_stage1_kernel<
                      128U>),
          grid, dim3(192U), 0U, b.stream, b.query.ptr, b.key.ptr, b.value.ptr,
          b.key_scales.ptr, b.value_scales.ptr, b.workspace_a.ptr, b.m, start,
          nullptr);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_wave_split_staged_stage1_kernel<
                      true, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 128U>),
          grid, dim3(32U), 0U, b.stream, b.query.ptr, b.key.ptr, b.value.ptr,
          b.key_scales.ptr, b.value_scales.ptr, nullptr, nullptr,
          b.workspace_a.ptr, b.m, start, kQHeads, kKvHeads, kHeadDim, 1.0F,
          1.0F, nullptr);
    }
  } else if (gqa_shared) {
    hipLaunchKernelGGL(sllm_causal_attention_kernel::
                           causal_attention_decode_gqa6_staged32_stage1_kernel,
                       grid, dim3(192U), 0U, b.stream, b.query.ptr, b.key.ptr,
                       b.value.ptr, b.key_scales.ptr, b.value_scales.ptr,
                       b.workspace_a.ptr, b.m, start);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_wave_split_staged_stage1_kernel<
                    true, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 32U>),
        grid, dim3(32U), 0U, b.stream, b.query.ptr, b.key.ptr, b.value.ptr,
        b.key_scales.ptr, b.value_scales.ptr, nullptr, nullptr,
        b.workspace_a.ptr, b.m, start, kQHeads, kKvHeads, kHeadDim, 1.0F, 1.0F,
        nullptr);
  }
  check(hipGetLastError(), "control decode stage1");
}

void launch_decode_stage2(const DeviceFixture &b, const bool paged) {
  const float *const workspace = paged ? b.workspace_b.ptr : b.workspace_a.ptr;
  uint16_t *const output = paged ? b.output_b.ptr : b.output_a.ptr;
  if (b.splits == 128U) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_gqa6_staged32_split_merge_kernel<128U>),
        dim3(b.m * kQHeads), dim3(256U), 0U, b.stream, workspace, output, b.m,
        nullptr);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            sllm_causal_attention_kernel::
                causal_attention_decode_wave_split_staged_stage2_kernel<32U>),
        dim3(b.m * kQHeads), dim3(256U), 0U, b.stream, workspace, output, b.m,
        kQHeads, kKvHeads, kHeadDim, nullptr);
  }
  check(hipGetLastError(), "decode merge");
}

bool uses_qtile8(const DeviceFixture &b) {
  return b.m >= 128U && b.length - b.m >= 1024U;
}

void launch_prefill(const DeviceFixture &b, const bool paged) {
  const uint64_t start = b.length - b.m;
  uint16_t *const output = paged ? b.output_b.ptr : b.output_a.ptr;
  const void *const key = paged ? static_cast<const void *>(b.paged_key.ptr)
                                : static_cast<const void *>(b.key.ptr);
  const void *const value = paged ? static_cast<const void *>(b.paged_value.ptr)
                                  : static_cast<const void *>(b.value.ptr);
  const void *const key_scales =
      paged ? static_cast<const void *>(b.paged_key_scales.ptr)
            : static_cast<const void *>(b.key_scales.ptr);
  const void *const value_scales =
      paged ? static_cast<const void *>(b.paged_value_scales.ptr)
            : static_cast<const void *>(b.value_scales.ptr);
  if (uses_qtile8(b)) {
    if (paged) {
      check(sllm_causal_attention_kernel::launch_wup1_paged_prefill_qtile8(
                b.query.ptr, key, value, key_scales, value_scales, b.table.ptr,
                output, b.m, start, b.stream),
            "paged prefill qtile8");
    } else {
      check(sllm_causal_attention_kernel::launch_gqa6_qtile8_w16(
                b.query.ptr, key, value, key_scales, value_scales, nullptr,
                nullptr, output, b.m, start, kQHeads, kKvHeads, kHeadDim,
                SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 1.0F, 1.0F, false, b.stream),
            "control prefill qtile8");
    }
  } else if (paged) {
    check(sllm_causal_attention_kernel::launch_wup1_paged_prefill_qtile4(
              b.query.ptr, key, value, key_scales, value_scales, b.table.ptr,
              output, b.m, start, b.stream),
          "paged prefill qtile4");
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sllm_causal_attention_kernel::
                            causal_attention_prefill_gqa4_qtile4_kernel<
                                SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 6U>),
        dim3(((b.m + 3U) / 4U) * kKvHeads), dim3(256U), 0U, b.stream,
        b.query.ptr, key, value, key_scales, value_scales, nullptr, nullptr,
        output, b.m, start, kQHeads, kKvHeads, kHeadDim, 1.0F, 1.0F);
    check(hipGetLastError(), "control prefill qtile4");
  }
}

void launch_variant(const DeviceFixture &b, const Kind kind, const bool paged,
                    const bool gqa_shared) {
  if (kind == Kind::Decode) {
    launch_decode_stage1(b, paged, gqa_shared);
    launch_decode_stage2(b, paged);
  } else {
    launch_prefill(b, paged);
  }
}

double median(std::vector<double> values) {
  if (values.empty())
    fail("empty measurement series");
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2U;
  return values.size() % 2U == 0U ? (values[middle - 1U] + values[middle]) * 0.5
                                  : values[middle];
}

double elapsed_ms(const DeviceFixture &b, const Kind kind, const bool paged,
                  const bool gqa_shared, const uint32_t replays) {
  hipEvent_t start = nullptr, end = nullptr;
  check(hipEventCreate(&start), "create start event");
  check(hipEventCreate(&end), "create end event");
  check(hipEventRecord(start, b.stream), "record start event");
  for (uint32_t iteration = 0U; iteration < replays; ++iteration)
    launch_variant(b, kind, paged, gqa_shared);
  check(hipEventRecord(end, b.stream), "record end event");
  check(hipEventSynchronize(end), "synchronize end event");
  float milliseconds = 0.0F;
  check(hipEventElapsedTime(&milliseconds, start, end), "elapsed event time");
  check(hipEventDestroy(start), "destroy start event");
  check(hipEventDestroy(end), "destroy end event");
  return static_cast<double>(milliseconds) / replays;
}

struct Options final {
  std::string target;
  Kind kind = Kind::Decode;
  uint64_t length = 0U;
  uint32_t m = 0U;
  uint32_t rounds = 3U;
  uint32_t samples = 5U;
  uint32_t replays = 0U;
};

Options parse_options(const int argc, char **const argv) {
  if (argc < 5 || argc > 8)
    fail("usage: probe TARGET decode|prefill KV_LENGTH M [ROUNDS SAMPLES "
         "REPLAYS]");
  Options result;
  result.target = argv[1];
  if (result.target != "gfx1030" && result.target != "gfx1201")
    fail("target must be exact gfx1030 or gfx1201");
  const std::string kind = argv[2];
  if (kind == "decode")
    result.kind = Kind::Decode;
  else if (kind == "prefill")
    result.kind = Kind::Prefill;
  else
    fail("kind must be decode or prefill");
  result.length = std::stoull(argv[3]);
  result.m = static_cast<uint32_t>(std::stoul(argv[4]));
  if (argc >= 6)
    result.rounds = static_cast<uint32_t>(std::stoul(argv[5]));
  if (argc >= 7)
    result.samples = static_cast<uint32_t>(std::stoul(argv[6]));
  result.replays = result.kind == Kind::Decode ? 16U : 1U;
  if (argc == 8)
    result.replays = static_cast<uint32_t>(std::stoul(argv[7]));
  if (result.length == 0U || result.length > 65536U || result.m == 0U ||
      result.m > result.length || result.rounds < 2U || result.samples == 0U ||
      result.replays == 0U)
    fail("invalid benchmark dimensions or repetitions");
  if (result.kind == Kind::Decode && result.m > 3U)
    fail("decode M must be 1..3");
  if (result.kind == Kind::Prefill && result.m > 219U)
    fail("prefill M must be <=219 for the reviewed fixture");
  return result;
}

struct Round final {
  const char *order;
  std::vector<double> control, paged;
  double control_median = 0.0;
  double paged_median = 0.0;
  double increase_percent = 0.0;
};

void print_samples(const std::vector<double> &samples) {
  std::printf("[");
  for (size_t index = 0U; index < samples.size(); ++index) {
    if (index != 0U)
      std::printf(",");
    std::printf("%.9g", samples[index]);
  }
  std::printf("]");
}

void report(const Options &options, const uint32_t splits,
            const uint64_t page_count, const size_t contiguous_bytes,
            const size_t pool_bytes, const double oracle_abs,
            const bool bitwise, const bool repeat, const bool guards,
            const bool performance_ok, const bool cleanup_zero,
            const std::vector<Round> &rounds) {
  const char *const kind = options.kind == Kind::Decode ? "decode" : "prefill";
  const char *const prefill_provider =
      options.kind != Kind::Prefill ? "none"
      : options.m >= 128U && options.length - options.m >= 1024U
          ? "gqa6_qtile8_w16"
          : "gqa6_qtile4";
  std::printf("{\"schema_version\":\"phase87-wup1-paged-probe-v1\","
              "\"target\":\"%s\",\"kind\":\"%s\",\"prefill_provider\":\"%s\","
              "\"kv_encoding\":\"mxfp8-e4\",\"q_heads\":24,\"kv_heads\":4,"
              "\"head_dim\":256,\"page_tokens\":128,"
              "\"page_table_layout\":\"reverse_by_page\",\"length\":%llu,"
              "\"m\":%u,\"splits\":%u,\"page_count\":%llu,"
              "\"table_bytes\":%llu,\"contiguous_kv_bytes\":%llu,"
              "\"paged_pool_bytes\":%llu,\"round_count\":%u,"
              "\"samples_per_variant\":%u,\"replays_per_sample\":%u,"
              "\"oracle_max_abs\":%.9g,\"candidate_bitwise\":%s,"
              "\"repeat\":%s,\"guards\":%s,\"performance_below_10pct\":%s,"
              "\"gpu_execution\":true,\"fallback_used\":false,"
              "\"cleanup_zero\":%s,\"rounds\":[",
              options.target.c_str(), kind, prefill_provider,
              static_cast<unsigned long long>(options.length), options.m,
              splits, static_cast<unsigned long long>(page_count),
              static_cast<unsigned long long>(page_count * sizeof(uint32_t)),
              static_cast<unsigned long long>(contiguous_bytes),
              static_cast<unsigned long long>(pool_bytes), options.rounds,
              options.samples, options.replays, oracle_abs,
              bitwise ? "true" : "false", repeat ? "true" : "false",
              guards ? "true" : "false", performance_ok ? "true" : "false",
              cleanup_zero ? "true" : "false");
  for (size_t index = 0U; index < rounds.size(); ++index) {
    const Round &round = rounds[index];
    if (index != 0U)
      std::printf(",");
    std::printf("{\"order\":\"%s\",\"control_ms\":", round.order);
    print_samples(round.control);
    std::printf(",\"paged_ms\":");
    print_samples(round.paged);
    std::printf(",\"control_median_ms\":%.9g,"
                "\"paged_median_ms\":%.9g,\"increase_percent\":%.9g}",
                round.control_median, round.paged_median,
                round.increase_percent);
  }
  const bool numeric_ok =
      bitwise && repeat && guards && cleanup_zero && oracle_abs <= 0.03125;
  std::printf("],\"state\":\"%s\"}\n",
              numeric_ok ? performance_ok ? "PASS" : "PERFORMANCE_REVIEW"
                         : "FAIL");
}

void run(const Options &options) {
  int count = 0;
  check(hipGetDeviceCount(&count), "hipGetDeviceCount");
  if (count != 1)
    fail("probe requires exactly one visible GPU");
  hipDeviceProp_t device{};
  check(hipGetDeviceProperties(&device, 0), "hipGetDeviceProperties");
  const std::string actual_target(device.gcnArchName);
  if (actual_target != options.target &&
      actual_target.rfind(options.target + ":", 0U) != 0U)
    fail("actual HIP target differs from the requested exact target");
  const bool gqa_shared = options.target == "gfx1030";
  const uint32_t splits =
      options.kind == Kind::Decode ? options.length >= 8192U ? 128U : 32U : 1U;
  Fixture fixture = make_fixture(options.length, options.m);
  PagedFixture paged = page_fixture(fixture);
  const uint64_t page_count = paged.table.size();
  const size_t contiguous_bytes = fixture.key.size() + fixture.value.size() +
                                  fixture.key_scales.size() +
                                  fixture.value_scales.size();
  const size_t pool_bytes = paged.key.size() + paged.value.size() +
                            paged.key_scales.size() + paged.value_scales.size();
  double oracle_abs = 0.0;
  bool bitwise = false, repeat = false, guards = false, performance_ok = true;
  std::vector<Round> rounds;
  {
    DeviceFixture buffers(fixture, paged, splits);
    launch_variant(buffers, options.kind, false, gqa_shared);
    launch_variant(buffers, options.kind, true, gqa_shared);
    check(hipStreamSynchronize(buffers.stream), "numerical synchronize");
    const auto control = buffers.output_a.download();
    const auto candidate = buffers.output_b.download();
    bitwise = control == candidate;
    oracle_abs = oracle_max_abs(fixture, control);
    guards = output_guard_intact(control, buffers.output_elements) &&
             output_guard_intact(candidate, buffers.output_elements) &&
             workspace_guard_intact(buffers.workspace_a.download(),
                                    buffers.workspace_elements) &&
             workspace_guard_intact(buffers.workspace_b.download(),
                                    buffers.workspace_elements);
    launch_variant(buffers, options.kind, false, gqa_shared);
    launch_variant(buffers, options.kind, true, gqa_shared);
    check(hipStreamSynchronize(buffers.stream), "repeat synchronize");
    repeat = control == buffers.output_a.download() &&
             candidate == buffers.output_b.download();
    if (!bitwise || !repeat || !guards || oracle_abs > 0.03125)
      fail("numerical, repeat, or guard check failed");
    launch_variant(buffers, options.kind, false, gqa_shared);
    launch_variant(buffers, options.kind, true, gqa_shared);
    check(hipStreamSynchronize(buffers.stream), "warmup synchronize");
    for (uint32_t index = 0U; index < options.rounds; ++index) {
      Round round{};
      round.order = index % 2U == 0U ? "AB" : "BA";
      const auto measure = [&](const bool paged_variant) {
        for (uint32_t sample = 0U; sample < options.samples; ++sample) {
          const double measured =
              elapsed_ms(buffers, options.kind, paged_variant, gqa_shared,
                         options.replays);
          (paged_variant ? round.paged : round.control).push_back(measured);
        }
      };
      if (index % 2U == 0U) {
        measure(false);
        measure(true);
      } else {
        measure(true);
        measure(false);
      }
      round.control_median = median(round.control);
      round.paged_median = median(round.paged);
      round.increase_percent =
          (round.paged_median / round.control_median - 1.0) * 100.0;
      performance_ok &= round.increase_percent < 10.0;
      rounds.push_back(std::move(round));
    }
  }
  const bool cleanup_zero = cleanup_ok && live_allocations == 0U;
  report(options, splits, page_count, contiguous_bytes, pool_bytes, oracle_abs,
         bitwise, repeat, guards, performance_ok, cleanup_zero, rounds);
  if (!cleanup_zero)
    fail("GPU allocation or stream cleanup failed");
}

} // namespace

int main(const int argc, char **const argv) {
  try {
    run(parse_options(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "WU-P1 probe failed: %s\n", error.what());
    return 2;
  }
}
