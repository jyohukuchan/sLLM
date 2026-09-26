#include "attention_preprocess_kernel_internal.hpp"
#include "causal_attention_kernel_internal.hpp"
#include "decode_control_kernel_internal.hpp"
#include "kv_state_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
namespace sllm_old_causal_attention_kernel {
hipError_t launch_old_decode_wave_split_staged32(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t encoding, float static_key_scale,
    float static_value_scale, void *workspace, uint64_t workspace_bytes,
    bool use_query_preload, bool use_gqa_shared, bool use_split128,
    hipStream_t stream) noexcept;
} // namespace sllm_old_causal_attention_kernel

namespace sllm_old_kv_state_kernel {
hipError_t launch(const uint16_t *key_input, const uint16_t *value_input,
                  void *key_output, void *value_output, void *key_scales,
                  void *value_scales, float *key_outer_scales,
                  float *value_outer_scales, uint32_t token_count,
                  uint64_t capacity_tokens, uint64_t start_position,
                  uint32_t head_count, uint32_t head_dim, uint32_t encoding,
                  float static_key_scale, float static_value_scale,
                  hipStream_t stream) noexcept;
} // namespace sllm_old_kv_state_kernel
#endif

namespace {
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kScaleBlocks = 8U;
constexpr uint32_t kMaxRows = 3U;
constexpr uint64_t kCapacity = 8200U;
constexpr uint64_t kAttentionTokens = 8193U;
constexpr uint8_t kByteCanary = 0xa5U;

[[noreturn]] void fail(const char *const text) {
  throw std::runtime_error(text);
}
void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
    fail(operation);
  }
}

uint16_t bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}
float bf16_to_float(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
float e4m3fn(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  uint32_t raw = sign;
  if (exponent == 0U) {
    const float subnormal = static_cast<float>(mantissa) * 0x1p-9F;
    std::memcpy(&raw, &subnormal, sizeof(raw));
    raw |= sign;
  } else if (magnitude == 0x7fU) {
    raw = sign | UINT32_C(0x7fc00000);
  } else {
    raw |= ((exponent + 120U) << 23U) | (mantissa << 20U);
  }
  float result = 0.0F;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}
float e8m0(const uint8_t bits) {
  return std::ldexp(1.0F, static_cast<int>(bits) - 127);
}
uint32_t bf16_ulp(const uint16_t lhs, const uint16_t rhs) {
  const auto ordered = [](const uint16_t value) {
    const uint32_t magnitude = value & UINT16_C(0x7fff);
    return (value & UINT16_C(0x8000)) != 0U ? UINT32_C(0x8000) - magnitude
                                            : UINT32_C(0x8000) + value;
  };
  const uint32_t left = ordered(lhs);
  const uint32_t right = ordered(rhs);
  return left >= right ? left - right : right - left;
}
uint8_t pattern(const uint64_t index, const uint64_t salt) {
  return static_cast<uint8_t>((index * 17U + salt * 29U + 7U) & 0xffU);
}

template <typename T> struct Device final {
  T *ptr = nullptr;
  size_t count = 0U;
  explicit Device(const size_t n) : count(n) {
    check(hipMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)),
          "hipMalloc");
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  ~Device() {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
    }
  }
  void fill(const uint8_t byte, hipStream_t stream = nullptr) {
    check(hipMemsetAsync(ptr, byte, count * sizeof(T), stream),
          "hipMemsetAsync");
  }
  std::vector<T> get() const {
    std::vector<T> result(count);
    check(
        hipMemcpy(result.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost),
        "hipMemcpy D2H");
    return result;
  }
};

struct Stream final {
  hipStream_t value = nullptr;
  Stream() { check(hipStreamCreate(&value), "hipStreamCreate"); }
  ~Stream() {
    if (value != nullptr) {
      (void)hipStreamDestroy(value);
    }
  }
};

struct ControlGraph final {
  Device<sllm_decode_control::ControlV1> control;
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  explicit ControlGraph() : control(1U) {}
  ~ControlGraph() {
    if (exec != nullptr) {
      (void)hipGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
      (void)hipGraphDestroy(graph);
    }
  }
};

void set_control(Device<sllm_decode_control::ControlV1> *const control,
                 const uint64_t position, const uint32_t rows,
                 const bool active, const bool halted = false) {
  sllm_decode_control::ControlV1 value{};
  value.version = sllm_decode_control::kVersion;
  value.mode = sllm_decode_control::kModeTargetOnly;
  value.phase_position = position;
  value.phase_rows = rows;
  value.phase_active = active ? 1U : 0U;
  value.halted = halted ? 1U : 0U;
  check(hipMemcpy(control->ptr, &value, sizeof(value), hipMemcpyHostToDevice),
        "upload ControlV1");
}

struct KvBuffers final {
  Device<uint16_t> key_input;
  Device<uint16_t> value_input;
  Device<uint8_t> key;
  Device<uint8_t> value;
  Device<uint8_t> key_scales;
  Device<uint8_t> value_scales;
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
  Device<uint8_t> old_key;
  Device<uint8_t> old_value;
  Device<uint8_t> old_key_scales;
  Device<uint8_t> old_value_scales;
#endif
  Stream stream;
  ControlGraph graph;
  KvBuffers(const uint32_t input_tokens, const uint32_t encoding)
      : key_input(static_cast<size_t>(input_tokens) * kKvHeads * kHeadDim),
        value_input(static_cast<size_t>(input_tokens) * kKvHeads * kHeadDim),
        key(static_cast<size_t>(kCapacity) * kKvHeads * kHeadDim *
            (encoding == SLLM_HIP_KV_ENCODING_FP16_V1 ? sizeof(uint16_t)
                                                      : sizeof(uint8_t))),
        value(static_cast<size_t>(kCapacity) * kKvHeads * kHeadDim *
              (encoding == SLLM_HIP_KV_ENCODING_FP16_V1 ? sizeof(uint16_t)
                                                        : sizeof(uint8_t))),
        key_scales(static_cast<size_t>(kCapacity) * kKvHeads * kScaleBlocks),
        value_scales(static_cast<size_t>(kCapacity) * kKvHeads * kScaleBlocks)
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
        ,
        old_key(key.count), old_value(value.count),
        old_key_scales(key_scales.count), old_value_scales(value_scales.count)
#endif
  {
  }
};

struct AttentionBuffers final {
  Device<uint16_t> query;
  Device<uint16_t> baseline;
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
  Device<uint16_t> old_baseline;
#endif
  Device<uint16_t> graph_output;
  Device<float> workspace;
  Stream stream;
  ControlGraph graph;
  explicit AttentionBuffers(const uint32_t rows)
      : query(static_cast<size_t>(rows) * kQHeads * kHeadDim),
        baseline(static_cast<size_t>(rows) * kQHeads * kHeadDim),
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
        old_baseline(static_cast<size_t>(rows) * kQHeads * kHeadDim),
#endif
        graph_output(static_cast<size_t>(rows) * kQHeads * kHeadDim),
        workspace(static_cast<size_t>(rows) * kQHeads * 128U *
                  (kHeadDim + 2U)) {
  }
};

void fill_input(const KvBuffers &kv, const AttentionBuffers &attention) {
  std::vector<uint16_t> key(kv.key_input.count);
  std::vector<uint16_t> value(kv.value_input.count);
  for (size_t index = 0U; index < key.size(); ++index) {
    const float source =
        0.05F + 0.003F * static_cast<float>(pattern(index, 3U) % 31U);
    key[index] = bf16_rne((index & 7U) == 0U ? -source : source);
    value[index] = bf16_rne(
        0.09F + 0.002F * static_cast<float>(pattern(index, 11U) % 37U));
  }
  check(hipMemcpy(kv.key_input.ptr, key.data(), key.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload key input");
  check(hipMemcpy(kv.value_input.ptr, value.data(),
                  value.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        "upload value input");
  std::vector<uint16_t> query(attention.query.count);
  for (size_t index = 0U; index < query.size(); ++index) {
    const float source =
        0.02F + 0.001F * static_cast<float>(pattern(index, 19U) % 23U);
    query[index] = bf16_rne((index % 13U) == 0U ? -source : source);
  }
  check(hipMemcpy(attention.query.ptr, query.data(),
                  query.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        "upload query");
}

void fill_kv_input(const KvBuffers &kv) {
  std::vector<uint16_t> key(kv.key_input.count);
  std::vector<uint16_t> value(kv.value_input.count);
  for (size_t index = 0U; index < key.size(); ++index) {
    const float source =
        0.05F + 0.003F * static_cast<float>(pattern(index, 3U) % 31U);
    key[index] = bf16_rne((index & 7U) == 0U ? -source : source);
    value[index] = bf16_rne(
        0.09F + 0.002F * static_cast<float>(pattern(index, 11U) % 37U));
  }
  check(hipMemcpy(kv.key_input.ptr, key.data(), key.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload KV key input");
  check(hipMemcpy(kv.value_input.ptr, value.data(),
                  value.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        "upload KV value input");
}

void launch_kv_normal(KvBuffers *const kv, const uint32_t rows,
                      const uint64_t position, const uint32_t encoding) {
  check(sllm_kv_state_kernel::launch(
            kv->key_input.ptr, kv->value_input.ptr, kv->key.ptr, kv->value.ptr,
            kv->key_scales.ptr, kv->value_scales.ptr, nullptr, nullptr, rows,
            kCapacity, position, kKvHeads, kHeadDim, encoding, 1.0F, 1.0F,
            kv->stream.value),
        "normal KV launch");
}

#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
void launch_kv_old(KvBuffers *const kv, const uint32_t rows,
                   const uint64_t position, const uint32_t encoding) {
  check(sllm_old_kv_state_kernel::launch(
            kv->key_input.ptr, kv->value_input.ptr, kv->old_key.ptr,
            kv->old_value.ptr, kv->old_key_scales.ptr, kv->old_value_scales.ptr,
            nullptr, nullptr, rows, kCapacity, position, kKvHeads, kHeadDim,
            encoding, 1.0F, 1.0F, kv->stream.value),
        "old HEAD KV launch");
}
#endif

void capture_kv(KvBuffers *const kv, const uint32_t encoding,
                const uint32_t captured_rows = kMaxRows,
                const uint64_t initial_position = 0U,
                const uint32_t initial_rows = kMaxRows) {
  set_control(&kv->graph.control, initial_position, initial_rows, true);
  check(hipStreamBeginCapture(kv->stream.value, hipStreamCaptureModeGlobal),
        "KV begin capture");
  check(sllm_kv_state_kernel::launch_device(
            kv->key_input.ptr, kv->value_input.ptr, kv->key.ptr, kv->value.ptr,
            kv->key_scales.ptr, kv->value_scales.ptr, nullptr, nullptr,
            captured_rows, kCapacity, kKvHeads, kHeadDim, encoding, 1.0F, 1.0F,
            kv->graph.control.ptr, kv->stream.value),
        "control KV launch");
  check(hipStreamEndCapture(kv->stream.value, &kv->graph.graph),
        "KV end capture");
  check(hipGraphInstantiate(&kv->graph.exec, kv->graph.graph, nullptr, nullptr,
                            0U),
        "KV graph instantiate");
}

void launch_kv_graph(KvBuffers *const kv) {
  check(hipGraphLaunch(kv->graph.exec, kv->stream.value), "KV graph launch");
}

void verify_attention_oracle(const KvBuffers &kv,
                             const AttentionBuffers &attention,
                             const std::vector<uint16_t> &observed) {
  const auto key = kv.key.get();
  const auto value = kv.value.get();
  const auto key_scales = kv.key_scales.get();
  const auto value_scales = kv.value_scales.get();
  const auto query = attention.query.get();
  constexpr uint32_t kOracleHeads = 2U;
  constexpr uint32_t kOracleDimensions = 8U;
  constexpr uint64_t kCommitted = 8192U;
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  for (uint32_t head = 0U; head < kOracleHeads; ++head) {
    const uint32_t kv_head = head / 6U;
    std::vector<float> scores(kCommitted);
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint64_t token = 0U; token < kCommitted; ++token) {
      const uint64_t row = token * kKvHeads + kv_head;
      float score = 0.0F;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const uint64_t state_index = row * kHeadDim + dimension;
        const float decoded =
            e4m3fn(key[state_index]) *
            e8m0(key_scales[row * kScaleBlocks + dimension / 32U]);
        score +=
            bf16_to_float(
                query[static_cast<uint64_t>(head) * kHeadDim + dimension]) *
            decoded;
      }
      scores[token] = score * (1.0F / 16.0F);
      maximum = std::max(maximum, scores[token]);
    }
    float denominator = 0.0F;
    for (float &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (uint32_t dimension = 0U; dimension < kOracleDimensions; ++dimension) {
      float accumulated = 0.0F;
      for (uint64_t token = 0U; token < kCommitted; ++token) {
        const uint64_t row = token * kKvHeads + kv_head;
        const uint64_t state_index = row * kHeadDim + dimension;
        const float decoded =
            e4m3fn(value[state_index]) *
            e8m0(value_scales[row * kScaleBlocks + dimension / 32U]);
        accumulated += (scores[token] / denominator) * decoded;
      }
      const uint16_t expected = bf16_rne(accumulated);
      const uint16_t actual =
          observed[static_cast<size_t>(head * kHeadDim + dimension)];
      max_ulp = std::max(max_ulp, bf16_ulp(expected, actual));
      max_abs = std::max(max_abs,
                         std::abs(static_cast<double>(bf16_to_float(expected)) -
                                  static_cast<double>(bf16_to_float(actual))));
    }
  }
  std::printf("attention_oracle heads=%u dims=%u max_ulp=%u max_abs=%.9g "
              "status=%s\n",
              kOracleHeads, kOracleDimensions, max_ulp, max_abs,
              max_ulp <= 8U && max_abs <= 0.03125 ? "PASS" : "FAIL");
  if (max_ulp > 8U || max_abs > 0.03125) {
    fail("attention numerical oracle");
  }
}

void compare_bytes(const std::vector<uint8_t> &lhs,
                   const std::vector<uint8_t> &rhs, const char *const label) {
  if (lhs != rhs) {
    size_t first = 0U;
    while (first < lhs.size() && lhs[first] == rhs[first]) {
      ++first;
    }
    std::fprintf(stderr, "%s mismatch at byte %zu\n", label, first);
    fail(label);
  }
}

void run_kv_encoding(const uint32_t encoding) {
  KvBuffers kv(kMaxRows, encoding);
  fill_kv_input(kv);
  kv.key.fill(kByteCanary);
  kv.value.fill(kByteCanary);
  kv.key_scales.fill(kByteCanary);
  kv.value_scales.fill(kByteCanary);
  check(hipStreamSynchronize(kv.stream.value), "KV direct clear");
  set_control(&kv.graph.control, 0U, kMaxRows, true);
  check(sllm_kv_state_kernel::launch_device(
            kv.key_input.ptr, kv.value_input.ptr, kv.key.ptr, kv.value.ptr,
            kv.key_scales.ptr, kv.value_scales.ptr, nullptr, nullptr, kMaxRows,
            kCapacity, kKvHeads, kHeadDim, encoding, 1.0F, 1.0F,
            kv.graph.control.ptr, kv.stream.value),
        "KV direct control launch");
  check(hipStreamSynchronize(kv.stream.value), "KV direct synchronize");
  capture_kv(&kv, encoding);
  const std::array<std::pair<uint64_t, uint32_t>, 6> cases = {{
      {8190U, 1U},
      {8191U, 1U},
      {8192U, 1U},
      {8189U, 3U},
      {8190U, 3U},
      {8191U, 3U},
  }};
  for (const auto &[position, rows] : cases) {
    kv.key.fill(kByteCanary);
    kv.value.fill(kByteCanary);
    kv.key_scales.fill(kByteCanary);
    kv.value_scales.fill(kByteCanary);
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    kv.old_key.fill(kByteCanary);
    kv.old_value.fill(kByteCanary);
    kv.old_key_scales.fill(kByteCanary);
    kv.old_value_scales.fill(kByteCanary);
#endif
    check(hipStreamSynchronize(kv.stream.value), "KV graph clear");
    set_control(&kv.graph.control, position, rows, true);
    launch_kv_graph(&kv);
    check(hipStreamSynchronize(kv.stream.value), "KV graph synchronize");
    const auto graph_key = kv.key.get();
    const auto graph_value = kv.value.get();
    const auto graph_key_scales = kv.key_scales.get();
    const auto graph_value_scales = kv.value_scales.get();

    kv.key.fill(kByteCanary);
    kv.value.fill(kByteCanary);
    kv.key_scales.fill(kByteCanary);
    kv.value_scales.fill(kByteCanary);
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    kv.old_key.fill(kByteCanary);
    kv.old_value.fill(kByteCanary);
    kv.old_key_scales.fill(kByteCanary);
    kv.old_value_scales.fill(kByteCanary);
#endif
    check(hipStreamSynchronize(kv.stream.value), "KV baseline clear");
    launch_kv_normal(&kv, rows, position, encoding);
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    launch_kv_old(&kv, rows, position, encoding);
#endif
    check(hipStreamSynchronize(kv.stream.value), "KV baseline synchronize");
    compare_bytes(graph_key, kv.key.get(), "KV key graph/baseline");
    compare_bytes(graph_value, kv.value.get(), "KV value graph/baseline");
    compare_bytes(graph_key_scales, kv.key_scales.get(),
                  "KV key scale graph/baseline");
    compare_bytes(graph_value_scales, kv.value_scales.get(),
                  "KV value scale graph/baseline");
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    compare_bytes(kv.key.get(), kv.old_key.get(), "KV key current/old HEAD");
    compare_bytes(kv.value.get(), kv.old_value.get(),
                  "KV value current/old HEAD");
    compare_bytes(kv.key_scales.get(), kv.old_key_scales.get(),
                  "KV key scale current/old HEAD");
    compare_bytes(kv.value_scales.get(), kv.old_value_scales.get(),
                  "KV value scale current/old HEAD");
#endif
  }
  kv.key.fill(kByteCanary);
  kv.value.fill(kByteCanary);
  kv.key_scales.fill(kByteCanary);
  kv.value_scales.fill(kByteCanary);
  check(hipStreamSynchronize(kv.stream.value), "KV inactive clear");
  set_control(&kv.graph.control, 8191U, 3U, false);
  launch_kv_graph(&kv);
  check(hipStreamSynchronize(kv.stream.value), "KV inactive synchronize");
  compare_bytes(kv.key.get(), std::vector<uint8_t>(kv.key.count, kByteCanary),
                "KV inactive key write");
  set_control(&kv.graph.control, kCapacity - 1U, 3U, true);
  launch_kv_graph(&kv);
  check(hipStreamSynchronize(kv.stream.value), "KV capacity synchronize");
  compare_bytes(kv.key.get(), std::vector<uint8_t>(kv.key.count, kByteCanary),
                "KV capacity key write");
  auto control = kv.graph.control.get().front();
  if (control.halted == 0U) {
    fail("KV capacity did not halt");
  }
  std::printf("kv_control encoding=%u cases=%zu status=PASS\n", encoding,
              cases.size());
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
  std::printf("kv_old_head encoding=%u cases=%zu payload_n0=PASS\n", encoding,
              cases.size());
#endif
}

void run_kv_capture_entry_tail() {
  constexpr uint32_t kCapturedRows = sllm_decode_control::kMaxWidth + 1U;
  constexpr uint32_t kEncoding = SLLM_HIP_KV_ENCODING_FP16_V1;
  KvBuffers kv(kCapturedRows, kEncoding);
  fill_kv_input(kv);
  // The graph is assembled while the live request has only one physical row
  // left, even though the captured launch keeps the configured maximum shape.
  capture_kv(&kv, kEncoding, kCapturedRows, kCapacity - 1U, 1U);

  for (uint32_t active_rows = 1U; active_rows <= kCapturedRows; ++active_rows) {
    const uint64_t position = kCapacity - active_rows;
    kv.key.fill(kByteCanary);
    kv.value.fill(kByteCanary);
    kv.key_scales.fill(kByteCanary);
    kv.value_scales.fill(kByteCanary);
    check(hipStreamSynchronize(kv.stream.value), "KV tail-entry graph clear");
    set_control(&kv.graph.control, position, active_rows, true);
    launch_kv_graph(&kv);
    check(hipStreamSynchronize(kv.stream.value),
          "KV tail-entry graph synchronize");
    const auto graph_key = kv.key.get();
    const auto graph_value = kv.value.get();
    const auto graph_key_scales = kv.key_scales.get();
    const auto graph_value_scales = kv.value_scales.get();
    const auto observed_control = kv.graph.control.get().front();
    if (observed_control.halted != 0U) {
      fail("KV tail-entry valid active prefix halted");
    }

    kv.key.fill(kByteCanary);
    kv.value.fill(kByteCanary);
    kv.key_scales.fill(kByteCanary);
    kv.value_scales.fill(kByteCanary);
    check(hipStreamSynchronize(kv.stream.value),
          "KV tail-entry baseline clear");
    launch_kv_normal(&kv, active_rows, position, kEncoding);
    check(hipStreamSynchronize(kv.stream.value),
          "KV tail-entry baseline synchronize");
    compare_bytes(graph_key, kv.key.get(), "KV tail-entry key graph/baseline");
    compare_bytes(graph_value, kv.value.get(),
                  "KV tail-entry value graph/baseline");
    compare_bytes(graph_key_scales, kv.key_scales.get(),
                  "KV tail-entry key scale graph/baseline");
    compare_bytes(graph_value_scales, kv.value_scales.get(),
                  "KV tail-entry value scale graph/baseline");
  }
  std::printf("kv_capture_entry configured_width=%u remaining_cases=%u "
              "status=PASS\n",
              kCapturedRows - 1U, kCapturedRows);
}

void run_attention_preprocess_capture_tail() {
  constexpr uint32_t kRows = sllm_decode_control::kMaxWidth + 1U;
  constexpr uint64_t kPhysicalCapacity = 22U;
  constexpr uint16_t kCanary = UINT16_C(0xa5a5);
  const size_t packed_count =
      static_cast<size_t>(kRows) * kQHeads * kHeadDim * 2U;
  const size_t key_count = static_cast<size_t>(kRows) * kKvHeads * kHeadDim;
  const size_t query_count = static_cast<size_t>(kRows) * kQHeads * kHeadDim;
  Device<uint16_t> packed(packed_count);
  Device<uint16_t> key(key_count);
  Device<uint16_t> query_scale(static_cast<size_t>(kQHeads) * kHeadDim);
  Device<uint16_t> key_scale(static_cast<size_t>(kKvHeads) * kHeadDim);
  Device<uint16_t> graph_query(query_count);
  Device<uint16_t> graph_gate(query_count);
  Device<uint16_t> graph_key(key_count);
  Device<uint16_t> baseline_query(query_count);
  Device<uint16_t> baseline_gate(query_count);
  Device<uint16_t> baseline_key(key_count);
  Device<sllm_decode_control::ControlV1> control(1U);
  Stream stream;

  std::vector<uint16_t> packed_host(packed_count);
  std::vector<uint16_t> key_host(key_count);
  std::vector<uint16_t> query_scale_host(query_scale.count);
  std::vector<uint16_t> key_scale_host(key_scale.count);
  for (size_t index = 0U; index < packed_host.size(); ++index) {
    packed_host[index] = bf16_rne(
        0.01F + 0.001F * static_cast<float>(pattern(index, 23U) % 31U));
  }
  for (size_t index = 0U; index < key_host.size(); ++index) {
    key_host[index] = bf16_rne(
        0.02F + 0.001F * static_cast<float>(pattern(index, 29U) % 29U));
  }
  for (size_t index = 0U; index < query_scale_host.size(); ++index) {
    query_scale_host[index] =
        bf16_rne(0.005F * static_cast<float>(pattern(index, 31U) % 7U));
  }
  for (size_t index = 0U; index < key_scale_host.size(); ++index) {
    key_scale_host[index] =
        bf16_rne(0.005F * static_cast<float>(pattern(index, 37U) % 7U));
  }
  check(hipMemcpy(packed.ptr, packed_host.data(),
                  packed_host.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        "attention preprocess packed upload");
  check(hipMemcpy(key.ptr, key_host.data(), key_host.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "attention preprocess key upload");
  check(hipMemcpy(query_scale.ptr, query_scale_host.data(),
                  query_scale_host.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "attention preprocess query scale upload");
  check(hipMemcpy(key_scale.ptr, key_scale_host.data(),
                  key_scale_host.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "attention preprocess key scale upload");

  const std::array<std::pair<uint64_t, uint32_t>, 4> cases = {{
      {21U, 1U},
      {20U, 2U},
      {19U, 3U},
      {13U, 9U},
  }};
  for (const auto &[position, active_rows] : cases) {
    graph_query.fill(0xa5U, stream.value);
    graph_gate.fill(0xa5U, stream.value);
    graph_key.fill(0xa5U, stream.value);
    baseline_query.fill(0xa5U, stream.value);
    baseline_gate.fill(0xa5U, stream.value);
    baseline_key.fill(0xa5U, stream.value);
    sllm_decode_control::ControlV1 host_control{};
    host_control.version = sllm_decode_control::kVersion;
    host_control.status =
        static_cast<uint32_t>(sllm_decode_control::Status::Ok);
    host_control.mode = sllm_decode_control::kModeMtp;
    host_control.width = sllm_decode_control::kMaxWidth;
    host_control.active_width = active_rows - 1U;
    host_control.capacity = kPhysicalCapacity;
    host_control.phase_position = position;
    host_control.phase_rows = active_rows;
    host_control.phase_active = 1U;
    check(hipMemcpyAsync(control.ptr, &host_control, sizeof(host_control),
                         hipMemcpyHostToDevice, stream.value),
          "attention preprocess control upload");
    check(sllm_attention_preprocess_kernel::launch(
              packed.ptr, key.ptr, query_scale.ptr, key_scale.ptr, nullptr,
              baseline_query.ptr, baseline_gate.ptr, baseline_key.ptr,
              active_rows, kQHeads, kKvHeads, kHeadDim, 1U,
              static_cast<uint32_t>(position),
              SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1, true,
              stream.value),
          "attention preprocess eager launch");
    check(sllm_attention_preprocess_kernel::launch_device(
              packed.ptr, key.ptr, query_scale.ptr, key_scale.ptr,
              graph_query.ptr, graph_gate.ptr, graph_key.ptr, kRows, kQHeads,
              kKvHeads, control.ptr, stream.value),
          "attention preprocess device launch");
    check(hipStreamSynchronize(stream.value),
          "attention preprocess tail synchronize");
    const auto graph_query_host = graph_query.get();
    const auto graph_gate_host = graph_gate.get();
    const auto graph_key_host = graph_key.get();
    const auto baseline_query_host = baseline_query.get();
    const auto baseline_gate_host = baseline_gate.get();
    const auto baseline_key_host = baseline_key.get();
    const size_t active_query =
        static_cast<size_t>(active_rows) * kQHeads * kHeadDim;
    const size_t active_key =
        static_cast<size_t>(active_rows) * kKvHeads * kHeadDim;
    if (!std::equal(graph_query_host.begin(),
                    graph_query_host.begin() + active_query,
                    baseline_query_host.begin()) ||
        !std::equal(graph_gate_host.begin(),
                    graph_gate_host.begin() + active_query,
                    baseline_gate_host.begin()) ||
        !std::equal(graph_key_host.begin(), graph_key_host.begin() + active_key,
                    baseline_key_host.begin()) ||
        !std::all_of(graph_query_host.begin() + active_query,
                     graph_query_host.end(),
                     [](const uint16_t value) { return value == kCanary; }) ||
        !std::all_of(graph_gate_host.begin() + active_query,
                     graph_gate_host.end(),
                     [](const uint16_t value) { return value == kCanary; }) ||
        !std::all_of(graph_key_host.begin() + active_key, graph_key_host.end(),
                     [](const uint16_t value) { return value == kCanary; })) {
      fail("attention preprocess active-prefix oracle");
    }
    const auto observed = control.get().front();
    if (observed.status !=
            static_cast<uint32_t>(sllm_decode_control::Status::Ok) ||
        observed.halted != 0U) {
      fail("attention preprocess valid tail control");
    }
  }

  graph_query.fill(0xa5U, stream.value);
  graph_gate.fill(0xa5U, stream.value);
  graph_key.fill(0xa5U, stream.value);
  sllm_decode_control::ControlV1 invalid{};
  invalid.version = sllm_decode_control::kVersion;
  invalid.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  invalid.mode = sllm_decode_control::kModeMtp;
  invalid.width = sllm_decode_control::kMaxWidth;
  invalid.capacity = kPhysicalCapacity;
  invalid.phase_position = kPhysicalCapacity;
  invalid.phase_rows = 1U;
  invalid.phase_active = 1U;
  check(hipMemcpyAsync(control.ptr, &invalid, sizeof(invalid),
                       hipMemcpyHostToDevice, stream.value),
        "attention preprocess invalid control upload");
  check(sllm_attention_preprocess_kernel::launch_device(
            packed.ptr, key.ptr, query_scale.ptr, key_scale.ptr,
            graph_query.ptr, graph_gate.ptr, graph_key.ptr, kRows, kQHeads,
            kKvHeads, control.ptr, stream.value),
        "attention preprocess invalid launch");
  check(hipStreamSynchronize(stream.value),
        "attention preprocess invalid synchronize");
  const auto observed = control.get().front();
  const auto invalid_query = graph_query.get();
  const auto invalid_gate = graph_gate.get();
  const auto invalid_key = graph_key.get();
  if (observed.status !=
          static_cast<uint32_t>(sllm_decode_control::Status::InvalidPosition) ||
      observed.halted == 0U || observed.phase_active != 0U ||
      !std::all_of(invalid_query.begin(), invalid_query.end(),
                   [](const uint16_t value) { return value == kCanary; }) ||
      !std::all_of(invalid_gate.begin(), invalid_gate.end(),
                   [](const uint16_t value) { return value == kCanary; }) ||
      !std::all_of(invalid_key.begin(), invalid_key.end(),
                   [](const uint16_t value) { return value == kCanary; })) {
    fail("attention preprocess invalid phase wrote output");
  }
  if (sllm_attention_preprocess_kernel::launch_device(
          packed.ptr, key.ptr, query_scale.ptr, key_scale.ptr, graph_query.ptr,
          graph_gate.ptr, graph_key.ptr, kRows + 1U, kQHeads, kKvHeads,
          control.ptr, stream.value) != hipErrorInvalidValue) {
    fail("attention preprocess accepted more than nine captured rows");
  }
  std::printf("attention_preprocess_tail cases=%zu invalid_guard=PASS "
              "status=PASS\n",
              cases.size());
}

void capture_attention(AttentionBuffers *const attention, const KvBuffers &kv,
                       const bool gqa_shared,
                       const uint32_t captured_rows = kMaxRows) {
  set_control(&attention->graph.control, 0U, captured_rows, true);
  check(hipStreamBeginCapture(attention->stream.value,
                              hipStreamCaptureModeGlobal),
        "attention begin capture");
  check(sllm_causal_attention_kernel::launch_decode_wave_split_staged32_device(
            attention->query.ptr, kv.key.ptr, kv.value.ptr, kv.key_scales.ptr,
            kv.value_scales.ptr, nullptr, nullptr, attention->graph_output.ptr,
            captured_rows, 0U, captured_rows, kQHeads, kKvHeads, kHeadDim,
            SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 1.0F, 1.0F,
            attention->workspace.ptr,
            attention->workspace.count * sizeof(float), true, gqa_shared,
            attention->graph.control.ptr, attention->stream.value),
        "control attention launch");
  check(hipStreamEndCapture(attention->stream.value, &attention->graph.graph),
        "attention end capture");
  check(hipGraphInstantiate(&attention->graph.exec, attention->graph.graph,
                            nullptr, nullptr, 0U),
        "attention graph instantiate");
}

void run_attention(const bool gqa_shared) {
  KvBuffers kv(static_cast<uint32_t>(kAttentionTokens),
               SLLM_HIP_KV_ENCODING_MXFP8_E4_V1);
  AttentionBuffers attention(kMaxRows);
  fill_input(kv, attention);
  kv.key.fill(0U);
  kv.value.fill(0U);
  kv.key_scales.fill(0U);
  kv.value_scales.fill(0U);
  check(hipStreamSynchronize(kv.stream.value), "attention state clear");
  launch_kv_normal(&kv, static_cast<uint32_t>(kAttentionTokens), 0U,
                   SLLM_HIP_KV_ENCODING_MXFP8_E4_V1);
  check(hipStreamSynchronize(kv.stream.value), "attention state append");
  capture_attention(&attention, kv, gqa_shared);

  const std::array<std::pair<uint64_t, uint32_t>, 6> cases = {{
      {127U, 1U},
      {8190U, 1U},
      {8191U, 1U},
      {8192U, 1U},
      {8190U, 3U},
      {8191U, 3U},
  }};
  std::vector<uint16_t> oracle_output;
  for (const auto &[position, rows] : cases) {
    attention.graph_output.fill(0U, attention.stream.value);
    attention.baseline.fill(0U, attention.stream.value);
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    attention.old_baseline.fill(0U, attention.stream.value);
#endif
    check(hipStreamSynchronize(attention.stream.value), "attention clear");
    set_control(&attention.graph.control, position, rows, true);
    // The attention graph owns its own stream and graph handle.
    check(hipGraphLaunch(attention.graph.exec, attention.stream.value),
          "attention graph launch");
    check(hipStreamSynchronize(attention.stream.value),
          "attention graph synchronize");
    const auto graph = attention.graph_output.get();
    const bool split128 = position + rows >= 8192U;
    check(sllm_causal_attention_kernel::launch_decode_wave_split_staged32(
              attention.query.ptr, kv.key.ptr, kv.value.ptr, kv.key_scales.ptr,
              kv.value_scales.ptr, nullptr, nullptr, attention.baseline.ptr,
              rows, position, position + rows, kQHeads, kKvHeads, kHeadDim,
              SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 1.0F, 1.0F,
              attention.workspace.ptr,
              attention.workspace.count * sizeof(float), true, gqa_shared,
              split128, attention.stream.value),
          "attention baseline launch");
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    check(
        sllm_old_causal_attention_kernel::launch_old_decode_wave_split_staged32(
            attention.query.ptr, kv.key.ptr, kv.value.ptr, kv.key_scales.ptr,
            kv.value_scales.ptr, nullptr, nullptr, attention.old_baseline.ptr,
            rows, position, position + rows, kQHeads, kKvHeads, kHeadDim,
            SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 1.0F, 1.0F,
            attention.workspace.ptr, attention.workspace.count * sizeof(float),
            true, gqa_shared, split128, attention.stream.value),
        "attention old HEAD baseline launch");
#endif
    check(hipStreamSynchronize(attention.stream.value),
          "attention baseline synchronize");
    if (graph != attention.baseline.get()) {
      fail("attention graph/baseline output mismatch");
    }
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    if (attention.baseline.get() != attention.old_baseline.get()) {
      fail("attention current/old HEAD baseline mismatch");
    }
#endif
    if (position == 8191U && rows == 1U) {
      oracle_output = graph;
    }
  }
  verify_attention_oracle(kv, attention, oracle_output);
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
  std::printf(
      "attention_old_head gqa_shared=%s cases=%zu full_output_n0=PASS\n",
      gqa_shared ? "true" : "false", cases.size());
#endif
  std::printf("attention_control gqa_shared=%s cases=%zu status=PASS\n",
              gqa_shared ? "true" : "false", cases.size());
}

void run_short_attention_n0(const bool gqa_shared,
                            const bool gfx1201_wave_provider,
                            const uint32_t captured_rows) {
  constexpr uint32_t kShortTokens = 64U;
  KvBuffers kv(kShortTokens, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1);
  AttentionBuffers attention(captured_rows);
  fill_input(kv, attention);
  kv.key.fill(0U);
  kv.value.fill(0U);
  kv.key_scales.fill(0U);
  kv.value_scales.fill(0U);
  check(hipStreamSynchronize(kv.stream.value), "short attention state clear");
  launch_kv_normal(&kv, kShortTokens, 0U, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1);
  check(hipStreamSynchronize(kv.stream.value), "short attention state append");
  capture_attention(&attention, kv, gqa_shared, captured_rows);

  for (uint32_t rows = 1U; rows <= captured_rows; ++rows) {
    const uint64_t position = static_cast<uint64_t>(rows) * 3U;
    attention.graph_output.fill(0U, attention.stream.value);
    attention.baseline.fill(0U, attention.stream.value);
    check(hipStreamSynchronize(attention.stream.value),
          "short attention output clear");
    set_control(&attention.graph.control, position, rows, true);
    check(hipGraphLaunch(attention.graph.exec, attention.stream.value),
          "short attention graph launch");
    check(sllm_causal_attention_kernel::launch(
              attention.query.ptr, kv.key.ptr, kv.value.ptr, kv.key_scales.ptr,
              kv.value_scales.ptr, nullptr, nullptr, attention.baseline.ptr,
              rows, kCapacity, position, position + rows, kQHeads, kKvHeads,
              kHeadDim, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 1.0F, 1.0F,
              gfx1201_wave_provider, false, false, false, false, false, 0U,
              1.0F / 16.0F, attention.stream.value),
          "short attention eager baseline launch");
    check(hipStreamSynchronize(attention.stream.value),
          "short attention synchronize");
    const auto graph = attention.graph_output.get();
    const auto baseline = attention.baseline.get();
    const size_t active_elements =
        static_cast<size_t>(rows) * kQHeads * kHeadDim;
    if (!std::equal(graph.begin(), graph.begin() + active_elements,
                    baseline.begin())) {
      std::fprintf(stderr,
                   "short attention N0 mismatch position=%llu rows=%u\n",
                   static_cast<unsigned long long>(position), rows);
      fail("short attention graph/eager N0");
    }
  }
  std::printf("attention_short_n0 gqa_shared=%s cases=%u status=PASS\n",
              gqa_shared ? "true" : "false", captured_rows);
}

} // namespace

int main(const int argc, char **const argv) {
  try {
    const std::string_view target = argc > 1 ? argv[1] : "gfx1030";
    const bool gqa_shared = target == "gfx1030";
#if defined(SLLM_STAGE5_HAS_OLD_HEAD)
    if (argc < 3 || std::string_view(argv[2]) != "--old-head") {
      fail("old HEAD build requires --old-head");
    }
#endif
    run_kv_encoding(SLLM_HIP_KV_ENCODING_FP16_V1);
    run_kv_encoding(SLLM_HIP_KV_ENCODING_MXFP8_E4_V1);
    run_kv_capture_entry_tail();
    run_attention_preprocess_capture_tail();
    run_attention(gqa_shared);
    run_short_attention_n0(gqa_shared, target == "gfx1201", kMaxRows);
    run_short_attention_n0(false, target == "gfx1201",
                           sllm_decode_control::kMaxWidth + 1U);
    std::puts("phase87_stage5_device_control_gpu_test: PASS");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87_stage5_device_control_gpu_test: %s\n",
                 error.what());
    return 1;
  }
}
// Historical Stage 5 device-control oracle. The token-major and contiguous
// provider calls are retired from current Stage 10 CTest; keep this source for
// the archived control evidence only.
