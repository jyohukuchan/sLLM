// Phase 83 MXFP8 GQA6 prefill public provider oracle.
//
// This probe compares the default packed/QTILE4/QTILE8-W16 causal-attention
// routes with an explicit GQA6 QTILE4 control at the production Qwen3.8
// geometry. It exercises the query-count boundary and non-zero prefix
// positions, including the gfx1030/gfx1201 QTILE8/W16 start-position boundary,
// through the public KV/attention API. The host oracle independently applies
// OCP MXFP8-E4M3 block-32 quantization, decodes the values, performs FP32
// causal softmax/value accumulation, and rounds the result to BF16.
//
// The probe is evidence only and does not change the production selector. Both
// provider results are required to stay within the existing Phase 83 oracle
// tolerance; provider-to-provider differences are checked separately.

#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

namespace {

constexpr uint32_t kQueryHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kGqaRatio = 6U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kBlockSize = 32U;
constexpr uint32_t kBlocksPerRow = kHeadDim / kBlockSize;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr uint32_t kTimeoutMs = 30'000U;
constexpr uint32_t kOracleMaxUlp = 4U;
constexpr float kOracleMaxAbs = 0.03125F;
constexpr float kOracleMaxRelative = 0.04F;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect(const sllm_status_t actual, const sllm_status_t expected,
            const char *const operation, const Error &error) {
  if (actual == expected)
    return true;
  std::fprintf(stderr, "%s returned %u, expected %u: %s\n", operation,
               static_cast<unsigned>(actual), static_cast<unsigned>(expected),
               error.message);
  return false;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::fprintf(stderr, "%s returned no completion\n", operation);
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(sllm_completion_wait(*completion, kTimeoutMs,
                                                  &result, &error.sink),
                             SLLM_STATUS_OK, operation, error) &&
                      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released = expect(sllm_completion_release(completion, &error.sink),
                               SLLM_STATUS_OK, "completion release", error) &&
                        *completion == nullptr;
  return waited && released;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const source, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer upload", error))
    return false;
  return wait_release(&completion, "buffer upload wait");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  const sllm_status_t copy_status =
      sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion, &error.sink);
  if (!expect(copy_status, SLLM_STATUS_OK, "buffer download", error) ||
      completion == nullptr)
    return false;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, kTimeoutMs, &result, &error.sink),
          SLLM_STATUS_OK, "buffer download wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  const bool read = expect(sllm_completion_read(completion, destination, bytes,
                                                &written, &error.sink),
                           SLLM_STATUS_OK, "buffer download read", error) &&
                    written == bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "buffer download completion release", error) &&
      completion == nullptr;
  return read && released;
}

bool make_buffer(const sllm_context_t *const context, const uint64_t bytes,
                 sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, output, &error.sink),
                SLLM_STATUS_OK, "buffer create", error) &&
         *output != nullptr;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr)
    return true;
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "buffer release", error) &&
         *buffer == nullptr;
}

bool release_state(sllm_kv_state_t **const state) {
  if (state == nullptr || *state == nullptr)
    return true;
  Error error;
  return expect(sllm_kv_state_release(state, &error.sink), SLLM_STATUS_OK,
                "KV state release", error) &&
         *state == nullptr;
}

bool create_context(sllm_context_t **const context,
                    sllm_queue_t **const queue) {
  Error error;
  uint32_t device_count = 0U;
  if (!expect(sllm_device_count(&device_count, &error.sink), SLLM_STATUS_OK,
              "device count", error) ||
      device_count != 1U) {
    std::fprintf(stderr, "expected one visible GPU, got %u\n", device_count);
    return false;
  }
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "device query", error) ||
      std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::fprintf(stderr, "visible target is not %s\n",
                 SLLM_TEST_EXPECTED_TARGET);
    return false;
  }
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  if (!expect(sllm_context_create(&context_info, context, &error.sink),
              SLLM_STATUS_OK, "context create", error) ||
      *context == nullptr)
    return false;
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  return expect(sllm_queue_create(*context, &queue_info, queue, &error.sink),
                SLLM_STATUS_OK, "queue create", error) &&
         *queue != nullptr;
}

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  bits += UINT32_C(0x7fff) + (upper & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint8_t mx_scale_code(const float maximum) {
  if (!(maximum > 0.0F) || !std::isfinite(maximum))
    return UINT8_C(127);
  const int exponent = std::clamp(std::ilogb(maximum) - 8, -127, 127);
  return static_cast<uint8_t>(exponent + 127);
}

uint8_t e4m3_encode(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : UINT8_C(0);
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

float e4m3_decode(const uint8_t value) {
  const float sign = (value & UINT8_C(0x80)) != 0U ? -1.0F : 1.0F;
  const uint32_t magnitude = value & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT32_C(7);
  if (exponent == 0U)
    return sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

struct QuantizedRow final {
  std::array<uint8_t, kHeadDim> values{};
  std::array<uint8_t, kBlocksPerRow> scales{};
};

QuantizedRow quantize_row(const std::vector<uint16_t> &source,
                          const uint64_t token, const uint32_t head) {
  QuantizedRow result{};
  const uint64_t row =
      (token * kKvHeads + head) * static_cast<uint64_t>(kHeadDim);
  for (uint32_t block = 0U; block != kBlocksPerRow; ++block) {
    float maximum = 0.0F;
    for (uint32_t lane = 0U; lane != kBlockSize; ++lane)
      maximum = std::max(
          maximum, std::fabs(bf16_to_f32(source[row + block * 32U + lane])));
    const uint8_t scale = mx_scale_code(maximum);
    result.scales[block] = scale;
    const float scale_value = std::ldexp(1.0F, static_cast<int>(scale) - 127);
    for (uint32_t lane = 0U; lane != kBlockSize; ++lane) {
      const float normalized =
          bf16_to_f32(source[row + block * 32U + lane]) / scale_value;
      result.values[block * 32U + lane] = e4m3_encode(normalized);
    }
  }
  return result;
}

float decoded(const QuantizedRow &row, const uint32_t dimension) {
  return e4m3_decode(row.values[dimension]) *
         std::ldexp(1.0F,
                    static_cast<int>(row.scales[dimension / kBlockSize]) - 127);
}

std::vector<uint16_t> make_kv(const uint64_t tokens, const bool value_plane) {
  std::vector<uint16_t> result(
      static_cast<size_t>(tokens * kKvHeads * static_cast<uint64_t>(kHeadDim)));
  for (uint64_t token = 0U; token != tokens; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const uint32_t block = dimension / kBlockSize;
        float magnitude =
            (value_plane ? 0.21F : 0.29F) +
            (value_plane ? 0.071F : 0.093F) * static_cast<float>(block) +
            (value_plane ? 0.019F : 0.023F) * static_cast<float>(head) +
            (value_plane ? 0.008F : 0.009F) * static_cast<float>(token % 11U) +
            (value_plane ? 0.0019F : 0.0023F) *
                static_cast<float>(dimension % 13U);
        if ((token + head + dimension) % 5U == 0U)
          magnitude = -magnitude;
        result[static_cast<size_t>((token * kKvHeads + head) * kHeadDim +
                                   dimension)] = f32_to_bf16(magnitude);
      }
    }
  }
  return result;
}

std::vector<uint16_t> make_query_with_row_offset(const uint32_t rows,
                                                 const uint64_t prefix,
                                                 const uint32_t row_offset) {
  std::vector<uint16_t> result(static_cast<size_t>(
      rows * kQueryHeads * static_cast<uint64_t>(kHeadDim)));
  for (uint32_t row = 0U; row != rows; ++row) {
    const uint32_t absolute_row = row_offset + row;
    for (uint32_t head = 0U; head != kQueryHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float value = 0.032F +
                      0.0021F * static_cast<float>(dimension / kBlockSize) +
                      0.0011F * static_cast<float>(head) +
                      0.00037F * static_cast<float>(dimension % 17U) +
                      0.0009F * static_cast<float>((prefix + row) % 7U);
        if ((absolute_row + head + dimension) % 7U == 0U)
          value = -value;
        result[static_cast<size_t>(
            (static_cast<uint64_t>(row) * kQueryHeads + head) * kHeadDim +
            dimension)] = f32_to_bf16(value);
      }
    }
  }
  return result;
}

std::vector<uint16_t> make_query(const uint32_t rows, const uint64_t prefix) {
  return make_query_with_row_offset(rows, prefix, 0U);
}

std::vector<uint16_t> oracle(const std::vector<uint16_t> &key,
                             const std::vector<uint16_t> &value,
                             const std::vector<uint16_t> &query,
                             const uint64_t prefix, const uint32_t rows) {
  const uint64_t context = prefix + rows;
  std::vector<QuantizedRow> key_rows;
  std::vector<QuantizedRow> value_rows;
  key_rows.reserve(static_cast<size_t>(context * kKvHeads));
  value_rows.reserve(static_cast<size_t>(context * kKvHeads));
  for (uint64_t token = 0U; token != context; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      key_rows.push_back(quantize_row(key, token, head));
      value_rows.push_back(quantize_row(value, token, head));
    }
  }
  std::vector<uint16_t> result(static_cast<size_t>(
      rows * kQueryHeads * static_cast<uint64_t>(kHeadDim)));
  for (uint32_t row = 0U; row != rows; ++row) {
    const uint64_t last_key = prefix + row;
    for (uint32_t qhead = 0U; qhead != kQueryHeads; ++qhead) {
      const uint32_t kv_head = qhead / kGqaRatio;
      std::vector<float> scores(static_cast<size_t>(last_key + 1U));
      float maximum = -std::numeric_limits<float>::infinity();
      const uint16_t *const qrow =
          query.data() + static_cast<size_t>(row) * kQueryHeads * kHeadDim +
          static_cast<size_t>(qhead) * kHeadDim;
      for (uint64_t token = 0U; token <= last_key; ++token) {
        float score = 0.0F;
        const QuantizedRow &krow =
            key_rows[static_cast<size_t>(token * kKvHeads + kv_head)];
        for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension)
          score = std::fmaf(bf16_to_f32(qrow[dimension]),
                            decoded(krow, dimension), score);
        score *= kAttentionScale;
        scores[static_cast<size_t>(token)] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float accumulated = 0.0F;
        for (uint64_t token = 0U; token <= last_key; ++token) {
          const QuantizedRow &vrow =
              value_rows[static_cast<size_t>(token * kKvHeads + kv_head)];
          accumulated =
              std::fmaf(scores[static_cast<size_t>(token)] / denominator,
                        decoded(vrow, dimension), accumulated);
        }
        result[static_cast<size_t>(
            (static_cast<uint64_t>(row) * kQueryHeads + qhead) * kHeadDim +
            dimension)] = f32_to_bf16(accumulated);
      }
    }
  }
  return result;
}

uint32_t bf16_ulp(const uint16_t left, const uint16_t right) {
  const auto ordered = [](const uint16_t value) {
    const uint32_t magnitude = value & UINT16_C(0x7fff);
    return (value & UINT16_C(0x8000)) != 0U ? UINT32_C(0x8000) - magnitude
                                            : UINT32_C(0x8000) + value;
  };
  const uint32_t a = ordered(left);
  const uint32_t b = ordered(right);
  return a >= b ? a - b : b - a;
}

struct Metrics final {
  uint32_t max_ulp = 0U;
  float max_abs = 0.0F;
  float max_relative = 0.0F;
  uint64_t over_tolerance = 0U;
};

struct DispatchMetadata final {
  uint32_t dispatch_count = 0U;
  uint32_t workgroup_size = 0U;
  uint32_t grid_size = 0U;
  uint64_t query_count = 0U;
  uint64_t start_position = 0U;
  uint64_t committed_kv_length = 0U;
  uint32_t fallback_allowed = 0U;
  uint32_t fallback_used = 0U;
  std::string logical_symbol;
  std::string device_symbol;
  std::string arch_name;
};

Metrics compare(const std::vector<uint16_t> &expected,
                const std::vector<uint16_t> &actual) {
  Metrics result{};
  if (expected.size() != actual.size()) {
    result.over_tolerance = 1U;
    return result;
  }
  for (size_t index = 0U; index != expected.size(); ++index) {
    const float wanted = bf16_to_f32(expected[index]);
    const float observed = bf16_to_f32(actual[index]);
    const float absolute = std::fabs(observed - wanted);
    result.max_ulp =
        std::max(result.max_ulp, bf16_ulp(expected[index], actual[index]));
    result.max_abs = std::max(result.max_abs, absolute);
    result.max_relative = std::max(
        result.max_relative, absolute / std::max(std::fabs(wanted), 1.0e-6F));
    if (!std::isfinite(observed) || !std::isfinite(wanted) ||
        result.max_ulp > kOracleMaxUlp || result.max_abs > kOracleMaxAbs ||
        result.max_relative > kOracleMaxRelative)
      ++result.over_tolerance;
  }
  return result;
}

struct Environment final {
  static constexpr std::array<const char *, 9> kNames = {
      "SLLM_CAUSAL_ATTENTION_FORCE_BASELINE",
      "SLLM_CAUSAL_ATTENTION_GFX1030_SCALED_PREFILL_GEMM",
      "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1030",
      "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1030_ROCBLAS_F32",
      "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4",
      "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K4_FP16",
      "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K8_FP16",
      "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K16_FP16",
      "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K32_FP16"};
  std::array<bool, kNames.size()> present{};
  std::array<std::string, kNames.size()> values{};

  Environment() {
    for (size_t index = 0U; index != kNames.size(); ++index) {
      const char *const value = std::getenv(kNames[index]);
      present[index] = value != nullptr;
      if (value != nullptr)
        values[index] = value;
    }
    set(false);
  }

  ~Environment() {
    for (size_t index = 0U; index != kNames.size(); ++index) {
      if (present[index])
        (void)setenv(kNames[index], values[index].c_str(), 1);
      else
        (void)unsetenv(kNames[index]);
    }
  }

  static void set(const bool qtile4) {
    (void)setenv(kNames[0], "0", 1);
    (void)setenv(kNames[1], "0", 1);
    (void)setenv(kNames[2], "0", 1);
    (void)setenv(kNames[3], "0", 1);
    // The default Q8/W16 provider is selected only when QTILE4 is absent.
    // Keep the explicit QTILE4 control at "1" and remove the variable for
    // the default route; "0" would still disable Q8/W16.
    if (qtile4)
      (void)setenv(kNames[4], "1", 1);
    else
      (void)unsetenv(kNames[4]);
    (void)setenv(kNames[5], "0", 1);
    (void)setenv(kNames[6], "0", 1);
    (void)setenv(kNames[7], "0", 1);
    (void)setenv(kNames[8], "0", 1);
  }
};

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint64_t rows,
                              const uint32_t columns,
                              const uint64_t offset = 0U) {
  const uint64_t shape[] = {rows, columns, kHeadDim};
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.byte_offset = offset;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = 3U;
  result.shape[0] = shape[0];
  result.shape[1] = shape[1];
  result.shape[2] = shape[2];
  result.stride_elements[2] = 1U;
  result.stride_elements[1] = kHeadDim;
  result.stride_elements[0] = static_cast<uint64_t>(columns) * kHeadDim;
  return result;
}

bool execute_attention(const sllm_context_t *const context,
                       const sllm_queue_t *const queue,
                       const sllm_kv_state_t *const state,
                       const sllm_buffer_t *const query,
                       const sllm_buffer_t *const output, const uint64_t prefix,
                       const uint64_t context_tokens, const uint32_t rows,
                       std::vector<uint16_t> *const result,
                       DispatchMetadata *const dispatch) {
  sllm_causal_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
  descriptor.start_position = prefix;
  descriptor.expected_kv_length = context_tokens;
  descriptor.kv_state = state;
  descriptor.query = binding(query, SLLM_TENSOR_DTYPE_BF16, rows, kQueryHeads);
  descriptor.output =
      binding(output, SLLM_TENSOR_DTYPE_BF16, rows, kQueryHeads);
  sllm_causal_attention_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                            &completion, &info, &error.sink),
              SLLM_STATUS_OK, "MXFP8 prefill", error) ||
      completion == nullptr || info.dispatch_count != 1U ||
      info.fallback_allowed != 0U || info.fallback_used != 0U ||
      !wait_release(&completion, "MXFP8 prefill wait"))
    return false;
  dispatch->dispatch_count = info.dispatch_count;
  dispatch->workgroup_size = info.workgroup_size_x;
  dispatch->grid_size = info.grid_size_x;
  dispatch->query_count = info.query_count;
  dispatch->start_position = info.start_position;
  dispatch->committed_kv_length = info.committed_kv_length;
  dispatch->fallback_allowed = info.fallback_allowed;
  dispatch->fallback_used = info.fallback_used;
  dispatch->logical_symbol.assign(info.kernel_symbol);
  dispatch->device_symbol.assign(info.device_symbol);
  dispatch->arch_name.assign(info.gcn_arch_name);
  result->resize(static_cast<size_t>(rows * kQueryHeads *
                                     static_cast<uint64_t>(kHeadDim)));
  return download(queue, output, result->data(),
                  static_cast<uint64_t>(result->size()) * sizeof(uint16_t));
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t prefix,
              const uint32_t rows, const bool qtile4,
              const std::vector<uint16_t> &key,
              const std::vector<uint16_t> &value,
              const std::vector<uint16_t> &query,
              const std::vector<uint16_t> &expected, Metrics *const metrics,
              DispatchMetadata *const dispatch,
              std::vector<uint16_t> *const actual_output) {
  const uint64_t context_tokens = prefix + rows;
  const uint64_t kv_bytes =
      context_tokens * kKvHeads * kHeadDim * sizeof(uint16_t);
  const uint64_t query_bytes =
      static_cast<uint64_t>(rows) * kQueryHeads * kHeadDim * sizeof(uint16_t);
  std::array<sllm_buffer_t *, 4> buffers{};
  sllm_kv_state_t *state = nullptr;
  Error error;
  bool ok = make_buffer(context, kv_bytes, &buffers[0]) &&
            make_buffer(context, kv_bytes, &buffers[1]) &&
            make_buffer(context, query_bytes, &buffers[2]) &&
            make_buffer(context, query_bytes, &buffers[3]) &&
            upload(queue, buffers[0], key.data(), kv_bytes) &&
            upload(queue, buffers[1], value.data(), kv_bytes) &&
            upload(queue, buffers[2], query.data(), query_bytes);
  if (ok) {
    sllm_kv_state_create_info_v2_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
    create.session_id = UINT64_C(0x8303) + prefix + rows;
    create.layer_id = 83U;
    create.capacity_tokens = context_tokens;
    create.head_count = kKvHeads;
    create.head_dim = kHeadDim;
    create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    create.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
    create.block_size = kBlockSize;
    create.scale_dtype = SLLM_TENSOR_DTYPE_U8;
    ok = expect(sllm_kv_state_create_v2(context, &create, &state, &error.sink),
                SLLM_STATUS_OK, "MXFP8 state create", error) &&
         state != nullptr;
  }
  if (ok) {
    sllm_kv_append_desc_t append{};
    append.struct_size = sizeof(append);
    append.abi_version = SLLM_HIP_ABI_VERSION;
    append.append_version = SLLM_HIP_KV_STATE_VERSION;
    append.expected_length = 0U;
    append.start_position = 0U;
    append.key_input =
        binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, context_tokens, kKvHeads);
    append.value_input =
        binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, context_tokens, kKvHeads);
    sllm_completion_t *append_completion = nullptr;
    sllm_kv_append_info_t append_info{};
    append_info.struct_size = sizeof(append_info);
    append_info.abi_version = SLLM_HIP_ABI_VERSION;
    append_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
    ok = expect(sllm_kv_state_append(state, queue, &append, &append_completion,
                                     &append_info, &error.sink),
                SLLM_STATUS_OK, "MXFP8 prefill append", error) &&
         append_completion != nullptr &&
         wait_release(&append_completion, "MXFP8 prefill append wait");
    if (append_completion != nullptr)
      (void)sllm_completion_release(&append_completion, &error.sink);
  }
  std::vector<uint16_t> actual;
  if (ok) {
    Environment::set(qtile4);
    ok = execute_attention(context, queue, state, buffers[2], buffers[3],
                           prefix, context_tokens, rows, &actual, dispatch);
    if (ok) {
      *metrics = compare(expected, actual);
      if (actual_output != nullptr)
        *actual_output = actual;
      std::printf(
          "oracle provider=%s prefix=%llu query_count=%u max_bf16_ulp=%u "
          "max_abs=%g max_relative=%g over_tolerance=%llu\n",
          qtile4 ? "qtile4" : "default",
          static_cast<unsigned long long>(prefix), rows, metrics->max_ulp,
          metrics->max_abs, metrics->max_relative,
          static_cast<unsigned long long>(metrics->over_tolerance));
    }
  }
  ok = release_state(&state) && ok;
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend(); ++iterator)
    ok = release_buffer(&*iterator) && ok;
  return ok;
}

bool dispatch_matches(const uint64_t prefix, const uint32_t rows,
                      const bool qtile4, const DispatchMetadata &actual) {
  const bool target_has_qtile8 =
      std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0 ||
      std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") == 0;
  const bool expected_qtile8 =
      !qtile4 && target_has_qtile8 && rows >= 128U && prefix >= 1024U;
  const bool expected_qtile4 = rows >= 128U && !expected_qtile8;
  const bool target_has_gfx1201_packed =
      std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") == 0;
  const bool expected_gfx1201_wave =
      target_has_gfx1201_packed && (rows == 1U || rows >= 32U);
  const char *const expected_logical =
      expected_qtile8   ? "causal_attention.prefill.gqa6_qtile8_w16.mxfp8.v1"
      : expected_qtile4 ? "causal_attention.prefill.gqa6_qtile4.v1"
      : expected_gfx1201_wave
          ? "causal_attention.online_softmax_gqa.packed_kv.gfx1201_wave.v4"
          : "causal_attention.online_softmax_gqa.packed_kv.v3";
  const char *const expected_device =
      expected_qtile8 ? "sllm_causal_attention_prefill_gqa6_qtile8_w16_mxfp8_v1"
      : expected_qtile4 ? "sllm_causal_attention_prefill_gqa6_qtile4_v1"
      : expected_gfx1201_wave
          ? "sllm_causal_attention_packed_gfx1201_wave_v4"
          : "sllm_causal_attention_online_softmax_gqa_packed_kv_v3";
  const uint32_t expected_workgroup = expected_qtile8 ? 512U : 256U;
  const uint32_t expected_grid =
      expected_qtile8
          ? static_cast<uint32_t>((static_cast<uint64_t>(rows) + 7U) / 8U) *
                kKvHeads
      : expected_qtile4
          ? static_cast<uint32_t>((static_cast<uint64_t>(rows) + 3U) / 4U) *
                kKvHeads
          : rows * kQueryHeads;
  const bool matches =
      actual.dispatch_count == 1U &&
      actual.workgroup_size == expected_workgroup &&
      actual.grid_size == expected_grid && actual.query_count == rows &&
      actual.start_position == prefix &&
      actual.committed_kv_length == prefix + rows &&
      actual.fallback_allowed == 0U && actual.fallback_used == 0U &&
      actual.logical_symbol == expected_logical &&
      actual.device_symbol == expected_device &&
      actual.arch_name == SLLM_TEST_EXPECTED_TARGET;
  if (!matches) {
    std::fprintf(
        stderr,
        "dispatch metadata mismatch prefix=%llu query_count=%u provider=%s "
        "expected=(logical=%s device=%s workgroup=%u grid=%u) "
        "actual=(logical=%s device=%s workgroup=%u grid=%u start=%llu "
        "committed=%llu arch=%s fallback=%u/%u)\n",
        static_cast<unsigned long long>(prefix), rows,
        qtile4 ? "qtile4" : "default", expected_logical, expected_device,
        expected_workgroup, expected_grid, actual.logical_symbol.c_str(),
        actual.device_symbol.c_str(), actual.workgroup_size, actual.grid_size,
        static_cast<unsigned long long>(actual.start_position),
        static_cast<unsigned long long>(actual.committed_kv_length),
        actual.arch_name.c_str(), actual.fallback_allowed,
        actual.fallback_used);
  }
  return matches;
}

bool run_decode_block_parity(const sllm_context_t *const context,
                             const sllm_queue_t *const queue,
                             const uint64_t prefix) {
  constexpr uint32_t kBlockRows = 3U;
  const uint64_t block_tokens = prefix + kBlockRows;
  const std::vector<uint16_t> block_key = make_kv(block_tokens, false);
  const std::vector<uint16_t> block_value = make_kv(block_tokens, true);
  const std::vector<uint16_t> block_query = make_query(kBlockRows, prefix);
  const std::vector<uint16_t> block_expected =
      oracle(block_key, block_value, block_query, prefix, kBlockRows);
  Metrics block_metrics{};
  DispatchMetadata block_dispatch{};
  std::vector<uint16_t> block_actual;
  bool success = run_case(context, queue, prefix, kBlockRows, false, block_key,
                          block_value, block_query, block_expected,
                          &block_metrics, &block_dispatch, &block_actual) &&
                 dispatch_matches(prefix, kBlockRows, false, block_dispatch);
  std::printf("decode_block_oracle prefix=%llu rows=%u max_bf16_ulp=%u "
              "max_abs=%g max_relative=%g over_tolerance=%llu\n",
              static_cast<unsigned long long>(prefix), kBlockRows,
              block_metrics.max_ulp, block_metrics.max_abs,
              block_metrics.max_relative,
              static_cast<unsigned long long>(block_metrics.over_tolerance));
  if (block_metrics.over_tolerance != 0U) {
    std::fprintf(stderr,
                 "decode block oracle mismatch prefix=%llu rows=%u "
                 "over_tolerance=%llu\n",
                 static_cast<unsigned long long>(prefix), kBlockRows,
                 static_cast<unsigned long long>(block_metrics.over_tolerance));
    success = false;
  }

  const size_t row_elements =
      static_cast<size_t>(kQueryHeads) * static_cast<size_t>(kHeadDim);
  for (uint32_t row = 0U; row != kBlockRows; ++row) {
    const uint64_t row_prefix = prefix + row;
    const uint64_t row_tokens = row_prefix + 1U;
    const std::vector<uint16_t> row_key = make_kv(row_tokens, false);
    const std::vector<uint16_t> row_value = make_kv(row_tokens, true);
    const std::vector<uint16_t> row_query =
        make_query_with_row_offset(1U, row_prefix, row);
    const std::vector<uint16_t> row_expected =
        oracle(row_key, row_value, row_query, row_prefix, 1U);
    Metrics row_metrics{};
    DispatchMetadata row_dispatch{};
    std::vector<uint16_t> row_actual;
    const bool row_success =
        run_case(context, queue, row_prefix, 1U, false, row_key, row_value,
                 row_query, row_expected, &row_metrics, &row_dispatch,
                 &row_actual) &&
        dispatch_matches(row_prefix, 1U, false, row_dispatch);
    success = row_success && success;
    std::printf("decode_row_oracle prefix=%llu row=%u max_bf16_ulp=%u "
                "max_abs=%g max_relative=%g over_tolerance=%llu\n",
                static_cast<unsigned long long>(prefix), row,
                row_metrics.max_ulp, row_metrics.max_abs,
                row_metrics.max_relative,
                static_cast<unsigned long long>(row_metrics.over_tolerance));
    if (row_metrics.over_tolerance != 0U) {
      std::fprintf(stderr,
                   "decode row oracle mismatch prefix=%llu row=%u "
                   "over_tolerance=%llu\n",
                   static_cast<unsigned long long>(row_prefix), row,
                   static_cast<unsigned long long>(row_metrics.over_tolerance));
      success = false;
    }
    if (!row_success || row_actual.size() != row_elements ||
        block_actual.size() < static_cast<size_t>(row + 1U) * row_elements) {
      continue;
    }
    const auto block_row_begin =
        block_actual.begin() + static_cast<size_t>(row) * row_elements;
    const std::vector<uint16_t> block_row(block_row_begin,
                                          block_row_begin + row_elements);
    const Metrics parity = compare(row_actual, block_row);
    std::printf("decode_block_parity prefix=%llu row=%u m1_provider=%s "
                "m3_provider=%s m1_vs_m3_max_bf16_ulp=%u max_abs=%g "
                "max_relative=%g pair_over_reference_tolerance=%llu\n",
                static_cast<unsigned long long>(prefix), row,
                row_dispatch.logical_symbol.c_str(),
                block_dispatch.logical_symbol.c_str(), parity.max_ulp,
                parity.max_abs, parity.max_relative,
                static_cast<unsigned long long>(parity.over_tolerance));
  }
  return success;
}

} // namespace

int main(int argc, char **argv) {
  const bool decode_block_parity =
      argc == 2 && std::strcmp(argv[1], "--decode-block-parity") == 0;
  if (argc > 1 && !decode_block_parity) {
    std::fprintf(stderr, "usage: %s [--decode-block-parity]\n", argv[0]);
    return EXIT_FAILURE;
  }
  Environment environment;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  bool success = create_context(&context, &queue);
  if (success && !decode_block_parity) {
    constexpr std::array<std::array<uint64_t, 2>, 13> cases = {
        {{{0U, 127U}},
         {{0U, 128U}},
         {{0U, 129U}},
         {{31U, 127U}},
         {{31U, 128U}},
         {{31U, 129U}},
         {{256U, 127U}},
         {{256U, 128U}},
         {{256U, 129U}},
         {{1023U, 127U}},
         {{1023U, 128U}},
         {{1024U, 128U}},
         {{1025U, 129U}}}};
    for (const auto &test_case : cases) {
      const uint64_t prefix = test_case[0];
      const uint32_t rows = static_cast<uint32_t>(test_case[1]);
      const uint64_t context_tokens = prefix + rows;
      const std::vector<uint16_t> key = make_kv(context_tokens, false);
      const std::vector<uint16_t> value = make_kv(context_tokens, true);
      const std::vector<uint16_t> query = make_query(rows, prefix);
      const std::vector<uint16_t> expected =
          oracle(key, value, query, prefix, rows);
      Metrics default_metrics{};
      Metrics qtile_metrics{};
      Metrics provider_metrics{};
      DispatchMetadata default_dispatch{};
      DispatchMetadata qtile_dispatch{};
      std::vector<uint16_t> default_actual;
      std::vector<uint16_t> qtile_actual;
      success =
          run_case(context, queue, prefix, rows, false, key, value, query,
                   expected, &default_metrics, &default_dispatch,
                   &default_actual) &&
          run_case(context, queue, prefix, rows, true, key, value, query,
                   expected, &qtile_metrics, &qtile_dispatch, &qtile_actual);
      if (success) {
        success = dispatch_matches(prefix, rows, false, default_dispatch) &&
                  dispatch_matches(prefix, rows, true, qtile_dispatch);
        std::printf(
            "dispatch prefix=%llu query_count=%u default=%s qtile4=%s\n",
            static_cast<unsigned long long>(prefix), rows,
            default_dispatch.logical_symbol.c_str(),
            qtile_dispatch.logical_symbol.c_str());
        provider_metrics = compare(default_actual, qtile_actual);
        std::printf(
            "provider_diff prefix=%llu query_count=%u max_bf16_ulp=%u "
            "max_abs=%g max_relative=%g over_tolerance=%llu\n",
            static_cast<unsigned long long>(prefix), rows,
            provider_metrics.max_ulp, provider_metrics.max_abs,
            provider_metrics.max_relative,
            static_cast<unsigned long long>(provider_metrics.over_tolerance));
        if (provider_metrics.over_tolerance != 0U)
          success = false;
      }
      if (default_metrics.over_tolerance != 0U ||
          qtile_metrics.over_tolerance != 0U)
        success = false;
      if (!success)
        break;
    }
  }
  if (success && decode_block_parity) {
    if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") != 0) {
      std::fprintf(stderr,
                   "--decode-block-parity requires SLLM_TEST_EXPECTED_TARGET="
                   "gfx1201\n");
      success = false;
    } else {
      constexpr std::array<uint64_t, 2> prefixes = {127U, 129U};
      for (const uint64_t prefix : prefixes) {
        if (!run_decode_block_parity(context, queue, prefix)) {
          success = false;
          break;
        }
      }
    }
  }
  Error error;
  if (queue != nullptr)
    success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                     "queue release", error) &&
              queue == nullptr && success;
  if (context != nullptr)
    success = expect(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_OK, "context release", error) &&
              context == nullptr && success;
  if (success)
    std::printf(
        decode_block_parity
            ? "phase83 MXFP8 decode block parity public GPU PASS target=%s\n"
            : "phase83 MXFP8 prefill qtile8/w16 public GPU PASS target=%s\n",
        SLLM_TEST_EXPECTED_TARGET);
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
