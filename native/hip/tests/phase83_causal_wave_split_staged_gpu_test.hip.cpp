#include "causal_attention_kernel_internal.hpp"
#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

// Phase 83 bounded native probe.  It compares the existing one-block wave8
// provider with the opt-in staged provider for the exact Qwen3.8 M=1 MXFP8
// shape.  The staged provider is expected to be bitwise identical; the second
// check is an independent scalar decode/attention oracle over the encoded
// E4M3FN/E8M0 planes.

namespace {

constexpr uint32_t kQueryHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kBlocksPerRow = kHeadDim / 32U;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr uint64_t kQueryValues = static_cast<uint64_t>(kQueryHeads) * kHeadDim;
constexpr uint64_t kWorkspaceBytes =
    static_cast<uint64_t>(kQueryHeads) * 8U * (kHeadDim + 2U) * sizeof(float);

bool hip_ok(const hipError_t status, const char *const operation) {
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

template <typename T>
bool upload(T *const device, const std::vector<T> &host,
            const char *const label) {
  return hip_ok(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                          hipMemcpyHostToDevice),
                label);
}

bool compare(const std::vector<uint16_t> &control,
             const std::vector<uint16_t> &staged,
             const std::vector<uint16_t> &expected) {
  uint32_t control_staged_ulp = 0U;
  uint32_t control_oracle_ulp = 0U;
  float max_abs = 0.0F;
  std::size_t first_difference = control.size();
  for (std::size_t index = 0U; index != control.size(); ++index) {
    if (control[index] != staged[index] && first_difference == control.size()) {
      first_difference = index;
    }
    control_staged_ulp = std::max(
        control_staged_ulp, bf16_ulp_distance(control[index], staged[index]));
    control_oracle_ulp = std::max(
        control_oracle_ulp, bf16_ulp_distance(control[index], expected[index]));
    max_abs = std::max(max_abs, std::fabs(bf16_to_f32(control[index]) -
                                          bf16_to_f32(expected[index])));
    if (!std::isfinite(bf16_to_f32(control[index])) ||
        !std::isfinite(bf16_to_f32(staged[index]))) {
      std::cerr << "nonfinite output at index " << index << '\n';
      return false;
    }
  }
  std::cout << "staged equality="
            << (first_difference == control.size() ? "bitwise" : "mismatch")
            << " control_staged_max_ulp=" << control_staged_ulp
            << " oracle_max_abs=" << max_abs
            << " oracle_max_ulp=" << control_oracle_ulp << '\n';
  if (first_difference != control.size()) {
    std::cerr << "first staged mismatch index=" << first_difference
              << " control=0x" << std::hex << control[first_difference]
              << " staged=0x" << staged[first_difference] << std::dec << '\n';
    return false;
  }
  return control_oracle_ulp <= 8U && max_abs <= 0.03125F;
}

struct LaunchBuffers final {
  const uint16_t *query;
  const uint8_t *key;
  const uint8_t *value;
  const uint8_t *key_scales;
  const uint8_t *value_scales;
  uint16_t *output;
  void *workspace;
  uint32_t query_count;
  uint64_t start_position;
  uint64_t committed_length;
  uint64_t workspace_bytes;
  hipStream_t stream;
};

hipError_t launch_control(const LaunchBuffers &buffers) {
  return sllm_causal_attention_kernel::launch(
      buffers.query, buffers.key, buffers.value, buffers.key_scales,
      buffers.value_scales, nullptr, nullptr, buffers.output,
      buffers.query_count, buffers.committed_length, buffers.start_position,
      buffers.committed_length, kQueryHeads, kKvHeads, kHeadDim,
      SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 1.0F, 1.0F, false, true, true, false,
      false, false, 0U, kAttentionScale, buffers.stream);
}

hipError_t launch_staged(const LaunchBuffers &buffers) {
  return sllm_causal_attention_kernel::launch_decode_wave_split_staged(
      buffers.query, buffers.key, buffers.value, buffers.key_scales,
      buffers.value_scales, nullptr, nullptr, buffers.output,
      buffers.query_count, buffers.start_position, buffers.committed_length,
      kQueryHeads, kKvHeads, kHeadDim, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 1.0F,
      1.0F, buffers.workspace, buffers.workspace_bytes, true, buffers.stream);
}

bool measure_provider(const std::vector<LaunchBuffers> &launches,
                      const bool staged, float *const median_ms) {
  constexpr std::size_t kWarmups = 2U;
  constexpr std::size_t kMeasured = 5U;
  if (launches.empty()) {
    return false;
  }
  const hipStream_t stream = launches.front().stream;
  const auto launch = staged ? launch_staged : launch_control;
  for (std::size_t iteration = 0U; iteration != kWarmups; ++iteration) {
    bool warmup_ok = true;
    for (const LaunchBuffers &buffers : launches) {
      warmup_ok = hip_ok(launch(buffers), staged ? "staged warmup launch"
                                                 : "control warmup launch") &&
                  warmup_ok;
    }
    if (!warmup_ok || !hip_ok(hipStreamSynchronize(stream),
                              staged ? "staged warmup synchronize"
                                     : "control warmup synchronize")) {
      return false;
    }
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!hip_ok(hipEventCreate(&start), "timing start event") ||
      !hip_ok(hipEventCreate(&stop), "timing stop event")) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return false;
  }
  std::vector<float> samples;
  samples.reserve(kMeasured);
  bool valid = true;
  for (std::size_t iteration = 0U; iteration != kMeasured; ++iteration) {
    valid = hip_ok(hipEventRecord(start, stream), "timing start") && valid;
    for (const LaunchBuffers &buffers : launches) {
      valid = hip_ok(launch(buffers), staged ? "staged measured launch"
                                             : "control measured launch") &&
              valid;
    }
    valid = hip_ok(hipEventRecord(stop, stream), "timing stop") && valid;
    valid = hip_ok(hipEventSynchronize(stop), "timing synchronize") && valid;
    float elapsed_ms = 0.0F;
    valid = hip_ok(hipEventElapsedTime(&elapsed_ms, start, stop),
                   "timing elapsed") &&
            valid;
    if (!valid) {
      break;
    }
    samples.push_back(elapsed_ms);
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  if (!valid || samples.empty()) {
    return false;
  }
  std::sort(samples.begin(), samples.end());
  *median_ms = samples[samples.size() / 2U];
  std::cout << (staged ? "staged" : "control")
            << " launches=" << launches.size() << " warmups=" << kWarmups
            << " measured=" << kMeasured << " median_ms=" << *median_ms << '\n';
  return true;
}

bool parse_unsigned(const char *const value, uint64_t *const parsed_value) {
  if (value == nullptr) {
    return false;
  }
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0U ||
      parsed > std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  *parsed_value = static_cast<uint64_t>(parsed);
  return true;
}

bool parse_args(const int argc, char **const argv, uint64_t *const length,
                uint32_t *const query_count) {
  *length = 2048U;
  *query_count = 1U;
  const char *length_value = std::getenv("SLLM_STAGED_LENGTH");
  const char *query_count_value = std::getenv("SLLM_STAGED_M");
  if (length_value != nullptr && !parse_unsigned(length_value, length)) {
    std::cerr << "invalid SLLM_STAGED_LENGTH: " << length_value << '\n';
    return false;
  }
  if (query_count_value != nullptr) {
    uint64_t parsed_query_count = 0U;
    if (!parse_unsigned(query_count_value, &parsed_query_count) ||
        parsed_query_count > 4U) {
      std::cerr << "invalid SLLM_STAGED_M: " << query_count_value << '\n';
      return false;
    }
    *query_count = static_cast<uint32_t>(parsed_query_count);
  }
  for (int index = 1; index < argc; ++index) {
    const char *value = nullptr;
    if (std::strcmp(argv[index], "--length") == 0) {
      if (++index >= argc) {
        return false;
      }
      value = argv[index];
      if (!parse_unsigned(value, length)) {
        std::cerr << "invalid length: " << value << '\n';
        return false;
      }
    } else if (std::strcmp(argv[index], "--m") == 0) {
      if (++index >= argc) {
        return false;
      }
      value = argv[index];
      uint64_t parsed_query_count = 0U;
      if (!parse_unsigned(value, &parsed_query_count) ||
          parsed_query_count > 4U) {
        std::cerr << "invalid M: " << value << '\n';
        return false;
      }
      *query_count = static_cast<uint32_t>(parsed_query_count);
    } else {
      std::cerr << "usage: " << argv[0] << " [--length TOKENS] [--m 1..4]\n";
      return false;
    }
  }
  return true;
}

} // namespace

int main(const int argc, char **const argv) {
  uint64_t committed_length = 0U;
  uint32_t query_count = 1U;
  if (!parse_args(argc, argv, &committed_length, &query_count) ||
      committed_length < query_count) {
    std::cerr << "length must be >= M\n";
    return 2;
  }
  hipDeviceProp_t properties{};
  int device = 0;
  bool ok = hip_ok(hipGetDevice(&device), "hipGetDevice") &&
            hip_ok(hipGetDeviceProperties(&properties, device),
                   "hipGetDeviceProperties");
  if (ok &&
      std::strcmp(properties.gcnArchName, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "visible target is " << properties.gcnArchName << ", expected "
              << SLLM_TEST_EXPECTED_TARGET << '\n';
    ok = false;
  }
  Fixture fixture = make_fixture(committed_length, query_count);
  const std::vector<uint16_t> expected =
      oracle(fixture, committed_length, query_count);
  uint16_t *query_device = nullptr;
  uint8_t *key_device = nullptr;
  uint8_t *value_device = nullptr;
  uint8_t *key_scales_device = nullptr;
  uint8_t *value_scales_device = nullptr;
  uint16_t *control_device = nullptr;
  uint16_t *staged_device = nullptr;
  void *workspace_device = nullptr;
  hipStream_t stream = nullptr;
  ok = hip_ok(hipStreamCreate(&stream), "hipStreamCreate") && ok;
  const uint64_t query_bytes =
      static_cast<uint64_t>(fixture.query.size()) * sizeof(uint16_t);
  const uint64_t workspace_bytes = kWorkspaceBytes * query_count;
  const uint64_t kv_values = fixture.key.size();
  const uint64_t kv_scales = fixture.key_scales.size();
  ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&query_device), query_bytes),
              "query allocation") &&
       ok;
  ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&key_device), kv_values),
              "key allocation") &&
       ok;
  ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&value_device), kv_values),
              "value allocation") &&
       ok;
  ok = hip_ok(
           hipMalloc(reinterpret_cast<void **>(&key_scales_device), kv_scales),
           "key scale allocation") &&
       ok;
  ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&value_scales_device),
                        kv_scales),
              "value scale allocation") &&
       ok;
  ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&control_device), query_bytes),
             "control allocation") &&
      ok;
  ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&staged_device), query_bytes),
              "staged allocation") &&
       ok;
  ok = hip_ok(hipMalloc(&workspace_device, workspace_bytes),
              "workspace allocation") &&
       ok;
  if (ok) {
    ok = upload(query_device, fixture.query, "query upload") && ok;
    ok = upload(key_device, fixture.key, "key upload") && ok;
    ok = upload(value_device, fixture.value, "value upload") && ok;
    ok = upload(key_scales_device, fixture.key_scales, "key scales upload") &&
         ok;
    ok = upload(value_scales_device, fixture.value_scales,
                "value scales upload") &&
         ok;
    ok = hip_ok(hipMemset(control_device, 0, query_bytes), "control memset") &&
         ok;
    ok =
        hip_ok(hipMemset(staged_device, 0, query_bytes), "staged memset") && ok;
  }
  if (ok) {
    const uint64_t start_position = committed_length - query_count;
    std::vector<LaunchBuffers> control_buffers;
    control_buffers.reserve(query_count);
    for (uint32_t query_index = 0U; query_index != query_count; ++query_index) {
      const std::size_t query_offset =
          static_cast<std::size_t>(query_index * kQueryValues);
      control_buffers.push_back(
          {query_device + query_offset, key_device, value_device,
           key_scales_device, value_scales_device,
           control_device + query_offset, workspace_device, 1U,
           start_position + query_index, start_position + query_index + 1U,
           kWorkspaceBytes, stream});
    }
    const LaunchBuffers staged_buffers{
        query_device,      key_device,          value_device,
        key_scales_device, value_scales_device, staged_device,
        workspace_device,  query_count,         start_position,
        committed_length,  workspace_bytes,     stream};
    float control_median_ms = 0.0F;
    float staged_median_ms = 0.0F;
    ok = measure_provider(control_buffers, false, &control_median_ms) && ok;
    ok = measure_provider(std::vector<LaunchBuffers>{staged_buffers}, true,
                          &staged_median_ms) &&
         ok;
    if (ok && control_median_ms > 0.0F) {
      std::cout << "length=" << committed_length << " M=" << query_count
                << " staged_over_control="
                << (staged_median_ms / control_median_ms) << '\n';
    }
  }
  std::vector<uint16_t> control(fixture.query.size(), 0U);
  std::vector<uint16_t> staged(fixture.query.size(), 0U);
  if (ok) {
    ok = hip_ok(hipMemcpy(control.data(), control_device,
                          control.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "control download") &&
         ok;
    ok = hip_ok(hipMemcpy(staged.data(), staged_device,
                          staged.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "staged download") &&
         ok;
    ok = compare(control, staged, expected) && ok;
  }

  (void)hipFree(workspace_device);
  (void)hipFree(staged_device);
  (void)hipFree(control_device);
  (void)hipFree(value_scales_device);
  (void)hipFree(key_scales_device);
  (void)hipFree(value_device);
  (void)hipFree(key_device);
  (void)hipFree(query_device);
  (void)hipStreamDestroy(stream);
  return ok ? 0 : 1;
}
