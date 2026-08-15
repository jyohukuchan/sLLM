#include "gemma_attention_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

struct Case final {
  const char *name;
  uint32_t query_count;
  uint64_t start_position;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint64_t sliding_window;
};

float bf16_to_f32(const uint16_t raw) {
  const uint32_t bits = static_cast<uint32_t>(raw) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

bool hip_ok(const hipError_t status, const char *const label) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << label << " failed: " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

std::vector<uint16_t> make_input(const std::size_t count, const uint32_t salt) {
  std::vector<uint16_t> values(count);
  for (std::size_t index = 0; index != count; ++index) {
    const uint64_t mixed =
        (static_cast<uint64_t>(index) * UINT64_C(53) + salt * 29U) % 257U;
    const float value =
        static_cast<float>(static_cast<int32_t>(mixed) - 128) / 1024.0F;
    values[index] = f32_to_bf16_rne(value);
  }
  return values;
}

std::vector<uint16_t> reference(const std::vector<uint16_t> &query,
                                const std::vector<uint16_t> &key,
                                const std::vector<uint16_t> &value,
                                const Case &test_case) {
  constexpr uint32_t q_heads = 16U;
  const uint64_t kv_length = test_case.start_position + test_case.query_count;
  const std::size_t output_count =
      static_cast<std::size_t>(test_case.query_count) * q_heads *
      test_case.head_dim;
  std::vector<uint16_t> output(output_count);
  for (uint32_t row = 0U; row != test_case.query_count; ++row) {
    const uint64_t query_position = test_case.start_position + row;
    uint64_t first_key = 0U;
    if (test_case.sliding_window != 0U &&
        query_position + 1U > test_case.sliding_window) {
      first_key = query_position + 1U - test_case.sliding_window;
    }
    const std::size_t key_count =
        static_cast<std::size_t>(query_position - first_key + 1U);
    for (uint32_t query_head = 0U; query_head != q_heads; ++query_head) {
      const uint32_t kv_head = query_head / (q_heads / test_case.kv_heads);
      const std::size_t query_base =
          (static_cast<std::size_t>(row) * q_heads + query_head) *
          test_case.head_dim;
      std::vector<float> scores(key_count);
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t local_key = 0; local_key != key_count; ++local_key) {
        const uint64_t key_position = first_key + local_key;
        const std::size_t key_base =
            (static_cast<std::size_t>(key_position) * test_case.kv_heads +
             kv_head) *
            test_case.head_dim;
        float score = 0.0F;
        for (uint32_t dimension = 0U; dimension != test_case.head_dim;
             ++dimension) {
          score += bf16_to_f32(query[query_base + dimension]) *
                   bf16_to_f32(key[key_base + dimension]);
        }
        scores[local_key] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (uint32_t dimension = 0U; dimension != test_case.head_dim;
           ++dimension) {
        float result = 0.0F;
        for (std::size_t local_key = 0; local_key != key_count; ++local_key) {
          const uint64_t key_position = first_key + local_key;
          const std::size_t value_index =
              (static_cast<std::size_t>(key_position) * test_case.kv_heads +
               kv_head) *
                  test_case.head_dim +
              dimension;
          result += scores[local_key] * bf16_to_f32(value[value_index]);
        }
        output[query_base + dimension] = f32_to_bf16_rne(result / denominator);
      }
    }
  }
  (void)kv_length;
  return output;
}

bool compare(const std::vector<uint16_t> &actual,
             const std::vector<uint16_t> &expected, float *const max_abs,
             float *const max_scaled_rel) {
  constexpr float atol = 0.015625F;
  constexpr float rtol = 0.03125F;
  for (std::size_t index = 0; index != actual.size(); ++index) {
    const float observed = bf16_to_f32(actual[index]);
    const float oracle = bf16_to_f32(expected[index]);
    if (!std::isfinite(observed) || !std::isfinite(oracle)) {
      std::cerr << "non-finite attention result at index " << index << '\n';
      return false;
    }
    const float absolute = std::abs(observed - oracle);
    const float scaled_relative = absolute / std::max(std::abs(oracle), atol);
    *max_abs = std::max(*max_abs, absolute);
    *max_scaled_rel = std::max(*max_scaled_rel, scaled_relative);
    if (absolute > atol + rtol * std::abs(oracle)) {
      std::cerr << "attention mismatch at index " << index
                << ": actual=" << observed << " expected=" << oracle
                << " abs=" << absolute << " scaled_rel=" << scaled_relative
                << '\n';
      return false;
    }
  }
  return true;
}

bool run_case(const Case &test_case, float *const max_abs,
              float *const max_scaled_rel) {
  constexpr uint32_t q_heads = 16U;
  const uint64_t kv_length = test_case.start_position + test_case.query_count;
  const std::size_t query_count =
      static_cast<std::size_t>(test_case.query_count) * q_heads *
      test_case.head_dim;
  const std::size_t kv_count = static_cast<std::size_t>(kv_length) *
                               test_case.kv_heads * test_case.head_dim;
  const std::vector<uint16_t> query = make_input(query_count, 3U);
  const std::vector<uint16_t> key = make_input(kv_count, 7U);
  const std::vector<uint16_t> value = make_input(kv_count, 13U);
  const std::vector<uint16_t> oracle = reference(query, key, value, test_case);
  std::vector<uint16_t> output(query_count);

  uint16_t *device_query = nullptr;
  uint16_t *device_key = nullptr;
  uint16_t *device_value = nullptr;
  uint16_t *device_output = nullptr;
  const std::size_t query_bytes = query_count * sizeof(uint16_t);
  const std::size_t kv_bytes = kv_count * sizeof(uint16_t);
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_query), query_bytes),
             "hipMalloc(query)") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_key), kv_bytes),
             "hipMalloc(key)") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_value), kv_bytes),
             "hipMalloc(value)") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_output), query_bytes),
             "hipMalloc(output)");
  if (ok) {
    ok = hip_ok(hipMemcpy(device_query, query.data(), query_bytes,
                          hipMemcpyHostToDevice),
                "hipMemcpy(query)") &&
         hip_ok(
             hipMemcpy(device_key, key.data(), kv_bytes, hipMemcpyHostToDevice),
             "hipMemcpy(key)") &&
         hip_ok(hipMemcpy(device_value, value.data(), kv_bytes,
                          hipMemcpyHostToDevice),
                "hipMemcpy(value)") &&
         hip_ok(sllm_gemma_attention_kernel::launch(
                    device_query, device_key, device_value, device_output,
                    test_case.query_count, test_case.start_position, kv_length,
                    q_heads, test_case.kv_heads, test_case.head_dim,
                    test_case.sliding_window, nullptr),
                "Gemma attention launch") &&
         hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize") &&
         hip_ok(hipMemcpy(output.data(), device_output, query_bytes,
                          hipMemcpyDeviceToHost),
                "hipMemcpy(output)");
  }
  if (ok) {
    ok = compare(output, oracle, max_abs, max_scaled_rel);
  }
  if (device_output != nullptr) {
    ok = hip_ok(hipFree(device_output), "hipFree(output)") && ok;
  }
  if (device_value != nullptr) {
    ok = hip_ok(hipFree(device_value), "hipFree(value)") && ok;
  }
  if (device_key != nullptr) {
    ok = hip_ok(hipFree(device_key), "hipFree(key)") && ok;
  }
  if (device_query != nullptr) {
    ok = hip_ok(hipFree(device_query), "hipFree(query)") && ok;
  }
  if (!ok) {
    std::cerr << "case failed: " << test_case.name << '\n';
  }
  return ok;
}

} // namespace

int main() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0),
              "hipGetDeviceProperties")) {
    return 1;
  }
  if (std::strcmp(properties.gcnArchName, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "target mismatch: actual=" << properties.gcnArchName
              << " expected=" << SLLM_TEST_EXPECTED_TARGET << '\n';
    return 1;
  }
  constexpr std::array<Case, 4> cases{{
      {"sliding-short", 1U, 0U, 8U, 256U, 1'024U},
      {"sliding-window-boundary", 3U, 1'022U, 8U, 256U, 1'024U},
      {"full-short", 3U, 17U, 1U, 512U, 0U},
      {"full-nonaligned", 17U, 240U, 1U, 512U, 0U},
  }};
  float max_abs = 0.0F;
  float max_scaled_rel = 0.0F;
  for (const Case &test_case : cases) {
    if (!run_case(test_case, &max_abs, &max_scaled_rel)) {
      return 1;
    }
  }
  std::cout << "Gemma causal attention PASS target=" << properties.gcnArchName
            << " cases=" << cases.size() << " max_abs=" << max_abs
            << " max_scaled_rel=" << max_scaled_rel << " fallback=false kernel="
            << sllm_gemma_attention_kernel::kLogicalKernelId
            << " symbol=" << sllm_gemma_attention_kernel::kDeviceSymbol << '\n';
  return 0;
}
