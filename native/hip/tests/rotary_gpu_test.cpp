#include "rotary_kernel_internal.hpp"

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
  uint32_t token_count;
  uint32_t start_position;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  float theta;
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
        (static_cast<uint64_t>(index) * UINT64_C(37) + salt * 19U) % 257U;
    const float value =
        (static_cast<float>(static_cast<int32_t>(mixed) - 128) / 96.0F);
    values[index] = f32_to_bf16_rne(value);
  }
  return values;
}

std::vector<uint16_t> reference(const std::vector<uint16_t> &input,
                                const std::vector<int32_t> &positions,
                                const uint32_t heads, const uint32_t head_dim,
                                const uint32_t rotary_dim, const float theta) {
  std::vector<uint16_t> output = input;
  const uint32_t half = head_dim / 2U;
  const uint32_t active_pairs = rotary_dim / 2U;
  for (std::size_t token = 0; token != positions.size(); ++token) {
    for (uint32_t head = 0U; head != heads; ++head) {
      const std::size_t base =
          (token * heads + static_cast<std::size_t>(head)) * head_dim;
      for (uint32_t pair = 0U; pair != active_pairs; ++pair) {
        const float exponent =
            -2.0F * static_cast<float>(pair) / static_cast<float>(head_dim);
        const float angle =
            static_cast<float>(positions[token]) * std::pow(theta, exponent);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float left = bf16_to_f32(input[base + pair]);
        const float right = bf16_to_f32(input[base + half + pair]);
        output[base + pair] = f32_to_bf16_rne(left * cosine - right * sine);
        output[base + half + pair] =
            f32_to_bf16_rne(right * cosine + left * sine);
      }
    }
  }
  return output;
}

bool compare(const std::vector<uint16_t> &actual,
             const std::vector<uint16_t> &expected,
             const std::vector<uint16_t> &input, const uint32_t head_dim,
             const uint32_t rotary_dim, float *const max_abs,
             float *const max_rel) {
  constexpr float atol = 0.03125F;
  constexpr float rtol = 0.03125F;
  const uint32_t half = head_dim / 2U;
  const uint32_t active_pairs = rotary_dim / 2U;
  for (std::size_t index = 0; index != actual.size(); ++index) {
    const uint32_t dimension = static_cast<uint32_t>(index % head_dim);
    const bool active = dimension < active_pairs ||
                        (dimension >= half && dimension < half + active_pairs);
    if (!active && actual[index] != input[index]) {
      std::cerr << "inactive rotary dimension changed at index " << index
                << '\n';
      return false;
    }
    const float observed = bf16_to_f32(actual[index]);
    const float oracle = bf16_to_f32(expected[index]);
    if (!std::isfinite(observed) || !std::isfinite(oracle)) {
      std::cerr << "non-finite rotary result at index " << index << '\n';
      return false;
    }
    const float absolute = std::abs(observed - oracle);
    const float relative = absolute / std::max(std::abs(oracle), atol);
    *max_abs = std::max(*max_abs, absolute);
    *max_rel = std::max(*max_rel, relative);
    if (absolute > atol + rtol * std::abs(oracle)) {
      std::cerr << "rotary mismatch at index " << index
                << ": actual=" << observed << " expected=" << oracle
                << " abs=" << absolute << " rel=" << relative << '\n';
      return false;
    }
  }
  return true;
}

bool run_case(const Case &test_case, float *const max_abs,
              float *const max_rel) {
  constexpr uint32_t q_heads = 16U;
  const std::size_t q_count = static_cast<std::size_t>(test_case.token_count) *
                              q_heads * test_case.head_dim;
  const std::size_t k_count = static_cast<std::size_t>(test_case.token_count) *
                              test_case.kv_heads * test_case.head_dim;
  const std::vector<uint16_t> query = make_input(q_count, 3U);
  const std::vector<uint16_t> key = make_input(k_count, 11U);
  std::vector<int32_t> positions(test_case.token_count);
  for (uint32_t index = 0U; index != test_case.token_count; ++index) {
    positions[index] = static_cast<int32_t>(test_case.start_position + index);
  }
  const std::vector<uint16_t> query_oracle =
      reference(query, positions, q_heads, test_case.head_dim,
                test_case.rotary_dim, test_case.theta);
  const std::vector<uint16_t> key_oracle =
      reference(key, positions, test_case.kv_heads, test_case.head_dim,
                test_case.rotary_dim, test_case.theta);
  std::vector<uint16_t> query_output(q_count);
  std::vector<uint16_t> key_output(k_count);

  uint16_t *device_query = nullptr;
  uint16_t *device_key = nullptr;
  uint16_t *device_query_output = nullptr;
  uint16_t *device_key_output = nullptr;
  int32_t *device_positions = nullptr;
  const std::size_t q_bytes = q_count * sizeof(uint16_t);
  const std::size_t k_bytes = k_count * sizeof(uint16_t);
  const std::size_t position_bytes = positions.size() * sizeof(int32_t);
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_query), q_bytes),
             "hipMalloc(query)") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_key), k_bytes),
             "hipMalloc(key)") &&
      hip_ok(
          hipMalloc(reinterpret_cast<void **>(&device_query_output), q_bytes),
          "hipMalloc(query output)") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_key_output), k_bytes),
             "hipMalloc(key output)") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_positions),
                       position_bytes),
             "hipMalloc(positions)");
  if (ok) {
    ok = hip_ok(hipMemcpy(device_query, query.data(), q_bytes,
                          hipMemcpyHostToDevice),
                "hipMemcpy(query)") &&
         hip_ok(
             hipMemcpy(device_key, key.data(), k_bytes, hipMemcpyHostToDevice),
             "hipMemcpy(key)") &&
         hip_ok(hipMemcpy(device_positions, positions.data(), position_bytes,
                          hipMemcpyHostToDevice),
                "hipMemcpy(positions)") &&
         hip_ok(sllm_rotary_kernel::launch(
                    device_query, device_key, device_positions,
                    device_query_output, device_key_output,
                    test_case.token_count, q_heads, test_case.kv_heads,
                    test_case.head_dim, test_case.rotary_dim, test_case.theta,
                    nullptr),
                "rotary launch") &&
         hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize") &&
         hip_ok(hipMemcpy(query_output.data(), device_query_output, q_bytes,
                          hipMemcpyDeviceToHost),
                "hipMemcpy(query output)") &&
         hip_ok(hipMemcpy(key_output.data(), device_key_output, k_bytes,
                          hipMemcpyDeviceToHost),
                "hipMemcpy(key output)");
  }
  if (ok) {
    ok = compare(query_output, query_oracle, query, test_case.head_dim,
                 test_case.rotary_dim, max_abs, max_rel) &&
         compare(key_output, key_oracle, key, test_case.head_dim,
                 test_case.rotary_dim, max_abs, max_rel);
  }
  if (device_positions != nullptr) {
    ok = hip_ok(hipFree(device_positions), "hipFree(positions)") && ok;
  }
  if (device_key_output != nullptr) {
    ok = hip_ok(hipFree(device_key_output), "hipFree(key output)") && ok;
  }
  if (device_query_output != nullptr) {
    ok = hip_ok(hipFree(device_query_output), "hipFree(query output)") && ok;
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
  constexpr std::array<Case, 6> cases{{
      {"sliding-m1-p0", 1U, 0U, 8U, 256U, 256U, 10'000.0F},
      {"sliding-m3-p255", 3U, 255U, 8U, 256U, 256U, 10'000.0F},
      {"sliding-m17-tail", 17U, 262'127U, 8U, 256U, 256U, 10'000.0F},
      {"full-m1-p0", 1U, 0U, 1U, 512U, 128U, 1'000'000.0F},
      {"full-m3-p255", 3U, 255U, 1U, 512U, 128U, 1'000'000.0F},
      {"full-m17-tail", 17U, 262'127U, 1U, 512U, 128U, 1'000'000.0F},
  }};
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  for (const Case &test_case : cases) {
    if (!run_case(test_case, &max_abs, &max_rel)) {
      return 1;
    }
  }
  std::cout << "split-half rotary PASS target=" << properties.gcnArchName
            << " cases=" << cases.size() << " max_abs=" << max_abs
            << " max_scaled_rel=" << max_rel
            << " fallback=false kernel=" << sllm_rotary_kernel::kLogicalKernelId
            << " symbol=" << sllm_rotary_kernel::kDeviceSymbol << '\n';
  return 0;
}
