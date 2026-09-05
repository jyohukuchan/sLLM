// Phase 78 GQA6 P64/P128 numerical probe.
//
// The input layout and the 24-query-head/4-KV-head/256-dimension contract are
// the same as phase78_gqa6_decode_partition_sweep_probe.hip.cpp.  This probe
// calls the production P64 and P128 launch wrappers from
// causal_attention_kernel_internal.hpp.  Its oracle is deliberately separate
// from that probe: score, stable softmax, and value accumulation are all done
// in double precision before the final BF16 rounding.
//
// P64 and P128 have the same real attention equation
//   O[d] = sum_t exp(q.k_t/sqrt(256)-m) V_t[d] / sum_t exp(...)
// and differ in the partition-induced FP32 online-softmax and merge order.
//
// The measured rows are evidence for an N2 candidate classification.  They do
// not establish a standard worst-case error bound, and absence of an N1 proof
// alone is not an N3 result.  GPU execution is intentionally left to the
// scheduled Phase 78 runner; compile-only checks are sufficient for this file.

#include "causal_attention_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

#include "sllm/hip.h"

namespace {

constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kGqa = 6U;
constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr uint32_t kSeedCount = 2U;
constexpr std::array<uint64_t, 4> kContexts = {UINT64_C(8191), UINT64_C(8192),
                                               UINT64_C(8193), UINT64_C(9435)};
constexpr std::array<uint32_t, kSeedCount> kSeeds = {0U, 7919U};

using LaunchFn = hipError_t (*)(const uint16_t *, const void *, const void *,
                                const void *, const void *, const float *,
                                const float *, uint16_t *, uint32_t, uint64_t,
                                uint64_t, uint32_t, uint32_t, uint32_t,
                                uint32_t, float, float, void *, uint64_t,
                                hipStream_t) noexcept;

struct Buffers final {
  uint16_t *query = nullptr;
  uint16_t *key = nullptr;
  uint16_t *value = nullptr;
  uint16_t *output = nullptr;
  float *workspace = nullptr;
  size_t workspace_bytes = 0U;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip_error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float host_bf16_to_float(const uint16_t bits) {
  const uint32_t value = static_cast<uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

uint16_t host_fp16_rne(const float value) {
  // The generated inputs are ordinary finite values in [-0.65, 0.65].
  const float absolute = std::fabs(value);
  if (absolute == 0.0F)
    return static_cast<uint16_t>(std::signbit(value) ? 0x8000U : 0U);
  const uint32_t sign = std::signbit(value) ? 0x8000U : 0U;
  int exponent = 0;
  const float normalized = std::frexp(absolute, &exponent);
  int half_exponent = exponent + 14;
  if (half_exponent <= 0) {
    const int mantissa = static_cast<int>(std::lrint(absolute * 16777216.0F));
    return static_cast<uint16_t>(
        sign | static_cast<uint32_t>(std::clamp(mantissa, 0, 1023)));
  }
  if (half_exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00U);
  const float scaled = std::ldexp(normalized, 11);
  int mantissa = static_cast<int>(std::lrint(scaled)) - 1024;
  if (mantissa >= 1024) {
    mantissa = 0;
    ++half_exponent;
  }
  return static_cast<uint16_t>(sign |
                               static_cast<uint32_t>(half_exponent << 10) |
                               static_cast<uint32_t>(mantissa));
}

float host_fp16_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 31U;
  const uint32_t mantissa = bits & 1023U;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      const float value = static_cast<float>(mantissa) * 0x1p-24F;
      uint32_t value_bits = 0U;
      std::memcpy(&value_bits, &value, sizeof(value_bits));
      result = (value_bits & 0x7fffffffU) | sign;
    }
  } else if (exponent == 31U) {
    result |= 0x7f800000U | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float output = 0.0F;
  std::memcpy(&output, &result, sizeof(output));
  return output;
}

double host_bf16_to_double(const uint16_t bits) {
  return static_cast<double>(host_bf16_to_float(bits));
}

double host_fp16_to_double(const uint16_t bits) {
  return static_cast<double>(host_fp16_to_float(bits));
}

int32_t bf16_order(const uint16_t bits) {
  return (bits & 0x8000U) != 0U ? 0x8000 - static_cast<int32_t>(bits & 0x7fffU)
                                : 0x8000 + static_cast<int32_t>(bits);
}

uint32_t bf16_ulp(const uint16_t left, const uint16_t right) {
  if ((left & 0x7fffU) == 0U && (right & 0x7fffU) == 0U)
    return 0U;
  return static_cast<uint32_t>(std::abs(bf16_order(left) - bf16_order(right)));
}

void fill_inputs(const uint64_t context, const uint32_t seed,
                 std::vector<uint16_t> *const query,
                 std::vector<uint16_t> *const key,
                 std::vector<uint16_t> *const value) {
  query->resize(static_cast<size_t>(kQHeads) * kHeadDim);
  const size_t kv_elements = static_cast<size_t>(context) * kKvHeads * kHeadDim;
  key->resize(kv_elements);
  value->resize(kv_elements);

  // seed=0 is byte-for-byte compatible with the existing partition probe's
  // deterministic fixture.  The second seed shifts the same bounded fixture
  // so the oracle is exercised on more than one finite input distribution.
  const uint32_t q_shift = seed % 97U;
  const uint32_t k_shift = seed % 131U;
  const uint32_t v_shift = seed % 127U;
  for (uint32_t head = 0U; head < kQHeads; ++head) {
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      const uint32_t phase = (head * 13U + dimension * 3U + q_shift) % 97U;
      const float source = 0.45F * std::sin(static_cast<float>(phase) * 0.071F);
      (*query)[static_cast<size_t>(head) * kHeadDim + dimension] =
          host_bf16_rne(source);
    }
  }
  for (uint64_t token = 0U; token < context; ++token) {
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const uint32_t key_phase = static_cast<uint32_t>(
            (token * 11U + head * 5U + dimension * 17U + k_shift) % 131U);
        const uint32_t value_phase = static_cast<uint32_t>(
            (token * 3U + head * 19U + dimension * 7U + v_shift) % 127U);
        const float key_source =
            0.65F * std::cos(static_cast<float>(key_phase) * 0.053F);
        const float value_source =
            0.55F * std::sin(static_cast<float>(value_phase) * 0.067F);
        const size_t index =
            (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim +
            dimension;
        (*key)[index] = host_fp16_rne(key_source);
        (*value)[index] = host_fp16_rne(value_source);
      }
    }
  }
}

// Independent double precision stable-softmax oracle.  It intentionally does
// not call or share the FP32 oracle in the partition sweep probe.
void fp64_oracle(const uint64_t context, const std::vector<uint16_t> &query,
                 const std::vector<uint16_t> &key,
                 const std::vector<uint16_t> &value,
                 std::vector<double> *const values,
                 std::vector<uint16_t> *const rounded) {
  values->assign(static_cast<size_t>(kQHeads) * kHeadDim, 0.0);
  rounded->assign(static_cast<size_t>(kQHeads) * kHeadDim, 0U);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  for (uint32_t head = 0U; head < kQHeads; ++head) {
    const uint32_t kv_head = head / kGqa;
    std::vector<double> scores(static_cast<size_t>(context));
    double maximum = -std::numeric_limits<double>::infinity();
    for (uint64_t token = 0U; token < context; ++token) {
      double dot = 0.0;
      const size_t query_base = static_cast<size_t>(head) * kHeadDim;
      const size_t key_base =
          (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        dot += host_bf16_to_double(query[query_base + dimension]) *
               host_fp16_to_double(key[key_base + dimension]);
      }
      scores[static_cast<size_t>(token)] = dot * scale;
      maximum = std::max(maximum, scores[static_cast<size_t>(token)]);
    }

    double denominator = 0.0;
    for (double &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      double numerator = 0.0;
      for (uint64_t token = 0U; token < context; ++token) {
        const size_t value_index =
            (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim +
            dimension;
        numerator += scores[static_cast<size_t>(token)] *
                     host_fp16_to_double(value[value_index]);
      }
      const double result = numerator / denominator;
      const size_t output_index =
          static_cast<size_t>(head) * kHeadDim + dimension;
      (*values)[output_index] = result;
      // The final BF16 conversion is the only float conversion in the oracle.
      // Every tested result is in the ordinary finite range, so the float
      // narrowing is exact enough to select the same BF16 RNE bin, including
      // the representable midpoint between adjacent BF16 values.
      (*rounded)[output_index] = host_bf16_rne(static_cast<float>(result));
    }
  }
}

struct ErrorStats final {
  double l1 = 0.0;
  double l2 = 0.0;
  double max_abs = 0.0;
  uint32_t max_ulp = 0U;
  uint64_t actual_nonfinite = 0U;
  uint64_t oracle_nonfinite = 0U;
  uint64_t bit_mismatch = 0U;
};

ErrorStats compare_output(const std::vector<double> &oracle,
                          const std::vector<uint16_t> &expected,
                          const std::vector<uint16_t> &actual) {
  ErrorStats result;
  if (oracle.size() != actual.size() || expected.size() != actual.size()) {
    result.actual_nonfinite = 1U;
    result.oracle_nonfinite = 1U;
    return result;
  }
  double l2_squared = 0.0;
  for (size_t index = 0U; index < actual.size(); ++index) {
    const double expected_value = oracle[index];
    const double actual_value = host_bf16_to_double(actual[index]);
    if (!std::isfinite(expected_value))
      ++result.oracle_nonfinite;
    if (!std::isfinite(actual_value))
      ++result.actual_nonfinite;
    const double error = std::fabs(actual_value - expected_value);
    result.l1 += error;
    l2_squared += error * error;
    result.max_abs = std::max(result.max_abs, error);
    result.max_ulp =
        std::max(result.max_ulp, bf16_ulp(expected[index], actual[index]));
    if (expected[index] != actual[index])
      ++result.bit_mismatch;
  }
  result.l2 = std::sqrt(l2_squared);
  return result;
}

void free_buffers(Buffers *const buffers) {
  if (buffers == nullptr)
    return;
  if (buffers->stop != nullptr)
    (void)hipEventDestroy(buffers->stop);
  if (buffers->start != nullptr)
    (void)hipEventDestroy(buffers->start);
  if (buffers->stream != nullptr)
    (void)hipStreamDestroy(buffers->stream);
  if (buffers->workspace != nullptr)
    (void)hipFree(buffers->workspace);
  if (buffers->output != nullptr)
    (void)hipFree(buffers->output);
  if (buffers->value != nullptr)
    (void)hipFree(buffers->value);
  if (buffers->key != nullptr)
    (void)hipFree(buffers->key);
  if (buffers->query != nullptr)
    (void)hipFree(buffers->query);
  *buffers = {};
}

bool make_buffers(const uint64_t context, Buffers *const buffers) {
  const size_t query_bytes =
      static_cast<size_t>(kQHeads) * kHeadDim * sizeof(uint16_t);
  const size_t kv_bytes =
      static_cast<size_t>(context) * kKvHeads * kHeadDim * sizeof(uint16_t);
  // Allocate enough for P128 and pass the same workspace to both providers.
  buffers->workspace_bytes =
      static_cast<size_t>(kQHeads) * 128U * (kHeadDim + 2U) * sizeof(float);
  if (!hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->query), query_bytes),
          "malloc query") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->key), kv_bytes),
              "malloc key") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->value), kv_bytes),
              "malloc value") ||
      !hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->output), query_bytes),
          "malloc output") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->workspace),
                        buffers->workspace_bytes),
              "malloc workspace") ||
      !hip_ok(hipStreamCreate(&buffers->stream), "create stream") ||
      !hip_ok(hipEventCreate(&buffers->start), "create start event") ||
      !hip_ok(hipEventCreate(&buffers->stop), "create stop event")) {
    free_buffers(buffers);
    return false;
  }
  return true;
}

bool upload_inputs(const std::vector<uint16_t> &query,
                   const std::vector<uint16_t> &key,
                   const std::vector<uint16_t> &value, Buffers *const buffers) {
  return hip_ok(hipMemcpy(buffers->query, query.data(),
                          query.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy query") &&
         hip_ok(hipMemcpy(buffers->key, key.data(),
                          key.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
                "copy key") &&
         hip_ok(hipMemcpy(buffers->value, value.data(),
                          value.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy value");
}

bool launch_provider(const LaunchFn provider, const uint64_t context,
                     Buffers *const buffers) {
  const hipError_t status = provider(
      buffers->query, buffers->key, buffers->value, nullptr, nullptr, nullptr,
      nullptr, buffers->output, 1U, context - 1U, context, kQHeads, kKvHeads,
      kHeadDim, SLLM_HIP_KV_ENCODING_FP16_V1, 1.0F, 1.0F, buffers->workspace,
      buffers->workspace_bytes, buffers->stream);
  return hip_ok(status, "production GQA6 launch");
}

bool copy_output(Buffers *const buffers, std::vector<uint16_t> *const output) {
  output->resize(static_cast<size_t>(kQHeads) * kHeadDim);
  return hip_ok(hipMemcpy(output->data(), buffers->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

bool compare_bitwise(const std::vector<uint16_t> &left,
                     const std::vector<uint16_t> &right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

bool measure_provider(const LaunchFn provider, const uint64_t context,
                      Buffers *const buffers, float *const median_us,
                      std::vector<uint16_t> *const output) {
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    if (!launch_provider(provider, context, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup synchronize"))
      return false;
  }
  std::array<float, kMeasured> samples{};
  for (float &sample : samples) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch_provider(provider, context, buffers) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "event synchronize") ||
        !hip_ok(hipEventElapsedTime(&sample, buffers->start, buffers->stop),
                "event elapsed"))
      return false;
    sample *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return copy_output(buffers, output);
}

bool repeat_provider(const LaunchFn provider, const uint64_t context,
                     Buffers *const buffers,
                     std::vector<uint16_t> *const output,
                     std::vector<uint16_t> *const repeat) {
  if (!launch_provider(provider, context, buffers) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "repeat synchronize 1") ||
      !copy_output(buffers, output) ||
      !launch_provider(provider, context, buffers) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "repeat synchronize 2") ||
      !copy_output(buffers, repeat))
    return false;
  return true;
}

void print_stats(const char *const candidate, const uint64_t context,
                 const uint32_t seed, const ErrorStats &stats,
                 const float median_us, const bool repeat_bitwise) {
  std::printf("oracle candidate=%s context=%llu seed=%u median_us=%.3f "
              "l1=%.9e l2=%.9e max_abs=%.9e max_bf16_ulp=%u "
              "bit_mismatch=%llu oracle_nonfinite=%llu actual_nonfinite=%llu "
              "repeat_bitwise=%s classification=N2_candidate\n",
              candidate, static_cast<unsigned long long>(context), seed,
              median_us, stats.l1, stats.l2, stats.max_abs, stats.max_ulp,
              static_cast<unsigned long long>(stats.bit_mismatch),
              static_cast<unsigned long long>(stats.oracle_nonfinite),
              static_cast<unsigned long long>(stats.actual_nonfinite),
              repeat_bitwise ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 2) {
    std::fprintf(stderr, "usage: phase78_gqa6_p64_p128_fp64_probe [device]\n");
    return EXIT_FAILURE;
  }
  uint32_t device = 0U;
  if (argc == 2) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || parsed > UINT32_MAX) {
      std::fprintf(stderr, "invalid device\n");
      return EXIT_FAILURE;
    }
    device = static_cast<uint32_t>(parsed);
  }
  if (!hip_ok(hipSetDevice(static_cast<int>(device)), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device), "device properties"))
    return EXIT_FAILURE;
  std::printf("target=%s device=%u pci=%04x:%02x:%02x name=%s "
              "contexts=8191,8192,8193,9435 q_heads=%u kv_heads=%u head_dim=%u "
              "seeds=0,7919 warmups=%u measured=%u "
              "oracle=fp64_stable_softmax_v1 classification=N2_candidate "
              "worst_case_bound=UNPROVEN\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name,
              kQHeads, kKvHeads, kHeadDim, kWarmups, kMeasured);
  if (std::string_view(properties.gcnArchName).find("gfx1030") ==
      std::string_view::npos) {
    std::fprintf(stderr, "unsupported target: %s (expected gfx1030)\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }

  const LaunchFn providers[] = {
      ::sllm_causal_attention_kernel::launch_decode_gqa6_split_p64,
      ::sllm_causal_attention_kernel::launch_decode_gqa6_split_p128};
  const char *const names[] = {"p64", "p128"};
  bool all_ok = true;
  double p64_weighted_us = 0.0;
  double p128_weighted_us = 0.0;
  for (const uint32_t seed : kSeeds) {
    for (const uint64_t context : kContexts) {
      std::vector<uint16_t> query, key, value, expected;
      std::vector<double> oracle;
      fill_inputs(context, seed, &query, &key, &value);
      fp64_oracle(context, query, key, value, &oracle, &expected);

      Buffers buffers;
      if (!make_buffers(context, &buffers) ||
          !upload_inputs(query, key, value, &buffers)) {
        free_buffers(&buffers);
        return EXIT_FAILURE;
      }
      std::vector<uint16_t> outputs[2];
      for (size_t provider_index = 0U; provider_index < 2U; ++provider_index) {
        float median_us = 0.0F;
        if (!measure_provider(providers[provider_index], context, &buffers,
                              &median_us, &outputs[provider_index])) {
          all_ok = false;
          break;
        }
        std::vector<uint16_t> repeat;
        if (!repeat_provider(providers[provider_index], context, &buffers,
                             &outputs[provider_index], &repeat)) {
          all_ok = false;
          break;
        }
        const bool repeat_bitwise =
            compare_bitwise(outputs[provider_index], repeat);
        const ErrorStats stats =
            compare_output(oracle, expected, outputs[provider_index]);
        print_stats(names[provider_index], context, seed, stats, median_us,
                    repeat_bitwise);
        if (!repeat_bitwise || stats.actual_nonfinite != 0U ||
            stats.oracle_nonfinite != 0U)
          all_ok = false;
        if (provider_index == 0U)
          p64_weighted_us += static_cast<double>(median_us);
        else
          p128_weighted_us += static_cast<double>(median_us);
      }
      if (outputs[0].size() == outputs[1].size() && !outputs[0].empty()) {
        uint32_t max_ulp = 0U;
        uint64_t mismatches = 0U;
        for (size_t index = 0U; index < outputs[0].size(); ++index) {
          max_ulp =
              std::max(max_ulp, bf16_ulp(outputs[0][index], outputs[1][index]));
          if (outputs[0][index] != outputs[1][index])
            ++mismatches;
        }
        std::printf("candidate_compare context=%llu seed=%u "
                    "p64_vs_p128_max_bf16_ulp=%u "
                    "bit_mismatch=%llu\n",
                    static_cast<unsigned long long>(context), seed, max_ulp,
                    static_cast<unsigned long long>(mismatches));
      }
      free_buffers(&buffers);
      if (!all_ok)
        break;
    }
    if (!all_ok)
      break;
  }
  std::printf(
      "weighted_summary rows=%zu p64_median_sum_us=%.3f "
      "p128_median_sum_us=%.3f p128_vs_p64=%.6f status=%s\n",
      kSeeds.size() * kContexts.size(), p64_weighted_us, p128_weighted_us,
      p128_weighted_us == 0.0 ? 0.0 : p64_weighted_us / p128_weighted_us,
      all_ok ? "PASS" : "FAIL");
  std::printf(
      "classification=N2_candidate reason=finite measured FP64 errors; "
      "same GQA6 equation; no worst_case_proof; measured samples are not a "
      "worst_case bound; N1 absence alone is not N3\n");
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
