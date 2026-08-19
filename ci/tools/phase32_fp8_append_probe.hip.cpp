#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kBaselineThreads = 256U;
constexpr uint32_t kPackedThreads = 128U;
constexpr uint32_t kWarmups = 5U;
constexpr uint32_t kMeasured = 31U;

enum class Provider : uint32_t {
  Software = 0U,
  NativeScalar = 1U,
  NativePacked = 2U,
};

const char *provider_name(const Provider provider) {
  switch (provider) {
  case Provider::Software:
    return "software";
  case Provider::NativeScalar:
    return "native-scalar";
  case Provider::NativePacked:
    return "native-packed";
  }
  return "invalid";
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}

__host__ __device__ float bf16_to_float(const uint16_t value) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
#else
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
#endif
}

uint16_t float_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t exponent = bits & UINT32_C(0x7f800000);
  const uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponent == UINT32_C(0x7f800000)) {
    if (fraction != 0U) {
      return static_cast<uint16_t>((bits >> 16U) | UINT32_C(0x0040));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  const uint32_t bias = UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>((bits + bias) >> 16U);
}

__device__ float e4m3fn_to_float(const uint8_t bits) {
  const float sign = (bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F;
  const uint8_t exponent = static_cast<uint8_t>((bits >> 3U) & 0x0fU);
  const uint8_t mantissa = static_cast<uint8_t>(bits & 0x07U);
  if (exponent == 0U) {
    return mantissa == 0U
               ? copysignf(0.0F, sign)
               : sign * static_cast<float>(mantissa) * ldexpf(1.0F, -9);
  }
  if (exponent == 0x0fU && mantissa == 0x07U) {
    return NAN;
  }
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         ldexpf(1.0F, static_cast<int>(exponent) - 7);
}

__device__ uint8_t software_encode(float value) {
  const uint8_t sign = signbit(value) ? UINT8_C(0x80) : 0U;
  if (isnan(value)) {
    return UINT8_C(0x7f);
  }
  value = fabsf(value);
  if (value == 0.0F) {
    return sign;
  }
  if (!isfinite(value) || value >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  uint32_t low = 0U;
  uint32_t high = UINT32_C(0x7e);
  while (low < high) {
    const uint32_t middle = (low + high) >> 1U;
    if (e4m3fn_to_float(static_cast<uint8_t>(middle)) < value) {
      low = middle + 1U;
    } else {
      high = middle;
    }
  }
  const uint8_t upper = static_cast<uint8_t>(low);
  const uint8_t lower = upper == 0U ? 0U : static_cast<uint8_t>(upper - 1U);
  const float lower_error = value - e4m3fn_to_float(lower);
  const float upper_error = e4m3fn_to_float(upper) - value;
  const bool upper_selected =
      upper_error < lower_error ||
      (upper_error == lower_error && (upper & 1U) == 0U && (lower & 1U) != 0U);
  return static_cast<uint8_t>(sign | (upper_selected ? upper : lower));
}

__device__ uint8_t native_scalar_encode(const float value) {
#if defined(__gfx1201__)
  if (isnan(value)) {
    return UINT8_C(0x7f);
  }
  const uint8_t sign = signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = fabsf(value);
  if (magnitude == 0.0F) {
    return sign;
  }
  if (!isfinite(magnitude) || magnitude >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  const uint32_t packed =
      __builtin_amdgcn_cvt_pk_fp8_f32(value, value, 0, false);
  return static_cast<uint8_t>(packed & UINT32_C(0xff));
#else
  return software_encode(value);
#endif
}

__device__ uint16_t native_pair_encode(const float first, const float second) {
#if defined(__gfx1201__)
  const bool first_regular = isfinite(first) && fabsf(first) < 448.0F;
  const bool second_regular = isfinite(second) && fabsf(second) < 448.0F;
  if (first_regular && second_regular) {
    return static_cast<uint16_t>(
        __builtin_amdgcn_cvt_pk_fp8_f32(first, second, 0, false));
  }
  return static_cast<uint16_t>(native_scalar_encode(first)) |
         static_cast<uint16_t>(native_scalar_encode(second)) << 8U;
#else
  return static_cast<uint16_t>(software_encode(first)) |
         static_cast<uint16_t>(software_encode(second)) << 8U;
#endif
}

template <Provider Selected>
__global__ __launch_bounds__(kBaselineThreads, 1) void append_scalar_kernel(
    const uint16_t *const key_input, const uint16_t *const value_input,
    uint8_t *const key_output, uint8_t *const value_output,
    float *const key_scales, float *const value_scales,
    const uint32_t token_count, const float fixed_key_scale,
    const float fixed_value_scale, const uint32_t static_mode) {
  const uint64_t row = blockIdx.x;
  if (row >= static_cast<uint64_t>(token_count) * kHeads) {
    return;
  }
  const uint64_t input_base = row * kHeadDim;
  __shared__ float key_maxima[kBaselineThreads];
  __shared__ float value_maxima[kBaselineThreads];
  const uint32_t dimension = threadIdx.x;
  const float key_value = bf16_to_float(key_input[input_base + dimension]);
  const float value_value = bf16_to_float(value_input[input_base + dimension]);
  key_maxima[dimension] = isfinite(key_value) ? fabsf(key_value) : 0.0F;
  value_maxima[dimension] = isfinite(value_value) ? fabsf(value_value) : 0.0F;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (dimension < stride) {
      key_maxima[dimension] =
          fmaxf(key_maxima[dimension], key_maxima[dimension + stride]);
      value_maxima[dimension] =
          fmaxf(value_maxima[dimension], value_maxima[dimension + stride]);
    }
    __syncthreads();
  }
  const float key_scale =
      static_mode != 0U
          ? fixed_key_scale
          : (key_maxima[0] == 0.0F ? 1.0F : key_maxima[0] / 448.0F);
  const float value_scale =
      static_mode != 0U
          ? fixed_value_scale
          : (value_maxima[0] == 0.0F ? 1.0F : value_maxima[0] / 448.0F);
  if (dimension == 0U) {
    key_scales[row] = key_scale;
    value_scales[row] = value_scale;
  }
  const float normalized_key = key_value / key_scale;
  const float normalized_value = value_value / value_scale;
  if constexpr (Selected == Provider::Software) {
    key_output[input_base + dimension] = software_encode(normalized_key);
    value_output[input_base + dimension] = software_encode(normalized_value);
  } else {
    key_output[input_base + dimension] = native_scalar_encode(normalized_key);
    value_output[input_base + dimension] =
        native_scalar_encode(normalized_value);
  }
}

__global__ __launch_bounds__(kPackedThreads, 1) void append_packed_kernel(
    const uint16_t *const key_input, const uint16_t *const value_input,
    uint8_t *const key_output, uint8_t *const value_output,
    float *const key_scales, float *const value_scales,
    const uint32_t token_count, const float fixed_key_scale,
    const float fixed_value_scale, const uint32_t static_mode) {
  const uint64_t row = blockIdx.x;
  if (row >= static_cast<uint64_t>(token_count) * kHeads) {
    return;
  }
  const uint64_t input_base = row * kHeadDim;
  const uint32_t pair = threadIdx.x;
  const uint32_t first_dimension = pair * 2U;
  const uint32_t second_dimension = first_dimension + 1U;
  const float key_first =
      bf16_to_float(key_input[input_base + first_dimension]);
  const float key_second =
      bf16_to_float(key_input[input_base + second_dimension]);
  const float value_first =
      bf16_to_float(value_input[input_base + first_dimension]);
  const float value_second =
      bf16_to_float(value_input[input_base + second_dimension]);
  __shared__ float key_maxima[kPackedThreads];
  __shared__ float value_maxima[kPackedThreads];
  key_maxima[pair] = fmaxf(isfinite(key_first) ? fabsf(key_first) : 0.0F,
                           isfinite(key_second) ? fabsf(key_second) : 0.0F);
  value_maxima[pair] =
      fmaxf(isfinite(value_first) ? fabsf(value_first) : 0.0F,
            isfinite(value_second) ? fabsf(value_second) : 0.0F);
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (pair < stride) {
      key_maxima[pair] = fmaxf(key_maxima[pair], key_maxima[pair + stride]);
      value_maxima[pair] =
          fmaxf(value_maxima[pair], value_maxima[pair + stride]);
    }
    __syncthreads();
  }
  const float key_scale =
      static_mode != 0U
          ? fixed_key_scale
          : (key_maxima[0] == 0.0F ? 1.0F : key_maxima[0] / 448.0F);
  const float value_scale =
      static_mode != 0U
          ? fixed_value_scale
          : (value_maxima[0] == 0.0F ? 1.0F : value_maxima[0] / 448.0F);
  if (pair == 0U) {
    key_scales[row] = key_scale;
    value_scales[row] = value_scale;
  }
  const uint16_t key_packed =
      native_pair_encode(key_first / key_scale, key_second / key_scale);
  const uint16_t value_packed =
      native_pair_encode(value_first / value_scale, value_second / value_scale);
  reinterpret_cast<uint16_t *>(key_output + input_base)[pair] = key_packed;
  reinterpret_cast<uint16_t *>(value_output + input_base)[pair] = value_packed;
}

hipError_t launch(const Provider provider, const uint16_t *const key_input,
                  const uint16_t *const value_input, uint8_t *const key_output,
                  uint8_t *const value_output, float *const key_scales,
                  float *const value_scales, const uint32_t token_count,
                  const bool static_mode, const hipStream_t stream) {
  const dim3 grid(token_count * kHeads, 1U, 1U);
  if (provider == Provider::Software) {
    hipLaunchKernelGGL(append_scalar_kernel<Provider::Software>, grid,
                       dim3(kBaselineThreads), 0U, stream, key_input,
                       value_input, key_output, value_output, key_scales,
                       value_scales, token_count, 1.0F, 1.0F,
                       static_mode ? 1U : 0U);
  } else if (provider == Provider::NativeScalar) {
    hipLaunchKernelGGL(append_scalar_kernel<Provider::NativeScalar>, grid,
                       dim3(kBaselineThreads), 0U, stream, key_input,
                       value_input, key_output, value_output, key_scales,
                       value_scales, token_count, 1.0F, 1.0F,
                       static_mode ? 1U : 0U);
  } else {
    hipLaunchKernelGGL(append_packed_kernel, grid, dim3(kPackedThreads), 0U,
                       stream, key_input, value_input, key_output, value_output,
                       key_scales, value_scales, token_count, 1.0F, 1.0F,
                       static_mode ? 1U : 0U);
  }
  return hipGetLastError();
}

struct DeviceBuffers {
  uint16_t *key_input = nullptr;
  uint16_t *value_input = nullptr;
  uint8_t *key_output = nullptr;
  uint8_t *value_output = nullptr;
  float *key_scales = nullptr;
  float *value_scales = nullptr;

  ~DeviceBuffers() {
    static_cast<void>(hipFree(value_scales));
    static_cast<void>(hipFree(key_scales));
    static_cast<void>(hipFree(value_output));
    static_cast<void>(hipFree(key_output));
    static_cast<void>(hipFree(value_input));
    static_cast<void>(hipFree(key_input));
  }
};

struct HostResult {
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<float> key_scales;
  std::vector<float> value_scales;
};

struct Timing {
  double median_ns = 0.0;
  double mad_ns = 0.0;
  double p10_ns = 0.0;
  double p90_ns = 0.0;
  uint32_t inner_iterations = 0U;
};

double percentile(const std::vector<double> &sorted, const double fraction) {
  const double position = fraction * static_cast<double>(sorted.size() - 1U);
  const size_t low = static_cast<size_t>(position);
  const size_t high = std::min(low + 1U, sorted.size() - 1U);
  const double weight = position - static_cast<double>(low);
  return sorted[low] * (1.0 - weight) + sorted[high] * weight;
}

bool same_float_bits(const std::vector<float> &left,
                     const std::vector<float> &right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0;
}

bool copy_result(const DeviceBuffers &buffers, const uint64_t elements,
                 const uint64_t rows, HostResult *const result) {
  result->key.resize(elements);
  result->value.resize(elements);
  result->key_scales.resize(rows);
  result->value_scales.resize(rows);
  return hip_ok(hipMemcpy(result->key.data(), buffers.key_output, elements,
                          hipMemcpyDeviceToHost),
                "hipMemcpy key output") &&
         hip_ok(hipMemcpy(result->value.data(), buffers.value_output, elements,
                          hipMemcpyDeviceToHost),
                "hipMemcpy value output") &&
         hip_ok(hipMemcpy(result->key_scales.data(), buffers.key_scales,
                          rows * sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy key scales") &&
         hip_ok(hipMemcpy(result->value_scales.data(), buffers.value_scales,
                          rows * sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy value scales");
}

Timing measure(const Provider provider, const DeviceBuffers &buffers,
               const uint32_t token_count, const bool static_mode,
               const hipStream_t stream) {
  const uint32_t inner = std::min(
      UINT32_C(64), std::max(UINT32_C(1), UINT32_C(8192) / token_count));
  for (uint32_t warmup = 0U; warmup != kWarmups; ++warmup) {
    for (uint32_t iteration = 0U; iteration != inner; ++iteration) {
      if (!hip_ok(launch(provider, buffers.key_input, buffers.value_input,
                         buffers.key_output, buffers.value_output,
                         buffers.key_scales, buffers.value_scales, token_count,
                         static_mode, stream),
                  "warmup launch")) {
        std::exit(2);
      }
    }
  }
  if (!hip_ok(hipStreamSynchronize(stream), "warmup synchronize")) {
    std::exit(2);
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!hip_ok(hipEventCreate(&start), "hipEventCreate start") ||
      !hip_ok(hipEventCreate(&stop), "hipEventCreate stop")) {
    std::exit(2);
  }
  std::vector<double> samples;
  samples.reserve(kMeasured);
  for (uint32_t sample = 0U; sample != kMeasured; ++sample) {
    if (!hip_ok(hipEventRecord(start, stream), "hipEventRecord start")) {
      std::exit(2);
    }
    for (uint32_t iteration = 0U; iteration != inner; ++iteration) {
      if (!hip_ok(launch(provider, buffers.key_input, buffers.value_input,
                         buffers.key_output, buffers.value_output,
                         buffers.key_scales, buffers.value_scales, token_count,
                         static_mode, stream),
                  "measured launch")) {
        std::exit(2);
      }
    }
    if (!hip_ok(hipEventRecord(stop, stream), "hipEventRecord stop") ||
        !hip_ok(hipEventSynchronize(stop), "hipEventSynchronize stop")) {
      std::exit(2);
    }
    float elapsed_ms = 0.0F;
    if (!hip_ok(hipEventElapsedTime(&elapsed_ms, start, stop),
                "hipEventElapsedTime")) {
      std::exit(2);
    }
    samples.push_back(static_cast<double>(elapsed_ms) * 1000000.0 /
                      static_cast<double>(inner));
  }
  static_cast<void>(hipEventDestroy(stop));
  static_cast<void>(hipEventDestroy(start));
  std::sort(samples.begin(), samples.end());
  const double median = percentile(samples, 0.5);
  std::vector<double> deviations;
  deviations.reserve(samples.size());
  for (const double sample : samples) {
    deviations.push_back(std::fabs(sample - median));
  }
  std::sort(deviations.begin(), deviations.end());
  return Timing{median, percentile(deviations, 0.5), percentile(samples, 0.1),
                percentile(samples, 0.9), inner};
}

bool run_case(const std::string &target, const uint32_t token_count,
              const bool static_mode, const bool exhaustive) {
  const uint64_t rows = static_cast<uint64_t>(token_count) * kHeads;
  const uint64_t elements = rows * kHeadDim;
  std::vector<uint16_t> key(elements);
  std::vector<uint16_t> value(elements);
  for (uint64_t element = 0U; element != elements; ++element) {
    if (exhaustive) {
      key[element] = static_cast<uint16_t>(element & UINT16_MAX);
      value[element] = static_cast<uint16_t>(
          (element + static_cast<uint64_t>(elements)) & UINT16_MAX);
    } else {
      const int32_t key_integer =
          static_cast<int32_t>((element * 17U + 13U) % 2001U) - 1000;
      const int32_t value_integer =
          static_cast<int32_t>((element * 29U + 7U) % 1601U) - 800;
      key[element] = float_to_bf16_rne(static_cast<float>(key_integer) / 32.0F);
      value[element] =
          float_to_bf16_rne(static_cast<float>(value_integer) / 16.0F);
    }
  }

  DeviceBuffers buffers;
  if (!hip_ok(hipMalloc(&buffers.key_input, elements * sizeof(uint16_t)),
              "hipMalloc key input") ||
      !hip_ok(hipMalloc(&buffers.value_input, elements * sizeof(uint16_t)),
              "hipMalloc value input") ||
      !hip_ok(hipMalloc(&buffers.key_output, elements),
              "hipMalloc key output") ||
      !hip_ok(hipMalloc(&buffers.value_output, elements),
              "hipMalloc value output") ||
      !hip_ok(hipMalloc(&buffers.key_scales, rows * sizeof(float)),
              "hipMalloc key scales") ||
      !hip_ok(hipMalloc(&buffers.value_scales, rows * sizeof(float)),
              "hipMalloc value scales") ||
      !hip_ok(hipMemcpy(buffers.key_input, key.data(),
                        elements * sizeof(uint16_t), hipMemcpyHostToDevice),
              "hipMemcpy key input") ||
      !hip_ok(hipMemcpy(buffers.value_input, value.data(),
                        elements * sizeof(uint16_t), hipMemcpyHostToDevice),
              "hipMemcpy value input")) {
    return false;
  }

  hipStream_t stream = nullptr;
  if (!hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
              "hipStreamCreateWithFlags")) {
    return false;
  }
  const Provider providers[] = {Provider::Software, Provider::NativeScalar,
                                Provider::NativePacked};
  const bool native_available = target.rfind("gfx1201", 0U) == 0U;
  const size_t provider_count =
      native_available ? sizeof(providers) / sizeof(providers[0]) : 1U;
  HostResult baseline;
  bool passed = true;
  for (size_t provider_index = 0U; provider_index != provider_count;
       ++provider_index) {
    const Provider provider = providers[provider_index];
    if (!hip_ok(launch(provider, buffers.key_input, buffers.value_input,
                       buffers.key_output, buffers.value_output,
                       buffers.key_scales, buffers.value_scales, token_count,
                       static_mode, stream),
                "correctness launch") ||
        !hip_ok(hipStreamSynchronize(stream), "correctness synchronize")) {
      passed = false;
      break;
    }
    HostResult result;
    if (!copy_result(buffers, elements, rows, &result)) {
      passed = false;
      break;
    }
    uint64_t mismatches = 0U;
    if (provider == Provider::Software) {
      baseline = std::move(result);
    } else {
      for (uint64_t element = 0U; element != elements; ++element) {
        mismatches += baseline.key[element] != result.key[element] ? 1U : 0U;
        mismatches +=
            baseline.value[element] != result.value[element] ? 1U : 0U;
      }
      if (!same_float_bits(baseline.key_scales, result.key_scales) ||
          !same_float_bits(baseline.value_scales, result.value_scales)) {
        mismatches += 1U;
      }
    }
    const Timing timing =
        measure(provider, buffers, token_count, static_mode, stream);
    std::printf("{\"schema\":\"phase32-fp8-append-probe-v1\",\"target\":\"%s\","
                "\"tokens\":%u,\"heads\":%u,\"head_dim\":%u,"
                "\"encoding\":\"%s\",\"fixture\":\"%s\","
                "\"native_available\":%s,\"provider\":\"%s\","
                "\"mismatches\":%llu,"
                "\"median_ns\":%.3f,\"mad_ns\":%.3f,\"p10_ns\":%.3f,"
                "\"p90_ns\":%.3f,\"inner_iterations\":%u,"
                "\"warmups\":%u,\"measured\":%u}\n",
                target.c_str(), token_count, kHeads, kHeadDim,
                static_mode ? "fp8-static" : "fp8-dynamic",
                exhaustive ? "exhaustive-bf16" : "finite-production-like",
                native_available ? "true" : "false", provider_name(provider),
                static_cast<unsigned long long>(mismatches), timing.median_ns,
                timing.mad_ns, timing.p10_ns, timing.p90_ns,
                timing.inner_iterations, kWarmups, kMeasured);
    passed = passed && mismatches == 0U;
  }
  static_cast<void>(hipStreamDestroy(stream));
  return passed;
}

std::vector<uint32_t> parse_tokens(const int argc, char **argv) {
  if (argc == 1) {
    return {1U,     31U,    32U,    33U,    255U,  256U,  257U,
            511U,   512U,   513U,   2047U,  2048U, 2049U, 9999U,
            10000U, 10001U, 16383U, 16384U, 16385U};
  }
  std::vector<uint32_t> tokens;
  for (int index = 1; index != argc; ++index) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(argv[index], &end, 10);
    if (argv[index][0] == '\0' || end == nullptr || *end != '\0' ||
        parsed == 0UL || parsed > UINT32_MAX) {
      std::fprintf(stderr, "invalid token count: %s\n", argv[index]);
      std::exit(2);
    }
    tokens.push_back(static_cast<uint32_t>(parsed));
  }
  return tokens;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDevice(&device), "hipGetDevice") ||
      !hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties")) {
    return 2;
  }
  const std::string target(properties.gcnArchName);
  const bool exact_gfx1201 = target.rfind("gfx1201", 0U) == 0U;
  const bool exact_gfx1030 = target.rfind("gfx1030", 0U) == 0U;
  if (!exact_gfx1201 && !exact_gfx1030) {
    std::fprintf(stderr, "unsupported exact target: %s\n", target.c_str());
    return 2;
  }
  bool passed = true;
  if (exact_gfx1201) {
    passed = run_case(target, 32U, false, true) && passed;
    passed = run_case(target, 32U, true, true) && passed;
  }
  for (const uint32_t tokens : parse_tokens(argc, argv)) {
    passed = run_case(target, tokens, false, false) && passed;
    passed = run_case(target, tokens, true, false) && passed;
  }
  if (!hip_ok(hipDeviceSynchronize(), "terminal synchronize")) {
    return 2;
  }
  std::fprintf(stderr,
               "phase32_fp8_append_probe: %s target=%s fallback=false\n",
               passed ? "PASS" : "FAIL", target.c_str());
  return passed ? 0 : 1;
}
