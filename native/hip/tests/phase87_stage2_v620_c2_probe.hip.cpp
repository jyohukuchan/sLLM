// Phase 87 WU-2V C2 standalone gfx1030 FP8 M=1 probe.
//
// The linked production ID82 launcher is the control.  C2 stages the
// activation row once into shared FP16 and reuses it across all eight wave32s;
// the weight path and arithmetic order remain the ID82 four-column LUT path.

#include "../src/matmul_kernel_internal.hpp"
#include "phase87_stage2_v620_c2_kernel.hpp"
#include <lowp/detail/lowp_kernel_internal.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using phase87_stage2_v620_c2::kK;
using phase87_stage2_v620_c2::kN;

void need(const bool ok, const char *const message) {
  if (!ok)
    throw std::runtime_error(message);
}

void hipcheck(const hipError_t status, const char *const operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

std::size_t live_allocations = 0U;
bool frees_ok = true;

template <typename T> struct DeviceBuffer final {
  T *pointer = nullptr;
  std::size_t count = 0U;
  explicit DeviceBuffer(const std::size_t elements) : count(elements) {
    hipcheck(hipMalloc(reinterpret_cast<void **>(&pointer),
                       std::max<std::size_t>(1U, elements * sizeof(T))),
             "hipMalloc");
    ++live_allocations;
  }
  ~DeviceBuffer() {
    if (pointer != nullptr) {
      if (hipFree(pointer) == hipSuccess)
        --live_allocations;
      else
        frees_ok = false;
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  void put(const std::vector<T> &host) const {
    need(host.size() == count, "host/device size mismatch");
    hipcheck(hipMemcpy(pointer, host.data(), count * sizeof(T),
                       hipMemcpyHostToDevice),
             "hipMemcpy H2D");
  }
  std::vector<T> get() const {
    std::vector<T> host(count);
    hipcheck(hipMemcpy(host.data(), pointer, count * sizeof(T),
                       hipMemcpyDeviceToHost),
             "hipMemcpy D2H");
    return host;
  }
};

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
  uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float e4m3fn_decode(const uint8_t code) {
  const uint32_t exponent = (code >> 3U) & 15U;
  const uint32_t mantissa = code & 7U;
  if (exponent == 15U && mantissa == 7U)
    return std::numeric_limits<float>::quiet_NaN();
  const float magnitude =
      exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -9)
                     : std::ldexp(1.0F + static_cast<float>(mantissa) * 0.125F,
                                  static_cast<int>(exponent) - 7);
  return (code & 0x80U) == 0U ? magnitude : -magnitude;
}

uint8_t e4m3fn_encode(const float value) {
  const bool negative = std::signbit(value);
  const float magnitude = std::abs(value);
  float best = std::numeric_limits<float>::infinity();
  uint8_t result = 0U;
  for (uint32_t code = 0U; code <= 126U; ++code) {
    const float distance =
        std::abs(magnitude - e4m3fn_decode(static_cast<uint8_t>(code)));
    if (distance < best || (distance == best && (code & 1U) == 0U)) {
      best = distance;
      result = static_cast<uint8_t>(code);
    }
  }
  return static_cast<uint8_t>(result | (negative ? 0x80U : 0U));
}

__host__ __device__ uint8_t weight_code(const uint64_t column,
                                        const uint64_t index,
                                        const int pattern) {
  if (pattern == 1)
    return static_cast<uint8_t>(0x38U | ((index & 1U) ? 0x80U : 0U));
  uint32_t state = static_cast<uint32_t>(column) * 1664525U +
                   static_cast<uint32_t>(index) * 1013904223U;
  state ^= state >> 13U;
  const uint8_t magnitude = static_cast<uint8_t>(0x20U + (state % 32U));
  return static_cast<uint8_t>(magnitude | ((state % 13U == 0U) ? 0x80U : 0U));
}

__global__ void fill_weights(uint8_t *const destination, const uint64_t bytes,
                             const uint64_t k, const uint64_t n,
                             const int pattern) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < bytes) {
    const uint64_t local = index % (k * n);
    destination[index] = weight_code(local / k, local % k, pattern);
  }
}

struct Buffers final {
  const uint64_t copies;
  DeviceBuffer<uint16_t> activation{kK};
  DeviceBuffer<uint8_t> activation_fp8{kK};
  DeviceBuffer<float> activation_scale{1U};
  DeviceBuffer<uint8_t> weights;
  DeviceBuffer<float> weight_scales{kN};
  DeviceBuffer<uint16_t> control_output{kN + 32U};
  DeviceBuffer<uint16_t> candidate_output{kN + 32U};

  explicit Buffers(const uint64_t copy_count)
      : copies(copy_count),
        weights(static_cast<std::size_t>(copy_count * kK * kN)) {}

  void quantize() {
    hipcheck(sllm_matmul_kernel::launch_fp8_quantize(
                 activation.pointer, activation_fp8.pointer,
                 activation_scale.pointer, 1U, kK, false, nullptr),
             "production FP8 quantizer");
  }

  void control(const uint64_t copy) {
    hipcheck(
        sllm_matmul_kernel::launch_fp8_outer_decode_gfx1030_lds_lut_wave4col32(
            activation_fp8.pointer, activation_scale.pointer,
            weights.pointer + copy * kK * kN, weight_scales.pointer,
            control_output.pointer, 1U, kK, kN, nullptr),
        "production ID82 launcher");
  }

  void candidate(const uint64_t copy) {
    hipLaunchKernelGGL(phase87_stage2_v620_c2::candidate_kernel,
                       dim3((kN + 31U) / 32U), dim3(256U), 0U, nullptr,
                       activation_fp8.pointer, activation_scale.pointer,
                       weights.pointer + copy * kK * kN, weight_scales.pointer,
                       candidate_output.pointer, 1U, kK, kN);
    hipcheck(hipGetLastError(), "C2 candidate launch");
  }
};

struct Timing final {
  float quant_ms;
  float dot_ms;
  float total_ms;
};

Timing timed(Buffers &buffers, const bool candidate, const uint64_t copy) {
  hipEvent_t start = nullptr, quant_end = nullptr, end = nullptr;
  hipcheck(hipEventCreate(&start), "hipEventCreate start");
  hipcheck(hipEventCreate(&quant_end), "hipEventCreate quant");
  hipcheck(hipEventCreate(&end), "hipEventCreate end");
  hipcheck(hipEventRecord(start), "hipEventRecord start");
  buffers.quantize();
  hipcheck(hipEventRecord(quant_end), "hipEventRecord quant");
  if (candidate)
    buffers.candidate(copy);
  else
    buffers.control(copy);
  hipcheck(hipEventRecord(end), "hipEventRecord end");
  hipcheck(hipEventSynchronize(end), "hipEventSynchronize");
  Timing result{};
  hipcheck(hipEventElapsedTime(&result.quant_ms, start, quant_end),
           "hipEventElapsedTime quant");
  hipcheck(hipEventElapsedTime(&result.dot_ms, quant_end, end),
           "hipEventElapsedTime dot");
  hipcheck(hipEventElapsedTime(&result.total_ms, start, end),
           "hipEventElapsedTime total");
  hipcheck(hipEventDestroy(start), "hipEventDestroy start");
  hipcheck(hipEventDestroy(quant_end), "hipEventDestroy quant");
  hipcheck(hipEventDestroy(end), "hipEventDestroy end");
  return result;
}

void warm(Buffers &buffers, const bool candidate) {
  const auto start = std::chrono::steady_clock::now();
  uint64_t copy = 0U;
  do {
    for (unsigned index = 0U; index < 32U; ++index) {
      buffers.quantize();
      if (candidate)
        buffers.candidate(copy++ % buffers.copies);
      else
        buffers.control(copy++ % buffers.copies);
    }
    hipcheck(hipDeviceSynchronize(), "warmup synchronize");
  } while (std::chrono::steady_clock::now() - start <
           std::chrono::milliseconds(300));
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

void print_samples(const char *const name, const std::vector<float> &values) {
  std::cout << ",\"" << name << "\":[";
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

struct Compare final {
  bool finite = true;
  bool guard = true;
  uint32_t max_ulp = 0U;
  float max_abs = 0.0F;
};

Compare compare_output(const std::vector<uint16_t> &actual,
                       const std::vector<uint16_t> &expected) {
  constexpr uint16_t kCanary = UINT16_C(0x5a5a);
  Compare result;
  need(actual.size() == kN + 32U && expected.size() == kN, "compare size");
  for (uint64_t index = 0U; index < kN; ++index) {
    result.finite &= std::isfinite(bf16_to_f32(actual[index]));
    const uint32_t ulp = actual[index] >= expected[index]
                             ? actual[index] - expected[index]
                             : expected[index] - actual[index];
    result.max_ulp = std::max(result.max_ulp, ulp);
    result.max_abs =
        std::max(result.max_abs, std::abs(bf16_to_f32(actual[index]) -
                                          bf16_to_f32(expected[index])));
  }
  for (uint64_t index = kN; index < actual.size(); ++index)
    result.guard &= actual[index] == kCanary;
  return result;
}

std::vector<uint16_t> make_activation(const int pattern) {
  std::vector<uint16_t> result(kK);
  for (uint64_t index = 0U; index < kK; ++index) {
    float value = 0.0F;
    if (pattern == 1)
      value = (index & 1U) == 0U ? 1.0F : -1.0F;
    else if (pattern == 2)
      value = (index % 7U == 0U) ? -0.75F : ((index % 5U) * 0.25F);
    else
      value = static_cast<float>(16U + (index * 17U) % 49U) / 64.0F;
    result[index] = f32_to_bf16(value);
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string target = "gfx1030";
    int pattern = 0;
    bool bench = true;
    for (int index = 1; index < argc; ++index) {
      need(index + 1 < argc, "argument value");
      const std::string key = argv[index];
      const std::string value = argv[++index];
      if (key == "--target")
        target = value;
      else if (key == "--pattern")
        pattern = std::stoi(value);
      else if (key == "--bench")
        bench = std::stoi(value) != 0;
      else
        throw std::runtime_error("unknown argument");
    }
    need(target == "gfx1030", "C2 is exact gfx1030 only");
    need(pattern >= 0 && pattern <= 2, "pattern must be 0, 1 or 2");
    int device_index = 0;
    hipcheck(hipGetDevice(&device_index), "hipGetDevice");
    hipDeviceProp_t properties{};
    hipcheck(hipGetDeviceProperties(&properties, device_index),
             "hipGetDeviceProperties");
    need(target == properties.gcnArchName, "visible GPU target mismatch");

    const uint64_t weight_bytes = kK * kN;
    const uint64_t copies = std::max<uint64_t>(
        1U, ((UINT64_C(512) << 20) + weight_bytes - 1U) / weight_bytes);
    auto owner = std::make_unique<Buffers>(copies);
    Buffers &buffers = *owner;
    const std::vector<uint16_t> activation = make_activation(pattern);
    std::vector<float> weight_scales(kN);
    for (uint64_t column = 0U; column < kN; ++column)
      weight_scales[column] =
          pattern == 1 ? 1.0F : static_cast<float>(8U + column % 7U) / 256.0F;
    buffers.activation.put(activation);
    buffers.weight_scales.put(weight_scales);
    hipLaunchKernelGGL(
        fill_weights, dim3((buffers.weights.count + 255U) / 256U), dim3(256U),
        0U, nullptr, buffers.weights.pointer,
        static_cast<uint64_t>(buffers.weights.count), kK, kN, pattern);
    hipcheck(hipGetLastError(), "fill weights launch");
    hipcheck(hipDeviceSynchronize(), "fill weights synchronize");
    buffers.quantize();
    hipcheck(hipDeviceSynchronize(), "quantizer synchronize");
    const auto quantized = buffers.activation_fp8.get();
    const auto scales = buffers.activation_scale.get();
    float maximum = 0.0F;
    for (const auto value : activation)
      maximum = std::max(maximum, std::abs(bf16_to_f32(value)));
    const float expected_scale = maximum == 0.0F ? 1.0F : maximum / 448.0F;
    std::vector<uint8_t> expected_quantized(kK);
    for (uint64_t index = 0U; index < kK; ++index)
      expected_quantized[index] =
          e4m3fn_encode(bf16_to_f32(activation[index]) / expected_scale);
    const bool quantizer_bitwise =
        quantized == expected_quantized && scales.size() == 1U &&
        std::memcmp(scales.data(), &expected_scale, sizeof(float)) == 0;
    need(quantizer_bitwise, "production quantizer differs from oracle");

    std::vector<uint16_t> expected(kN);
    for (uint64_t column = 0U; column < kN; ++column) {
      float sum = 0.0F;
      for (uint64_t index = 0U; index < kK; ++index)
        sum += e4m3fn_decode(quantized[index]) *
               e4m3fn_decode(weight_code(column, index, pattern));
      expected[column] = f32_to_bf16(sum * scales[0] * weight_scales[column]);
    }

    hipcheck(hipMemset(buffers.control_output.pointer, 0x5a,
                       (kN + 32U) * sizeof(uint16_t)),
             "control canary");
    buffers.control(0U);
    hipcheck(hipDeviceSynchronize(), "control synchronize");
    const auto control_first = buffers.control_output.get();
    hipcheck(hipMemset(buffers.control_output.pointer, 0x5a,
                       (kN + 32U) * sizeof(uint16_t)),
             "control repeat canary");
    buffers.control(0U);
    hipcheck(hipDeviceSynchronize(), "control repeat synchronize");
    const auto control_second = buffers.control_output.get();
    hipcheck(hipMemset(buffers.candidate_output.pointer, 0x5a,
                       (kN + 32U) * sizeof(uint16_t)),
             "candidate canary");
    buffers.candidate(0U);
    hipcheck(hipDeviceSynchronize(), "candidate synchronize");
    const auto candidate_first = buffers.candidate_output.get();
    hipcheck(hipMemset(buffers.candidate_output.pointer, 0x5a,
                       (kN + 32U) * sizeof(uint16_t)),
             "candidate repeat canary");
    buffers.candidate(0U);
    hipcheck(hipDeviceSynchronize(), "candidate repeat synchronize");
    const auto candidate_second = buffers.candidate_output.get();
    const Compare control_oracle = compare_output(control_first, expected);
    const Compare candidate_oracle = compare_output(candidate_first, expected);
    const bool repeat =
        control_first == control_second && candidate_first == candidate_second;
    const bool bitwise = control_first == candidate_first;
    const bool oracle_ok = control_oracle.finite && candidate_oracle.finite &&
                           control_oracle.guard && candidate_oracle.guard &&
                           control_oracle.max_ulp <= 4U &&
                           candidate_oracle.max_ulp <= 4U;
    const bool all_ok = quantizer_bitwise && repeat && bitwise && oracle_ok;

    std::cout << std::defaultfloat << std::setprecision(9);
    std::cout << "{\"kind\":\"identity\",\"state\":\""
              << (all_ok ? "PASS" : "FAIL") << "\",\"target\":\"" << target
              << "\",\"m\":1,\"k\":" << kK << ",\"n\":" << kN
              << ",\"pattern\":" << pattern
              << ",\"candidate\":\"activation_shared_wave4_4col\""
              << ",\"control_launcher\":\"production_public_id82\""
              << ",\"control_symbol\":\"sllm_matmul_fp8_outer_decode_gfx1030_"
                 "lds_lut_m1_k6144n5120_v1\""
              << ",\"gpu_execution\":true,\"fallback_used\":false"
              << ",\"shared_bytes\":" << phase87_stage2_v620_c2::kStagedBytes
              << ",\"weight_pool_bytes\":"
              << static_cast<uint64_t>(buffers.weights.count)
              << ",\"copies\":" << copies << "}\n";
    std::cout << "{\"kind\":\"oracle\",\"state\":\""
              << (all_ok ? "PASS" : "FAIL") << "\",\"finite\":"
              << ((control_oracle.finite && candidate_oracle.finite) ? "true"
                                                                     : "false")
              << ",\"guard\":"
              << ((control_oracle.guard && candidate_oracle.guard) ? "true"
                                                                   : "false")
              << ",\"repeat\":" << (repeat ? "true" : "false")
              << ",\"candidate_control_bitwise\":"
              << (bitwise ? "true" : "false") << ",\"quantizer_bitwise\":"
              << (quantizer_bitwise ? "true" : "false")
              << ",\"max_ulp\":" << candidate_oracle.max_ulp
              << ",\"max_abs\":" << candidate_oracle.max_abs
              << ",\"control_max_ulp\":" << control_oracle.max_ulp
              << ",\"control_max_abs\":" << control_oracle.max_abs << "}\n";

    if (all_ok && bench) {
      std::vector<float> cq, cd, ct, aq, ad, at;
      for (unsigned round = 0U; round < 3U; ++round) {
        for (unsigned position = 0U; position < 2U; ++position) {
          const bool candidate = ((round % 2U == 0U) == (position == 1U));
          warm(buffers, candidate);
          for (unsigned sample = 0U; sample < 9U; ++sample) {
            const uint64_t copy =
                ((static_cast<uint64_t>(round) * 9U + sample) *
                 std::max<uint64_t>(1U, copies / 9U)) %
                copies;
            const Timing timing = timed(buffers, candidate, copy);
            auto &quant = candidate ? aq : cq;
            auto &dot = candidate ? ad : cd;
            auto &total = candidate ? at : ct;
            quant.push_back(timing.quant_ms);
            dot.push_back(timing.dot_ms);
            total.push_back(timing.total_ms);
          }
        }
      }
      std::cout << "{\"kind\":\"performance\",\"state\":\"PASS\","
                << "\"warmup_ms\":300,\"samples\":27,\"rounds\":3,"
                << "\"order\":\"AB-BA-AB\",\"control_quant_ms\":" << median(cq)
                << ",\"control_dot_ms\":" << median(cd)
                << ",\"control_total_ms\":" << median(ct)
                << ",\"candidate_quant_ms\":" << median(aq)
                << ",\"candidate_dot_ms\":" << median(ad)
                << ",\"candidate_total_ms\":" << median(at);
      print_samples("control_quant_samples_ms", cq);
      print_samples("control_dot_samples_ms", cd);
      print_samples("control_total_samples_ms", ct);
      print_samples("candidate_quant_samples_ms", aq);
      print_samples("candidate_dot_samples_ms", ad);
      print_samples("candidate_total_samples_ms", at);
      std::cout << "}\n";
    }
    owner.reset();
    need(live_allocations == 0U && frees_ok, "cleanup");
    std::cout << "{\"kind\":\"cleanup\",\"state\":\"PASS\","
              << "\"live_allocations\":0}\n";
    return all_ok ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "phase87_stage2_v620_c2: " << error.what() << '\n';
    return 1;
  }
}
