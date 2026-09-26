// Phase 87 WU-2V/C3 bounded probe for exact gfx1030, M=1, K=6144, N=5120.
//
// The control is the production ID82 public launcher from the linked lowp
// archive.  The candidate is test-only and uses a one-pair direct-wave loop
// with the same ID82 LUT and dot order.
// Both paths consume the production activation quantizer and use the same
// E4M3FN LUT, dot order, reduction, and BF16 epilogue.

#include "../src/matmul_kernel_internal.hpp"
#include <lowp/detail/bf16_helpers.inc>
#include <lowp/detail/low_precision_block_codec.hpp>
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

namespace phase87_stage2_v620 {

constexpr uint64_t kK = UINT64_C(6144);
constexpr uint64_t kN = UINT64_C(5120);
constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kColumnsPerWave = 4U;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroups = 12U;
constexpr uint32_t kPrefetchChunks = 2U;

// This is the production ID82 table.  It is intentionally resident in the
// candidate code object so that the candidate and the linked control consume
// identical FP16 ingress values.
__device__ __constant__ uint16_t kLut[256U] = {
    0x0000U, 0x1800U, 0x1c00U, 0x1e00U, 0x2000U, 0x2100U, 0x2200U, 0x2300U,
    0x2400U, 0x2480U, 0x2500U, 0x2580U, 0x2600U, 0x2680U, 0x2700U, 0x2780U,
    0x2800U, 0x2880U, 0x2900U, 0x2980U, 0x2a00U, 0x2a80U, 0x2b00U, 0x2b80U,
    0x2c00U, 0x2c80U, 0x2d00U, 0x2d80U, 0x2e00U, 0x2e80U, 0x2f00U, 0x2f80U,
    0x3000U, 0x3080U, 0x3100U, 0x3180U, 0x3200U, 0x3280U, 0x3300U, 0x3380U,
    0x3400U, 0x3480U, 0x3500U, 0x3580U, 0x3600U, 0x3680U, 0x3700U, 0x3780U,
    0x3800U, 0x3880U, 0x3900U, 0x3980U, 0x3a00U, 0x3a80U, 0x3b00U, 0x3b80U,
    0x3c00U, 0x3c80U, 0x3d00U, 0x3d80U, 0x3e00U, 0x3e80U, 0x3f00U, 0x3f80U,
    0x4000U, 0x4080U, 0x4100U, 0x4180U, 0x4200U, 0x4280U, 0x4300U, 0x4380U,
    0x4400U, 0x4480U, 0x4500U, 0x4580U, 0x4600U, 0x4680U, 0x4700U, 0x4780U,
    0x4800U, 0x4880U, 0x4900U, 0x4980U, 0x4a00U, 0x4a80U, 0x4b00U, 0x4b80U,
    0x4c00U, 0x4c80U, 0x4d00U, 0x4d80U, 0x4e00U, 0x4e80U, 0x4f00U, 0x4f80U,
    0x5000U, 0x5080U, 0x5100U, 0x5180U, 0x5200U, 0x5280U, 0x5300U, 0x5380U,
    0x5400U, 0x5480U, 0x5500U, 0x5580U, 0x5600U, 0x5680U, 0x5700U, 0x5780U,
    0x5800U, 0x5880U, 0x5900U, 0x5980U, 0x5a00U, 0x5a80U, 0x5b00U, 0x5b80U,
    0x5c00U, 0x5c80U, 0x5d00U, 0x5d80U, 0x5e00U, 0x5e80U, 0x5f00U, 0x7e00U,
    0x8000U, 0x9800U, 0x9c00U, 0x9e00U, 0xa000U, 0xa100U, 0xa200U, 0xa300U,
    0xa400U, 0xa480U, 0xa500U, 0xa580U, 0xa600U, 0xa680U, 0xa700U, 0xa780U,
    0xa800U, 0xa880U, 0xa900U, 0xa980U, 0xaa00U, 0xaa80U, 0xab00U, 0xab80U,
    0xac00U, 0xac80U, 0xad00U, 0xad80U, 0xae00U, 0xae80U, 0xaf00U, 0xaf80U,
    0xb000U, 0xb080U, 0xb100U, 0xb180U, 0xb200U, 0xb280U, 0xb300U, 0xb380U,
    0xb400U, 0xb480U, 0xb500U, 0xb580U, 0xb600U, 0xb680U, 0xb700U, 0xb780U,
    0xb800U, 0xb880U, 0xb900U, 0xb980U, 0xba00U, 0xba80U, 0xbb00U, 0xbb80U,
    0xbc00U, 0xbc80U, 0xbd00U, 0xbd80U, 0xbe00U, 0xbe80U, 0xbf00U, 0xbf80U,
    0xc000U, 0xc080U, 0xc100U, 0xc180U, 0xc200U, 0xc280U, 0xc300U, 0xc380U,
    0xc400U, 0xc480U, 0xc500U, 0xc580U, 0xc600U, 0xc680U, 0xc700U, 0xc780U,
    0xc800U, 0xc880U, 0xc900U, 0xc980U, 0xca00U, 0xca80U, 0xcb00U, 0xcb80U,
    0xcc00U, 0xcc80U, 0xcd00U, 0xcd80U, 0xce00U, 0xce80U, 0xcf00U, 0xcf80U,
    0xd000U, 0xd080U, 0xd100U, 0xd180U, 0xd200U, 0xd280U, 0xd300U, 0xd380U,
    0xd400U, 0xd480U, 0xd500U, 0xd580U, 0xd600U, 0xd680U, 0xd700U, 0xd780U,
    0xd800U, 0xd880U, 0xd900U, 0xd980U, 0xda00U, 0xda80U, 0xdb00U, 0xdb80U,
    0xdc00U, 0xdc80U, 0xdd00U, 0xdd80U, 0xde00U, 0xde80U, 0xdf00U, 0xfe00U};

__device__ __forceinline__ uint32_t lut_slot(const uint32_t code) noexcept {
  return code + (code >> 5U);
}

__device__ __forceinline__ uint32_t
lut_pair(const uint32_t packed, const uint16_t *const lut) noexcept {
  const uint8_t first = static_cast<uint8_t>(packed & UINT32_C(0xff));
  const uint8_t second = static_cast<uint8_t>((packed >> 8U) & UINT32_C(0xff));
  return static_cast<uint32_t>(lut[lut_slot(first)]) |
         (static_cast<uint32_t>(lut[lut_slot(second)]) << 16U);
}

__device__ __forceinline__ __half2 packed_half2(const uint32_t bits) noexcept {
  return *reinterpret_cast<const __half2 *>(&bits);
}

// C3 uses one direct-wave pair per loop iteration instead of the ID82 two-chunk
// prefetch. The FP8 LUT and each column's four dot/add operations remain in
// the same K order as the current production ID82 kernel.
template <uint64_t TupleK, uint64_t TupleN, uint32_t TupleGroups>
__device__ __forceinline__ void
candidate_body(const uint8_t *const activation,
               const float *const activation_scales,
               const uint8_t *const weight, const float *const weight_scales,
               uint16_t *const output, const uint64_t m, const uint64_t k,
               const uint64_t n, uint16_t *const lut) {
  static_assert(TupleK == kK && TupleN == kN && TupleGroups == kGroups);
  constexpr uint32_t values_per_iteration = 8U;
  static_assert(TupleK / values_per_iteration ==
                kWave * kPrefetchChunks * TupleGroups);
  if (m != 1U || k != TupleK || n != TupleN)
    return;

  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x >> 5U;
  const uint64_t column_base =
      (static_cast<uint64_t>(blockIdx.x) * kWaves + wave) * kColumnsPerWave;
  float accumulators[kColumnsPerWave] = {};
  const auto *const activation_dwords =
      reinterpret_cast<const uint32_t *>(activation);

#pragma unroll 1
  for (uint32_t iteration = lane; iteration < TupleK / values_per_iteration;
       iteration += kWave) {
    const uint32_t activation_first =
        __builtin_nontemporal_load(activation_dwords + iteration * 2U);
    const uint32_t activation_second =
        __builtin_nontemporal_load(activation_dwords + iteration * 2U + 1U);
    uint32_t weight_first[kColumnsPerWave];
    uint32_t weight_second[kColumnsPerWave];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      const auto *const column_dwords =
          reinterpret_cast<const uint32_t *>(weight + column * TupleK);
      weight_first[local_column] =
          __builtin_nontemporal_load(column_dwords + iteration * 2U);
      weight_second[local_column] =
          __builtin_nontemporal_load(column_dwords + iteration * 2U + 1U);
    }
    const __half2 activation_first_low =
        packed_half2(lut_pair(activation_first, lut));
    const __half2 activation_first_high =
        packed_half2(lut_pair(activation_first >> 16U, lut));
    const __half2 activation_second_low =
        packed_half2(lut_pair(activation_second, lut));
    const __half2 activation_second_high =
        packed_half2(lut_pair(activation_second >> 16U, lut));
#pragma unroll
    for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
         ++local_column) {
      const __half2 weight_first_low =
          packed_half2(lut_pair(weight_first[local_column], lut));
      const __half2 weight_first_high =
          packed_half2(lut_pair(weight_first[local_column] >> 16U, lut));
      const __half2 weight_second_low =
          packed_half2(lut_pair(weight_second[local_column], lut));
      const __half2 weight_second_high =
          packed_half2(lut_pair(weight_second[local_column] >> 16U, lut));
      accumulators[local_column] =
          amd_mixed_dot(activation_first_low, weight_first_low,
                        accumulators[local_column], false);
      accumulators[local_column] =
          amd_mixed_dot(activation_first_high, weight_first_high,
                        accumulators[local_column], false);
      accumulators[local_column] =
          amd_mixed_dot(activation_second_low, weight_second_low,
                        accumulators[local_column], false);
      accumulators[local_column] =
          amd_mixed_dot(activation_second_high, weight_second_high,
                        accumulators[local_column], false);
    }
  }

#pragma unroll
  for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U) {
#pragma unroll
    for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
         ++local_column)
      accumulators[local_column] +=
          __shfl_down(accumulators[local_column], offset, kWave);
  }
  if (lane == 0U) {
    const float activation_scale = activation_scales[0];
#pragma unroll
    for (uint32_t local_column = 0U; local_column < kColumnsPerWave;
         ++local_column) {
      const uint64_t column = column_base + local_column;
      output[column] =
          float_to_bf16_rne_bits(accumulators[local_column] * activation_scale *
                                 weight_scales[column]);
    }
  }
}

extern "C" __global__
__launch_bounds__(kThreads, 1) void phase87_stage2_v620_candidate(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n) {
  __shared__ __align__(16) uint16_t lut[272U];
  lut[lut_slot(threadIdx.x)] = kLut[threadIdx.x];
  __syncthreads();
  candidate_body<kK, kN, kGroups>(activation, activation_scales, weight,
                                  weight_scales, output, m, k, n, lut);
}

} // namespace phase87_stage2_v620

namespace {

using phase87_stage2_v620::kK;
using phase87_stage2_v620::kN;

void need(const bool value, const char *const message) {
  if (!value)
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

struct Timing final {
  float quant_ms = 0.0F;
  float dot_ms = 0.0F;
  float total_ms = 0.0F;
};

struct Buffers final {
  const uint64_t copies;
  const int pattern;
  DeviceBuffer<uint16_t> activation{kK};
  DeviceBuffer<uint8_t> activation_fp8{kK};
  DeviceBuffer<float> activation_scale{1U};
  DeviceBuffer<uint8_t> weights;
  DeviceBuffer<float> weight_scales{kN};
  DeviceBuffer<uint16_t> control_output{kN + 32U};
  DeviceBuffer<uint16_t> candidate_output{kN + 32U};

  Buffers(const uint64_t copy_count, const int input_pattern)
      : copies(copy_count), pattern(input_pattern),
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
    hipLaunchKernelGGL(phase87_stage2_v620::phase87_stage2_v620_candidate,
                       dim3((kN + 31U) / 32U), dim3(256U), 0U, nullptr,
                       activation_fp8.pointer, activation_scale.pointer,
                       weights.pointer + copy * kK * kN, weight_scales.pointer,
                       candidate_output.pointer, 1U, kK, kN);
    hipcheck(hipGetLastError(), "candidate launch");
  }
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
  Timing timing;
  hipcheck(hipEventElapsedTime(&timing.quant_ms, start, quant_end),
           "hipEventElapsedTime quant");
  hipcheck(hipEventElapsedTime(&timing.dot_ms, quant_end, end),
           "hipEventElapsedTime dot");
  hipcheck(hipEventElapsedTime(&timing.total_ms, start, end),
           "hipEventElapsedTime total");
  hipcheck(hipEventDestroy(start), "hipEventDestroy start");
  hipcheck(hipEventDestroy(quant_end), "hipEventDestroy quant");
  hipcheck(hipEventDestroy(end), "hipEventDestroy end");
  return timing;
}

void warm(Buffers &buffers, const bool candidate) {
  const auto start = std::chrono::steady_clock::now();
  uint64_t copy = 0U;
  do {
    for (unsigned i = 0U; i < 32U; ++i) {
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

Compare compare_oracle(const std::vector<uint16_t> &actual,
                       const std::vector<uint16_t> &oracle,
                       const uint16_t guard_value) {
  Compare result;
  need(actual.size() == kN + 32U && oracle.size() == kN, "compare size");
  for (uint64_t index = 0U; index < kN; ++index) {
    result.finite &= std::isfinite(bf16_to_f32(actual[index]));
    const uint32_t ulp = actual[index] >= oracle[index]
                             ? actual[index] - oracle[index]
                             : oracle[index] - actual[index];
    result.max_ulp = std::max(result.max_ulp, ulp);
    result.max_abs =
        std::max(result.max_abs, std::abs(bf16_to_f32(actual[index]) -
                                          bf16_to_f32(oracle[index])));
  }
  for (uint64_t index = kN; index < actual.size(); ++index)
    result.guard &= actual[index] == guard_value;
  return result;
}

std::vector<uint16_t> make_activation(const int pattern) {
  std::vector<uint16_t> activation(kK);
  for (uint64_t index = 0U; index < kK; ++index) {
    float value = 0.0F;
    if (pattern == 1)
      value = (index & 1U) == 0U ? 1.0F : -1.0F;
    else if (pattern == 2)
      value = (index % 7U == 0U) ? -0.75F : ((index % 5U) * 0.25F);
    else
      value = static_cast<float>(16U + (index * 17U) % 49U) / 64.0F;
    activation[index] = f32_to_bf16(value);
  }
  return activation;
}

std::vector<uint16_t>
make_quantized_oracle(const std::vector<uint16_t> &activation,
                      std::vector<float> *scale) {
  float maximum = 0.0F;
  for (const uint16_t value : activation)
    maximum = std::max(maximum, std::abs(bf16_to_f32(value)));
  const float quant_scale = maximum == 0.0F ? 1.0F : maximum / 448.0F;
  scale->assign(1U, quant_scale);
  std::vector<uint16_t> result(kK);
  for (uint64_t index = 0U; index < kK; ++index)
    result[index] = e4m3fn_encode(bf16_to_f32(activation[index]) / quant_scale);
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
    need(target == "gfx1030", "this probe is exact gfx1030 only");
    need(pattern >= 0 && pattern <= 2, "pattern must be 0, 1 or 2");

    hipDeviceProp_t properties{};
    int device_index = 0;
    hipcheck(hipGetDevice(&device_index), "hipGetDevice");
    hipcheck(hipGetDeviceProperties(&properties, device_index),
             "hipGetDeviceProperties");
    need(target == properties.gcnArchName, "visible GPU target mismatch");

    const uint64_t weight_bytes = kK * kN;
    const uint64_t copies = std::max<uint64_t>(
        1U, ((UINT64_C(512) << 20) + weight_bytes - 1U) / weight_bytes);
    auto buffers_owner = std::make_unique<Buffers>(copies, pattern);
    Buffers &buffers = *buffers_owner;
    const std::vector<uint16_t> activation = make_activation(pattern);
    std::vector<float> oracle_scale;
    const std::vector<uint16_t> oracle_activation =
        make_quantized_oracle(activation, &oracle_scale);
    std::vector<float> host_weight_scales(kN);
    for (uint64_t column = 0U; column < kN; ++column)
      host_weight_scales[column] =
          pattern == 1 ? 1.0F : static_cast<float>(8U + column % 7U) / 256.0F;
    buffers.activation.put(activation);
    buffers.weight_scales.put(host_weight_scales);
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
    bool quantizer_bitwise =
        quantized == std::vector<uint8_t>(oracle_activation.begin(),
                                          oracle_activation.end());
    quantizer_bitwise &=
        scales.size() == oracle_scale.size() &&
        std::memcmp(scales.data(), oracle_scale.data(), sizeof(float)) == 0;
    need(quantizer_bitwise, "production quantizer differs from oracle");

    constexpr uint16_t kCanary = UINT16_C(0x5a5a);
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
    const bool control_repeat = control_first == control_second;

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
    const bool candidate_repeat = candidate_first == candidate_second;
    const bool candidate_control_bitwise = candidate_first == control_first;
    // Rebuild the independent FP32 oracle with the actual encoded input and
    // weight pattern, then compare both outputs to it.
    std::vector<uint16_t> expected(kN);
    for (uint64_t column = 0U; column < kN; ++column) {
      float sum = 0.0F;
      for (uint64_t index = 0U; index < kK; ++index)
        sum += e4m3fn_decode(quantized[index]) *
               e4m3fn_decode(weight_code(column, index, pattern));
      expected[column] =
          f32_to_bf16(sum * scales[0] * host_weight_scales[column]);
    }
    const Compare control_oracle =
        compare_oracle(control_first, expected, kCanary);
    const Compare candidate_oracle =
        compare_oracle(candidate_first, expected, kCanary);
    const bool finite = control_oracle.finite && candidate_oracle.finite;
    const bool guard = control_oracle.guard && candidate_oracle.guard;
    const bool oracle_ok = finite && guard && control_oracle.max_ulp <= 4U &&
                           candidate_oracle.max_ulp <= 4U;
    const bool all_ok = quantizer_bitwise && control_repeat &&
                        candidate_repeat && candidate_control_bitwise &&
                        oracle_ok;

    std::cout << std::defaultfloat << std::setprecision(9);
    std::cout << "{\"kind\":\"identity\",\"state\":\""
              << (all_ok ? "PASS" : "FAIL") << "\",\"target\":\"" << target
              << "\",\"m\":1,\"k\":" << kK << ",\"n\":" << kN
              << ",\"pattern\":" << pattern
              << ",\"candidate\":\"direct_wave4_pair2_id82_lut\""
              << ",\"control_symbol\":\"sllm_matmul_fp8_outer_decode_gfx1030_"
                 "lds_lut_m1_k6144n5120_v1\""
              << ",\"control_launcher\":\"production_public_id82\""
              << ",\"gpu_execution\":true,\"fallback_used\":false"
              << ",\"weight_pool_bytes\":"
              << static_cast<uint64_t>(buffers.weights.count)
              << ",\"copies\":" << copies << "}\n";
    std::cout << "{\"kind\":\"oracle\",\"state\":\""
              << (all_ok ? "PASS" : "FAIL")
              << "\",\"variant\":\"candidate\",\"finite\":"
              << (finite ? "true" : "false")
              << ",\"guard\":" << (guard ? "true" : "false") << ",\"repeat\":"
              << ((control_repeat && candidate_repeat) ? "true" : "false")
              << ",\"candidate_control_bitwise\":"
              << (candidate_control_bitwise ? "true" : "false")
              << ",\"quantizer_bitwise\":"
              << (quantizer_bitwise ? "true" : "false")
              << ",\"max_ulp\":" << candidate_oracle.max_ulp
              << ",\"max_abs\":" << candidate_oracle.max_abs
              << ",\"control_max_ulp\":" << control_oracle.max_ulp
              << ",\"control_max_abs\":" << control_oracle.max_abs << "}\n";

    if (all_ok && bench) {
      std::vector<float> control_quant, control_dot, control_total;
      std::vector<float> candidate_quant, candidate_dot, candidate_total;
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
            auto &quant = candidate ? candidate_quant : control_quant;
            auto &dot = candidate ? candidate_dot : control_dot;
            auto &total = candidate ? candidate_total : control_total;
            quant.push_back(timing.quant_ms);
            dot.push_back(timing.dot_ms);
            total.push_back(timing.total_ms);
          }
        }
      }
      std::cout << "{\"kind\":\"performance\",\"state\":\"PASS\","
                << "\"warmup_ms\":300,\"samples\":27,\"rounds\":3,"
                << "\"order\":\"AB-BA-AB\",\"control_quant_ms\":"
                << median(control_quant)
                << ",\"control_dot_ms\":" << median(control_dot)
                << ",\"control_total_ms\":" << median(control_total)
                << ",\"candidate_quant_ms\":" << median(candidate_quant)
                << ",\"candidate_dot_ms\":" << median(candidate_dot)
                << ",\"candidate_total_ms\":" << median(candidate_total);
      print_samples("control_quant_samples_ms", control_quant);
      print_samples("control_dot_samples_ms", control_dot);
      print_samples("control_total_samples_ms", control_total);
      print_samples("candidate_quant_samples_ms", candidate_quant);
      print_samples("candidate_dot_samples_ms", candidate_dot);
      print_samples("candidate_total_samples_ms", candidate_total);
      std::cout << "}\n";
    }
    buffers_owner.reset();
    need(live_allocations == 0U && frees_ok, "cleanup");
    std::cout << "{\"kind\":\"cleanup\",\"state\":\"PASS\","
              << "\"live_allocations\":0}\n";
    return all_ok ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "phase87_stage2_v620: " << error.what() << '\n';
    return 1;
  }
}
