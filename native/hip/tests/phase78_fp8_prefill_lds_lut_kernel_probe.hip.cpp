// Phase 78 ID85 standalone kernel probe.
//
// This probe directly includes the production candidate include and declares
// the existing ID71 kernel as an external C symbol.  The tiny case has a
// direct host FP32 oracle; the requested M-tail cases only compare the two
// device kernels bit-for-bit.  A separate tiny oracle deliberately offsets the
// weight base by 1/2/3 bytes while K remains divisible by four, exercising the
// ID85 scalar fallback for an unaligned source.  The Qwen shape run is an
// operator timing probe and is deliberately separate from both correctness
// checks.

#include <hip/hip_runtime.h>

#include "../src/fp8_prefill_lds_lut.inc"
#include "low_precision_block_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

extern "C" __global__ void sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1(
    const uint8_t *const activation, const float *const activation_scales,
    const uint8_t *const weight, const float *const weight_scales,
    uint16_t *const output, const uint64_t m, const uint64_t k,
    const uint64_t n);

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kTileM = 64U;
constexpr uint32_t kTileN = 64U;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;
constexpr uint32_t kGuardWords = 32U;
constexpr uint8_t kGuardByte = 0xa5U;

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

bool exact_gfx1030(const char *const arch) {
  if (arch == nullptr)
    return false;
  const std::string_view value(arch);
  return value == "gfx1030" ||
         (value.size() > 7U && value.compare(0U, 7U, "gfx1030") == 0 &&
          value[7U] == ':');
}

struct DeviceBuffers final {
  uint8_t *activation = nullptr;
  float *activation_scales = nullptr;
  uint8_t *weight_allocation = nullptr;
  uint8_t *weight = nullptr;
  float *weight_scales = nullptr;
  uint16_t *output = nullptr;
  uint64_t output_words = 0U;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void release(DeviceBuffers *const buffers) {
  if (buffers == nullptr)
    return;
  if (buffers->stop != nullptr)
    (void)hipEventDestroy(buffers->stop);
  if (buffers->start != nullptr)
    (void)hipEventDestroy(buffers->start);
  if (buffers->stream != nullptr)
    (void)hipStreamDestroy(buffers->stream);
  if (buffers->output != nullptr)
    (void)hipFree(buffers->output);
  if (buffers->weight_scales != nullptr)
    (void)hipFree(buffers->weight_scales);
  if (buffers->weight_allocation != nullptr)
    (void)hipFree(buffers->weight_allocation);
  if (buffers->activation_scales != nullptr)
    (void)hipFree(buffers->activation_scales);
  if (buffers->activation != nullptr)
    (void)hipFree(buffers->activation);
  *buffers = {};
}

bool checked_matrix_sizes(const uint64_t m, const uint64_t k, const uint64_t n,
                          uint64_t *const activation_bytes,
                          uint64_t *const weight_bytes,
                          uint64_t *const output_words) {
  if (m == 0U || k == 0U || n == 0U || m > UINT64_MAX / k ||
      n > UINT64_MAX / k || m > UINT64_MAX / n) {
    return false;
  }
  const uint64_t activation_size = m * k;
  const uint64_t weight_size = n * k;
  const uint64_t output_size = m * n;
  if (output_size > UINT64_MAX - kGuardWords)
    return false;
  *activation_bytes = activation_size;
  *weight_bytes = weight_size;
  *output_words = output_size + kGuardWords;
  return true;
}

bool allocate(const uint64_t m, const uint64_t k, const uint64_t n,
              DeviceBuffers *const buffers,
              const uint64_t weight_byte_offset = 0U) {
  if (buffers == nullptr)
    return false;
  uint64_t activation_bytes = 0U;
  uint64_t weight_bytes = 0U;
  uint64_t output_words = 0U;
  if (!checked_matrix_sizes(m, k, n, &activation_bytes, &weight_bytes,
                            &output_words) ||
      weight_byte_offset > UINT64_MAX - weight_bytes ||
      weight_byte_offset + weight_bytes > SIZE_MAX) {
    return false;
  }
  buffers->output_words = output_words;
  const bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation),
                       static_cast<size_t>(activation_bytes)),
             "malloc activation") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->activation_scales),
                       static_cast<size_t>(m * sizeof(float))),
             "malloc activation scales") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_allocation),
                       static_cast<size_t>(weight_byte_offset + weight_bytes)),
             "malloc weight") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->weight_scales),
                       static_cast<size_t>(n * sizeof(float))),
             "malloc weight scales") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->output),
                       static_cast<size_t>(output_words * sizeof(uint16_t))),
             "malloc output") &&
      hip_ok(hipStreamCreate(&buffers->stream), "create stream") &&
      hip_ok(hipEventCreate(&buffers->start), "create start") &&
      hip_ok(hipEventCreate(&buffers->stop), "create stop");
  if (!ok)
    release(buffers);
  else
    buffers->weight = buffers->weight_allocation + weight_byte_offset;
  return ok;
}

struct HostInputs final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> weight;
  std::vector<float> activation_scales;
  std::vector<float> weight_scales;
};

uint8_t finite_code(const uint64_t value) {
  const uint8_t code = static_cast<uint8_t>(value & UINT64_C(0xff));
  return code == UINT8_C(0x7f) || code == UINT8_C(0xff) ? UINT8_C(0x7e) : code;
}

bool make_inputs(const uint64_t m, const uint64_t k, const uint64_t n,
                 const uint32_t seed, HostInputs *const inputs) {
  if (inputs == nullptr || m > SIZE_MAX / k || n > SIZE_MAX / k) {
    return false;
  }
  inputs->activation.resize(static_cast<size_t>(m * k));
  inputs->weight.resize(static_cast<size_t>(n * k));
  inputs->activation_scales.resize(static_cast<size_t>(m));
  inputs->weight_scales.resize(static_cast<size_t>(n));
  for (uint64_t row = 0U; row < m; ++row) {
    inputs->activation_scales[static_cast<size_t>(row)] =
        0.75F + static_cast<float>((row + seed) % 13U) * 0.03125F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint64_t code = row * 37U + inner * 11U + seed * 17U + 5U;
      inputs->activation[static_cast<size_t>(row * k + inner)] =
          finite_code(code);
    }
  }
  for (uint64_t column = 0U; column < n; ++column) {
    inputs->weight_scales[static_cast<size_t>(column)] =
        0.625F + static_cast<float>((column + seed) % 17U) * 0.0234375F;
    for (uint64_t inner = 0U; inner < k; ++inner) {
      const uint64_t code = column * 19U + inner * 7U + seed * 29U + 13U;
      inputs->weight[static_cast<size_t>(column * k + inner)] =
          finite_code(code);
    }
  }
  return true;
}

bool upload(const uint64_t m, const uint64_t k, const uint64_t n,
            const HostInputs &inputs, DeviceBuffers *const buffers) {
  return hip_ok(hipMemcpy(buffers->activation, inputs.activation.data(),
                          static_cast<size_t>(m * k), hipMemcpyHostToDevice),
                "copy activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales,
                          inputs.activation_scales.data(),
                          static_cast<size_t>(m * sizeof(float)),
                          hipMemcpyHostToDevice),
                "copy activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, inputs.weight.data(),
                          static_cast<size_t>(n * k), hipMemcpyHostToDevice),
                "copy weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, inputs.weight_scales.data(),
                          static_cast<size_t>(n * sizeof(float)),
                          hipMemcpyHostToDevice),
                "copy weight scales") &&
         hip_ok(hipMemset(buffers->output, kGuardByte,
                          static_cast<size_t>(buffers->output_words *
                                              sizeof(uint16_t))),
                "clear output guard");
}

enum class Kernel { Id71, Id85 };

bool launch(const Kernel kernel, const uint64_t m, const uint64_t k,
            const uint64_t n, DeviceBuffers *const buffers) {
  const uint64_t rows = (m + kTileM - 1U) / kTileM;
  const uint64_t columns = (n + kTileN - 1U) / kTileN;
  if (rows == 0U || columns == 0U || rows > UINT64_MAX / columns ||
      rows * columns > UINT32_MAX) {
    return false;
  }
  const dim3 grid(static_cast<uint32_t>(rows * columns));
  const dim3 block(kThreads);
  if (kernel == Kernel::Id71) {
    hipLaunchKernelGGL(sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1,
                       grid, block, 0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  } else {
    hipLaunchKernelGGL((sllm_phase78_fp8_prefill_lds_lut::
                            sllm_matmul_fp8_outer_prefill_gfx1030_lds_lut_v1),
                       grid, block, 0U, buffers->stream, buffers->activation,
                       buffers->activation_scales, buffers->weight,
                       buffers->weight_scales, buffers->output, m, k, n);
  }
  return hip_ok(hipGetLastError(), "launch kernel");
}

bool synchronize(const DeviceBuffers &buffers, const char *const operation) {
  return hip_ok(hipStreamSynchronize(buffers.stream), operation);
}

bool copy_output(const uint64_t m, const uint64_t n,
                 const DeviceBuffers &buffers,
                 std::vector<uint16_t> *const out) {
  if (out == nullptr || m > SIZE_MAX / n)
    return false;
  out->resize(static_cast<size_t>(m * n));
  return hip_ok(hipMemcpy(out->data(), buffers.output,
                          static_cast<size_t>(m * n * sizeof(uint16_t)),
                          hipMemcpyDeviceToHost),
                "copy output");
}

bool guard_ok(const uint64_t m, const uint64_t n,
              const DeviceBuffers &buffers) {
  if (m > UINT64_MAX / n)
    return false;
  std::array<uint16_t, kGuardWords> guard{};
  const uint64_t offset = m * n;
  if (!hip_ok(hipMemcpy(guard.data(), buffers.output + offset,
                        kGuardWords * sizeof(uint16_t), hipMemcpyDeviceToHost),
              "copy guard")) {
    return false;
  }
  for (const uint16_t value : guard) {
    if (value != UINT16_C(0xa5a5))
      return false;
  }
  return true;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT16_C(0x7fc0) |
                                   ((bits >> 16U) & UINT32_C(0x003f)));
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

float host_fp16_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & UINT16_C(0x8000)) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & UINT16_C(0x03ff);
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      float value = static_cast<float>(mantissa) * 0x1p-24F;
      std::memcpy(&result, &value, sizeof(result));
      result = (result & UINT32_C(0x7fffffff)) | sign;
    }
  } else if (exponent == 31U) {
    result |= UINT32_C(0x7f800000) | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

// This is an independent scalar FP32 reference: it decodes through the
// authoritative host codec, performs one fmaf per K element, applies the two
// scales after the reduction, and rounds only at the BF16 store boundary.
bool cpu_fp32_oracle(const uint64_t m, const uint64_t k, const uint64_t n,
                     const HostInputs &inputs,
                     std::vector<uint16_t> *const expected) {
  if (expected == nullptr || m > SIZE_MAX / n)
    return false;
  expected->resize(static_cast<size_t>(m * n));
  for (uint64_t row = 0U; row < m; ++row) {
    for (uint64_t column = 0U; column < n; ++column) {
      float accumulator = 0.0F;
      for (uint64_t inner = 0U; inner < k; ++inner) {
        const uint16_t activation_bits = sllm_lowp::e4m3fn_to_fp16_bits(
            inputs.activation[static_cast<size_t>(row * k + inner)]);
        const uint16_t weight_bits = sllm_lowp::e4m3fn_to_fp16_bits(
            inputs.weight[static_cast<size_t>(column * k + inner)]);
        accumulator = std::fmaf(host_fp16_to_float(activation_bits),
                                host_fp16_to_float(weight_bits), accumulator);
      }
      const float scaled = accumulator *
                           inputs.activation_scales[static_cast<size_t>(row)] *
                           inputs.weight_scales[static_cast<size_t>(column)];
      (*expected)[static_cast<size_t>(row * n + column)] =
          host_bf16_rne(scaled);
    }
  }
  return true;
}

size_t mismatch_count(const std::vector<uint16_t> &left,
                      const std::vector<uint16_t> &right) {
  if (left.size() != right.size())
    return std::numeric_limits<size_t>::max();
  size_t mismatches = 0U;
  for (size_t index = 0U; index < left.size(); ++index) {
    if (left[index] != right[index])
      ++mismatches;
  }
  return mismatches;
}

bool run_tiny() {
  constexpr uint64_t m = 3U;
  constexpr uint64_t k = 37U;
  constexpr uint64_t n = 5U;
  HostInputs inputs;
  DeviceBuffers buffers;
  if (!make_inputs(m, k, n, 1U, &inputs) || !allocate(m, k, n, &buffers) ||
      !upload(m, k, n, inputs, &buffers)) {
    release(&buffers);
    return false;
  }
  std::vector<uint16_t> id71;
  std::vector<uint16_t> id85;
  std::vector<uint16_t> expected;
  bool ok = launch(Kernel::Id71, m, k, n, &buffers) &&
            synchronize(buffers, "tiny ID71 sync") &&
            copy_output(m, n, buffers, &id71) && guard_ok(m, n, buffers);
  if (ok) {
    ok = hip_ok(
        hipMemset(buffers.output, kGuardByte,
                  static_cast<size_t>(buffers.output_words * sizeof(uint16_t))),
        "tiny reset output");
  }
  if (ok) {
    ok = launch(Kernel::Id85, m, k, n, &buffers) &&
         synchronize(buffers, "tiny ID85 sync") &&
         copy_output(m, n, buffers, &id85) && guard_ok(m, n, buffers) &&
         cpu_fp32_oracle(m, k, n, inputs, &expected);
  }
  const size_t id71_oracle = ok ? mismatch_count(id71, expected) : 1U;
  const size_t id85_oracle = ok ? mismatch_count(id85, expected) : 1U;
  const size_t id71_id85 = ok ? mismatch_count(id71, id85) : 1U;
  std::printf(
      "tiny m=%llu k=%llu n=%llu cpu_fp32_id71_mismatches=%zu "
      "cpu_fp32_id85_mismatches=%zu id71_id85_bit_mismatches=%zu status=%s\n",
      static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
      static_cast<unsigned long long>(n), id71_oracle, id85_oracle, id71_id85,
      ok && id71_oracle == 0U && id85_oracle == 0U && id71_id85 == 0U ? "PASS"
                                                                      : "FAIL");
  release(&buffers);
  return ok && id71_oracle == 0U && id85_oracle == 0U && id71_id85 == 0U;
}

bool run_weight_alignment_oracle() {
  constexpr uint64_t m = 3U;
  constexpr uint64_t k = 36U;
  constexpr uint64_t n = 5U;
  bool all_ok = true;
  for (const uint64_t weight_byte_offset : {1U, 2U, 3U}) {
    HostInputs inputs;
    DeviceBuffers buffers;
    std::vector<uint16_t> actual;
    std::vector<uint16_t> expected;
    const bool prepared =
        make_inputs(m, k, n, static_cast<uint32_t>(weight_byte_offset),
                    &inputs) &&
        allocate(m, k, n, &buffers, weight_byte_offset) &&
        upload(m, k, n, inputs, &buffers);
    const uint32_t pointer_mod4 =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(buffers.weight) &
                              static_cast<uintptr_t>(3U));
    const bool pointer_is_unaligned = pointer_mod4 != 0U;
    const bool ok = prepared && pointer_is_unaligned &&
                    launch(Kernel::Id85, m, k, n, &buffers) &&
                    synchronize(buffers, "alignment ID85 sync") &&
                    copy_output(m, n, buffers, &actual) &&
                    guard_ok(m, n, buffers) &&
                    cpu_fp32_oracle(m, k, n, inputs, &expected);
    const size_t oracle_mismatches = ok ? mismatch_count(actual, expected) : 1U;
    std::printf(
        "alignment weight_base_offset=%llu weight_ptr_mod4=%u m=%llu k=%llu "
        "n=%llu cpu_fp32_id85_mismatches=%zu status=%s\n",
        static_cast<unsigned long long>(weight_byte_offset), pointer_mod4,
        static_cast<unsigned long long>(m), static_cast<unsigned long long>(k),
        static_cast<unsigned long long>(n), oracle_mismatches,
        ok && oracle_mismatches == 0U ? "PASS" : "FAIL");
    all_ok = ok && oracle_mismatches == 0U && all_ok;
    release(&buffers);
  }
  return all_ok;
}

struct TailShape final {
  uint64_t m;
  const char *name;
};

bool run_tail(const TailShape shape) {
  constexpr uint64_t k = 70U;
  constexpr uint64_t n = 65U;
  HostInputs inputs;
  DeviceBuffers buffers;
  if (!make_inputs(shape.m, k, n, static_cast<uint32_t>(shape.m), &inputs) ||
      !allocate(shape.m, k, n, &buffers) ||
      !upload(shape.m, k, n, inputs, &buffers)) {
    release(&buffers);
    return false;
  }
  std::vector<uint16_t> id71;
  std::vector<uint16_t> id85;
  bool ok = launch(Kernel::Id71, shape.m, k, n, &buffers) &&
            synchronize(buffers, "tail ID71 sync") &&
            copy_output(shape.m, n, buffers, &id71) &&
            guard_ok(shape.m, n, buffers);
  if (ok) {
    ok = hip_ok(
        hipMemset(buffers.output, kGuardByte,
                  static_cast<size_t>(buffers.output_words * sizeof(uint16_t))),
        "tail reset output");
  }
  if (ok) {
    ok = launch(Kernel::Id85, shape.m, k, n, &buffers) &&
         synchronize(buffers, "tail ID85 sync") &&
         copy_output(shape.m, n, buffers, &id85) &&
         guard_ok(shape.m, n, buffers);
  }
  const size_t bit_mismatches = ok ? mismatch_count(id71, id85) : 1U;
  std::printf("tail name=%s m=%llu k=%llu n=%llu id71_id85_bit_mismatches=%zu "
              "status=%s\n",
              shape.name, static_cast<unsigned long long>(shape.m),
              static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(n), bit_mismatches,
              ok && bit_mismatches == 0U ? "PASS" : "FAIL");
  release(&buffers);
  return ok && bit_mismatches == 0U;
}

struct QwenShape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  const char *name;
  uint32_t occurrences;
};

constexpr std::array<QwenShape, 8> kQwenShapes = {{
    {1024U, 5120U, 17408U, "layers56-63.mlp.gate+up", 16U},
    {1024U, 17408U, 5120U, "layers56-63.mlp.down", 8U},
    {1024U, 5120U, 12288U, "16.full-attn.q", 16U},
    {1024U, 5120U, 1024U, "16.full-attn.k+v", 32U},
    {1024U, 6144U, 5120U, "full-attn.o+linear-attn.out", 64U},
    {1024U, 5120U, 10240U, "48.linear-attn.qkv", 48U},
    {1024U, 5120U, 6144U, "48.linear-attn.z", 48U},
    {1024U, 5120U, 248320U, "lm_head", 1U},
}};

constexpr uint32_t qwen_occurrences() {
  uint32_t result = 0U;
  for (const QwenShape &shape : kQwenShapes)
    result += shape.occurrences;
  return result;
}

static_assert(qwen_occurrences() == 233U);

struct Timing final {
  float median_us = 0.0F;
  float mad_us = 0.0F;
};

bool measure(const Kernel kernel, const QwenShape shape,
             DeviceBuffers *const buffers, Timing *const timing) {
  for (int iteration = 0; iteration < kWarmups; ++iteration) {
    if (!launch(kernel, shape.m, shape.k, shape.n, buffers) ||
        !synchronize(*buffers, "warmup sync")) {
      return false;
    }
  }
  std::array<float, kMeasured> samples{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "record start") ||
        !launch(kernel, shape.m, shape.k, shape.n, buffers) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                "record stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "sync stop") ||
        !hip_ok(hipEventElapsedTime(&samples[static_cast<size_t>(iteration)],
                                    buffers->start, buffers->stop),
                "elapsed time")) {
      return false;
    }
    samples[static_cast<size_t>(iteration)] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  timing->median_us = samples[kMeasured / 2];
  std::array<float, kMeasured> deviations{};
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    deviations[static_cast<size_t>(iteration)] =
        std::fabs(samples[static_cast<size_t>(iteration)] - timing->median_us);
  }
  std::sort(deviations.begin(), deviations.end());
  timing->mad_us = deviations[kMeasured / 2];
  return true;
}

bool run_qwen_performance() {
  double weighted_id71 = 0.0;
  double weighted_id85 = 0.0;
  uint32_t occurrences = 0U;
  bool ok = true;
  for (const QwenShape shape : kQwenShapes) {
    HostInputs inputs;
    DeviceBuffers buffers;
    const bool prepared =
        make_inputs(shape.m, shape.k, shape.n, shape.occurrences, &inputs) &&
        allocate(shape.m, shape.k, shape.n, &buffers) &&
        upload(shape.m, shape.k, shape.n, inputs, &buffers);
    Timing id71_timing;
    Timing id85_timing;
    bool shape_ok =
        prepared && measure(Kernel::Id71, shape, &buffers, &id71_timing);
    if (shape_ok) {
      shape_ok = measure(Kernel::Id85, shape, &buffers, &id85_timing) &&
                 guard_ok(shape.m, shape.n, buffers);
    }
    const double speedup = id85_timing.median_us > 0.0F
                               ? static_cast<double>(id71_timing.median_us) /
                                     static_cast<double>(id85_timing.median_us)
                               : 0.0;
    std::printf("qwen8 name=%s m=%llu k=%llu n=%llu occurrences=%u "
                "id71_median_us=%.3f id71_mad_us=%.3f id85_median_us=%.3f "
                "id85_mad_us=%.3f speedup=%.6f status=%s\n",
                shape.name, static_cast<unsigned long long>(shape.m),
                static_cast<unsigned long long>(shape.k),
                static_cast<unsigned long long>(shape.n), shape.occurrences,
                id71_timing.median_us, id71_timing.mad_us,
                id85_timing.median_us, id85_timing.mad_us, speedup,
                shape_ok ? "PASS" : "FAIL");
    if (shape_ok) {
      weighted_id71 += static_cast<double>(id71_timing.median_us) *
                       static_cast<double>(shape.occurrences);
      weighted_id85 += static_cast<double>(id85_timing.median_us) *
                       static_cast<double>(shape.occurrences);
      occurrences += shape.occurrences;
    }
    release(&buffers);
    ok = shape_ok && ok;
  }
  const double weighted_speedup =
      weighted_id85 > 0.0 ? weighted_id71 / weighted_id85 : 0.0;
  std::printf(
      "qwen8-summary occurrences=%u warmups=%d measured=%d "
      "weighted_id71_us=%.3f weighted_id85_us=%.3f weighted_speedup=%.6f "
      "status=%s\n",
      occurrences, kWarmups, kMeasured, weighted_id71, weighted_id85,
      weighted_speedup,
      ok && occurrences == qwen_occurrences() ? "PASS" : "FAIL");
  return ok && occurrences == qwen_occurrences();
}

} // namespace

int main(int argc, char **argv) {
  bool run_tiny_case = false;
  bool run_alignment_case = false;
  bool run_tail_cases = false;
  bool run_qwen_case = false;
  if (argc == 1) {
    run_tiny_case = true;
    run_alignment_case = true;
    run_tail_cases = true;
    run_qwen_case = true;
  }
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--tiny") {
      run_tiny_case = true;
    } else if (argument == "--alignment") {
      run_alignment_case = true;
    } else if (argument == "--tails") {
      run_tail_cases = true;
    } else if (argument == "--qwen-performance") {
      run_qwen_case = true;
    } else if (argument == "--all") {
      run_tiny_case = true;
      run_alignment_case = true;
      run_tail_cases = true;
      run_qwen_case = true;
    } else {
      std::fprintf(
          stderr,
          "usage: %s [--tiny] [--alignment] [--tails] [--qwen-performance] "
          "[--all]\n",
          argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (!hip_ok(hipSetDevice(0), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "device properties") ||
      !exact_gfx1030(properties.gcnArchName)) {
    std::fprintf(stderr, "exact gfx1030 required; got %s\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }
  std::printf(
      "identity target=%s logical_device=0 pci=%04x:%02x:%02x name=%s\n",
      properties.gcnArchName, properties.pciDomainID, properties.pciBusID,
      properties.pciDeviceID, properties.name);

  bool ok = true;
  if (run_tiny_case)
    ok = run_tiny() && ok;
  if (run_alignment_case)
    ok = run_weight_alignment_oracle() && ok;
  if (run_tail_cases) {
    constexpr std::array<TailShape, 7> tails = {{
        {17U, "m17"},
        {127U, "m127"},
        {128U, "m128"},
        {129U, "m129"},
        {219U, "m219"},
        {512U, "m512"},
        {1024U, "m1024"},
    }};
    for (const TailShape shape : tails)
      ok = run_tail(shape) && ok;
  }
  if (run_qwen_case)
    ok = run_qwen_performance() && ok;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
