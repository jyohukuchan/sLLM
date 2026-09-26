// WU-12C private M=5 NVFP4 candidate probe.
//
// The candidate is deliberately reached through the private lowp launcher;
// it is not a production provider or selector.  This probe compares it with
// the current M=5 row8 fallback on the two Qwen3.8 projection tuples.  The
// host oracle decodes the packed E2M1/E4M3FN values independently and checks
// selected rows/columns, while the device run checks finite/repeat/guard
// behavior and records AB-BA-AB device-event timings.

#include "../../lowp/include/lowp/detail/lowp_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "unknown"
#endif

// The device symbol is defined in lowp's HIP translation unit.  This
// declaration is used only for HIP resource attributes; launches go through
// the private host launcher below so the test cannot become a production ABI.
extern "C" __global__ void sllm_nvfp4_w4a4_small_m_vgpr_reuse_m5_probe_v1(
    const uint8_t *packed_activation, const uint8_t *activation_block_scales,
    const uint8_t *packed_weight, const uint8_t *weight_block_scales,
    const float *weight_tensor_scale, const float *input_tensor_scale,
    uint16_t *output, uint64_t m, uint64_t k, uint64_t n);

namespace {

constexpr uint64_t kRows = 5U;
constexpr uint32_t kThreads = 256U;
constexpr uint32_t kGuardElements = 64U;
constexpr int kWarmups = 2;
constexpr int kMeasured = 5;
constexpr std::array<uint64_t, 18> kSampleColumns = {
    0U,    1U,    15U,   16U,   31U,   32U,   127U,  128U,   1023U,
    1024U, 2560U, 4095U, 4096U, 5118U, 5119U, 8702U, 17406U, 17407U};

struct Shape final {
  const char *name;
  uint64_t k;
  uint64_t n;
};

struct DeviceBuffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  uint8_t *weight_scales = nullptr;
  float *weight_tensor_scale = nullptr;
  float *input_tensor_scale = nullptr;
  uint16_t *output_guard = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

struct HostInputs final {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weight;
  std::vector<uint8_t> weight_scales;
  float weight_tensor_scale = 0.75F;
  float input_tensor_scale = 1.25F;
};

struct Resources final {
  int num_regs = -1;
  int static_lds = -1;
  int private_bytes = -1;
  int max_threads = -1;
  int active_blocks = -1;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}

bool check_status(const hipError_t status, const char *const operation) {
  return hip_ok(status, operation);
}

template <typename T> bool alloc(T **const pointer, const size_t bytes) {
  return check_status(hipMalloc(reinterpret_cast<void **>(pointer), bytes),
                      "hipMalloc");
}

template <typename T> bool free_device(T **const pointer) {
  if (*pointer == nullptr) {
    return true;
  }
  const hipError_t status = hipFree(*pointer);
  *pointer = nullptr;
  return check_status(status, "hipFree");
}

uint16_t f32_to_bf16_rne(const float value) {
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

bool finite_bf16(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

uint32_t bf16_ulp(const uint16_t actual, const uint16_t expected) {
  return actual >= expected ? static_cast<uint32_t>(actual - expected)
                            : static_cast<uint32_t>(expected - actual);
}

// Independent host decode of the OCP NVFP4 scalar code.  The kernel uses the
// shared lowp helpers; this oracle intentionally does not call them.
float oracle_e4m3fn(const uint8_t code) {
  const uint8_t magnitude = code & UINT8_C(0x7f);
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & UINT8_C(0x07);
  float value = 0.0F;
  if (exponent == 0U) {
    value = static_cast<float>(mantissa) * 0x1p-9F;
  } else if (magnitude == UINT8_C(0x7f)) {
    value = std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                       static_cast<int>(exponent) - 7);
  }
  return (code & UINT8_C(0x80)) == 0U ? value : -value;
}

int32_t oracle_scaled_e2m1(const uint8_t code) {
  constexpr std::array<int32_t, 8> magnitude = {0, 1, 2, 3, 4, 6, 8, 12};
  const int32_t value = magnitude[code & UINT8_C(0x07)];
  return (code & UINT8_C(0x08)) == 0U ? value : -value;
}

uint8_t nibble(const uint8_t *const packed, const uint64_t index) {
  const uint8_t value = packed[index / 2U];
  return (index & 1U) == 0U ? value & UINT8_C(0x0f) : value >> 4U;
}

HostInputs make_inputs(const Shape &shape) {
  const uint64_t blocks = shape.k / UINT64_C(16);
  const uint64_t packed_row_bytes = shape.k / UINT64_C(2);
  HostInputs inputs;
  inputs.activation.resize(static_cast<size_t>(kRows * packed_row_bytes));
  inputs.activation_scales.resize(static_cast<size_t>(kRows * blocks));
  inputs.weight.resize(static_cast<size_t>(shape.n * packed_row_bytes));
  inputs.weight_scales.resize(static_cast<size_t>(shape.n * blocks));

  constexpr std::array<uint8_t, 8> activation_codes = {
      UINT8_C(0x01), UINT8_C(0x2f), UINT8_C(0x47), UINT8_C(0x68),
      UINT8_C(0x7e), UINT8_C(0x83), UINT8_C(0x9a), UINT8_C(0xcf)};
  constexpr std::array<uint8_t, 8> weight_codes = {
      UINT8_C(0x00), UINT8_C(0x17), UINT8_C(0x2e), UINT8_C(0x38),
      UINT8_C(0x49), UINT8_C(0x5d), UINT8_C(0x76), UINT8_C(0xef)};
  constexpr std::array<uint8_t, 5> scale_codes = {UINT8_C(0x30), UINT8_C(0x38),
                                                  UINT8_C(0x40), UINT8_C(0xb8),
                                                  UINT8_C(0x48)};

  for (uint64_t row = 0U; row < kRows; ++row) {
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.activation_scales[static_cast<size_t>(row * blocks + block)] =
          scale_codes[(row + block) % scale_codes.size()];
      for (uint64_t pair = 0U; pair < 8U; ++pair) {
        const uint8_t low =
            activation_codes[(row + block + pair) % activation_codes.size()];
        const uint8_t high = activation_codes[(row * 3U + block + pair + 1U) %
                                              activation_codes.size()];
        inputs.activation[static_cast<size_t>(row * packed_row_bytes +
                                              block * 8U + pair)] =
            static_cast<uint8_t>((high << 4U) | low);
      }
    }
  }
  for (uint64_t column = 0U; column < shape.n; ++column) {
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.weight_scales[static_cast<size_t>(column * blocks + block)] =
          scale_codes[(column * 5U + block * 3U) % scale_codes.size()];
      for (uint64_t pair = 0U; pair < 8U; ++pair) {
        const uint8_t low =
            weight_codes[(column + block * 5U + pair) % weight_codes.size()];
        const uint8_t high = weight_codes[(column * 3U + block + pair + 2U) %
                                          weight_codes.size()];
        inputs.weight[static_cast<size_t>(column * packed_row_bytes +
                                          block * 8U + pair)] =
            static_cast<uint8_t>((high << 4U) | low);
      }
    }
  }
  return inputs;
}

bool make_device_buffers(const Shape &shape, DeviceBuffers *const buffers) {
  const uint64_t blocks = shape.k / UINT64_C(16);
  const uint64_t packed_row_bytes = shape.k / UINT64_C(2);
  const uint64_t output_elements = kRows * shape.n;
  bool valid =
      hip_ok(hipStreamCreate(&buffers->stream), "hipStreamCreate") &&
      hip_ok(hipEventCreate(&buffers->start), "hipEventCreate(start)") &&
      hip_ok(hipEventCreate(&buffers->stop), "hipEventCreate(stop)");
  valid =
      valid &&
      alloc(&buffers->activation,
            static_cast<size_t>(kRows * packed_row_bytes)) &&
      alloc(&buffers->activation_scales, static_cast<size_t>(kRows * blocks)) &&
      alloc(&buffers->weight,
            static_cast<size_t>(shape.n * packed_row_bytes)) &&
      alloc(&buffers->weight_scales, static_cast<size_t>(shape.n * blocks)) &&
      alloc(&buffers->weight_tensor_scale, sizeof(float)) &&
      alloc(&buffers->input_tensor_scale, sizeof(float)) &&
      alloc(&buffers->output_guard,
            static_cast<size_t>(output_elements + 2U * kGuardElements) *
                sizeof(uint16_t));
  return valid;
}

bool release_device_buffers(DeviceBuffers *const buffers) {
  bool valid = true;
  valid = free_device(&buffers->activation) && valid;
  valid = free_device(&buffers->activation_scales) && valid;
  valid = free_device(&buffers->weight) && valid;
  valid = free_device(&buffers->weight_scales) && valid;
  valid = free_device(&buffers->weight_tensor_scale) && valid;
  valid = free_device(&buffers->input_tensor_scale) && valid;
  valid = free_device(&buffers->output_guard) && valid;
  if (buffers->start != nullptr)
    valid = hip_ok(hipEventDestroy(buffers->start), "hipEventDestroy(start)") &&
            valid;
  if (buffers->stop != nullptr)
    valid = hip_ok(hipEventDestroy(buffers->stop), "hipEventDestroy(stop)") &&
            valid;
  if (buffers->stream != nullptr)
    valid =
        hip_ok(hipStreamDestroy(buffers->stream), "hipStreamDestroy") && valid;
  buffers->start = nullptr;
  buffers->stop = nullptr;
  buffers->stream = nullptr;
  return valid;
}

bool upload_inputs(const Shape &shape, const HostInputs &inputs,
                   DeviceBuffers *const buffers) {
  const uint64_t output_elements = kRows * shape.n;
  return hip_ok(hipMemcpy(buffers->activation, inputs.activation.data(),
                          inputs.activation.size(), hipMemcpyHostToDevice),
                "hipMemcpy activation") &&
         hip_ok(hipMemcpy(
                    buffers->activation_scales, inputs.activation_scales.data(),
                    inputs.activation_scales.size(), hipMemcpyHostToDevice),
                "hipMemcpy activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, inputs.weight.data(),
                          inputs.weight.size(), hipMemcpyHostToDevice),
                "hipMemcpy weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, inputs.weight_scales.data(),
                          inputs.weight_scales.size(), hipMemcpyHostToDevice),
                "hipMemcpy weight scales") &&
         hip_ok(hipMemcpy(buffers->weight_tensor_scale,
                          &inputs.weight_tensor_scale, sizeof(float),
                          hipMemcpyHostToDevice),
                "hipMemcpy weight tensor scale") &&
         hip_ok(hipMemcpy(buffers->input_tensor_scale,
                          &inputs.input_tensor_scale, sizeof(float),
                          hipMemcpyHostToDevice),
                "hipMemcpy input tensor scale") &&
         hip_ok(hipMemset(
                    buffers->output_guard, static_cast<int>(0xa5),
                    static_cast<size_t>(output_elements + 2U * kGuardElements) *
                        sizeof(uint16_t)),
                "hipMemset output guard");
}

uint16_t *output_pointer(const DeviceBuffers &buffers) {
  return buffers.output_guard + kGuardElements;
}

enum class Path : uint32_t { C1M5, C2Split4Plus1, Row8Baseline };

bool launch_split4_plus1(const Shape &shape, DeviceBuffers *const buffers) {
  const uint64_t blocks = shape.k / UINT64_C(16);
  const uint64_t packed_row_bytes = shape.k / UINT64_C(2);
  const uint64_t row4 = UINT64_C(4);
  const hipError_t first = sllm_matmul_kernel::launch_nvfp4_w4a4(
      buffers->activation, buffers->activation_scales, buffers->weight,
      buffers->weight_scales, buffers->weight_tensor_scale,
      buffers->input_tensor_scale, output_pointer(*buffers), row4, shape.k,
      shape.n, sllm_matmul_kernel::KernelVariant::Nvfp4W4A4SmallMVgprReuse,
      buffers->stream);
  if (!hip_ok(first, "launch C2 split M4")) {
    return false;
  }
  return hip_ok(
      sllm_matmul_kernel::launch_nvfp4_w4a4(
          buffers->activation + static_cast<size_t>(row4 * packed_row_bytes),
          buffers->activation_scales + static_cast<size_t>(row4 * blocks),
          buffers->weight, buffers->weight_scales, buffers->weight_tensor_scale,
          buffers->input_tensor_scale,
          output_pointer(*buffers) + static_cast<size_t>(row4 * shape.n),
          UINT64_C(1), shape.k, shape.n,
          sllm_matmul_kernel::KernelVariant::Nvfp4W4A4DecodeScaleLut,
          buffers->stream),
      "launch C2 split M1");
}

bool launch_candidate(const Shape &shape, DeviceBuffers *const buffers) {
  return hip_ok(sllm_matmul_kernel::launch_nvfp4_w4a4_small_m_m5_probe(
                    buffers->activation, buffers->activation_scales,
                    buffers->weight, buffers->weight_scales,
                    buffers->weight_tensor_scale, buffers->input_tensor_scale,
                    output_pointer(*buffers), kRows, shape.k, shape.n,
                    buffers->stream),
                "launch M5 candidate");
}

bool launch_baseline(const Shape &shape, DeviceBuffers *const buffers) {
  return hip_ok(
      sllm_matmul_kernel::launch_nvfp4_w4a4(
          buffers->activation, buffers->activation_scales, buffers->weight,
          buffers->weight_scales, buffers->weight_tensor_scale,
          buffers->input_tensor_scale, output_pointer(*buffers), kRows, shape.k,
          shape.n,
          sllm_matmul_kernel::KernelVariant::Nvfp4W4A4PrefillRow8Tiled256,
          buffers->stream),
      "launch M5 row8 baseline");
}

bool wait_stream(DeviceBuffers *const buffers, const char *const operation) {
  return hip_ok(hipStreamSynchronize(buffers->stream), operation);
}

bool launch_path(const Shape &shape, DeviceBuffers *const buffers,
                 const Path path) {
  switch (path) {
  case Path::C1M5:
    return launch_candidate(shape, buffers);
  case Path::C2Split4Plus1:
    return launch_split4_plus1(shape, buffers);
  case Path::Row8Baseline:
    return launch_baseline(shape, buffers);
  }
  return false;
}

bool record_timed(const Shape &shape, DeviceBuffers *const buffers,
                  const Path path, float *const elapsed_ms) {
  const bool launched = launch_path(shape, buffers, path);
  return launched &&
         hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                "hipEventRecord(stop)") &&
         hip_ok(hipEventSynchronize(buffers->stop),
                "hipEventSynchronize(stop)") &&
         hip_ok(hipEventElapsedTime(elapsed_ms, buffers->start, buffers->stop),
                "hipEventElapsedTime");
}

bool check_guards(const Shape &shape, const DeviceBuffers &buffers) {
  const uint64_t output_elements = kRows * shape.n;
  std::vector<uint16_t> observed(
      static_cast<size_t>(output_elements + 2U * kGuardElements));
  if (!hip_ok(hipMemcpy(observed.data(), buffers.output_guard,
                        observed.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy output guard") ||
      !std::all_of(observed.begin(), observed.begin() + kGuardElements,
                   [](const uint16_t value) { return value == 0xa5a5U; }) ||
      !std::all_of(observed.end() - kGuardElements, observed.end(),
                   [](const uint16_t value) { return value == 0xa5a5U; })) {
    return false;
  }
  return true;
}

std::vector<uint16_t> download_output(const Shape &shape,
                                      const DeviceBuffers &buffers) {
  std::vector<uint16_t> output(static_cast<size_t>(kRows * shape.n));
  if (!hip_ok(hipMemcpy(output.data(), output_pointer(buffers),
                        output.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy output")) {
    output.clear();
  }
  return output;
}

uint16_t oracle_value(const Shape &shape, const HostInputs &inputs,
                      const uint64_t row, const uint64_t column) {
  const uint64_t blocks = shape.k / UINT64_C(16);
  const uint64_t packed_row_bytes = shape.k / UINT64_C(2);
  float accumulator = 0.0F;
  const uint8_t *const activation_row =
      inputs.activation.data() + row * packed_row_bytes;
  for (uint64_t block = 0U; block < blocks; ++block) {
    int32_t dot = 0;
    const uint8_t *const weight_row =
        inputs.weight.data() + column * packed_row_bytes;
    for (uint64_t index = 0U; index < 16U; ++index) {
      dot += oracle_scaled_e2m1(nibble(activation_row, block * 16U + index)) *
             oracle_scaled_e2m1(nibble(weight_row, block * 16U + index));
    }
    const float activation_scale = oracle_e4m3fn(
        inputs.activation_scales[static_cast<size_t>(row * blocks + block)]);
    const float weight_scale = oracle_e4m3fn(
        inputs.weight_scales[static_cast<size_t>(column * blocks + block)]);
    accumulator = std::fma(static_cast<float>(dot) * 0.25F * activation_scale,
                           weight_scale, accumulator);
  }
  return f32_to_bf16_rne(accumulator * inputs.weight_tensor_scale *
                         inputs.input_tensor_scale);
}

bool check_oracle(const Shape &shape, const HostInputs &inputs,
                  const std::vector<uint16_t> &output, uint32_t *const max_ulp,
                  uint32_t *const mismatches) {
  const size_t expected_size = static_cast<size_t>(kRows * shape.n);
  if (output.size() != expected_size) {
    return false;
  }
  bool valid = true;
  for (uint64_t row = 0U; row < kRows; ++row) {
    for (const uint64_t column : kSampleColumns) {
      if (column >= shape.n)
        continue;
      const uint16_t expected = oracle_value(shape, inputs, row, column);
      const uint16_t actual =
          output[static_cast<size_t>(row * shape.n + column)];
      *max_ulp = std::max(*max_ulp, bf16_ulp(actual, expected));
      if (!finite_bf16(actual) || bf16_ulp(actual, expected) > 2U)
        ++*mismatches;
      valid = valid && finite_bf16(actual) && bf16_ulp(actual, expected) <= 2U;
    }
  }
  return valid;
}

Resources query_resources() {
  Resources result;
  hipFuncAttributes attributes{};
  const hipError_t status = hipFuncGetAttributes(
      &attributes, reinterpret_cast<const void *>(
                       sllm_nvfp4_w4a4_small_m_vgpr_reuse_m5_probe_v1));
  if (status != hipSuccess) {
    std::fprintf(stderr, "hipFuncGetAttributes(M5) failed: %s\n",
                 hipGetErrorString(status));
    return result;
  }
  result.num_regs = attributes.numRegs;
  result.static_lds = static_cast<int>(attributes.sharedSizeBytes);
  result.private_bytes = static_cast<int>(attributes.localSizeBytes);
  result.max_threads = attributes.maxThreadsPerBlock;
  int active = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active,
          reinterpret_cast<const void *>(
              sllm_nvfp4_w4a4_small_m_vgpr_reuse_m5_probe_v1),
          static_cast<int>(kThreads), 0U) == hipSuccess) {
    result.active_blocks = active;
  }
  return result;
}

bool run_shape(const Shape &shape) {
  const HostInputs inputs = make_inputs(shape);
  DeviceBuffers buffers;
  if (!make_device_buffers(shape, &buffers) ||
      !upload_inputs(shape, inputs, &buffers)) {
    (void)release_device_buffers(&buffers);
    return false;
  }

  const Resources resources = query_resources();
  const auto rejected_row = [&](const uint64_t rows) {
    return sllm_matmul_kernel::launch_nvfp4_w4a4_small_m_m5_probe(
               buffers.activation, buffers.activation_scales, buffers.weight,
               buffers.weight_scales, buffers.weight_tensor_scale,
               buffers.input_tensor_scale, output_pointer(buffers), rows,
               shape.k, shape.n, buffers.stream) == hipErrorInvalidValue;
  };
  const bool invalid_neighbor_rows_rejected =
      rejected_row(4U) && rejected_row(6U);
  std::printf("m5_resources target=%s shape=%s vgpr=%d lds_static=%d "
              "private_bytes=%d max_threads=%d active_blocks=%d\n",
              SLLM_TEST_EXPECTED_TARGET, shape.name, resources.num_regs,
              resources.static_lds, resources.private_bytes,
              resources.max_threads, resources.active_blocks);

  bool valid = invalid_neighbor_rows_rejected && resources.num_regs > 0 &&
               resources.max_threads >= static_cast<int>(kThreads) &&
               resources.active_blocks > 0;
  for (int warmup = 0; warmup < kWarmups && valid; ++warmup) {
    valid = launch_candidate(shape, &buffers) &&
            wait_stream(&buffers, "candidate warmup") &&
            launch_baseline(shape, &buffers) &&
            wait_stream(&buffers, "baseline warmup") &&
            launch_candidate(shape, &buffers) &&
            wait_stream(&buffers, "candidate warmup repeat");
  }
  for (int warmup = 0; warmup < kWarmups && valid; ++warmup) {
    valid = launch_split4_plus1(shape, &buffers) &&
            wait_stream(&buffers, "C2 split warmup") &&
            launch_baseline(shape, &buffers) &&
            wait_stream(&buffers, "C2 baseline warmup") &&
            launch_split4_plus1(shape, &buffers) &&
            wait_stream(&buffers, "C2 split warmup repeat");
  }

  std::array<float, kMeasured> candidate_ms{};
  std::array<float, kMeasured> candidate_repeat_ms{};
  std::array<float, kMeasured> split4_plus1_ms{};
  std::array<float, kMeasured> split4_plus1_repeat_ms{};
  std::array<float, kMeasured> baseline_ms{};
  std::array<float, kMeasured> c2_baseline_ms{};
  for (int round = 0; round < kMeasured && valid; ++round) {
    valid = hip_ok(hipEventRecord(buffers.start, buffers.stream),
                   "hipEventRecord(start)") &&
            record_timed(shape, &buffers, Path::C1M5,
                         &candidate_ms[static_cast<size_t>(round)]);
    valid = valid && wait_stream(&buffers, "candidate AB sync");
    valid = valid &&
            hip_ok(hipEventRecord(buffers.start, buffers.stream),
                   "hipEventRecord(start BA") &&
            record_timed(shape, &buffers, Path::Row8Baseline,
                         &baseline_ms[static_cast<size_t>(round)]);
    valid = valid && wait_stream(&buffers, "baseline BA sync");
    valid = valid &&
            hip_ok(hipEventRecord(buffers.start, buffers.stream),
                   "hipEventRecord(start AB repeat)") &&
            record_timed(shape, &buffers, Path::C1M5,
                         &candidate_repeat_ms[static_cast<size_t>(round)]);
    valid = valid && wait_stream(&buffers, "candidate AB repeat sync");
  }

  for (int round = 0; round < kMeasured && valid; ++round) {
    valid = hip_ok(hipEventRecord(buffers.start, buffers.stream),
                   "hipEventRecord(C2 start)") &&
            record_timed(shape, &buffers, Path::C2Split4Plus1,
                         &split4_plus1_ms[static_cast<size_t>(round)]);
    valid = valid && wait_stream(&buffers, "C2 split AB sync");
    valid = valid &&
            hip_ok(hipEventRecord(buffers.start, buffers.stream),
                   "hipEventRecord(C2 start BA)") &&
            record_timed(shape, &buffers, Path::Row8Baseline,
                         &c2_baseline_ms[static_cast<size_t>(round)]);
    valid = valid && wait_stream(&buffers, "C2 baseline BA sync");
    valid = valid &&
            hip_ok(hipEventRecord(buffers.start, buffers.stream),
                   "hipEventRecord(C2 start AB repeat)") &&
            record_timed(shape, &buffers, Path::C2Split4Plus1,
                         &split4_plus1_repeat_ms[static_cast<size_t>(round)]);
    valid = valid && wait_stream(&buffers, "C2 split AB repeat sync");
  }

  std::vector<float> candidate_sorted(candidate_ms.begin(), candidate_ms.end());
  std::vector<float> candidate_repeat_sorted(candidate_repeat_ms.begin(),
                                             candidate_repeat_ms.end());
  std::vector<float> split4_plus1_sorted(split4_plus1_ms.begin(),
                                         split4_plus1_ms.end());
  std::vector<float> split4_plus1_repeat_sorted(split4_plus1_repeat_ms.begin(),
                                                split4_plus1_repeat_ms.end());
  std::vector<float> baseline_sorted(baseline_ms.begin(), baseline_ms.end());
  std::vector<float> c2_baseline_sorted(c2_baseline_ms.begin(),
                                        c2_baseline_ms.end());
  std::sort(candidate_sorted.begin(), candidate_sorted.end());
  std::sort(candidate_repeat_sorted.begin(), candidate_repeat_sorted.end());
  std::sort(split4_plus1_sorted.begin(), split4_plus1_sorted.end());
  std::sort(split4_plus1_repeat_sorted.begin(),
            split4_plus1_repeat_sorted.end());
  std::sort(baseline_sorted.begin(), baseline_sorted.end());
  std::sort(c2_baseline_sorted.begin(), c2_baseline_sorted.end());
  bool c1_all_rounds_faster = true;
  bool c2_all_rounds_faster = true;
  for (size_t round = 0U; round < static_cast<size_t>(kMeasured); ++round) {
    c1_all_rounds_faster = c1_all_rounds_faster &&
                           candidate_ms[round] < baseline_ms[round] &&
                           candidate_repeat_ms[round] < baseline_ms[round];
    c2_all_rounds_faster =
        c2_all_rounds_faster &&
        split4_plus1_ms[round] < c2_baseline_ms[round] &&
        split4_plus1_repeat_ms[round] < c2_baseline_ms[round];
    std::printf("m5_round target=%s shape=%s round=%zu c1_ms=%.6f "
                "baseline_ms=%.6f c1_repeat_ms=%.6f c2_ms=%.6f "
                "c2_baseline_ms=%.6f c2_repeat_ms=%.6f\n",
                SLLM_TEST_EXPECTED_TARGET, shape.name, round,
                candidate_ms[round], baseline_ms[round],
                candidate_repeat_ms[round], split4_plus1_ms[round],
                c2_baseline_ms[round], split4_plus1_repeat_ms[round]);
  }

  // Re-run each path after timing so C1 and C2 each have an explicit first
  // output, baseline output, and repeat output for the full finite/guard and
  // bitwise repeat checks.
  valid = valid && launch_candidate(shape, &buffers) &&
          wait_stream(&buffers, "candidate correctness") &&
          check_guards(shape, buffers);
  const std::vector<uint16_t> candidate_output =
      download_output(shape, buffers);
  uint32_t candidate_max_ulp = 0U;
  uint32_t candidate_mismatches = 0U;

  // Run baseline once for an independent output comparison, then restore C1
  // to exercise repeat determinism.
  valid = valid && launch_baseline(shape, &buffers) &&
          wait_stream(&buffers, "baseline correctness") &&
          check_guards(shape, buffers);
  const std::vector<uint16_t> baseline_output = download_output(shape, buffers);
  uint32_t baseline_max_ulp = 0U;
  uint32_t baseline_mismatches = 0U;
  valid = valid && check_oracle(shape, inputs, baseline_output,
                                &baseline_max_ulp, &baseline_mismatches);
  valid = valid && launch_candidate(shape, &buffers) &&
          wait_stream(&buffers, "candidate repeat") &&
          check_guards(shape, buffers);
  const std::vector<uint16_t> repeat_output = download_output(shape, buffers);
  valid = valid && check_oracle(shape, inputs, candidate_output,
                                &candidate_max_ulp, &candidate_mismatches);

  valid = valid && launch_split4_plus1(shape, &buffers) &&
          wait_stream(&buffers, "C2 split correctness") &&
          check_guards(shape, buffers);
  const std::vector<uint16_t> split4_plus1_output =
      download_output(shape, buffers);
  uint32_t split4_plus1_max_ulp = 0U;
  uint32_t split4_plus1_mismatches = 0U;
  valid =
      valid && check_oracle(shape, inputs, split4_plus1_output,
                            &split4_plus1_max_ulp, &split4_plus1_mismatches);
  valid = valid && launch_baseline(shape, &buffers) &&
          wait_stream(&buffers, "C2 baseline correctness") &&
          check_guards(shape, buffers);
  valid = valid && launch_split4_plus1(shape, &buffers) &&
          wait_stream(&buffers, "C2 split repeat") &&
          check_guards(shape, buffers);
  const std::vector<uint16_t> split4_plus1_repeat_output =
      download_output(shape, buffers);

  uint32_t candidate_baseline_mismatches = 0U;
  uint32_t candidate_repeat_mismatches = 0U;
  uint32_t split4_plus1_baseline_mismatches = 0U;
  uint32_t split4_plus1_candidate_mismatches = 0U;
  uint32_t split4_plus1_repeat_mismatches = 0U;
  const size_t output_size = static_cast<size_t>(kRows * shape.n);
  if (candidate_output.size() != output_size ||
      baseline_output.size() != output_size ||
      repeat_output.size() != output_size ||
      split4_plus1_output.size() != output_size ||
      split4_plus1_repeat_output.size() != output_size) {
    valid = false;
  } else {
    for (size_t index = 0U; index < output_size; ++index) {
      if (candidate_output[index] != baseline_output[index])
        ++candidate_baseline_mismatches;
      if (candidate_output[index] != repeat_output[index])
        ++candidate_repeat_mismatches;
      if (split4_plus1_output[index] != baseline_output[index])
        ++split4_plus1_baseline_mismatches;
      if (split4_plus1_output[index] != candidate_output[index])
        ++split4_plus1_candidate_mismatches;
      if (split4_plus1_output[index] != split4_plus1_repeat_output[index])
        ++split4_plus1_repeat_mismatches;
      valid = valid && finite_bf16(candidate_output[index]) &&
              finite_bf16(baseline_output[index]) &&
              finite_bf16(repeat_output[index]) &&
              finite_bf16(split4_plus1_output[index]) &&
              finite_bf16(split4_plus1_repeat_output[index]);
    }
  }
  std::printf(
      "m5_result target=%s shape=%s M=5 K=%llu N=%llu candidate_median_ms=%.6f "
      "candidate_repeat_median_ms=%.6f "
      "c2_split4_plus1_median_ms=%.6f "
      "c2_split4_plus1_repeat_median_ms=%.6f "
      "baseline_row8_median_ms=%.6f "
      "c2_baseline_row8_median_ms=%.6f "
      "candidate_max_ulp=%u baseline_max_ulp=%u "
      "c2_max_ulp=%u "
      "candidate_oracle_mismatches=%u baseline_oracle_mismatches=%u "
      "c2_oracle_mismatches=%u "
      "candidate_baseline_mismatches=%u candidate_repeat_mismatches=%u "
      "c2_baseline_mismatches=%u c2_candidate_mismatches=%u "
      "c2_repeat_mismatches=%u "
      "c1_all_rounds_faster=%d c2_all_rounds_faster=%d "
      "invalid_neighbor_rows_rejected=%d "
      "status=%s\n",
      SLLM_TEST_EXPECTED_TARGET, shape.name,
      static_cast<unsigned long long>(shape.k),
      static_cast<unsigned long long>(shape.n), candidate_sorted[kMeasured / 2],
      candidate_repeat_sorted[kMeasured / 2],
      split4_plus1_sorted[kMeasured / 2],
      split4_plus1_repeat_sorted[kMeasured / 2], baseline_sorted[kMeasured / 2],
      c2_baseline_sorted[kMeasured / 2], candidate_max_ulp, baseline_max_ulp,
      split4_plus1_max_ulp, candidate_mismatches, baseline_mismatches,
      split4_plus1_mismatches, candidate_baseline_mismatches,
      candidate_repeat_mismatches, split4_plus1_baseline_mismatches,
      split4_plus1_candidate_mismatches, split4_plus1_repeat_mismatches,
      c1_all_rounds_faster ? 1 : 0, c2_all_rounds_faster ? 1 : 0,
      invalid_neighbor_rows_rejected ? 1 : 0, valid ? "PASS" : "FAIL");

  valid = release_device_buffers(&buffers) && valid;
  return valid;
}

} // namespace

int main() {
  const std::array<Shape, 2> shapes = {
      Shape{"gate_up", 5120U, 17408U},
      Shape{"down", 17408U, 5120U},
  };
  bool valid = true;
  for (const Shape &shape : shapes) {
    valid = run_shape(shape) && valid;
  }
  std::printf("m5_probe target=%s status=%s\n", SLLM_TEST_EXPECTED_TARGET,
              valid ? "PASS" : "FAIL");
  return valid ? 0 : 1;
}
