#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

// Phase 83.5 public NVFP4 prefill tiling boundaries.
// gfx1030 checks the M64/M128 tile switch at 511/512/513 and 1023/1024/1025;
// gfx1201 checks StageK32/StageK64 at 255/256/257. Public plan/execute must
// select the expected logical identity, device symbol, and grid. Independent
// boundary samples check the numerical oracle; full output and guard scans
// check finiteness, corruption, and exact repeat equality.

namespace {

constexpr uint32_t kTimeoutMs = 30'000U;
constexpr bool kGfx1030 =
    std::string_view{SLLM_TEST_EXPECTED_TARGET} == "gfx1030";
constexpr uint32_t kKernelId = kGfx1030 ? 87U : 89U;
constexpr uint32_t kDispatchCount = 2U;
constexpr uint32_t kWorkgroup = 256U;
constexpr uint32_t kGuardWords = 2048U;
constexpr uint16_t kGuardWord = UINT16_C(0x7e35);
constexpr std::size_t kWarmups = 1U;
constexpr std::size_t kMeasured = 2U;
constexpr const char *kLogicalSymbol =
    kGfx1030 ? "matmul.nvfp4.w4a4.block16.prefill.compensated64x64.v1"
             : "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.kahan.v1";
constexpr const char *kBaseSymbol =
    kGfx1030 ? "sllm_nvfp4_w4a4_prefill_compensated64x64_v1"
             : "sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_v1";
constexpr const char *kOptimizedSymbol =
    kGfx1030 ? "sllm_nvfp4_w4a4_prefill_compensated128x64_v1"
             : "sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_lookahead_v1";

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

struct Shape final {
  const char *name;
  uint64_t k;
  uint64_t n;
};

struct RunResult final {
  bool valid = false;
  bool full_finite = false;
  bool full_repeat = false;
  uint32_t max_oracle_ulp = 0U;
  double median_ms = 0.0;
  sllm_matmul_dispatch_info_t dispatch{};
};

bool expect(const sllm_status_t actual, const sllm_status_t wanted,
            const char *const operation, const Error &error) {
  if (actual == wanted) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << wanted
            << ": " << error.message << '\n';
  return false;
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint32_t bf16_ulp(const uint16_t actual, const uint16_t expected) {
  return actual >= expected ? static_cast<uint32_t>(actual - expected)
                            : static_cast<uint32_t>(expected - actual);
}

bool finite_bf16(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "sllm_buffer_release", error) &&
         *buffer == nullptr;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned no completion\n";
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
  const bool released =
      expect(sllm_completion_release(completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release", error) &&
      *completion == nullptr;
  return waited && released;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, output, &error.sink),
                SLLM_STATUS_OK, "sllm_buffer_create", error) &&
         output != nullptr && *output != nullptr;
}

bool copy_h2d(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, const void *const source,
              const uint64_t bytes) {
  const auto *const source_bytes = static_cast<const uint8_t *>(source);
  for (uint64_t offset = 0U; offset < bytes;) {
    const uint64_t count =
        std::min(bytes - offset, SLLM_HIP_MAX_TRANSFER_BYTES);
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = const_cast<uint8_t *>(source_bytes + offset);
    transfer.size_bytes = count;
    transfer.buffer_offset_bytes = offset;
    sllm_completion_t *completion = nullptr;
    Error error;
    if (!expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error) ||
        !wait_release(&completion, "sllm_buffer_copy_h2d wait")) {
      return false;
    }
    offset += count;
  }
  return true;
}

bool copy_d2h(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = destination;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error) ||
      completion == nullptr) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  bool valid =
      expect(sllm_completion_wait(completion, kTimeoutMs, &result, &error.sink),
             SLLM_STATUS_OK, "sllm_buffer_copy_d2h wait", error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  uint64_t bytes_written = 0U;
  valid = expect(sllm_completion_read(completion, destination, bytes,
                                      &bytes_written, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_read", error) &&
          bytes_written == bytes && valid;
  valid = expect(sllm_completion_release(&completion, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_release", error) &&
          completion == nullptr && valid;
  return valid;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint32_t encoding,
                              const uint64_t rows, const uint64_t columns) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = encoding;
  result.rank = 2U;
  result.shape[0] = rows;
  result.shape[1] = columns;
  result.stride_elements[0] = columns;
  result.stride_elements[1] = 1U;
  return result;
}

std::vector<uint16_t> make_activation(const Shape &shape, const uint64_t rows) {
  std::vector<uint16_t> result(static_cast<std::size_t>(rows * shape.k));
  for (uint64_t row = 0U; row < rows; ++row) {
    // 6.0 and 3.0 are exact E2M1 values after the public BF16->NVFP4 pack.
    const float value = (row & 1U) == 0U ? 6.0F : 3.0F;
    const uint16_t encoded = f32_to_bf16_rne(value);
    std::fill(result.begin() + static_cast<std::ptrdiff_t>(row * shape.k),
              result.begin() +
                  static_cast<std::ptrdiff_t>((row + 1U) * shape.k),
              encoded);
  }
  return result;
}

uint64_t nvfp4_weight_bytes(const Shape &shape) {
  const uint64_t values = shape.k * shape.n / 2U;
  const uint64_t scales = shape.n * (shape.k / 16U);
  return ((values + scales + 3U) & ~UINT64_C(3)) + 8U;
}

std::vector<uint8_t> make_nvfp4_weight(const Shape &shape) {
  const uint64_t values = shape.k * shape.n / 2U;
  const uint64_t scales = shape.n * (shape.k / 16U);
  const uint64_t scale_offset = (values + scales + 3U) & ~UINT64_C(3);
  std::vector<uint8_t> result(static_cast<std::size_t>(scale_offset + 8U), 0U);
  // E2M1 code 2 is +1.0; 0x22 fills both packed nibbles.
  std::fill(result.begin(),
            result.begin() + static_cast<std::ptrdiff_t>(values),
            UINT8_C(0x22));
  // E4M3FN 0x38 is the positive 1.0 block scale.
  std::fill(result.begin() + static_cast<std::ptrdiff_t>(values),
            result.begin() + static_cast<std::ptrdiff_t>(values + scales),
            UINT8_C(0x38));
  const float one = 1.0F;
  std::memcpy(result.data() + scale_offset, &one, sizeof(one));
  std::memcpy(result.data() + scale_offset + sizeof(one), &one, sizeof(one));
  return result;
}

sllm_matmul_desc_t descriptor(const Shape &shape, const uint64_t rows,
                              const sllm_buffer_t *const activation,
                              const sllm_buffer_t *const weight,
                              const sllm_buffer_t *const output) {
  sllm_matmul_desc_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.op_version = SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION;
  result.activation = binding(activation, SLLM_TENSOR_DTYPE_BF16,
                              SLLM_TENSOR_ENCODING_UNQUANTIZED, rows, shape.k);
  result.weight = binding(weight, SLLM_TENSOR_DTYPE_U8,
                          SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32,
                          shape.n, shape.k);
  result.output = binding(output, SLLM_TENSOR_DTYPE_BF16,
                          SLLM_TENSOR_ENCODING_UNQUANTIZED, rows, shape.n);
  return result;
}

bool check_dispatch(const Shape &shape, const uint64_t rows,
                    const sllm_matmul_dispatch_info_t &dispatch) {
  const bool optimized =
      kGfx1030 ? rows >= 512U && rows % 128U == 0U : rows >= 256U;
  const char *const expected_device =
      !kGfx1030 && rows >= 256U && rows % 128U == 0U
          ? (rows == 2048U
                 ? (shape.k == 5120U
                        ? "sllm_nvfp4_gfx1201_wmma128x64_pad68_k5120n17408_v1"
                        : "sllm_nvfp4_gfx1201_wmma128x64_pad68_k17408n5120_v1")
                 : (shape.k == 5120U
                        ? "sllm_nvfp4_gfx1201_wmma128x64_aligned_k5120n17408_v1"
                        : "sllm_nvfp4_gfx1201_wmma128x64_aligned_k17408n5120_"
                          "v1"))
          : (optimized ? kOptimizedSymbol : kBaseSymbol);
  const uint64_t tile_rows = optimized ? 128U : 64U;
  const uint64_t row_tiles =
      kGfx1030 ? (rows + tile_rows - 1U) / tile_rows : 1U;
  return dispatch.dispatch_id != 0U &&
         dispatch.dispatch_count == kDispatchCount &&
         dispatch.kernel_id == kKernelId &&
         dispatch.workgroup_size_x == kWorkgroup &&
         dispatch.grid_size_x ==
             static_cast<uint32_t>(row_tiles * ((shape.n + 63U) / 64U)) &&
         dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
         dispatch.m == rows && dispatch.k == shape.k && dispatch.n == shape.n &&
         dispatch.output_elements == rows * shape.n &&
         std::strcmp(dispatch.kernel_symbol, kLogicalSymbol) == 0 &&
         std::strcmp(dispatch.device_symbol, expected_device) == 0 &&
         std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
}

bool check_output(const Shape &shape, const uint64_t rows,
                  const std::vector<uint16_t> &activation,
                  const std::vector<uint16_t> &observed,
                  uint32_t *const max_ulp) {
  const std::size_t output_words = static_cast<std::size_t>(rows * shape.n);
  if (observed.size() != output_words + kGuardWords) {
    return false;
  }
  bool valid = true;
  for (std::size_t index = 0U; index < output_words; ++index) {
    valid = finite_bf16(observed[index]) && valid;
  }
  const std::array<uint64_t, 5> rows_to_check = {0U, 1U, rows / 2U, rows - 2U,
                                                 rows - 1U};
  const std::array<uint64_t, 5> columns_to_check = {0U, 1U, shape.n / 2U,
                                                    shape.n - 2U, shape.n - 1U};
  for (const uint64_t row : rows_to_check) {
    for (const uint64_t column : columns_to_check) {
      const std::size_t index =
          static_cast<std::size_t>(row * shape.n + column);
      const float input =
          bf16_to_f32(activation[static_cast<std::size_t>(row * shape.k)]);
      const uint16_t expected =
          f32_to_bf16_rne(input * static_cast<float>(shape.k));
      *max_ulp = std::max(*max_ulp, bf16_ulp(observed[index], expected));
      valid =
          finite_bf16(observed[index]) && observed[index] == expected && valid;
    }
  }
  for (uint32_t index = 0U; index < kGuardWords; ++index) {
    if (observed[output_words + index] != kGuardWord) {
      valid = false;
      break;
    }
  }
  return valid;
}

bool execute_once(const Shape &shape, const uint64_t rows,
                  const sllm_matmul_plan_t *const plan,
                  const sllm_queue_t *const queue,
                  const sllm_buffer_t *const output,
                  std::vector<uint16_t> *const observed,
                  sllm_matmul_dispatch_info_t *const dispatch,
                  double *const elapsed_ms) {
  dispatch->struct_size = sizeof(*dispatch);
  dispatch->abi_version = SLLM_HIP_ABI_VERSION;
  dispatch->info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  bool valid = expect(sllm_matmul_execute(plan, queue, &completion, dispatch,
                                          &error.sink),
                      SLLM_STATUS_OK, "sllm_matmul_execute", error) &&
               completion != nullptr;
  if (valid) {
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_completion_wait(completion, kTimeoutMs, &result,
                                        &error.sink),
                   SLLM_STATUS_OK, "sllm_matmul_execute wait", error) &&
            result.state == SLLM_COMPLETION_STATE_SUCCESS;
    sllm_completion_timing_t timing{};
    timing.struct_size = sizeof(timing);
    timing.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_completion_timing(completion, &timing, &error.sink),
                   SLLM_STATUS_OK, "sllm_completion_timing", error) &&
            timing.valid != 0U && valid;
    if (valid) {
      *elapsed_ms = static_cast<double>(timing.elapsed_ns) / 1.0e6;
    }
    valid = expect(sllm_completion_release(&completion, &error.sink),
                   SLLM_STATUS_OK, "sllm_completion_release", error) &&
            completion == nullptr && valid;
  }
  if (valid) {
    valid =
        check_dispatch(shape, rows, *dispatch) &&
        copy_d2h(queue, output, observed->data(),
                 static_cast<uint64_t>(observed->size() * sizeof(uint16_t)));
  }
  return valid;
}

RunResult run_case(const Shape &shape, const uint64_t rows,
                   const sllm_context_t *const context,
                   const sllm_queue_t *const queue) {
  RunResult result{};
  const uint64_t weight_bytes = nvfp4_weight_bytes(shape);
  const uint64_t activation_bytes = rows * shape.k * sizeof(uint16_t);
  const std::size_t output_words = static_cast<std::size_t>(rows * shape.n);
  const uint64_t output_bytes =
      static_cast<uint64_t>(output_words + kGuardWords) * sizeof(uint16_t);
  const std::vector<uint16_t> activation = make_activation(shape, rows);
  const std::vector<uint8_t> weight = make_nvfp4_weight(shape);
  std::vector<uint16_t> initial(output_words + kGuardWords, kGuardWord);
  std::vector<uint16_t> observed(initial.size(), 0U);
  std::vector<uint16_t> first_output;
  sllm_buffer_t *activation_buffer = nullptr;
  sllm_buffer_t *weight_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  sllm_matmul_plan_t *plan = nullptr;
  bool valid =
      weight.size() == static_cast<std::size_t>(weight_bytes) &&
      create_buffer(context, activation_bytes, &activation_buffer) &&
      create_buffer(context, weight_bytes, &weight_buffer) &&
      create_buffer(context, output_bytes, &output_buffer) &&
      copy_h2d(queue, activation_buffer, activation.data(), activation_bytes) &&
      copy_h2d(queue, weight_buffer, weight.data(), weight_bytes) &&
      copy_h2d(queue, output_buffer, initial.data(), output_bytes);
  if (valid) {
    const sllm_matmul_desc_t desc = descriptor(shape, rows, activation_buffer,
                                               weight_buffer, output_buffer);
    Error error;
    valid = expect(sllm_matmul_prepare(context, &desc, &plan, &error.sink),
                   SLLM_STATUS_OK, "sllm_matmul_prepare", error) &&
            plan != nullptr;
  }
  std::vector<double> timings;
  timings.reserve(kMeasured);
  bool all_finite = true;
  bool all_repeat = true;
  std::size_t completed_iterations = 0U;
  for (std::size_t iteration = 0U; valid && iteration < kWarmups + kMeasured;
       ++iteration) {
    double elapsed_ms = 0.0;
    sllm_matmul_dispatch_info_t dispatch{};
    valid = execute_once(shape, rows, plan, queue, output_buffer, &observed,
                         &dispatch, &elapsed_ms);
    if (!valid) {
      break;
    }
    ++completed_iterations;
    bool iteration_finite = observed.size() >= output_words;
    for (std::size_t index = 0U; iteration_finite && index < output_words;
         ++index) {
      iteration_finite = finite_bf16(observed[index]);
    }
    all_finite = all_finite && iteration_finite;
    uint32_t iteration_ulp = 0U;
    valid = check_output(shape, rows, activation, observed, &iteration_ulp);
    result.max_oracle_ulp = std::max(result.max_oracle_ulp, iteration_ulp);
    if (iteration >= kWarmups) {
      timings.push_back(elapsed_ms);
      if (first_output.empty()) {
        first_output = observed;
      } else {
        const bool repeat = observed == first_output;
        all_repeat = all_repeat && repeat;
        valid = repeat && valid;
      }
    }
    result.dispatch = dispatch;
  }
  result.full_finite =
      completed_iterations == kWarmups + kMeasured && all_finite;
  result.full_repeat = completed_iterations == kWarmups + kMeasured &&
                       timings.size() == kMeasured && all_repeat;
  if (valid && timings.size() == kMeasured) {
    std::sort(timings.begin(), timings.end());
    result.median_ms = timings[timings.size() / 2U];
  }
  result.valid =
      valid && timings.size() == kMeasured && result.max_oracle_ulp == 0U;
  if (plan != nullptr) {
    Error error;
    valid = expect(sllm_matmul_plan_release(&plan, &error.sink), SLLM_STATUS_OK,
                   "sllm_matmul_plan_release", error) &&
            valid;
  }
  valid = release_buffer(&output_buffer) && valid;
  valid = release_buffer(&weight_buffer) && valid;
  valid = release_buffer(&activation_buffer) && valid;
  result.valid = result.valid && valid && plan == nullptr &&
                 output_buffer == nullptr && weight_buffer == nullptr &&
                 activation_buffer == nullptr;
  std::cout << std::fixed << std::setprecision(6)
            << "phase83_5_nvfp4_prefill_tiling shape=" << shape.name
            << " M=" << rows << " K=" << shape.k << " N=" << shape.n
            << " provider_id=" << result.dispatch.kernel_id
            << " dispatch_count=" << result.dispatch.dispatch_count
            << " kernel=" << result.dispatch.kernel_symbol
            << " device=" << result.dispatch.device_symbol
            << " median_ms=" << result.median_ms
            << " oracle=sampled_boundary_rows_columns max_bf16_ulp="
            << result.max_oracle_ulp
            << " full_finite=" << (result.full_finite ? "PASS" : "FAIL")
            << " full_repeat=" << (result.full_repeat ? "PASS" : "FAIL")
            << " status=" << (result.valid ? "PASS" : "FAIL") << '\n';
  return result;
}

bool create_context_queue(sllm_context_t **const context,
                          sllm_queue_t **const queue) {
  Error error;
  uint32_t count = 0U;
  if (!expect(sllm_device_count(&count, &error.sink), SLLM_STATUS_OK,
              "sllm_device_count", error) ||
      count != 1U) {
    std::cerr << "expected exactly one visible GPU, got " << count << '\n';
    return false;
  }
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "sllm_device_query", error) ||
      std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "expected exact target " << SLLM_TEST_EXPECTED_TARGET
              << ", got " << device.gcn_arch_name << '\n';
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
              SLLM_STATUS_OK, "sllm_context_create", error) ||
      *context == nullptr) {
    return false;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_queue_create(*context, &queue_info, queue, &error.sink),
              SLLM_STATUS_OK, "sllm_queue_create", error) ||
      *queue == nullptr) {
    (void)sllm_context_release(context, &error.sink);
    return false;
  }
  return true;
}

void clear_selector_controls() {
  unsetenv("SLLM_MATMUL_FORCE_BASELINE");
  unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_WMMA_COMPENSATED");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_ROW8");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8");
  unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A");
  unsetenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID");
  unsetenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201");
}

} // namespace

int main() {
  try {
    if (!kGfx1030 && std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") != 0) {
      std::cerr << "phase83_5_nvfp4_prefill_tiling requires exact "
                   "gfx1030/gfx1201, got "
                << SLLM_TEST_EXPECTED_TARGET << '\n';
      return 1;
    }
    clear_selector_controls();
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    bool valid = create_context_queue(&context, &queue);
    constexpr std::array<Shape, 2> shapes = {{
        {"mlp_gate_up", 5120U, 17408U},
        {"mlp_down", 17408U, 5120U},
    }};
    const std::vector<uint64_t> rows =
        kGfx1030 ? std::vector<uint64_t>{511U, 512U, 513U, 1023U, 1024U, 1025U}
                 : std::vector<uint64_t>{255U, 256U, 257U, 2047U, 2048U, 2049U};
    bool completed_case = false;
    bool aggregate_finite = true;
    bool aggregate_repeat = true;
    for (const Shape &shape : shapes) {
      for (const uint64_t row_count : rows) {
        if (!valid) {
          break;
        }
        const RunResult result = run_case(shape, row_count, context, queue);
        completed_case = true;
        aggregate_finite = aggregate_finite && result.full_finite;
        aggregate_repeat = aggregate_repeat && result.full_repeat;
        valid = result.valid && valid;
      }
    }
    Error error;
    if (queue != nullptr) {
      valid = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                     "sllm_queue_release", error) &&
              valid;
    }
    if (context != nullptr) {
      valid = expect(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_OK, "sllm_context_release", error) &&
              valid;
    }
    clear_selector_controls();
    std::cout << "phase83_5_nvfp4_prefill_tiling status="
              << (valid ? "PASS" : "FAIL")
              << " target=" << SLLM_TEST_EXPECTED_TARGET
              << " public_plan_execute=1 dispatch_boundary=1"
              << " sampled_oracle=boundary_5x5" << " full_finite="
              << (completed_case && aggregate_finite ? "PASS" : "FAIL")
              << " full_repeat="
              << (completed_case && aggregate_repeat ? "PASS" : "FAIL")
              << " resources_released="
              << ((queue == nullptr && context == nullptr) ? 1 : 0) << '\n';
    return valid ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "phase83_5_nvfp4_prefill_tiling exception: " << error.what()
              << '\n';
    return 2;
  }
}
