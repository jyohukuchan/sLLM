#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

// Phase83.5 ID94 public NVFP4 small-M correctness and dispatch regression.
// This uses the public matmul ABI and an independent all-output BF16 oracle.

namespace {

constexpr std::array<uint64_t, 3> kRows = {2U, 3U, 4U};
constexpr std::size_t kWarmups = 1U;
constexpr std::size_t kMeasured = 2U;

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
  const char *family;
  const char *role;
  uint64_t k;
  uint64_t n;
  uint32_t op_version;
  uint32_t weight_dtype;
  uint32_t weight_encoding;
};

struct Run final {
  bool valid = false;
  uint32_t max_bf16_ulp = 0U;
  double median_ms = 0.0;
  sllm_matmul_dispatch_info_t dispatch{};
};

bool expect(const sllm_status_t actual, const sllm_status_t expected,
            const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  uint32_t rounded = upper;
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
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

uint32_t bf16_ulp(const uint16_t actual, const uint16_t expected) {
  return actual >= expected ? static_cast<uint32_t>(actual - expected)
                            : static_cast<uint32_t>(expected - actual);
}

bool finite_bf16(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
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

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, output, &error.sink),
                SLLM_STATUS_OK, "sllm_buffer_create", error);
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (*buffer == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "sllm_buffer_release", error);
}

bool upload_chunk(const sllm_queue_t *const queue,
                  const sllm_buffer_t *const buffer, const void *const source,
                  const uint64_t bytes, const uint64_t offset) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.size_bytes = bytes;
  transfer.buffer_offset_bytes = offset;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error) ||
      completion == nullptr) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited =
      expect(sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
             SLLM_STATUS_OK, "sllm_completion_wait(h2d)", error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(h2d)", error);
  return waited && released;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const source, const uint64_t bytes) {
  const auto *const data = static_cast<const uint8_t *>(source);
  for (uint64_t offset = 0; offset < bytes;) {
    const uint64_t count =
        std::min(bytes - offset, SLLM_HIP_MAX_TRANSFER_BYTES);
    if (!upload_chunk(queue, buffer, data + offset, count, offset))
      return false;
    offset += count;
  }
  return true;
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer,
              std::vector<uint16_t> *const output) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes =
      static_cast<uint64_t>(output->size()) * sizeof(uint16_t);
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
      expect(sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
             SLLM_STATUS_OK, "sllm_completion_wait(d2h)", error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  uint64_t written = 0U;
  valid =
      expect(sllm_completion_read(completion, output->data(),
                                  transfer.size_bytes, &written, &error.sink),
             SLLM_STATUS_OK, "sllm_completion_read", error) &&
      written == transfer.size_bytes && valid;
  valid = expect(sllm_completion_release(&completion, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_release(d2h)", error) &&
          valid;
  return valid;
}

bool wait_timed(sllm_completion_t **const completion,
                double *const elapsed_ms) {
  if (completion == nullptr || *completion == nullptr) {
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  bool valid = expect(sllm_completion_wait(*completion, UINT32_MAX, &result,
                                           &error.sink),
                      SLLM_STATUS_OK, "sllm_completion_wait(matmul)", error) &&
               result.state == SLLM_COMPLETION_STATE_SUCCESS;
  sllm_completion_timing_t timing{};
  timing.struct_size = sizeof(timing);
  timing.abi_version = SLLM_HIP_ABI_VERSION;
  valid = expect(sllm_completion_timing(*completion, &timing, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_timing(matmul)", error) &&
          timing.valid != 0U && valid;
  if (valid) {
    *elapsed_ms = static_cast<double>(timing.elapsed_ns) / 1.0e6;
  }
  valid = expect(sllm_completion_release(completion, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_release(matmul)", error) &&
          valid;
  return valid;
}

std::vector<uint16_t> make_activation(const Shape &shape, const uint64_t rows) {
  std::vector<uint16_t> result(static_cast<std::size_t>(rows * shape.k));
  for (uint64_t row = 0U; row != rows; ++row) {
    const float value = std::strcmp(shape.family, "NVFP4") == 0
                            ? ((row & 1U) == 0U ? 6.0F : 3.0F)
                            : ((row & 1U) == 0U ? 1.0F : 1.5F);
    std::fill(result.begin() + static_cast<std::ptrdiff_t>(row * shape.k),
              result.begin() +
                  static_cast<std::ptrdiff_t>((row + 1U) * shape.k),
              f32_to_bf16_rne(value));
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
  // E2M1 code 2 is +1.0; repeated nibbles make the sampled oracle exact.
  std::fill(result.begin(),
            result.begin() + static_cast<std::ptrdiff_t>(values),
            UINT8_C(0x22));
  std::fill(result.begin() + static_cast<std::ptrdiff_t>(values),
            result.begin() + static_cast<std::ptrdiff_t>(values + scales),
            UINT8_C(0x38)); // E4M3FN 1.0 block scale.
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
  result.op_version = shape.op_version;
  result.activation = binding(activation, SLLM_TENSOR_DTYPE_BF16,
                              SLLM_TENSOR_ENCODING_UNQUANTIZED, rows, shape.k);
  result.weight = binding(weight, shape.weight_dtype, shape.weight_encoding,
                          shape.n, shape.k);
  result.output = binding(output, SLLM_TENSOR_DTYPE_BF16,
                          SLLM_TENSOR_ENCODING_UNQUANTIZED, rows, shape.n);
  return result;
}

bool check_samples(const Shape &shape, const uint64_t rows,
                   const std::vector<uint16_t> &activation,
                   const std::vector<uint16_t> &output,
                   uint32_t *const max_ulp) {
  bool valid = output.size() == static_cast<std::size_t>(rows * shape.n);
  for (uint64_t row = 0U; row != rows && valid; ++row) {
    const float input =
        bf16_to_f32(activation[static_cast<std::size_t>(row * shape.k)]);
    const float expected_f32 = input * static_cast<float>(shape.k);
    const uint16_t expected = f32_to_bf16_rne(expected_f32);
    for (uint64_t column = 0U; column != shape.n; ++column) {
      const std::size_t index =
          static_cast<std::size_t>(row * shape.n + column);
      const uint16_t actual = output[index];
      *max_ulp = std::max(*max_ulp, bf16_ulp(actual, expected));
      valid = valid && finite_bf16(actual) && actual == expected;
    }
  }
  return valid;
}

Run run_plan(const Shape &shape, const uint64_t rows,
             const sllm_matmul_plan_t *const plan,
             const sllm_queue_t *const queue, const sllm_buffer_t *const output,
             const std::vector<uint16_t> &activation) {
  Run result{};
  std::vector<double> timings;
  timings.reserve(kMeasured);
  bool valid = true;
  std::vector<uint16_t> observed(static_cast<std::size_t>(rows * shape.n));
  std::vector<uint16_t> first_output;
  for (std::size_t iteration = 0U; iteration != kWarmups + kMeasured && valid;
       ++iteration) {
    sllm_matmul_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    Error error;
    valid = expect(sllm_matmul_execute(plan, queue, &completion, &dispatch,
                                       &error.sink),
                   SLLM_STATUS_OK, "sllm_matmul_execute", error) &&
            completion != nullptr;
    double elapsed_ms = 0.0;
    valid = wait_timed(&completion, &elapsed_ms) && valid;
    valid = valid && dispatch.dispatch_id != 0U &&
            dispatch.dispatch_count == 2U && dispatch.m == rows &&
            dispatch.k == shape.k && dispatch.n == shape.n &&
            dispatch.output_elements == rows * shape.n &&
            dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
            dispatch.workgroup_size_x == 256U &&
            dispatch.grid_size_x == (shape.n + 31U) / 32U &&
            dispatch.kernel_id == 94U &&
            std::strcmp(dispatch.kernel_symbol,
                        "matmul.nvfp4.w4a4.small_m.vgpr_reuse.v1") == 0 &&
            std::strcmp(dispatch.device_symbol,
                        "sllm_nvfp4_w4a4_small_m_vgpr_reuse_v1") == 0 &&
            std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
    if (!valid) {
      break;
    }
    if (iteration >= kWarmups) {
      timings.push_back(elapsed_ms);
      valid = download(queue, output, &observed) && valid;
      uint32_t iteration_max_ulp = 0U;
      valid = check_samples(shape, rows, activation, observed,
                            &iteration_max_ulp) &&
              valid;
      result.max_bf16_ulp = std::max(result.max_bf16_ulp, iteration_max_ulp);
      for (const uint16_t value : observed) {
        valid = valid && finite_bf16(value);
      }
      if (iteration == kWarmups) {
        first_output = observed;
      } else {
        valid = valid && observed == first_output;
      }
    }
    result.dispatch = dispatch;
  }
  if (valid && !timings.empty()) {
    std::sort(timings.begin(), timings.end());
    result.median_ms = timings[timings.size() / 2U];
  }
  result.valid = valid && timings.size() == kMeasured;
  return result;
}

bool run_shape(const Shape &shape, const sllm_context_t *const context,
               const sllm_queue_t *const queue) {
  const uint64_t weight_bytes = nvfp4_weight_bytes(shape);
  std::vector<uint8_t> host_weight = make_nvfp4_weight(shape);
  sllm_buffer_t *weight_buffer = nullptr;
  bool valid = create_buffer(context, weight_bytes, &weight_buffer) &&
               host_weight.size() == static_cast<std::size_t>(weight_bytes) &&
               upload(queue, weight_buffer, host_weight.data(), weight_bytes);
  for (const uint64_t rows : kRows) {
    sllm_buffer_t *activation_buffer = nullptr;
    sllm_buffer_t *output_buffer = nullptr;
    sllm_matmul_plan_t *plan = nullptr;
    const std::vector<uint16_t> activation = make_activation(shape, rows);
    valid = valid &&
            create_buffer(context, rows * shape.k * sizeof(uint16_t),
                          &activation_buffer) &&
            create_buffer(context, rows * shape.n * sizeof(uint16_t),
                          &output_buffer) &&
            upload(queue, activation_buffer, activation.data(),
                   static_cast<uint64_t>(activation.size()) * sizeof(uint16_t));
    if (valid) {
      const sllm_matmul_desc_t desc = descriptor(shape, rows, activation_buffer,
                                                 weight_buffer, output_buffer);
      Error error;
      valid = expect(sllm_matmul_prepare(context, &desc, &plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_matmul_prepare", error);
    }
    Run run{};
    if (valid) {
      run = run_plan(shape, rows, plan, queue, output_buffer, activation);
      valid = run.valid && valid;
      std::cout << std::fixed << std::setprecision(6)
                << "phase83_5_nvfp4_small_m_vgpr_reuse family=" << shape.family
                << " role=" << shape.role << " M=" << rows << " K=" << shape.k
                << " N=" << shape.n << " provider_id=" << run.dispatch.kernel_id
                << " dispatch_count=" << run.dispatch.dispatch_count
                << " kernel=" << run.dispatch.kernel_symbol
                << " device=" << run.dispatch.device_symbol
                << " median_ms=" << run.median_ms
                << " oracle=all_rows_all_columns max_bf16_ulp="
                << run.max_bf16_ulp
                << " status=" << (run.valid ? "PASS" : "FAIL") << '\n';
    }
    if (plan != nullptr) {
      Error error;
      valid = expect(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_matmul_plan_release", error) &&
              valid;
    }
    valid = release_buffer(&output_buffer) && valid;
    valid = release_buffer(&activation_buffer) && valid;
    if (!valid) {
      break;
    }
  }
  valid = release_buffer(&weight_buffer) && valid;
  return valid;
}

bool create_context_queue(sllm_context_t **const context,
                          sllm_queue_t **const queue) {
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  Error error;
  if (!expect(sllm_context_create(&context_info, context, &error.sink),
              SLLM_STATUS_OK, "sllm_context_create", error)) {
    return false;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  return expect(sllm_queue_create(*context, &queue_info, queue, &error.sink),
                SLLM_STATUS_OK, "sllm_queue_create", error);
}

} // namespace

int main() {
  try {
    if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") != 0 &&
        std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") != 0) {
      std::cerr << "phase83_5_nvfp4_small_m_vgpr_reuse requires exact gfx1030 "
                   "or gfx1201, got "
                << SLLM_TEST_EXPECTED_TARGET << '\n';
      return 1;
    }
    unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
    unsetenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID");
    unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED");
    unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_WMMA_COMPENSATED");
    unsetenv("SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201");
    unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_ROW8");
    unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8");
    unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A");
    unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA");
    unsetenv("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING");
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    bool valid = create_context_queue(&context, &queue);
    const std::array<Shape, 2> shapes = {{
        {"NVFP4", "mlp_gate_up", 5120U, 17408U,
         SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION, SLLM_TENSOR_DTYPE_U8,
         SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32},
        {"NVFP4", "mlp_down", 17408U, 5120U, SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION,
         SLLM_TENSOR_DTYPE_U8,
         SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32},
    }};
    for (const Shape &shape : shapes) {
      if (!valid) {
        break;
      }
      valid = run_shape(shape, context, queue) && valid;
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
    std::cout << "phase83_5_nvfp4_small_m_vgpr_reuse status="
              << (valid ? "PASS" : "FAIL")
              << " gpu_run_required=1 independent_all_output_oracle=1 "
                 "resources_released="
              << ((queue == nullptr && context == nullptr) ? 1 : 0) << '\n';
    return valid ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "phase83_5_nvfp4_small_m_vgpr_reuse exception: "
              << error.what() << '\n';
    return 2;
  }
}
