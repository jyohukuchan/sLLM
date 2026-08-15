#include "sllm/hip.h"

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

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
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

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t rank, const uint64_t rows,
                              const uint64_t columns) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = SLLM_TENSOR_DTYPE_BF16;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  if (rank == 1U) {
    result.shape[0] = columns;
    result.stride_elements[0] = 1U;
  } else {
    result.shape[0] = rows;
    result.shape[1] = columns;
    result.stride_elements[0] = columns;
    result.stride_elements[1] = 1U;
  }
  return result;
}

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, operation, error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    return false;
  }
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "sllm_completion_release", error);
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const data, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(data);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error) &&
         wait_and_release(&completion, "sllm_completion_wait(h2d)");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer,
              std::vector<uint16_t> *const output) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = output->size() * sizeof(uint16_t);
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "sllm_completion_wait(d2h)", error)) {
    return false;
  }
  uint64_t bytes_written = 0U;
  const bool read = expect(sllm_completion_read(completion, output->data(),
                                                transfer.size_bytes,
                                                &bytes_written, &error.sink),
                           SLLM_STATUS_OK, "sllm_completion_read", error);
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(d2h)", error);
  return read && bytes_written == transfer.size_bytes && released;
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

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t rows,
              const uint64_t columns, float *const cumulative_max_abs,
              float *const cumulative_max_rel) {
  const uint64_t element_count = rows * columns;
  std::vector<uint16_t> activation(static_cast<std::size_t>(element_count));
  std::vector<uint16_t> scale(static_cast<std::size_t>(columns));
  constexpr std::array<float, 5> scale_pattern = {0.0F, 0.5F, 1.0F, -0.5F,
                                                  2.0F};
  for (uint64_t index = 0U; index != element_count; ++index) {
    const int64_t centered =
        static_cast<int64_t>((index * 13U + 7U) % 29U) - 14;
    activation[static_cast<std::size_t>(index)] =
        f32_to_bf16_rne(static_cast<float>(centered) / 8.0F);
  }
  for (uint64_t column = 0U; column != columns; ++column) {
    scale[static_cast<std::size_t>(column)] = f32_to_bf16_rne(
        scale_pattern[static_cast<std::size_t>(column % scale_pattern.size())]);
  }

  std::vector<uint16_t> expected(static_cast<std::size_t>(element_count));
  for (uint64_t row = 0U; row != rows; ++row) {
    float sum = 0.0F;
    for (uint64_t column = 0U; column != columns; ++column) {
      const float value = bf16_to_f32(
          activation[static_cast<std::size_t>(row * columns + column)]);
      sum += value * value;
    }
    const float inverse_rms =
        1.0F / std::sqrt(sum / static_cast<float>(columns) + 1.0e-6F);
    for (uint64_t column = 0U; column != columns; ++column) {
      expected[static_cast<std::size_t>(row * columns + column)] =
          f32_to_bf16_rne(bf16_to_f32(activation[static_cast<std::size_t>(
                              row * columns + column)]) *
                          inverse_rms *
                          bf16_to_f32(scale[static_cast<std::size_t>(column)]));
    }
  }

  const uint64_t activation_bytes = element_count * sizeof(uint16_t);
  const uint64_t scale_bytes = columns * sizeof(uint16_t);
  sllm_buffer_t *activation_buffer = nullptr;
  sllm_buffer_t *scale_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  Error error;
  if (!create_buffer(context, activation_bytes, &activation_buffer) ||
      !create_buffer(context, scale_bytes, &scale_buffer) ||
      !create_buffer(context, activation_bytes, &output_buffer) ||
      !upload(queue, activation_buffer, activation.data(), activation_bytes) ||
      !upload(queue, scale_buffer, scale.data(), scale_bytes)) {
    return false;
  }

  sllm_rmsnorm_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_RMSNORM_VERSION;
  descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
  descriptor.scale_mode = SLLM_RMSNORM_SCALE_MODE_DIRECT;
  descriptor.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
  const float epsilon = 1.0e-6F;
  std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
  descriptor.activation = binding(activation_buffer, 2U, rows, columns);
  descriptor.raw_scale = binding(scale_buffer, 1U, 1U, columns);
  descriptor.output = binding(output_buffer, 2U, rows, columns);
  sllm_rmsnorm_plan_t *plan = nullptr;
  bool success =
      expect(sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
             SLLM_STATUS_OK, "sllm_rmsnorm_prepare(direct)", error);
  sllm_rmsnorm_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  success =
      success &&
      expect(sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
             SLLM_STATUS_OK, "sllm_rmsnorm_execute(direct)", error) &&
      wait_and_release(&completion, "sllm_completion_wait(rmsnorm)");
  const uint32_t expected_kernel =
      SLLM_HIP_RMSNORM_KERNEL_ID_BASELINE_WAVE32_V1;
  success =
      success && info.backend == SLLM_BACKEND_HIP &&
      info.dispatch_count == 1U && info.kernel_id == expected_kernel &&
      info.workgroup_size_x == SLLM_HIP_RMSNORM_WORKGROUP_SIZE &&
      info.grid_size_x == rows && info.row_count == rows &&
      info.normalized_size == columns && info.fallback_allowed == 0U &&
      info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol, "rmsnorm.baseline.wave32.v1") == 0 &&
      std::strcmp(info.device_symbol, "sllm_rmsnorm_baseline_wave32_v1") == 0 &&
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;

  std::vector<uint16_t> observed(static_cast<std::size_t>(element_count));
  success = success && download(queue, output_buffer, &observed);
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  if (success) {
    for (uint64_t index = 0U; index != element_count; ++index) {
      const float actual =
          bf16_to_f32(observed[static_cast<std::size_t>(index)]);
      const float oracle =
          bf16_to_f32(expected[static_cast<std::size_t>(index)]);
      const float absolute = std::abs(actual - oracle);
      const float relative =
          absolute /
          std::max(std::abs(oracle), std::numeric_limits<float>::min());
      max_abs = std::max(max_abs, absolute);
      max_rel = std::max(max_rel, relative);
      if (!std::isfinite(actual) || (absolute > 0.03125F && relative > 0.02F)) {
        std::cerr << "direct RMSNorm oracle mismatch rows=" << rows
                  << " columns=" << columns << " index=" << index
                  << " actual=" << actual << " expected=" << oracle << '\n';
        success = false;
        break;
      }
    }
  }
  if (success && (observed.front() & UINT16_C(0x7fff)) != 0U) {
    std::cerr << "zero direct scale did not produce an exact signed zero\n";
    success = false;
  }
  *cumulative_max_abs = std::max(*cumulative_max_abs, max_abs);
  *cumulative_max_rel = std::max(*cumulative_max_rel, max_rel);

  if (plan != nullptr) {
    success = expect(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_rmsnorm_plan_release", error) &&
              success;
  }
  success = expect(sllm_buffer_release(&output_buffer, &error.sink),
                   SLLM_STATUS_OK, "sllm_buffer_release(output)", error) &&
            success;
  success = expect(sllm_buffer_release(&scale_buffer, &error.sink),
                   SLLM_STATUS_OK, "sllm_buffer_release(scale)", error) &&
            success;
  success = expect(sllm_buffer_release(&activation_buffer, &error.sink),
                   SLLM_STATUS_OK, "sllm_buffer_release(activation)", error) &&
            success;
  return success;
}

} // namespace

int main() {
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
  Error error;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "sllm_context_create", error)) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  if (!expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
              SLLM_STATUS_OK, "sllm_queue_create", error)) {
    return 1;
  }

  constexpr std::array<uint64_t, 11> widths = {
      1U, 3U, 17U, 255U, 256U, 257U, 3839U, 3840U, 3841U, 4095U, 4096U};
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  bool success = true;
  for (std::size_t index = 0U; index != widths.size(); ++index) {
    const uint64_t rows = index % 2U == 0U ? 1U : 3U;
    if (!run_case(context, queue, rows, widths[index], &max_abs, &max_rel)) {
      success = false;
      break;
    }
  }
  success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "sllm_queue_release", error) &&
            success;
  success = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "sllm_context_release", error) &&
            success;
  if (success) {
    std::cout << "direct RMSNorm GPU test: PASS target="
              << SLLM_TEST_EXPECTED_TARGET << " cases=" << widths.size()
              << " max_abs=" << max_abs << " max_rel=" << max_rel << '\n';
  }
  return success ? 0 : 1;
}
