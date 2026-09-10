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

bool run_residual_case(const sllm_context_t *const context,
                       const sllm_queue_t *const queue, const uint64_t rows,
                       const uint64_t columns, const uint32_t scale_mode) {
  const uint64_t element_count = rows * columns;
  std::vector<uint16_t> residual(static_cast<std::size_t>(element_count));
  std::vector<uint16_t> addend(static_cast<std::size_t>(element_count));
  std::vector<uint16_t> scale(static_cast<std::size_t>(columns));
  constexpr std::array<float, 5> scale_pattern = {0.0F, 0.25F, 1.0F, -0.5F,
                                                  2.0F};
  for (uint64_t index = 0U; index != element_count; ++index) {
    const int64_t residual_centered =
        static_cast<int64_t>((index * 17U + 11U) % 61U) - 30;
    const int64_t addend_centered =
        static_cast<int64_t>((index * 29U + 5U) % 47U) - 23;
    residual[static_cast<std::size_t>(index)] =
        f32_to_bf16_rne(static_cast<float>(residual_centered) / 13.0F);
    addend[static_cast<std::size_t>(index)] =
        f32_to_bf16_rne(static_cast<float>(addend_centered) / 19.0F);
  }
  for (uint64_t column = 0U; column != columns; ++column) {
    scale[static_cast<std::size_t>(column)] = f32_to_bf16_rne(
        scale_pattern[static_cast<std::size_t>(column % scale_pattern.size())]);
  }

  /* The baseline comparator is explicit: F32 add, one BF16-RNE intermediate,
   * then the ordinary RMSNorm stats/scale path over that BF16 intermediate.
   * The fused kernel must match this bit-for-bit; no tolerance is accepted. */
  std::vector<uint16_t> baseline_add(static_cast<std::size_t>(element_count));
  std::vector<uint16_t> expected(static_cast<std::size_t>(element_count));
  for (uint64_t row = 0U; row != rows; ++row) {
    float sum = 0.0F;
    long double precise_sum = 0.0L;
    for (uint64_t column = 0U; column != columns; ++column) {
      const std::size_t index =
          static_cast<std::size_t>(row * columns + column);
      const float added =
          bf16_to_f32(residual[index]) + bf16_to_f32(addend[index]);
      baseline_add[index] = f32_to_bf16_rne(added);
      const float intermediate = bf16_to_f32(baseline_add[index]);
      sum += intermediate * intermediate;
      precise_sum += static_cast<long double>(intermediate) *
                     static_cast<long double>(intermediate);
    }
    const float inverse_rms =
        1.0F / std::sqrt(sum / static_cast<float>(columns) + 1.0e-6F);
    const long double precise_inverse_rms =
        1.0L / std::sqrt(precise_sum / static_cast<long double>(columns) +
                         static_cast<long double>(1.0e-6F));
    for (uint64_t column = 0U; column != columns; ++column) {
      const std::size_t index =
          static_cast<std::size_t>(row * columns + column);
      const float raw_scale =
          bf16_to_f32(scale[static_cast<std::size_t>(column)]);
      const float effective_scale =
          scale_mode == SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE ? 1.0F + raw_scale
                                                           : raw_scale;
      if (columns == 5120U) {
        const long double precise_value =
            static_cast<long double>(bf16_to_f32(baseline_add[index])) *
            precise_inverse_rms * static_cast<long double>(effective_scale);
        expected[index] = f32_to_bf16_rne(static_cast<float>(precise_value));
      } else {
        expected[index] = f32_to_bf16_rne(bf16_to_f32(baseline_add[index]) *
                                          inverse_rms * effective_scale);
      }
    }
  }

  const uint64_t matrix_bytes = element_count * sizeof(uint16_t);
  const uint64_t scale_bytes = columns * sizeof(uint16_t);
  sllm_buffer_t *residual_buffer = nullptr;
  sllm_buffer_t *addend_buffer = nullptr;
  sllm_buffer_t *scale_buffer = nullptr;
  sllm_buffer_t *residual_output_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  Error error;
  bool success =
      create_buffer(context, matrix_bytes, &residual_buffer) &&
      create_buffer(context, matrix_bytes, &addend_buffer) &&
      create_buffer(context, scale_bytes, &scale_buffer) &&
      create_buffer(context, matrix_bytes, &residual_output_buffer) &&
      create_buffer(context, matrix_bytes, &output_buffer) &&
      upload(queue, residual_buffer, residual.data(), matrix_bytes) &&
      upload(queue, addend_buffer, addend.data(), matrix_bytes) &&
      upload(queue, scale_buffer, scale.data(), scale_bytes);
  if (!success) {
    return false;
  }

  sllm_residual_rmsnorm_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_RESIDUAL_RMSNORM_VERSION;
  descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
  descriptor.scale_mode = scale_mode;
  descriptor.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
  constexpr float epsilon = 1.0e-6F;
  std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
  descriptor.residual = binding(residual_buffer, 2U, rows, columns);
  descriptor.addend = binding(addend_buffer, 2U, rows, columns);
  descriptor.raw_scale = binding(scale_buffer, 1U, 1U, columns);
  descriptor.residual_output =
      binding(residual_output_buffer, 2U, rows, columns);
  descriptor.output = binding(output_buffer, 2U, rows, columns);
  sllm_residual_rmsnorm_plan_t *plan = nullptr;
  success = expect(
      sllm_residual_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
      SLLM_STATUS_OK, "sllm_residual_rmsnorm_prepare", error);
  sllm_residual_rmsnorm_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_RESIDUAL_RMSNORM_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  success =
      success &&
      expect(sllm_residual_rmsnorm_execute(plan, queue, &completion, &info,
                                           &error.sink),
             SLLM_STATUS_OK, "sllm_residual_rmsnorm_execute", error) &&
      wait_and_release(&completion, "sllm_completion_wait(residual RMSNorm)");
  success = success && info.backend == SLLM_BACKEND_HIP &&
            info.dispatch_count == 1U &&
            info.kernel_id == SLLM_HIP_RESIDUAL_RMSNORM_KERNEL_ID_WAVE32_V1 &&
            info.workgroup_size_x == SLLM_HIP_RMSNORM_WORKGROUP_SIZE &&
            info.grid_size_x == rows && info.row_count == rows &&
            info.normalized_size == columns && info.fallback_allowed == 0U &&
            info.fallback_used == 0U &&
            std::strcmp(info.kernel_symbol,
                        "rmsnorm.residual_fused.wave32.v1") == 0 &&
            std::strcmp(info.device_symbol,
                        "sllm_rmsnorm_residual_fused_wave32_v1") == 0 &&
            std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;

  std::vector<uint16_t> observed_residual(
      static_cast<std::size_t>(element_count));
  std::vector<uint16_t> observed_output(
      static_cast<std::size_t>(element_count));
  const bool compare_decomposed = columns == 5120U;
  sllm_buffer_t *decomposed_output_buffer = nullptr;
  sllm_rmsnorm_plan_t *decomposed_plan = nullptr;
  std::vector<uint16_t> observed_decomposed;
  success = success &&
            download(queue, residual_output_buffer, &observed_residual) &&
            download(queue, output_buffer, &observed_output);
  if (success) {
    for (uint64_t index = 0U; index != element_count; ++index) {
      const std::size_t position = static_cast<std::size_t>(index);
      if (observed_residual[position] != baseline_add[position] ||
          observed_output[position] != expected[position]) {
        std::cerr << "residual RMSNorm bitwise oracle mismatch rows=" << rows
                  << " columns=" << columns << " index=" << index
                  << " scale_mode=" << scale_mode << " intermediate=0x"
                  << std::hex << observed_residual[position] << "/0x"
                  << baseline_add[position] << " output=0x"
                  << observed_output[position] << "/0x" << expected[position]
                  << std::dec << '\n';
        success = false;
        break;
      }
    }
  }
  if (success && compare_decomposed) {
    observed_decomposed.resize(static_cast<std::size_t>(element_count));
    success = create_buffer(context, matrix_bytes, &decomposed_output_buffer);
    if (success) {
      /* Feed the exact BF16 add oracle to the ordinary RMSNorm path.  This
       * makes the comparison an actual decomposed GPU execution while the
       * fused intermediate remains checked above. */
      success = upload(queue, residual_output_buffer, baseline_add.data(),
                       matrix_bytes);
    }
    sllm_rmsnorm_desc_t decomposed_descriptor{};
    decomposed_descriptor.struct_size = sizeof(decomposed_descriptor);
    decomposed_descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    decomposed_descriptor.op_version = SLLM_HIP_RMSNORM_VERSION;
    decomposed_descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
    decomposed_descriptor.scale_mode = scale_mode;
    decomposed_descriptor.alias_policy =
        SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
    std::memcpy(&decomposed_descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
    decomposed_descriptor.activation =
        binding(residual_output_buffer, 2U, rows, columns);
    decomposed_descriptor.raw_scale = binding(scale_buffer, 1U, 1U, columns);
    decomposed_descriptor.output =
        binding(decomposed_output_buffer, 2U, rows, columns);
    sllm_rmsnorm_dispatch_info_t decomposed_info{};
    decomposed_info.struct_size = sizeof(decomposed_info);
    decomposed_info.abi_version = SLLM_HIP_ABI_VERSION;
    decomposed_info.info_version = SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION;
    sllm_completion_t *decomposed_completion = nullptr;
    success =
        success &&
        expect(sllm_rmsnorm_prepare(context, &decomposed_descriptor,
                                    &decomposed_plan, &error.sink),
               SLLM_STATUS_OK, "sllm_rmsnorm_prepare(decomposed)", error) &&
        expect(sllm_rmsnorm_execute(decomposed_plan, queue,
                                    &decomposed_completion, &decomposed_info,
                                    &error.sink),
               SLLM_STATUS_OK, "sllm_rmsnorm_execute(decomposed)", error) &&
        wait_and_release(&decomposed_completion,
                         "sllm_completion_wait(decomposed RMSNorm)");
    success =
        success && decomposed_info.backend == SLLM_BACKEND_HIP &&
        decomposed_info.dispatch_count == 1U &&
        decomposed_info.kernel_id ==
            SLLM_HIP_RMSNORM_KERNEL_ID_BASELINE_WAVE32_V1 &&
        decomposed_info.workgroup_size_x == SLLM_HIP_RMSNORM_WORKGROUP_SIZE &&
        decomposed_info.grid_size_x == rows &&
        decomposed_info.row_count == rows &&
        decomposed_info.normalized_size == columns &&
        decomposed_info.fallback_allowed == 0U &&
        decomposed_info.fallback_used == 0U &&
        std::strcmp(decomposed_info.kernel_symbol,
                    "rmsnorm.baseline.wave32.v1") == 0 &&
        std::strcmp(decomposed_info.device_symbol,
                    "sllm_rmsnorm_baseline_wave32_v1") == 0 &&
        std::strcmp(decomposed_info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) ==
            0;
    success = success &&
              download(queue, decomposed_output_buffer, &observed_decomposed);
    if (success) {
      for (uint64_t index = 0U; index != element_count; ++index) {
        const std::size_t position = static_cast<std::size_t>(index);
        if (observed_decomposed[position] != expected[position] ||
            observed_decomposed[position] != observed_output[position]) {
          std::cerr << "decomposed RMSNorm bitwise oracle mismatch rows="
                    << rows << " columns=" << columns << " index=" << index
                    << " scale_mode=" << scale_mode << " fused=0x" << std::hex
                    << observed_output[position] << " decomposed=0x"
                    << observed_decomposed[position] << " expected=0x"
                    << expected[position] << std::dec << '\n';
          success = false;
          break;
        }
      }
    }
  }
  if (decomposed_plan != nullptr) {
    success = expect(sllm_rmsnorm_plan_release(&decomposed_plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_rmsnorm_plan_release(decomposed)",
                     error) &&
              success;
  }
  if (plan != nullptr) {
    success =
        expect(sllm_residual_rmsnorm_plan_release(&plan, &error.sink),
               SLLM_STATUS_OK, "sllm_residual_rmsnorm_plan_release", error) &&
        success;
  }
  if (decomposed_output_buffer != nullptr) {
    success =
        expect(sllm_buffer_release(&decomposed_output_buffer, &error.sink),
               SLLM_STATUS_OK, "sllm_buffer_release(decomposed output)",
               error) &&
        success;
  }
  success =
      expect(sllm_buffer_release(&output_buffer, &error.sink), SLLM_STATUS_OK,
             "sllm_buffer_release(residual output)", error) &&
      success;
  success =
      expect(sllm_buffer_release(&residual_output_buffer, &error.sink),
             SLLM_STATUS_OK, "sllm_buffer_release(intermediate)", error) &&
      success;
  success =
      expect(sllm_buffer_release(&scale_buffer, &error.sink), SLLM_STATUS_OK,
             "sllm_buffer_release(residual scale)", error) &&
      success;
  success = expect(sllm_buffer_release(&addend_buffer, &error.sink),
                   SLLM_STATUS_OK, "sllm_buffer_release(addend)", error) &&
            success;
  success = expect(sllm_buffer_release(&residual_buffer, &error.sink),
                   SLLM_STATUS_OK, "sllm_buffer_release(residual)", error) &&
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
  struct ResidualCase final {
    uint64_t rows;
    uint64_t columns;
    uint32_t scale_mode;
  };
  constexpr std::array<ResidualCase, 10> residual_cases = {
      {{1U, 2560U, SLLM_RMSNORM_SCALE_MODE_DIRECT},
       {2U, 255U, SLLM_RMSNORM_SCALE_MODE_DIRECT},
       {3U, 256U, SLLM_RMSNORM_SCALE_MODE_DIRECT},
       {3U, 257U, SLLM_RMSNORM_SCALE_MODE_DIRECT},
       {1U, 5120U, SLLM_RMSNORM_SCALE_MODE_DIRECT},
       {2U, 5120U, SLLM_RMSNORM_SCALE_MODE_DIRECT},
       {3U, 5120U, SLLM_RMSNORM_SCALE_MODE_DIRECT},
       {1U, 5120U, SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE},
       {2U, 5120U, SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE},
       {3U, 5120U, SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE}}};
  for (const auto &residual_case : residual_cases) {
    if (!run_residual_case(context, queue, residual_case.rows,
                           residual_case.columns, residual_case.scale_mode)) {
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
              << SLLM_TEST_EXPECTED_TARGET
              << " baseline_cases=" << widths.size()
              << " residual_cases=" << residual_cases.size()
              << " max_abs=" << max_abs << " max_rel=" << max_rel << '\n';
  }
  return success ? 0 : 1;
}
