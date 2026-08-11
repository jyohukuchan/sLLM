#include "sllm/hip.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
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

float bf16_to_f32(const uint16_t bits) {
  const uint32_t raw = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

uint16_t mode_value(const uint32_t mode, const uint64_t row,
                    const uint64_t column, const uint64_t v) {
  switch (mode) {
  case 0U:
    return column == 0U ? UINT16_C(0x40a0) : UINT16_C(0xc040);
  case 1U:
    return column + 1U == v ? UINT16_C(0x40e0) : UINT16_C(0xc040);
  case 2U:
    return column == 0U || column + 1U == v ? UINT16_C(0x4080)
                                            : UINT16_C(0xc040);
  case 3U:
    return column == (row & 1U) ? UINT16_C(0x8000) : UINT16_C(0x0000);
  case 4U:
    return column + 1U == v ? UINT16_C(0x7f80) : UINT16_C(0x4120);
  case 5U:
    return UINT16_C(0xff80);
  default:
    return column == (v / 2U) ? UINT16_C(0x7fc1) : UINT16_C(0x3f80);
  }
}

int32_t scalar_oracle(const std::vector<uint16_t> &logits, const uint64_t row,
                      const uint64_t v) {
  float maximum = 0.0F;
  uint64_t index = 0U;
  bool valid = false;
  bool has_nan = false;
  for (uint64_t column = 0U; column != v; ++column) {
    const float value = bf16_to_f32(logits[row * v + column]);
    if (std::isnan(value)) {
      has_nan = true;
    } else if (!valid || value > maximum ||
               (value == maximum && column < index)) {
      maximum = value;
      index = column;
      valid = true;
    }
  }
  return has_nan ? -1 : static_cast<int32_t>(index);
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint32_t rank,
                              const uint64_t first,
                              const uint64_t second = 0U) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  result.shape[0] = first;
  result.stride_elements[0] = rank == 2U ? second : 1U;
  if (rank == 2U) {
    result.shape[1] = second;
    result.stride_elements[1] = 1U;
  }
  return result;
}

bool wait_and_release(sllm_completion_t **const completion) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "sllm_completion_wait", error)) {
    return false;
  }
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "sllm_completion_release", error);
}

bool wait_timing_and_release(sllm_completion_t **const completion) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "sllm_completion_wait(argmax timing)", error)) {
    return false;
  }
  sllm_completion_timing_t timing{};
  timing.struct_size = sizeof(timing);
  timing.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_completion_timing(*completion, &timing, &error.sink),
              SLLM_STATUS_OK, "sllm_completion_timing(argmax)", error) ||
      timing.valid != 1U || timing.elapsed_ns == 0U) {
    std::cerr << "argmax completion timing was not positive\n";
    return false;
  }
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "sllm_completion_release(argmax timing)",
                error);
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
  if (!expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error)) {
    return false;
  }
  return wait_and_release(&completion);
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer,
              std::vector<int32_t> *const output) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = output->size() * sizeof(int32_t);
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
  const bool read_ok = expect(sllm_completion_read(completion, output->data(),
                                                   transfer.size_bytes,
                                                   &bytes_written, &error.sink),
                              SLLM_STATUS_OK, "sllm_completion_read", error);
  const bool size_ok = bytes_written == transfer.size_bytes;
  const bool release_ok =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(d2h)", error);
  return read_ok && size_ok && release_ok;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t m,
              const uint64_t v, const uint32_t mode) {
  std::vector<uint16_t> host_logits(static_cast<std::size_t>(m * v));
  for (uint64_t row = 0U; row != m; ++row) {
    for (uint64_t column = 0U; column != v; ++column) {
      host_logits[static_cast<std::size_t>(row * v + column)] =
          mode_value(mode, row, column, v);
    }
  }
  std::vector<int32_t> expected(static_cast<std::size_t>(m));
  for (uint64_t row = 0U; row != m; ++row) {
    expected[static_cast<std::size_t>(row)] =
        scalar_oracle(host_logits, row, v);
  }

  sllm_buffer_create_info_t logits_info{};
  logits_info.struct_size = sizeof(logits_info);
  logits_info.abi_version = SLLM_HIP_ABI_VERSION;
  logits_info.size_bytes = host_logits.size() * sizeof(uint16_t);
  sllm_buffer_create_info_t output_info{};
  output_info.struct_size = sizeof(output_info);
  output_info.abi_version = SLLM_HIP_ABI_VERSION;
  output_info.size_bytes = expected.size() * sizeof(int32_t);
  sllm_buffer_t *logits_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  Error error;
  if (!expect(sllm_buffer_create(context, &logits_info, &logits_buffer,
                                 &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_create(logits)", error) ||
      !expect(sllm_buffer_create(context, &output_info, &output_buffer,
                                 &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_create(output)", error)) {
    return false;
  }
  const bool uploaded =
      upload(queue, logits_buffer, host_logits.data(), logits_info.size_bytes);
  sllm_argmax_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_ARGMAX_VERSION;
  descriptor.logits = binding(logits_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, m, v);
  descriptor.output = binding(output_buffer, SLLM_TENSOR_DTYPE_I32, 1U, m);
  sllm_argmax_plan_t *plan = nullptr;
  const bool prepared =
      uploaded &&
      expect(sllm_argmax_prepare(context, &descriptor, &plan, &error.sink),
             SLLM_STATUS_OK, "sllm_argmax_prepare", error);
  sllm_argmax_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ARGMAX_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  const bool executed =
      prepared &&
      expect(sllm_argmax_execute(plan, queue, &completion, &info, &error.sink),
             SLLM_STATUS_OK, "sllm_argmax_execute", error);
  bool correct = executed && wait_timing_and_release(&completion);
  std::vector<int32_t> actual(static_cast<std::size_t>(m));
  if (correct) {
    correct = download(queue, output_buffer, &actual);
  }
  if (correct &&
      (info.backend != SLLM_BACKEND_HIP || info.dispatch_count != 1U ||
       info.kernel_id != SLLM_HIP_ARGMAX_KERNEL_ID_BASELINE_BF16_V1 ||
       info.workgroup_size_x != SLLM_HIP_ARGMAX_WORKGROUP_SIZE ||
       info.grid_size_x != m || info.row_count != m || info.vocab_size != v ||
       info.fallback_allowed != 0U || info.fallback_used != 0U ||
       std::strcmp(info.kernel_symbol, "argmax.bf16_f32.v1") != 0 ||
       std::strcmp(info.device_symbol, "sllm_argmax_bf16_f32_v1") != 0 ||
       std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0)) {
    std::cerr << "argmax dispatch evidence mismatch for M=" << m << " V=" << v
              << " mode=" << mode << '\n';
    correct = false;
  }
  if (correct && actual != expected) {
    std::cerr << "argmax oracle mismatch for M=" << m << " V=" << v
              << " mode=" << mode << '\n';
    correct = false;
  }
  if (plan != nullptr) {
    expect(sllm_argmax_plan_release(&plan, &error.sink), SLLM_STATUS_OK,
           "sllm_argmax_plan_release", error);
  }
  expect(sllm_buffer_release(&output_buffer, &error.sink), SLLM_STATUS_OK,
         "sllm_buffer_release(output)", error);
  expect(sllm_buffer_release(&logits_buffer, &error.sink), SLLM_STATUS_OK,
         "sllm_buffer_release(logits)", error);
  return correct;
}

bool prepare_fails(const sllm_context_t *const context,
                   const sllm_argmax_desc_t &descriptor,
                   const sllm_status_t expected, const char *const label) {
  sllm_argmax_plan_t *plan = nullptr;
  Error error;
  const bool failed =
      expect(sllm_argmax_prepare(context, &descriptor, &plan, &error.sink),
             expected, label, error) &&
      plan == nullptr;
  if (plan != nullptr) {
    (void)sllm_argmax_plan_release(&plan, &error.sink);
  }
  return failed;
}

bool run_negative_contract(const sllm_context_t *const context) {
  sllm_buffer_create_info_t logits_info{};
  logits_info.struct_size = sizeof(logits_info);
  logits_info.abi_version = SLLM_HIP_ABI_VERSION;
  logits_info.size_bytes = 3U * 17U * sizeof(uint16_t);
  sllm_buffer_create_info_t output_info{};
  output_info.struct_size = sizeof(output_info);
  output_info.abi_version = SLLM_HIP_ABI_VERSION;
  output_info.size_bytes = 3U * sizeof(int32_t);
  sllm_buffer_t *logits = nullptr;
  sllm_buffer_t *output = nullptr;
  Error error;
  if (!expect(sllm_buffer_create(context, &logits_info, &logits, &error.sink),
              SLLM_STATUS_OK, "negative logits buffer", error) ||
      !expect(sllm_buffer_create(context, &output_info, &output, &error.sink),
              SLLM_STATUS_OK, "negative output buffer", error)) {
    return false;
  }

  sllm_argmax_desc_t valid{};
  valid.struct_size = sizeof(valid);
  valid.abi_version = SLLM_HIP_ABI_VERSION;
  valid.op_version = SLLM_HIP_ARGMAX_VERSION;
  valid.logits = binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 3U, 17U);
  valid.output = binding(output, SLLM_TENSOR_DTYPE_I32, 1U, 3U);

  bool success = true;
  auto mutation = valid;
  mutation.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  success = success &&
            prepare_fails(context, mutation, SLLM_STATUS_INVALID_ABI_VERSION,
                          "argmax wrong ABI version");
  mutation = valid;
  mutation.op_version = SLLM_HIP_ARGMAX_VERSION + 1U;
  success = success && prepare_fails(context, mutation,
                                     SLLM_STATUS_INVALID_ARGMAX_DESCRIPTOR,
                                     "argmax wrong operation version");
  mutation = valid;
  mutation.reserved[0] = 1U;
  success =
      success && prepare_fails(context, mutation, SLLM_STATUS_RESERVED_NONZERO,
                               "argmax nonzero reserved field");
  mutation = valid;
  mutation.logits.dtype = SLLM_TENSOR_DTYPE_F16;
  success =
      success && prepare_fails(context, mutation, SLLM_STATUS_UNSUPPORTED_DTYPE,
                               "argmax wrong logits dtype");
  mutation = valid;
  mutation.output.rank = 2U;
  mutation.output.shape[1] = 1U;
  mutation.output.stride_elements[0] = 1U;
  mutation.output.stride_elements[1] = 1U;
  success = success &&
            prepare_fails(context, mutation, SLLM_STATUS_INVALID_TENSOR_BINDING,
                          "argmax wrong output rank");
  mutation = valid;
  mutation.logits.stride_elements[0] = 18U;
  success =
      success && prepare_fails(context, mutation, SLLM_STATUS_STRIDE_MISMATCH,
                               "argmax noncontiguous logits");
  mutation = valid;
  mutation.logits.shape[0] = 0U;
  success = success && prepare_fails(context, mutation, SLLM_STATUS_ZERO_EXTENT,
                                     "argmax zero row count");
  mutation = valid;
  mutation.output.shape[0] = 2U;
  success =
      success && prepare_fails(context, mutation, SLLM_STATUS_SHAPE_MISMATCH,
                               "argmax output shape mismatch");
  mutation = valid;
  mutation.logits.shape[1] = SLLM_HIP_ARGMAX_MAX_V + 1U;
  mutation.logits.stride_elements[0] = SLLM_HIP_ARGMAX_MAX_V + 1U;
  success = success && prepare_fails(context, mutation, SLLM_STATUS_UNSUPPORTED,
                                     "argmax vocabulary limit");
  mutation = valid;
  mutation.output = binding(logits, SLLM_TENSOR_DTYPE_I32, 1U, 3U);
  success =
      success && prepare_fails(context, mutation, SLLM_STATUS_ALIAS_OVERLAP,
                               "argmax overlapping bindings");

  const bool output_released =
      expect(sllm_buffer_release(&output, &error.sink), SLLM_STATUS_OK,
             "negative output buffer release", error);
  const bool logits_released =
      expect(sllm_buffer_release(&logits, &error.sink), SLLM_STATUS_OK,
             "negative logits buffer release", error);
  return success && output_released && logits_released;
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

  const uint64_t cases[][2] = {{1U, 1U},      {3U, 3U},   {17U, 17U},
                               {1U, 255U},    {3U, 256U}, {17U, 257U},
                               {1U, 248320U}, {3U, 17U},  {17U, 255U}};
  bool success = true;
  success = run_negative_contract(context);
  for (uint32_t case_index = 0U; case_index != sizeof(cases) / sizeof(cases[0]);
       ++case_index) {
    if (!success) {
      break;
    }
    if (!run_case(context, queue, cases[case_index][0], cases[case_index][1],
                  case_index % 7U)) {
      success = false;
      break;
    }
  }
  expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
         "sllm_queue_release", error);
  expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
         "sllm_context_release", error);
  return success ? 0 : 1;
}
