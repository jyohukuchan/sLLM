#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

namespace {

constexpr uint32_t kTimeoutMs = 30'000U;
constexpr uint16_t kGuardWord = UINT16_C(0x7e35);

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
                 SLLM_STATUS_OK, "sllm_buffer_copy_d2h read", error) &&
          bytes_written == bytes && valid;
  valid = expect(sllm_completion_release(&completion, &error.sink),
                 SLLM_STATUS_OK, "sllm_completion_release", error) &&
          completion == nullptr && valid;
  return valid;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint64_t rows, const uint64_t columns) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = SLLM_TENSOR_DTYPE_BF16;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = 2U;
  result.shape[0] = rows;
  result.shape[1] = columns;
  result.stride_elements[0] = columns;
  result.stride_elements[1] = 1U;
  return result;
}

bool create_context_queue(sllm_context_t **const context,
                          sllm_queue_t **const queue) {
  Error error;
  uint32_t device_count = 0U;
  if (!expect(sllm_device_count(&device_count, &error.sink), SLLM_STATUS_OK,
              "sllm_device_count", error) ||
      device_count != 1U) {
    std::cerr << "expected one visible GPU, got " << device_count << '\n';
    return false;
  }
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "sllm_device_query", error) ||
      std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "visible target is not " << SLLM_TEST_EXPECTED_TARGET << '\n';
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

std::vector<uint16_t> make_activation(const uint64_t rows, const uint64_t k) {
  std::vector<uint16_t> result(static_cast<std::size_t>(rows * k));
  for (uint64_t row = 0U; row < rows; ++row) {
    // The row-dependent value makes missing row tiles observable while staying
    // finite for the largest admitted M=1024 case.
    const float value = 1.0F + static_cast<float>(row) * 0.125F;
    const uint16_t encoded = f32_to_bf16_rne(value);
    std::fill(result.begin() + static_cast<std::ptrdiff_t>(row * k),
              result.begin() + static_cast<std::ptrdiff_t>((row + 1U) * k),
              encoded);
  }
  return result;
}

std::vector<uint16_t> make_weight(const uint64_t k, const uint64_t n) {
  return std::vector<uint16_t>(static_cast<std::size_t>(k * n),
                               f32_to_bf16_rne(1.0F));
}

bool check_output(const Shape &shape, const uint64_t rows,
                  const std::vector<uint16_t> &activation,
                  const std::vector<uint16_t> &observed,
                  const uint64_t guard_words, const uint32_t kernel_id) {
  const bool id91 = rows >= 64U;
  const uint32_t expected_id = UINT32_C(91);
  if ((kernel_id == expected_id) != id91) {
    std::cerr << shape.name << " M=" << rows
              << " unexpected provider_id=" << kernel_id << '\n';
    return false;
  }
  const std::array<uint64_t, 6> candidates = {
      0U, rows > 63U ? 63U : rows - 1U, 64U, 65U, rows - 2U, rows - 1U};
  bool valid = true;
  for (const uint64_t row : candidates) {
    if (row >= rows) {
      continue;
    }
    const float input =
        bf16_to_f32(activation[static_cast<std::size_t>(row * shape.k)]);
    const uint16_t expected =
        f32_to_bf16_rne(input * static_cast<float>(shape.k));
    for (const uint64_t column : {UINT64_C(0), shape.n / 2U, shape.n - 1U}) {
      const uint16_t actual =
          observed[static_cast<std::size_t>(row * shape.n + column)];
      if (!finite_bf16(actual) || actual != expected) {
        std::cerr << shape.name << " M=" << rows << " row=" << row
                  << " col=" << column << " actual=0x" << std::hex << actual
                  << " expected=0x" << expected << std::dec << '\n';
        valid = false;
      }
    }
  }
  const std::size_t output_words = static_cast<std::size_t>(rows * shape.n);
  for (uint64_t index = 0U; index < guard_words; ++index) {
    if (observed[output_words + static_cast<std::size_t>(index)] !=
        kGuardWord) {
      std::cerr << shape.name << " M=" << rows
                << " output guard overwritten at word " << index << '\n';
      valid = false;
      break;
    }
  }
  return valid;
}

bool run_case(const Shape &shape, const uint64_t rows,
              const sllm_context_t *const context,
              const sllm_queue_t *const queue) {
  constexpr uint64_t guard_words = 2048U;
  const uint64_t activation_bytes = rows * shape.k * sizeof(uint16_t);
  const uint64_t weight_bytes = shape.k * shape.n * sizeof(uint16_t);
  const uint64_t output_words = rows * shape.n;
  const uint64_t output_bytes = output_words * sizeof(uint16_t);
  const uint64_t output_storage_bytes =
      output_bytes + guard_words * sizeof(uint16_t);
  const std::vector<uint16_t> activation = make_activation(rows, shape.k);
  const std::vector<uint16_t> weight = make_weight(shape.k, shape.n);
  std::vector<uint16_t> initial(
      static_cast<std::size_t>(output_words + guard_words), kGuardWord);
  sllm_buffer_t *activation_buffer = nullptr;
  sllm_buffer_t *weight_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  sllm_matmul_plan_t *plan = nullptr;
  bool valid =
      create_buffer(context, activation_bytes, &activation_buffer) &&
      create_buffer(context, weight_bytes, &weight_buffer) &&
      create_buffer(context, output_storage_bytes, &output_buffer) &&
      copy_h2d(queue, activation_buffer, activation.data(), activation_bytes) &&
      copy_h2d(queue, weight_buffer, weight.data(), weight_bytes) &&
      copy_h2d(queue, output_buffer, initial.data(), output_storage_bytes);
  if (valid) {
    sllm_matmul_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_MATMUL_VERSION;
    descriptor.activation = binding(activation_buffer, rows, shape.k);
    descriptor.weight = binding(weight_buffer, shape.n, shape.k);
    descriptor.output = binding(output_buffer, rows, shape.n);
    Error error;
    valid =
        expect(sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
               SLLM_STATUS_OK, "sllm_matmul_prepare", error);
  }
  std::vector<uint16_t> observed(initial.size(), 0U);
  uint32_t kernel_id = 0U;
  if (valid) {
    sllm_matmul_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    Error error;
    valid = expect(sllm_matmul_execute(plan, queue, &completion, &dispatch,
                                       &error.sink),
                   SLLM_STATUS_OK, "sllm_matmul_execute", error) &&
            completion != nullptr &&
            wait_release(&completion, "sllm_matmul_execute wait");
    kernel_id = dispatch.kernel_id;
    valid = valid && dispatch.dispatch_id != 0U &&
            dispatch.dispatch_count == 1U && dispatch.m == rows &&
            dispatch.k == shape.k && dispatch.n == shape.n &&
            dispatch.output_elements == output_words &&
            dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
            std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
    if (valid) {
      valid =
          copy_d2h(queue, output_buffer, observed.data(), output_storage_bytes);
    }
  }
  if (valid) {
    valid =
        check_output(shape, rows, activation, observed, guard_words, kernel_id);
  }
  std::cout << "phase83_bf16_prefill shape=" << shape.name << " M=" << rows
            << " K=" << shape.k << " N=" << shape.n
            << " provider_id=" << kernel_id
            << " status=" << (valid ? "PASS" : "FAIL") << '\n';
  if (plan != nullptr) {
    Error error;
    valid = expect(sllm_matmul_plan_release(&plan, &error.sink), SLLM_STATUS_OK,
                   "sllm_matmul_plan_release", error) &&
            valid;
  }
  valid = release_buffer(&output_buffer) && valid;
  valid = release_buffer(&weight_buffer) && valid;
  valid = release_buffer(&activation_buffer) && valid;
  return valid;
}

} // namespace

int main() {
  // Keep this public regression deterministic even when a developer shell has
  // candidate selectors enabled for other Phase83 experiments.
  (void)unsetenv("SLLM_MATMUL_FORCE_BASELINE");
  (void)unsetenv("SLLM_MATMUL_GFX1030_SHORT_SERIAL");
  (void)unsetenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  bool valid = create_context_queue(&context, &queue);
  constexpr std::array<Shape, 3> shapes = {{
      {"fc", 10240U, 5120U},
      {"up", 5120U, 17408U},
      {"down", 17408U, 5120U},
  }};
  constexpr std::array<uint64_t, 4> rows = {63U, 64U, 65U, 1024U};
  for (const Shape &shape : shapes) {
    for (const uint64_t row_count : rows) {
      if (!valid) {
        break;
      }
      valid = run_case(shape, row_count, context, queue) && valid;
    }
  }
  Error error;
  if (queue != nullptr) {
    valid = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "sllm_queue_release", error) &&
            valid;
  }
  if (context != nullptr) {
    valid = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "sllm_context_release", error) &&
            valid;
  }
  std::cout << "phase83_bf16_prefill status=" << (valid ? "PASS" : "FAIL")
            << " public_launcher=1 oracle=1 guard_check=1\n";
  return valid ? 0 : 1;
}
