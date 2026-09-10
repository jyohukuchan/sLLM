#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

extern "C" sllm_status_t sllm_concat_bf16_rows_v1(
    const sllm_context_t *context, const sllm_queue_t *queue,
    const sllm_buffer_t *left_buffer, uint64_t left_offset,
    const sllm_buffer_t *right_buffer, uint64_t right_offset,
    const sllm_buffer_t *output_buffer, uint64_t output_offset, uint64_t rows,
    uint64_t left_columns, uint64_t right_columns,
    sllm_error_sink_t *error_sink) noexcept;

namespace {

constexpr uint16_t kGuard = UINT16_C(0x6d3b);
constexpr uint64_t kLeftOffset = 6U;
constexpr uint64_t kRightOffset = 10U;
constexpr uint64_t kOutputOffset = 14U;
constexpr uint64_t kGuardWords = 9U;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};

  void reset() {
    std::memset(message, 0, sizeof(message));
    sink.message_length = 0U;
  }
};

bool expect_status(const sllm_status_t actual, const sllm_status_t expected,
                   const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
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
  result.state = SLLM_COMPLETION_STATE_PENDING;
  bool ok = expect_status(
      sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
      SLLM_STATUS_OK, operation, error);
  ok = ok && result.state == SLLM_COMPLETION_STATE_SUCCESS;
  error.reset();
  ok = expect_status(sllm_completion_release(completion, &error.sink),
                     SLLM_STATUS_OK, "sllm_completion_release", error) &&
       ok && *completion == nullptr;
  return ok;
}

bool copy_h2d(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, const uint64_t offset,
              const void *const source, const uint64_t bytes,
              const char *const operation) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                            &completion, &error.sink),
                       SLLM_STATUS_OK, operation, error) &&
         wait_release(&completion, operation);
}

bool copy_d2h(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, const uint64_t offset,
              void *const destination, const uint64_t bytes,
              const char *const operation) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                          &error.sink),
                     SLLM_STATUS_OK, operation, error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.state = SLLM_COMPLETION_STATE_PENDING;
  bool ok = expect_status(
      sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
      SLLM_STATUS_OK, operation, error);
  uint64_t written = 0U;
  if (ok && result.state == SLLM_COMPLETION_STATE_SUCCESS) {
    error.reset();
    ok = expect_status(sllm_completion_read(completion, destination, bytes,
                                            &written, &error.sink),
                       SLLM_STATUS_OK, operation, error) &&
         written == bytes;
  }
  error.reset();
  ok = expect_status(sllm_completion_release(&completion, &error.sink),
                     SLLM_STATUS_OK, "sllm_completion_release", error) &&
       ok && completion == nullptr;
  return ok;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  // The private wrapper validates the device address's natural BF16 alignment;
  // this test deliberately leaves allocator alignment unconstrained.
  info.alignment_bytes = 0U;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, output, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_create", error) &&
         output != nullptr && *output != nullptr;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr) {
    return true;
  }
  Error error;
  return expect_status(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                       "sllm_buffer_release", error) &&
         *buffer == nullptr;
}

uint16_t pattern(const uint64_t index) {
  static constexpr std::array<uint16_t, 8U> values = {
      UINT16_C(0x0000), UINT16_C(0x8000), UINT16_C(0x0001), UINT16_C(0x7fc1),
      UINT16_C(0xffc2), UINT16_C(0x3f80), UINT16_C(0x4120), UINT16_C(0xbf40)};
  return values[static_cast<std::size_t>(index % values.size())];
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint64_t rows,
              const uint64_t left_columns, const uint64_t right_columns) {
  const uint64_t left_words = rows * left_columns;
  const uint64_t right_words = rows * right_columns;
  const uint64_t output_words = rows * (left_columns + right_columns);
  const uint64_t left_bytes = left_words * sizeof(uint16_t);
  const uint64_t right_bytes = right_words * sizeof(uint16_t);
  const uint64_t output_bytes = output_words * sizeof(uint16_t);
  const uint64_t left_size = kLeftOffset + left_bytes + kGuardWords * 2U;
  const uint64_t right_size = kRightOffset + right_bytes + kGuardWords * 2U;
  const uint64_t output_size = kOutputOffset + output_bytes + kGuardWords * 2U;
  std::vector<uint16_t> left_host(static_cast<std::size_t>(left_size / 2U),
                                  kGuard);
  std::vector<uint16_t> right_host(static_cast<std::size_t>(right_size / 2U),
                                   kGuard);
  std::vector<uint16_t> output_host(static_cast<std::size_t>(output_size / 2U),
                                    kGuard);
  std::vector<uint16_t> expected = output_host;
  for (uint64_t row = 0U; row != rows; ++row) {
    for (uint64_t column = 0U; column != left_columns; ++column) {
      const uint64_t source = row * left_columns + column;
      const uint16_t value = pattern(source + 3U * row);
      left_host[static_cast<std::size_t>(kLeftOffset / 2U + source)] = value;
      expected[static_cast<std::size_t>(
          kOutputOffset / 2U + row * (left_columns + right_columns) + column)] =
          value;
    }
    for (uint64_t column = 0U; column != right_columns; ++column) {
      const uint64_t source = row * right_columns + column;
      const uint16_t value = pattern(source + 5U * row + 1U);
      right_host[static_cast<std::size_t>(kRightOffset / 2U + source)] = value;
      expected[static_cast<std::size_t>(kOutputOffset / 2U +
                                        row * (left_columns + right_columns) +
                                        left_columns + column)] = value;
    }
  }
  sllm_buffer_t *left = nullptr;
  sllm_buffer_t *right = nullptr;
  sllm_buffer_t *output = nullptr;
  bool ok = create_buffer(context, left_size, &left) &&
            create_buffer(context, right_size, &right) &&
            create_buffer(context, output_size, &output);
  if (ok) {
    ok = copy_h2d(queue, left, 0U, left_host.data(), left_size,
                  "row concat left upload") &&
         copy_h2d(queue, right, 0U, right_host.data(), right_size,
                  "row concat right upload") &&
         copy_h2d(queue, output, 0U, output_host.data(), output_size,
                  "row concat output guard upload");
  }
  if (ok) {
    Error error;
    ok = expect_status(
        sllm_concat_bf16_rows_v1(context, queue, left, kLeftOffset, right,
                                 kRightOffset, output, kOutputOffset, rows,
                                 left_columns, right_columns, &error.sink),
        SLLM_STATUS_OK, "sllm_concat_bf16_rows_v1", error);
  }
  if (ok) {
    std::vector<uint16_t> actual(output_host.size(), 0U);
    ok = copy_d2h(queue, output, 0U, actual.data(), output_size,
                  "row concat output download");
    if (ok && actual != expected) {
      std::cerr << "row concat output differs for rows=" << rows
                << " left=" << left_columns << " right=" << right_columns
                << '\n';
      ok = false;
    }
  }
  ok = release_buffer(&output) && ok;
  ok = release_buffer(&right) && ok;
  ok = release_buffer(&left) && ok;
  return ok;
}

bool run_read_alias_case(const sllm_context_t *const context,
                         const sllm_queue_t *const queue) {
  constexpr uint64_t rows = 3U;
  constexpr uint64_t columns = 7U;
  const uint64_t source_words = rows * columns;
  const uint64_t source_bytes = source_words * sizeof(uint16_t);
  const uint64_t source_size = kLeftOffset + source_bytes + kGuardWords * 2U;
  const uint64_t output_words = rows * columns * 2U;
  const uint64_t output_bytes = output_words * sizeof(uint16_t);
  const uint64_t output_size = kOutputOffset + output_bytes + kGuardWords * 2U;
  std::vector<uint16_t> source(static_cast<std::size_t>(source_size / 2U),
                               kGuard);
  std::vector<uint16_t> output(static_cast<std::size_t>(output_size / 2U),
                               kGuard);
  std::vector<uint16_t> expected = output;
  for (uint64_t row = 0U; row != rows; ++row) {
    for (uint64_t column = 0U; column != columns; ++column) {
      const uint64_t index = row * columns + column;
      const uint16_t value = pattern(index + 11U);
      source[static_cast<std::size_t>(kLeftOffset / 2U + index)] = value;
      expected[static_cast<std::size_t>(kOutputOffset / 2U +
                                        row * columns * 2U + column)] = value;
      expected[static_cast<std::size_t>(
          kOutputOffset / 2U + row * columns * 2U + columns + column)] = value;
    }
  }
  sllm_buffer_t *source_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  bool ok = create_buffer(context, source_size, &source_buffer) &&
            create_buffer(context, output_size, &output_buffer);
  if (ok) {
    ok = copy_h2d(queue, source_buffer, 0U, source.data(), source_size,
                  "row concat alias source upload") &&
         copy_h2d(queue, output_buffer, 0U, output.data(), output_size,
                  "row concat alias guard upload");
  }
  if (ok) {
    Error error;
    ok = expect_status(sllm_concat_bf16_rows_v1(
                           context, queue, source_buffer, kLeftOffset,
                           source_buffer, kLeftOffset, output_buffer,
                           kOutputOffset, rows, columns, columns, &error.sink),
                       SLLM_STATUS_OK, "row concat read-alias", error);
  }
  if (ok) {
    std::vector<uint16_t> actual(output.size(), 0U);
    ok = copy_d2h(queue, output_buffer, 0U, actual.data(), output_size,
                  "row concat alias output download") &&
         actual == expected;
  }
  ok = release_buffer(&output_buffer) && ok;
  ok = release_buffer(&source_buffer) && ok;
  return ok;
}

bool run_error_cases(const sllm_context_t *const context,
                     const sllm_queue_t *const queue) {
  sllm_buffer_t *left = nullptr;
  sllm_buffer_t *right = nullptr;
  sllm_buffer_t *output = nullptr;
  bool ok = create_buffer(context, 256U, &left) &&
            create_buffer(context, 256U, &right) &&
            create_buffer(context, 512U, &output);
  if (!ok) {
    release_buffer(&output);
    release_buffer(&right);
    release_buffer(&left);
    return false;
  }
  Error error;
  ok = expect_status(sllm_concat_bf16_rows_v1(context, queue, left, 0U, right,
                                              0U, output, 0U, 0U, 3U, 5U,
                                              &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "zero rows", error) &&
       ok;
  error.reset();
  ok = expect_status(sllm_concat_bf16_rows_v1(context, queue, left, 1U, right,
                                              0U, output, 0U, 1U, 3U, 5U,
                                              &error.sink),
                     SLLM_STATUS_MISALIGNED_OFFSET, "odd left offset", error) &&
       ok;
  error.reset();
  ok = expect_status(
           sllm_concat_bf16_rows_v1(context, queue, left, 0U, right, 0U, output,
                                    0U, 2U, 100U, 100U, &error.sink),
           SLLM_STATUS_BUFFER_OUT_OF_BOUNDS, "range overflow", error) &&
       ok;
  error.reset();
  ok = expect_status(sllm_concat_bf16_rows_v1(context, queue, left, 0U, right,
                                              0U, left, 2U, 1U, 3U, 5U,
                                              &error.sink),
                     SLLM_STATUS_ALIAS_OVERLAP, "overlapping output", error) &&
       ok;
  ok = release_buffer(&output) && ok;
  ok = release_buffer(&right) && ok;
  ok = release_buffer(&left) && ok;
  return ok;
}

} // namespace

int main() {
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  bool ok = expect_status(sllm_device_query(0U, &device, &error.sink),
                          SLLM_STATUS_OK, "sllm_device_query", error);
  if (ok && std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "expected target " << SLLM_TEST_EXPECTED_TARGET << ", got "
              << device.gcn_arch_name << '\n';
    ok = false;
  }
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  if (ok) {
    sllm_context_create_info_t context_info{};
    context_info.struct_size = sizeof(context_info);
    context_info.abi_version = SLLM_HIP_ABI_VERSION;
    context_info.device_index = 0U;
    std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
                 sizeof(context_info.expected_gcn_arch_name) - 1U);
    ok =
        expect_status(sllm_context_create(&context_info, &context, &error.sink),
                      SLLM_STATUS_OK, "sllm_context_create", error);
  }
  if (ok) {
    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    ok = expect_status(
        sllm_queue_create(context, &queue_info, &queue, &error.sink),
        SLLM_STATUS_OK, "sllm_queue_create", error);
  }
  if (ok) {
    constexpr std::array<std::array<uint64_t, 3U>, 6U> cases = {{
        {{1U, 3U, 5U}},
        {{2U, 7U, 11U}},
        {{3U, 13U, 17U}},
        {{127U, 31U, 37U}},
        {{128U, 63U, 65U}},
        {{129U, 5U, 9U}},
    }};
    for (const auto &test_case : cases) {
      if (!run_case(context, queue, test_case[0], test_case[1], test_case[2])) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    ok = run_read_alias_case(context, queue);
  }
  if (ok) {
    ok = run_error_cases(context, queue);
  }
  if (queue != nullptr) {
    error.reset();
    ok = expect_status(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                       "sllm_queue_release", error) &&
         ok;
  }
  if (context != nullptr) {
    error.reset();
    ok = expect_status(sllm_context_release(&context, &error.sink),
                       SLLM_STATUS_OK, "sllm_context_release", error) &&
         ok;
  }
  std::cout << "phase83_5_row_concat status=" << (ok ? "PASS" : "FAIL")
            << " target=" << SLLM_TEST_EXPECTED_TARGET
            << " oracle=bitwise boundaries=1\n";
  return ok ? 0 : 1;
}
