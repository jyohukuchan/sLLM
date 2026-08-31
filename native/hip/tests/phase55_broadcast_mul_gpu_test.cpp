#include "sllm/hip.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

constexpr uint32_t kWidth = 2816U;
constexpr uint32_t kWorkgroupSize = 256U;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

struct Device final {
  int index = 0;
  int count = 0;
  hipDeviceProp_t properties{};
};

struct NumericalResult final {
  uint64_t finite_mismatches = 0U;
  uint64_t nonfinite_mismatches = 0U;
  uint64_t propagated_infinities = 0U;
  uint64_t propagated_nans = 0U;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
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

bool hip_expect(const hipError_t actual, const hipError_t expected,
                const char *const operation) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << static_cast<int>(actual) << ": "
            << hipGetErrorString(actual) << '\n';
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

bool is_bf16_nan(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) == UINT16_C(0x7f80) &&
         (value & UINT16_C(0x007f)) != 0U;
}

bool is_bf16_quiet_nan(const uint16_t value) {
  return is_bf16_nan(value) && (value & UINT16_C(0x0040)) != 0U;
}

bool parse_index(const char *const text, int *const result) {
  if (text == nullptr || text[0] == '\0' || text[0] == '-') {
    return false;
  }
  uint64_t value = 0U;
  for (const char *cursor = text; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') {
      return false;
    }
    const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
    if (value >
        (static_cast<uint64_t>(std::numeric_limits<int>::max()) - digit) /
            10U) {
      return false;
    }
    value = value * 10U + digit;
  }
  *result = static_cast<int>(value);
  return true;
}

bool select_device(const int index, Device *const device) {
  if (!hip_expect(hipGetDeviceCount(&device->count), hipSuccess,
                  "hipGetDeviceCount") ||
      device->count <= 0 || index < 0 || index >= device->count) {
    std::cerr << "device index " << index << " is not in the visible HIP "
              << "device range [0, " << device->count << ")\n";
    return false;
  }
  device->index = index;
  return hip_expect(hipSetDevice(index), hipSuccess, "hipSetDevice") &&
         hip_expect(hipGetDeviceProperties(&device->properties, index),
                    hipSuccess, "hipGetDeviceProperties");
}

sllm_tensor_binding_t
tensor_binding(const sllm_buffer_t *const buffer,
               const std::initializer_list<uint64_t> shape) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = SLLM_TENSOR_DTYPE_BF16;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = static_cast<uint32_t>(shape.size());
  uint64_t stride = 1U;
  std::size_t dimension = shape.size();
  for (auto iterator = shape.end(); iterator != shape.begin();) {
    --iterator;
    --dimension;
    result.shape[dimension] = *iterator;
    result.stride_elements[dimension] = stride;
    stride *= *iterator;
  }
  return result;
}

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned a null completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(
      sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
      SLLM_STATUS_OK, operation, error);
  const bool successful =
      waited && result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released =
      expect(sllm_completion_release(completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release", error);
  return successful && released && *completion == nullptr;
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
         *output != nullptr;
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
              SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error) ||
      completion == nullptr) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, "sllm_completion_wait(d2h)", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t bytes_written = 0U;
  const bool read = expect(sllm_completion_read(completion, output->data(),
                                                transfer.size_bytes,
                                                &bytes_written, &error.sink),
                           SLLM_STATUS_OK, "sllm_completion_read", error) &&
                    bytes_written == transfer.size_bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(d2h)", error);
  return read && released && completion == nullptr;
}

std::vector<uint16_t> make_vector() {
  std::vector<uint16_t> result(kWidth);
  for (uint32_t column = 0U; column != kWidth; ++column) {
    const int32_t signed_bucket =
        static_cast<int32_t>((column * 19U) % 37U) - 18;
    const int32_t exponent = static_cast<int32_t>(column % 9U) - 4;
    const float value =
        std::ldexp(static_cast<float>(signed_bucket) / 7.0F, exponent);
    result[column] = f32_to_bf16_rne(value);
  }
  result[0] = f32_to_bf16_rne(-1.0F);
  result[1] = f32_to_bf16_rne(2.0F);
  result[31] = f32_to_bf16_rne(-0.5F);
  result[32] = f32_to_bf16_rne(0.25F);
  result[255] = f32_to_bf16_rne(-17.0F);
  result[256] = f32_to_bf16_rne(0.03125F);
  result[kWidth - 1U] = f32_to_bf16_rne(-3.5F);
  return result;
}

std::vector<uint16_t> make_input(const uint32_t rows) {
  std::vector<uint16_t> result(static_cast<std::size_t>(rows) * kWidth);
  for (uint32_t row = 0U; row != rows; ++row) {
    for (uint32_t column = 0U; column != kWidth; ++column) {
      const int32_t signed_bucket =
          static_cast<int32_t>((column * 29U + row * 11U) % 53U) - 26;
      const int32_t exponent = static_cast<int32_t>((column + row) % 11U) - 5;
      float value =
          std::ldexp(static_cast<float>(signed_bucket) / 9.0F, exponent);
      if ((row & 1U) != 0U) {
        value = -value;
      }
      result[static_cast<std::size_t>(row) * kWidth + column] =
          f32_to_bf16_rne(value);
    }
    const std::size_t offset = static_cast<std::size_t>(row) * kWidth;
    const float row_value = static_cast<float>(row);
    result[offset] = UINT16_C(0x7f80);      // +infinity * -1 -> -infinity.
    result[offset + 1U] = UINT16_C(0x7fc1); // Quiet NaN must stay quiet NaN.
    result[offset + 31U] = f32_to_bf16_rne(-7.0F - row_value);
    result[offset + 32U] = f32_to_bf16_rne(7.0F + row_value);
    result[offset + 255U] = f32_to_bf16_rne(31.0F + row_value);
    result[offset + 256U] = f32_to_bf16_rne(-31.0F - row_value);
    result[offset + kWidth - 1U] = f32_to_bf16_rne(0.001953125F);
  }
  return result;
}

std::vector<uint16_t> reference(const std::vector<uint16_t> &input,
                                const std::vector<uint16_t> &vector) {
  std::vector<uint16_t> result(input.size());
  for (std::size_t index = 0U; index != input.size(); ++index) {
    result[index] = f32_to_bf16_rne(bf16_to_f32(input[index]) *
                                    bf16_to_f32(vector[index % kWidth]));
  }
  return result;
}

bool compare(const std::vector<uint16_t> &output,
             const std::vector<uint16_t> &oracle,
             NumericalResult *const numerical) {
  for (std::size_t index = 0U; index != output.size(); ++index) {
    const uint16_t expected = oracle[index];
    const uint16_t actual = output[index];
    const float expected_f32 = bf16_to_f32(expected);
    if (std::isnan(expected_f32)) {
      ++numerical->propagated_nans;
      if (!is_bf16_quiet_nan(actual)) {
        ++numerical->nonfinite_mismatches;
      }
      continue;
    }
    if (std::isinf(expected_f32)) {
      ++numerical->propagated_infinities;
      if (actual != expected) {
        ++numerical->nonfinite_mismatches;
      }
      continue;
    }
    const float actual_f32 = bf16_to_f32(actual);
    const double abs_error =
        std::abs(static_cast<double>(actual_f32) - expected_f32);
    const double denominator =
        std::max(std::abs(static_cast<double>(expected_f32)), 1.0e-30);
    numerical->max_abs_error = std::max(numerical->max_abs_error, abs_error);
    numerical->max_rel_error =
        std::max(numerical->max_rel_error, abs_error / denominator);
    if (actual != expected) {
      ++numerical->finite_mismatches;
    }
  }
  return numerical->finite_mismatches == 0U &&
         numerical->nonfinite_mismatches == 0U;
}

bool validate_metadata(const sllm_elementwise_dispatch_info_t &info,
                       const uint32_t rows) {
  const uint64_t element_count = static_cast<uint64_t>(rows) * kWidth;
  const uint32_t expected_grid = static_cast<uint32_t>(
      (element_count + kWorkgroupSize - 1U) / kWorkgroupSize);
  const bool valid =
      info.backend == SLLM_BACKEND_HIP && info.dispatch_id != 0U &&
      info.operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL &&
      info.dispatch_count > 0U &&
      info.kernel_id == SLLM_HIP_ELEMENTWISE_KERNEL_ID_BROADCAST_MUL_V1 &&
      info.workgroup_size_x == kWorkgroupSize &&
      info.grid_size_x == expected_grid && info.fallback_allowed == 0U &&
      info.fallback_used == 0U && info.element_count == element_count &&
      std::strcmp(info.kernel_symbol,
                  "elementwise.broadcast_mul.bf16_fp32.v1") == 0 &&
      std::strcmp(info.device_symbol,
                  "sllm_elementwise_broadcast_mul_bf16_fp32_v1") == 0 &&
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
  if (!valid) {
    std::cerr << "broadcast_mul dispatch mismatch: backend=" << info.backend
              << " dispatch_id=" << info.dispatch_id
              << " operation=" << info.operation
              << " dispatch_count=" << info.dispatch_count
              << " kernel_id=" << info.kernel_id
              << " workgroup=" << info.workgroup_size_x
              << " grid=" << info.grid_size_x
              << " fallback_allowed=" << info.fallback_allowed
              << " fallback_used=" << info.fallback_used
              << " element_count=" << info.element_count
              << " target=" << info.gcn_arch_name << '\n';
  }
  return valid;
}

bool run_case(const sllm_context_t *const context,
              const sllm_queue_t *const queue, const uint32_t rows,
              NumericalResult *const numerical) {
  const std::vector<uint16_t> input = make_input(rows);
  const std::vector<uint16_t> vector = make_vector();
  const std::vector<uint16_t> oracle = reference(input, vector);
  const uint64_t matrix_bytes = input.size() * sizeof(uint16_t);
  const uint64_t vector_bytes = vector.size() * sizeof(uint16_t);
  std::array<sllm_buffer_t *, 3> buffers{};
  bool success = create_buffer(context, matrix_bytes, &buffers[0]) &&
                 create_buffer(context, vector_bytes, &buffers[1]) &&
                 create_buffer(context, matrix_bytes, &buffers[2]) &&
                 upload(queue, buffers[0], input.data(), matrix_bytes) &&
                 upload(queue, buffers[1], vector.data(), vector_bytes);
  sllm_elementwise_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (success) {
    sllm_elementwise_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_ELEMENTWISE_VERSION;
    descriptor.operation = SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL;
    descriptor.input0 = tensor_binding(buffers[0], {rows, kWidth});
    descriptor.input1 = tensor_binding(buffers[1], {kWidth});
    descriptor.output = tensor_binding(buffers[2], {rows, kWidth});
    success = expect(sllm_elementwise_prepare(context, &descriptor, &plan,
                                              &error.sink),
                     SLLM_STATUS_OK, "sllm_elementwise_prepare", error) &&
              plan != nullptr;
  }
  sllm_elementwise_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION;
  if (success) {
    success =
        expect(sllm_elementwise_execute(plan, queue, &completion, &info,
                                        &error.sink),
               SLLM_STATUS_OK, "sllm_elementwise_execute", error) &&
        wait_and_release(&completion, "sllm_completion_wait(elementwise)") &&
        validate_metadata(info, rows);
  }
  std::vector<uint16_t> output(input.size());
  if (success) {
    success = download(queue, buffers[2], &output) &&
              compare(output, oracle, numerical);
  }
  if (plan != nullptr) {
    success = expect(sllm_elementwise_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "sllm_elementwise_plan_release", error) &&
              success;
  }
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    if (*iterator != nullptr) {
      success = expect(sllm_buffer_release(&*iterator, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_release", error) &&
                success;
    }
  }
  const uint64_t live_handles =
      static_cast<uint64_t>(plan != nullptr) +
      static_cast<uint64_t>(completion != nullptr) +
      static_cast<uint64_t>(std::count_if(
          buffers.begin(), buffers.end(),
          [](const sllm_buffer_t *const buffer) { return buffer != nullptr; }));
  if (live_handles != 0U) {
    std::cerr << "case cleanup retained " << live_handles << " handle(s)\n";
    success = false;
  }
  return success;
}

} // namespace

int main(const int argc, char **const argv) {
  if (argc > 2) {
    std::cerr << "usage: " << argv[0] << " [visible-device-index]\n";
    return 2;
  }
  int device_index = 0;
  if (argc == 2 && !parse_index(argv[1], &device_index)) {
    std::cerr << "device index must be a non-negative decimal integer\n";
    return 2;
  }
  Device device;
  if (!select_device(device_index, &device)) {
    return 1;
  }
  const std::string actual_arch(device.properties.gcnArchName);
  const char *const visible_devices = std::getenv("HIP_VISIBLE_DEVICES");
  std::cout << "device index=" << device.index
            << " visible_count=" << device.count << " name=\""
            << device.properties.name << "\" arch=" << actual_arch
            << " expected=" << SLLM_TEST_EXPECTED_TARGET
            << " HIP_VISIBLE_DEVICES=\""
            << (visible_devices == nullptr ? "" : visible_devices) << "\"\n";
  if (actual_arch != SLLM_TEST_EXPECTED_TARGET || device.index != 0 ||
      device.count != 1) {
    std::cerr << "exact single visible GPU contract mismatch\n";
    return 1;
  }
  if (!hip_expect(hipDeviceSynchronize(), hipSuccess, "hipDeviceSynchronize")) {
    return 1;
  }
  std::size_t free_before = 0U;
  std::size_t total_before = 0U;
  if (!hip_expect(hipMemGetInfo(&free_before, &total_before), hipSuccess,
                  "hipMemGetInfo(before)")) {
    return 1;
  }

  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
  Error error;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "sllm_context_create", error) ||
      context == nullptr) {
    return 1;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  bool success =
      expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
             SLLM_STATUS_OK, "sllm_queue_create", error) &&
      queue != nullptr;
  NumericalResult numerical;
  constexpr std::array<uint32_t, 2> rows{{1U, 3U}};
  for (const uint32_t row_count : rows) {
    if (success && !run_case(context, queue, row_count, &numerical)) {
      success = false;
    }
  }
  if (queue != nullptr) {
    success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                     "sllm_queue_release", error) &&
              success;
  }
  if (context != nullptr) {
    success = expect(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_OK, "sllm_context_release", error) &&
              success;
  }
  success = success && queue == nullptr && context == nullptr;
  if (!hip_expect(hipDeviceSynchronize(), hipSuccess,
                  "hipDeviceSynchronize(cleanup)")) {
    success = false;
  }
  std::size_t free_after = 0U;
  std::size_t total_after = 0U;
  const bool memory_read = hip_expect(hipMemGetInfo(&free_after, &total_after),
                                      hipSuccess, "hipMemGetInfo(after)");
  constexpr std::size_t kRuntimeCacheTolerance =
      static_cast<std::size_t>(512U) * 1024U * 1024U;
  const bool memory_cleanup =
      memory_read && total_before == total_after &&
      (free_after >= free_before ||
       free_before - free_after <= kRuntimeCacheTolerance);
  success = success && memory_cleanup && numerical.finite_mismatches == 0U &&
            numerical.nonfinite_mismatches == 0U &&
            numerical.propagated_infinities == 4U &&
            numerical.propagated_nans == 4U;
  if (success) {
    std::cout << "phase55 broadcast_mul GPU PASS target="
              << SLLM_TEST_EXPECTED_TARGET << " device_index=" << device.index
              << " device_name=\"" << device.properties.name
              << "\" cases=2 M=1,3 H=" << kWidth
              << " finite_mismatches=" << numerical.finite_mismatches
              << " nonfinite_mismatches=" << numerical.nonfinite_mismatches
              << " propagated_infinities=" << numerical.propagated_infinities
              << " propagated_quiet_nans=" << numerical.propagated_nans
              << " max_abs_error=" << numerical.max_abs_error
              << " max_rel_error=" << numerical.max_rel_error
              << " fallback_allowed=0 fallback_used=0 cleanup=0"
              << " free_before=" << free_before << " free_after=" << free_after
              << " kernel=elementwise.broadcast_mul.bf16_fp32.v1"
              << " symbol=sllm_elementwise_broadcast_mul_bf16_fp32_v1\n";
  } else {
    std::cerr << "phase55 broadcast_mul GPU FAIL target="
              << SLLM_TEST_EXPECTED_TARGET
              << " finite_mismatches=" << numerical.finite_mismatches
              << " nonfinite_mismatches=" << numerical.nonfinite_mismatches
              << " propagated_infinities=" << numerical.propagated_infinities
              << " propagated_nans=" << numerical.propagated_nans
              << " max_abs_error=" << numerical.max_abs_error
              << " max_rel_error=" << numerical.max_rel_error
              << " memory_cleanup=" << memory_cleanup << '\n';
  }
  return success ? 0 : 1;
}
