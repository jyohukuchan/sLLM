#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

namespace {

struct Shape final {
  const char *name;
  uint64_t k;
  uint64_t n;
};

constexpr std::array<Shape, 5> kShapes = {{
    {"k5120n12288", 5120U, 12288U},
    {"k5120n1024", 5120U, 1024U},
    {"k5120n17408", 5120U, 17408U},
    {"k17408n5120", 17408U, 5120U},
    {"k6144n5120", 6144U, 5120U},
}};
constexpr std::array<Shape, 2> kGfx1201Shapes = {
    {{"k5120n6144", 5120U, 6144U}, {"k5120n10240", 5120U, 10240U}}};
constexpr std::array<uint64_t, 5> kRows = {1U, 2U, 3U, 4U, 5U};
constexpr std::size_t kRepeats = 3U;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

struct ExpectedSelection final {
  uint32_t kernel_id;
  const char *kernel_symbol;
  const char *device_symbol;
  uint32_t grid_x;
};

ExpectedSelection expected_selection(const Shape &shape, const uint64_t rows) {
  if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") == 0) {
    return {5U, "matmul.fp8.outer.hipblaslt.v1", "hipblasLtMatmul",
            static_cast<uint32_t>(shape.n)};
  }
  const uint32_t decode_grid = static_cast<uint32_t>((shape.n + 31U) / 32U);
  const bool short_n64 =
      shape.k == 5120U && (shape.n == 12288U || shape.n == 17408U);
  if (rows == 1U) {
    const char *device = nullptr;
    if (shape.k == 5120U && shape.n == 12288U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_m1_k5120n12288_v1";
    } else if (shape.k == 5120U && shape.n == 1024U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_m1_k5120n1024_v1";
    } else if (shape.k == 5120U && shape.n == 17408U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_m1_k5120n17408_v1";
    } else if (shape.k == 17408U && shape.n == 5120U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_m1_k17408n5120_v1";
    } else if (shape.k == 6144U && shape.n == 5120U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_m1_k6144n5120_v1";
    }
    if (device != nullptr) {
      return {82U, "matmul.fp8.outer.decode.gfx1030.lds_lut.wave4col32.v1",
              device, decode_grid};
    }
    return {68, "matmul.fp8.outer.decode.gfx1030.dword8.wave4col32.v1",
            "sllm_matmul_fp8_outer_decode_gfx1030_dword8_wave4col32_v1",
            decode_grid};
  }
  if (rows <= 4U) {
    const char *device = nullptr;
    if (shape.k == 5120U && shape.n == 12288U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_fused_k5120n12288_v1";
    } else if (shape.k == 5120U && shape.n == 1024U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_fused_k5120n1024_v1";
    } else if (shape.k == 5120U && shape.n == 17408U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_fused_k5120n17408_v1";
    } else if (shape.k == 17408U && shape.n == 5120U) {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_fused_k17408n5120_v1";
    } else {
      device = "sllm_matmul_fp8_outer_decode_gfx1030_fused_k6144n5120_v1";
    }
    return {SLLM_HIP_MATMUL_KERNEL_ID_FP8_OUTER_GFX1030_FUSED_M2_4_V1,
            "matmul.fp8.outer.decode.gfx1030.fused.m2_4.v1", device,
            decode_grid};
  }
  return {71U, "matmul.fp8.outer.prefill.gfx1030.half2.64x64.v1",
          "sllm_matmul_fp8_outer_prefill_gfx1030_half2_64x64_v1",
          static_cast<uint32_t>(
              ((rows + 31U) / 32U) *
              (short_n64 ? (shape.n + 63U) / 64U : (shape.n + 31U) / 32U))};
}

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
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

uint32_t bf16_ulp(const uint16_t actual, const uint16_t expected) {
  return actual >= expected ? static_cast<uint32_t>(actual - expected)
                            : static_cast<uint32_t>(expected - actual);
}

bool finite_bf16(const uint16_t value) {
  return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

float activation_value(const uint64_t row) {
  return 0.5F + 0.5F * static_cast<float>(row % 5U);
}

float column_scale(const uint64_t column) {
  return (column & 1U) == 0U ? 0.5F : 1.0F;
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

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(sllm_completion_wait(*completion, UINT32_MAX,
                                                  &result, &error.sink),
                             SLLM_STATUS_OK, operation, error) &&
                      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released =
      expect(sllm_completion_release(completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release", error);
  return waited && released;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const source, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  const bool copied = expect(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                                  &completion, &error.sink),
                             SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error) &&
                      completion != nullptr;
  return copied && wait_and_release(&completion, "sllm_completion_wait(h2d)");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer,
              std::vector<uint16_t> *const output) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes =
      static_cast<uint64_t>(output->size()) * sizeof(output->front());
  sllm_completion_t *completion = nullptr;
  Error error;
  bool valid = expect(sllm_buffer_copy_d2h(queue, buffer, &transfer,
                                           &completion, &error.sink),
                      SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error) &&
               completion != nullptr;
  if (valid) {
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_completion_wait(completion, UINT32_MAX, &result,
                                        &error.sink),
                   SLLM_STATUS_OK, "sllm_completion_wait(d2h)", error) &&
            result.state == SLLM_COMPLETION_STATE_SUCCESS;
    uint64_t written = 0U;
    valid =
        expect(sllm_completion_read(completion, output->data(),
                                    transfer.size_bytes, &written, &error.sink),
               SLLM_STATUS_OK, "sllm_completion_read", error) &&
        written == transfer.size_bytes && valid;
  }
  if (completion != nullptr) {
    valid = expect(sllm_completion_release(&completion, &error.sink),
                   SLLM_STATUS_OK, "sllm_completion_release(d2h)", error) &&
            valid;
  }
  return valid;
}

std::vector<uint16_t> make_activation(const uint64_t rows, const uint64_t k) {
  std::vector<uint16_t> activation(static_cast<std::size_t>(rows * k));
  for (uint64_t row = 0U; row != rows; ++row) {
    const uint16_t value = f32_to_bf16_rne(activation_value(row));
    std::fill(activation.begin() + static_cast<std::ptrdiff_t>(row * k),
              activation.begin() + static_cast<std::ptrdiff_t>((row + 1U) * k),
              value);
  }
  return activation;
}

std::vector<uint8_t> make_weight(const Shape &shape) {
  const uint64_t value_bytes = shape.k * shape.n;
  std::vector<uint8_t> weight(
      static_cast<std::size_t>(value_bytes + shape.n * 4U), UINT8_C(0x38));
  for (uint64_t column = 0U; column != shape.n; ++column) {
    const float scale = column_scale(column);
    std::memcpy(weight.data() + value_bytes + column * sizeof(float), &scale,
                sizeof(scale));
  }
  return weight;
}

std::vector<uint16_t> expected_output(const Shape &shape, const uint64_t rows) {
  std::vector<uint16_t> expected(static_cast<std::size_t>(rows * shape.n));
  for (uint64_t row = 0U; row != rows; ++row) {
    const float activation = activation_value(row);
    for (uint64_t column = 0U; column != shape.n; ++column) {
      const float value =
          activation * static_cast<float>(shape.k) * column_scale(column);
      expected[static_cast<std::size_t>(row * shape.n + column)] =
          f32_to_bf16_rne(value);
    }
  }
  return expected;
}

sllm_matmul_desc_t descriptor(const Shape &shape, const uint64_t rows,
                              const sllm_buffer_t *const activation,
                              const sllm_buffer_t *const weight,
                              const sllm_buffer_t *const output) {
  sllm_matmul_desc_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.op_version = SLLM_HIP_MATMUL_FP8_VERSION;
  result.activation = binding(activation, SLLM_TENSOR_DTYPE_BF16,
                              SLLM_TENSOR_ENCODING_UNQUANTIZED, rows, shape.k);
  result.weight = binding(weight, SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                          SLLM_TENSOR_ENCODING_FP8_OUTER_F32, shape.n, shape.k);
  result.output = binding(output, SLLM_TENSOR_DTYPE_BF16,
                          SLLM_TENSOR_ENCODING_UNQUANTIZED, rows, shape.n);
  return result;
}

bool audit_dispatch(const sllm_matmul_dispatch_info_t &dispatch,
                    const Shape &shape, const uint64_t rows,
                    const ExpectedSelection &expected) {
  return dispatch.struct_size == sizeof(dispatch) &&
         dispatch.abi_version == SLLM_HIP_ABI_VERSION &&
         dispatch.info_version == SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION &&
         dispatch.backend == SLLM_BACKEND_HIP && dispatch.dispatch_id != 0U &&
         dispatch.dispatch_count == 2U &&
         dispatch.kernel_id == expected.kernel_id &&
         dispatch.workgroup_size_x == 256U &&
         dispatch.grid_size_x == expected.grid_x && dispatch.m == rows &&
         dispatch.k == shape.k && dispatch.n == shape.n &&
         dispatch.output_elements == rows * shape.n &&
         dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
         std::strcmp(dispatch.kernel_symbol, expected.kernel_symbol) == 0 &&
         std::strcmp(dispatch.device_symbol, expected.device_symbol) == 0 &&
         std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
}

bool check_oracle(const std::vector<uint16_t> &actual,
                  const std::vector<uint16_t> &expected, const Shape &shape,
                  const uint64_t rows, uint32_t *const max_ulp,
                  uint64_t *const mismatches) {
  bool valid = actual.size() == expected.size();
  *max_ulp = 0U;
  *mismatches = 0U;
  for (std::size_t index = 0U;
       index != actual.size() && index < expected.size(); ++index) {
    const uint32_t ulp = bf16_ulp(actual[index], expected[index]);
    *max_ulp = std::max(*max_ulp, ulp);
    if (!finite_bf16(actual[index]) || actual[index] != expected[index]) {
      ++*mismatches;
      valid = false;
    }
  }
  const std::size_t last_column = static_cast<std::size_t>(shape.n - 1U);
  for (uint64_t row = 0U; row != rows; ++row) {
    const std::size_t index =
        static_cast<std::size_t>(row * shape.n) + last_column;
    valid = valid && index < actual.size() && actual[index] == expected[index];
  }
  return valid;
}

bool run_case(const Shape &shape, const sllm_context_t *const context,
              const sllm_queue_t *const queue,
              const sllm_buffer_t *const weight, const uint64_t rows) {
  const ExpectedSelection expected = expected_selection(shape, rows);
  const std::vector<uint16_t> activation = make_activation(rows, shape.k);
  const std::vector<uint16_t> oracle = expected_output(shape, rows);
  sllm_buffer_t *activation_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  sllm_matmul_plan_t *plan = nullptr;
  bool valid =
      create_buffer(context, rows * shape.k * sizeof(uint16_t),
                    &activation_buffer) &&
      create_buffer(context, rows * shape.n * sizeof(uint16_t),
                    &output_buffer) &&
      upload(queue, activation_buffer, activation.data(),
             static_cast<uint64_t>(activation.size()) * sizeof(uint16_t));
  if (valid) {
    const sllm_matmul_desc_t desc =
        descriptor(shape, rows, activation_buffer, weight, output_buffer);
    Error error;
    valid = expect(sllm_matmul_prepare(context, &desc, &plan, &error.sink),
                   SLLM_STATUS_OK, "sllm_matmul_prepare", error) &&
            plan != nullptr;
  }

  std::vector<uint16_t> first_output;
  uint32_t max_ulp = 0U;
  uint64_t mismatches = 0U;
  for (std::size_t repeat = 0U; repeat != kRepeats && valid; ++repeat) {
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
            wait_and_release(&completion, "sllm_completion_wait(matmul)") &&
            audit_dispatch(dispatch, shape, rows, expected);
    std::vector<uint16_t> observed(static_cast<std::size_t>(rows * shape.n));
    if (valid) {
      valid =
          download(queue, output_buffer, &observed) &&
          check_oracle(observed, oracle, shape, rows, &max_ulp, &mismatches);
      if (repeat == 0U) {
        first_output = observed;
      } else {
        valid = valid && observed == first_output;
      }
    }
  }

  if (plan != nullptr) {
    Error error;
    valid = expect(sllm_matmul_plan_release(&plan, &error.sink), SLLM_STATUS_OK,
                   "sllm_matmul_plan_release", error) &&
            valid;
  }
  valid = release_buffer(&output_buffer) && valid;
  valid = release_buffer(&activation_buffer) && valid;
  std::cout << "phase83_5_fp8_small_m shape=" << shape.name << " M=" << rows
            << " K=" << shape.k << " N=" << shape.n
            << " provider_id=" << expected.kernel_id
            << " kernel=" << expected.kernel_symbol
            << " device=" << expected.device_symbol
            << " grid_x=" << expected.grid_x << " oracle_rows=" << rows
            << " oracle_last_column=" << (shape.n - 1U)
            << " max_bf16_ulp=" << max_ulp << " mismatches=" << mismatches
            << " status=" << (valid ? "PASS" : "FAIL") << '\n';
  return valid;
}

} // namespace

int main() {
  if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") != 0 &&
      std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") != 0) {
    std::cerr << "phase83_5_fp8_small_m requires exact gfx1030 or gfx1201, got "
              << SLLM_TEST_EXPECTED_TARGET << '\n';
    return 1;
  }

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *weight_buffer = nullptr;
  bool valid = true;
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  valid = expect(sllm_context_create(&context_info, &context, &error.sink),
                 SLLM_STATUS_OK, "sllm_context_create", error);
  if (valid) {
    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "sllm_queue_create", error);
  }

  std::vector<Shape> shapes(kShapes.begin(), kShapes.end());
  if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") == 0) {
    shapes.insert(shapes.end(), kGfx1201Shapes.begin(), kGfx1201Shapes.end());
  }
  for (const Shape &shape : shapes) {
    const std::vector<uint8_t> weight = make_weight(shape);
    if (valid) {
      valid = create_buffer(context, static_cast<uint64_t>(weight.size()),
                            &weight_buffer) &&
              upload(queue, weight_buffer, weight.data(),
                     static_cast<uint64_t>(weight.size()));
    }
    for (const uint64_t rows : kRows) {
      if (!valid) {
        break;
      }
      valid = run_case(shape, context, queue, weight_buffer, rows);
    }
    valid = release_buffer(&weight_buffer) && valid;
  }

  valid = release_buffer(&weight_buffer) && valid;
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
  std::cout << "phase83_5_fp8_small_m status=" << (valid ? "PASS" : "FAIL")
            << " target=" << SLLM_TEST_EXPECTED_TARGET << " resources_released="
            << ((context == nullptr && queue == nullptr &&
                 weight_buffer == nullptr)
                    ? 1
                    : 0)
            << '\n';
  return valid ? 0 : 1;
}
