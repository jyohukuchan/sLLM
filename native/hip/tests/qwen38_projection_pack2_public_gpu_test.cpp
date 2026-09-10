#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

constexpr uint64_t kM = 1U;
constexpr uint64_t kK = 5120U;
constexpr uint64_t kN = 17408U;
constexpr uint64_t kFp8GdnQkvN = 10240U;
constexpr uint64_t kFp8GdnZN = 6144U;
constexpr uint64_t kFp8GdnWorkspaceBytes = kK + sizeof(float);
constexpr uint64_t kPackedWeightBytes = kN * kK / 2U;
constexpr uint64_t kWeightScaleBytes = kN * (kK / 16U);
constexpr uint64_t kWeightTensorScaleOffset =
    (kPackedWeightBytes + kWeightScaleBytes + 3U) & ~UINT64_C(3);
constexpr uint64_t kWeightBytes = kWeightTensorScaleOffset + 8U;

static_assert(kPackedWeightBytes == UINT64_C(44564480));
static_assert(kWeightScaleBytes == UINT64_C(5570560));
static_assert(kWeightBytes == UINT64_C(50135048));
static_assert(SLLM_HIP_QWEN38_PROJECTION_PACK2_WORKSPACE_BYTES ==
              UINT64_C(2880));
static_assert(SLLM_HIP_QWEN38_PROJECTION_PACK2_FP8_GDN_WORKSPACE_BYTES ==
              kFp8GdnWorkspaceBytes);

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
  if (actual == expected)
    return true;
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint32_t kExponentMask = UINT32_C(0x7f800000);
  constexpr uint32_t kFractionMask = UINT32_C(0x007fffff);
  if ((bits & kExponentMask) == kExponentMask) {
    if ((bits & kFractionMask) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT32_C(0x7fc0) |
                                   ((bits >> 16U) & UINT32_C(0x003f)));
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
  transfer.size_bytes =
      static_cast<uint64_t>(output->size()) * sizeof(uint16_t);
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
  uint64_t written = 0U;
  const bool read =
      expect(sllm_completion_read(completion, output->data(),
                                  transfer.size_bytes, &written, &error.sink),
             SLLM_STATUS_OK, "sllm_completion_read", error);
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(d2h)", error);
  return read && released && written == transfer.size_bytes;
}

bool execute_fp8_gdn_graph_once(
    const sllm_queue_t *const queue, const sllm_graph_span_t *const graph,
    const sllm_buffer_t *const activation_buffer,
    const std::vector<uint16_t> &activation,
    const std::array<sllm_buffer_t *, 2> &output_buffers,
    std::array<std::vector<uint16_t>, 2> *const outputs) {
  Error error;
  if (!upload(queue, activation_buffer, activation.data(),
              static_cast<uint64_t>(activation.size()) * sizeof(uint16_t))) {
    return false;
  }
  sllm_completion_t *graph_completion = nullptr;
  if (!expect(sllm_graph_span_execute(graph, &graph_completion, &error.sink),
              SLLM_STATUS_OK, "FP8 GDN graph execute", error) ||
      graph_completion == nullptr) {
    return false;
  }
  sllm_completion_t *fence = nullptr;
  if (!expect(sllm_queue_fence(queue, &fence, &error.sink), SLLM_STATUS_OK,
              "FP8 GDN graph fence", error) ||
      fence == nullptr) {
    (void)sllm_completion_release(&graph_completion, &error.sink);
    return false;
  }
  sllm_completion_result_t fence_result{};
  fence_result.struct_size = sizeof(fence_result);
  fence_result.abi_version = SLLM_HIP_ABI_VERSION;
  bool valid = expect(
      sllm_completion_wait(fence, UINT32_MAX, &fence_result, &error.sink),
      SLLM_STATUS_OK, "FP8 GDN graph fence wait", error);
  sllm_completion_result_t graph_result{};
  graph_result.struct_size = sizeof(graph_result);
  graph_result.abi_version = SLLM_HIP_ABI_VERSION;
  valid = expect(sllm_completion_finalize_after(graph_completion, fence,
                                                &graph_result, &error.sink),
                 SLLM_STATUS_OK, "FP8 GDN graph finalize", error) &&
          graph_result.state == SLLM_COMPLETION_STATE_SUCCESS && valid;
  valid = expect(sllm_completion_release(&graph_completion, &error.sink),
                 SLLM_STATUS_OK, "FP8 GDN graph completion release", error) &&
          valid;
  valid = expect(sllm_completion_release(&fence, &error.sink), SLLM_STATUS_OK,
                 "FP8 GDN graph fence release", error) &&
          valid;
  for (std::size_t index = 0U; index != output_buffers.size() && valid;
       ++index) {
    (*outputs)[index].resize(output_buffers[index] == nullptr
                                 ? 0U
                                 : (index == 0U ? kFp8GdnQkvN : kFp8GdnZN));
    valid = download(queue, output_buffers[index], &(*outputs)[index]) && valid;
  }
  return valid;
}

std::vector<uint8_t> make_weight(const uint8_t e2m1_code) {
  const uint8_t packed = static_cast<uint8_t>(e2m1_code | (e2m1_code << 4U));
  std::vector<uint8_t> weight(static_cast<std::size_t>(kWeightBytes), 0U);
  std::fill_n(weight.begin(), static_cast<std::size_t>(kPackedWeightBytes),
              packed);
  std::fill_n(weight.begin() + static_cast<std::ptrdiff_t>(kPackedWeightBytes),
              static_cast<std::size_t>(kWeightScaleBytes), UINT8_C(0x38));
  constexpr float kOne = 1.0F;
  std::memcpy(weight.data() + kWeightTensorScaleOffset, &kOne, sizeof(kOne));
  std::memcpy(weight.data() + kWeightTensorScaleOffset + sizeof(kOne), &kOne,
              sizeof(kOne));
  return weight;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (*buffer == nullptr)
    return true;
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "sllm_buffer_release", error);
}

struct MatmulRunResult final {
  bool valid = false;
  bool deterministic = true;
  uint32_t max_bf16_ulp = 0U;
  double median_ms = 0.0;
  std::vector<uint16_t> output;
  sllm_matmul_dispatch_info_t dispatch{};
};

sllm_matmul_desc_t
nvfp4_matmul_descriptor(const sllm_buffer_t *const activation,
                        const sllm_buffer_t *const weight,
                        const sllm_buffer_t *const output, const uint64_t k,
                        const uint64_t n, const uint64_t m = UINT64_C(1)) {
  sllm_matmul_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION;
  descriptor.activation = binding(activation, SLLM_TENSOR_DTYPE_BF16,
                                  SLLM_TENSOR_ENCODING_UNQUANTIZED, m, k);
  descriptor.weight =
      binding(weight, SLLM_TENSOR_DTYPE_U8,
              SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32, n, k);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16,
                              SLLM_TENSOR_ENCODING_UNQUANTIZED, m, n);
  return descriptor;
}

sllm_matmul_desc_t
fp8_outer_matmul_descriptor(const sllm_buffer_t *const activation,
                            const sllm_buffer_t *const weight,
                            const sllm_buffer_t *const output, const uint64_t k,
                            const uint64_t n, const uint64_t m = 1U) {
  sllm_matmul_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_MATMUL_FP8_VERSION;
  descriptor.activation = binding(activation, SLLM_TENSOR_DTYPE_BF16,
                                  SLLM_TENSOR_ENCODING_UNQUANTIZED, m, k);
  descriptor.weight = binding(weight, SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                              SLLM_TENSOR_ENCODING_FP8_OUTER_F32, n, k);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16,
                              SLLM_TENSOR_ENCODING_UNQUANTIZED, m, n);
  return descriptor;
}

std::vector<uint8_t> make_fp8_outer_weight(const uint64_t k, const uint64_t n,
                                           const uint8_t value,
                                           const float even_scale,
                                           const float odd_scale) {
  const uint64_t value_bytes = k * n;
  const uint64_t scale_bytes = n * sizeof(float);
  std::vector<uint8_t> weight(
      static_cast<std::size_t>(value_bytes + scale_bytes), value);
  for (uint64_t row = 0U; row != n; ++row) {
    const float scale = (row & 1U) == 0U ? even_scale : odd_scale;
    std::memcpy(weight.data() + value_bytes + row * sizeof(float), &scale,
                sizeof(scale));
  }
  return weight;
}

std::vector<uint16_t> fp8_gdn_expected_output(
    const uint64_t k, const uint64_t n, const float activation_value,
    const uint8_t weight_value, const float even_scale, const float odd_scale) {
  const float decoded_weight = weight_value == UINT8_C(0x38) ? 1.0F : -1.0F;
  std::vector<uint16_t> expected(static_cast<std::size_t>(n));
  for (uint64_t row = 0U; row != n; ++row) {
    const float scale = (row & 1U) == 0U ? even_scale : odd_scale;
    expected[static_cast<std::size_t>(row)] = f32_to_bf16_rne(
        activation_value * decoded_weight * scale * static_cast<float>(k));
  }
  return expected;
}

struct Fp8GdnMatmulRun final {
  bool valid = false;
  bool deterministic = true;
  uint32_t max_bf16_ulp = 0U;
  std::vector<uint16_t> output;
  sllm_matmul_dispatch_info_t dispatch{};
};

Fp8GdnMatmulRun run_fp8_gdn_matmul_plan(const sllm_matmul_plan_t *const plan,
                                        const sllm_queue_t *const queue,
                                        const sllm_buffer_t *const output,
                                        const uint64_t k, const uint64_t n,
                                        const std::vector<uint16_t> &expected,
                                        const uint64_t m = 1U) {
  constexpr std::size_t kRepeats = 3U;
  std::vector<uint16_t> first(static_cast<std::size_t>(m * n));
  std::vector<uint16_t> observed(static_cast<std::size_t>(m * n));
  Fp8GdnMatmulRun result{};
  result.valid = true;
  for (std::size_t repeat = 0U; repeat != kRepeats && result.valid; ++repeat) {
    sllm_matmul_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    Error error;
    result.valid =
        expect(sllm_matmul_execute(plan, queue, &completion, &dispatch,
                                   &error.sink),
               SLLM_STATUS_OK, "FP8 GDN direct matmul execute", error) &&
        completion != nullptr &&
        wait_and_release(&completion, "FP8 GDN direct completion");
    result.valid =
        result.valid && dispatch.dispatch_count == 2U &&
        dispatch.dispatch_id != 0U && dispatch.m == m && dispatch.k == k &&
        dispatch.n == n && dispatch.output_elements == m * n &&
        dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
        std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
    const bool gfx1030 = std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0;
    // M1 uses the retained forced-LDS control; new rows use default selectors.
    const uint32_t expected_provider =
        gfx1030 ? (m == 1U ? 82U : (m <= 4U ? 92U : 71U)) : 5U;
    result.valid = result.valid && dispatch.kernel_id == expected_provider;
    if (!result.valid) {
      std::cerr << "FP8 direct metadata failure M=" << m << " N=" << n
                << " provider=" << dispatch.kernel_id << " rows=" << dispatch.m
                << " count=" << dispatch.dispatch_count << '\n';
      break;
    }
    result.valid = download(queue, output, &observed);
    for (std::size_t index = 0U; index != observed.size(); ++index) {
      const uint16_t value = observed[index];
      const uint16_t expected_value = expected[index];
      const uint32_t ulp = value >= expected_value
                               ? static_cast<uint32_t>(value - expected_value)
                               : static_cast<uint32_t>(expected_value - value);
      result.max_bf16_ulp = std::max(result.max_bf16_ulp, ulp);
      if (value != expected_value) {
        std::cerr << "FP8 direct oracle mismatch M=" << m << " N=" << n
                  << " index=" << index << " actual=" << value
                  << " expected=" << expected_value << " ulp=" << ulp << '\n';
        result.valid = false;
        break;
      }
    }
    if (repeat == 0U) {
      first = observed;
      result.output = observed;
    } else if (observed != first) {
      result.valid = false;
      result.deterministic = false;
    }
    result.dispatch = dispatch;
  }
  return result;
}

MatmulRunResult run_matmul_plan(
    const sllm_matmul_plan_t *const plan, const sllm_queue_t *const queue,
    const sllm_buffer_t *const output, const uint64_t k, const uint64_t n,
    const uint32_t expected_kernel_id, const char *const expected_kernel,
    const char *const expected_device, const uint32_t expected_grid) {
  constexpr std::size_t kWarmups = 3U;
  constexpr std::size_t kMeasured = 10U;
  const uint16_t expected_value = f32_to_bf16_rne(static_cast<float>(k) * 6.0F);
  std::vector<uint16_t> first_output;
  std::vector<uint16_t> observed(static_cast<std::size_t>(n));
  std::vector<double> measured_ms;
  measured_ms.reserve(kMeasured);
  MatmulRunResult result{};
  result.valid = true;
  for (std::size_t iteration = 0U;
       iteration != kWarmups + kMeasured && result.valid; ++iteration) {
    sllm_matmul_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    Error error;
    const auto start = std::chrono::steady_clock::now();
    const bool submitted = expect(
        sllm_matmul_execute(plan, queue, &completion, &dispatch, &error.sink),
        SLLM_STATUS_OK, "sllm_matmul_execute", error);
    bool waited = false;
    if (completion != nullptr) {
      waited = wait_and_release(&completion, "sllm_matmul completion");
    }
    const auto finish = std::chrono::steady_clock::now();
    result.valid =
        submitted && waited && dispatch.dispatch_id != 0U &&
        dispatch.dispatch_count == 2U &&
        dispatch.kernel_id == expected_kernel_id &&
        dispatch.workgroup_size_x == 256U &&
        dispatch.grid_size_x == expected_grid && dispatch.m == 1U &&
        dispatch.k == k && dispatch.n == n && dispatch.output_elements == n &&
        dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
        std::strcmp(dispatch.kernel_symbol, expected_kernel) == 0 &&
        std::strcmp(dispatch.device_symbol, expected_device) == 0 &&
        std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
    if (!result.valid) {
      std::cerr << "NVFP4 matmul dispatch audit failed for kernel "
                << expected_kernel_id << " K=" << k << " N=" << n << '\n';
      break;
    }
    if (iteration >= kWarmups) {
      measured_ms.push_back(
          std::chrono::duration<double, std::milli>(finish - start).count());
      result.valid = download(queue, output, &observed) && result.valid;
      for (std::size_t index = 0U; index != observed.size(); ++index) {
        const uint16_t value = observed[index];
        const uint32_t ulp =
            value >= expected_value
                ? static_cast<uint32_t>(value - expected_value)
                : static_cast<uint32_t>(expected_value - value);
        result.max_bf16_ulp = std::max(result.max_bf16_ulp, ulp);
        if (value != expected_value) {
          std::cerr << "NVFP4 matmul numerical mismatch kernel="
                    << expected_kernel_id << " K=" << k << " N=" << n
                    << " iteration=" << iteration << " index=" << index
                    << " actual=0x" << std::hex << value << " expected=0x"
                    << expected_value << std::dec << " ulp=" << ulp << '\n';
          result.valid = false;
          break;
        }
      }
      if (iteration == kWarmups) {
        first_output = observed;
        result.output = observed;
      } else if (observed != first_output) {
        result.deterministic = false;
        result.valid = false;
        std::cerr << "NVFP4 matmul repeat is not bitwise deterministic: "
                  << "kernel=" << expected_kernel_id << " K=" << k << " N=" << n
                  << '\n';
      }
    }
    result.dispatch = dispatch;
  }
  if (!measured_ms.empty()) {
    std::sort(measured_ms.begin(), measured_ms.end());
    const std::size_t middle = measured_ms.size() / 2U;
    result.median_ms =
        measured_ms.size() % 2U == 0U
            ? (measured_ms[middle - 1U] + measured_ms[middle]) / 2.0
            : measured_ms[middle];
  }
  return result;
}

bool run_fp8_gdn_shared_public_gpu_oracle(const uint64_t m = 1U) {
  constexpr const char *kLdsLutEnvironment =
      "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT";
  const char *const old_lds_lut = std::getenv(kLdsLutEnvironment);
  const std::string old_lds_lut_value =
      old_lds_lut != nullptr ? old_lds_lut : "";
  if (m == 1U && std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0)
    setenv(kLdsLutEnvironment, "1", 1);
  else
    unsetenv(kLdsLutEnvironment);

  constexpr std::array<uint64_t, 2> kWidths = {kFp8GdnQkvN, kFp8GdnZN};
  std::array<uint64_t, 2> weight_bytes{};
  for (std::size_t index = 0U; index != kWidths.size(); ++index)
    weight_bytes[index] = kK * kWidths[index] + kWidths[index] * sizeof(float);
  // Distinct row magnitudes/signs expose a stale or misplaced row-scale plane.
  const auto row_factor = [](const uint64_t row) {
    constexpr std::array<float, 4> factors = {1.0F, 0.5F, 0.25F, -1.0F};
    return factors[row % factors.size()];
  };
  std::vector<uint16_t> activation(static_cast<std::size_t>(m * kK));
  for (uint64_t row = 0U; row != m; ++row)
    std::fill_n(activation.data() + row * kK, kK,
                f32_to_bf16_rne(6.0F * row_factor(row)));
  const auto expected_rows = [&](const uint64_t n, const float mean,
                                 const uint8_t weight, const float even,
                                 const float odd) {
    std::vector<uint16_t> result;
    result.reserve(static_cast<std::size_t>(m * n));
    for (uint64_t row = 0U; row != m; ++row) {
      const auto values = fp8_gdn_expected_output(kK, n, mean * row_factor(row),
                                                  weight, even, odd);
      result.insert(result.end(), values.begin(), values.end());
    }
    return result;
  };
  std::array<std::vector<uint8_t>, 2> weights = {
      make_fp8_outer_weight(kK, kWidths[0], UINT8_C(0x38), 0.5F, 1.0F),
      make_fp8_outer_weight(kK, kWidths[1], UINT8_C(0xb8), 1.0F, 0.5F)};
  const std::array<std::vector<uint16_t>, 2> expected_initial = {
      expected_rows(kWidths[0], 6.0F, UINT8_C(0x38), 0.5F, 1.0F),
      expected_rows(kWidths[1], 6.0F, UINT8_C(0xb8), 1.0F, 0.5F)};

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation_buffer = nullptr;
  std::array<sllm_buffer_t *, 2> weight_buffers{};
  std::array<sllm_buffer_t *, 2> direct_outputs{};
  std::array<sllm_buffer_t *, 2> shared_outputs{};
  std::array<sllm_matmul_plan_t *, 2> direct_plans{};
  sllm_qwen38_projection_pack2_plan_t *shared_plan = nullptr;
  bool valid = true;
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  valid = expect(sllm_context_create(&context_info, &context, &error.sink),
                 SLLM_STATUS_OK, "FP8 GDN context create", error);
  if (valid) {
    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "FP8 GDN queue create", error);
  }
  valid = valid &&
          create_buffer(context, m * kK * sizeof(uint16_t), &activation_buffer);
  for (std::size_t index = 0U; index != kWidths.size(); ++index) {
    valid = valid &&
            create_buffer(context, weight_bytes[index], &weight_buffers[index]);
    valid =
        valid && create_buffer(context, m * kWidths[index] * sizeof(uint16_t),
                               &direct_outputs[index]);
    valid =
        valid && create_buffer(context, m * kWidths[index] * sizeof(uint16_t),
                               &shared_outputs[index]);
  }
  valid = valid &&
          upload(queue, activation_buffer, activation.data(),
                 static_cast<uint64_t>(activation.size()) * sizeof(uint16_t));
  for (std::size_t index = 0U; index != kWidths.size(); ++index)
    valid = valid && upload(queue, weight_buffers[index], weights[index].data(),
                            static_cast<uint64_t>(weights[index].size()));

  if (valid) {
    for (std::size_t index = 0U; index != kWidths.size(); ++index) {
      const auto descriptor = fp8_outer_matmul_descriptor(
          activation_buffer, weight_buffers[index], direct_outputs[index], kK,
          kWidths[index], m);
      valid = expect(sllm_matmul_prepare(context, &descriptor,
                                         &direct_plans[index], &error.sink),
                     SLLM_STATUS_OK, "FP8 GDN direct matmul prepare", error) &&
              valid;
    }
    sllm_qwen38_projection_pack2_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_QWEN38_PROJECTION_PACK2_VERSION;
    descriptor.role = SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_FP8_GDN_QKV_Z;
    descriptor.input_global_scale_f32_bits = 0U;
    descriptor.activation = binding(activation_buffer, SLLM_TENSOR_DTYPE_BF16,
                                    SLLM_TENSOR_ENCODING_UNQUANTIZED, m, kK);
    descriptor.gate_weight =
        binding(weight_buffers[0], SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                SLLM_TENSOR_ENCODING_FP8_OUTER_F32, kWidths[0], kK);
    descriptor.up_weight =
        binding(weight_buffers[1], SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                SLLM_TENSOR_ENCODING_FP8_OUTER_F32, kWidths[1], kK);
    descriptor.gate_output =
        binding(shared_outputs[0], SLLM_TENSOR_DTYPE_BF16,
                SLLM_TENSOR_ENCODING_UNQUANTIZED, m, kWidths[0]);
    descriptor.up_output =
        binding(shared_outputs[1], SLLM_TENSOR_DTYPE_BF16,
                SLLM_TENSOR_ENCODING_UNQUANTIZED, m, kWidths[1]);
    valid = expect(sllm_qwen38_projection_pack2_prepare(
                       context, &descriptor, &shared_plan, &error.sink),
                   SLLM_STATUS_OK, "FP8 GDN shared prepare", error) &&
            valid;
  }

  std::array<Fp8GdnMatmulRun, 2> direct_runs{};
  if (valid) {
    for (std::size_t index = 0U; index != kWidths.size(); ++index) {
      direct_runs[index] = run_fp8_gdn_matmul_plan(
          direct_plans[index], queue, direct_outputs[index], kK, kWidths[index],
          expected_initial[index], m);
      valid = valid && direct_runs[index].valid &&
              direct_runs[index].deterministic &&
              direct_runs[index].max_bf16_ulp == 0U;
    }
  }

  std::array<std::vector<uint16_t>, 2> shared_first{};
  uint32_t shared_max_ulp = 0U;
  bool shared_deterministic = true;
  sllm_qwen38_projection_pack2_dispatch_info_t last_shared_dispatch{};
  if (valid) {
    for (std::size_t repeat = 0U; repeat != 3U && valid; ++repeat) {
      sllm_qwen38_projection_pack2_dispatch_info_t dispatch{};
      dispatch.struct_size = sizeof(dispatch);
      dispatch.abi_version = SLLM_HIP_ABI_VERSION;
      dispatch.info_version =
          SLLM_HIP_QWEN38_PROJECTION_PACK2_DISPATCH_INFO_VERSION;
      sllm_completion_t *completion = nullptr;
      valid =
          expect(sllm_qwen38_projection_pack2_execute(
                     shared_plan, queue, &completion, &dispatch, &error.sink),
                 SLLM_STATUS_OK, "FP8 GDN shared execute", error) &&
          completion != nullptr &&
          wait_and_release(&completion, "FP8 GDN shared completion");
      valid =
          valid && dispatch.dispatch_id != 0U &&
          dispatch.dispatch_count == 3U &&
          dispatch.kernel_id ==
              SLLM_HIP_QWEN38_PROJECTION_PACK2_KERNEL_ID_FP8_GDN_SHARED_ACTIVATION_V1 &&
          dispatch.workgroup_size_x == 256U && dispatch.grid_size_x != 0U &&
          dispatch.m == m && dispatch.k == kK && dispatch.n == kWidths[0] &&
          dispatch.output_elements == m * (kWidths[0] + kWidths[1]) &&
          dispatch.workspace_bytes ==
              m * SLLM_HIP_QWEN38_PROJECTION_PACK2_FP8_GDN_WORKSPACE_BYTES &&
          dispatch.role ==
              SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_FP8_GDN_QKV_Z &&
          dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
          std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
      for (std::size_t index = 0U; index != kWidths.size() && valid; ++index) {
        std::vector<uint16_t> observed(
            static_cast<std::size_t>(m * kWidths[index]));
        valid = download(queue, shared_outputs[index], &observed) && valid;
        for (std::size_t output_index = 0U; output_index != observed.size();
             ++output_index) {
          const uint16_t value = observed[output_index];
          const uint16_t expected = expected_initial[index][output_index];
          const uint32_t ulp = value >= expected
                                   ? static_cast<uint32_t>(value - expected)
                                   : static_cast<uint32_t>(expected - value);
          shared_max_ulp = std::max(shared_max_ulp, ulp);
          if (value != expected) {
            std::cerr << "FP8 pair oracle mismatch M=" << m
                      << " member=" << index << " index=" << output_index
                      << " actual=" << value << " expected=" << expected
                      << " ulp=" << ulp << '\n';
            valid = false;
            break;
          }
        }
        if (repeat == 0U)
          shared_first[index] = observed;
        else if (observed != shared_first[index])
          shared_deterministic = false;
        if (observed != direct_runs[index].output) {
          std::cerr << "FP8 GDN shared/direct output mismatch member=" << index
                    << '\n';
          valid = false;
        }
      }
      last_shared_dispatch = dispatch;
    }
    valid = valid && shared_deterministic;
  }

  // Change the request input after the initial repeats and compare one more
  // shared/direct execution. This catches accidental reuse of stale quantized
  // bytes or row scale while preserving the same prepared plans/providers.
  if (valid) {
    for (std::size_t index = 0U; index != activation.size(); ++index) {
      activation[index] = f32_to_bf16_rne(((index & 1U) == 0U ? 3.0F : 1.5F) *
                                          row_factor(index / kK));
    }
    valid = upload(queue, activation_buffer, activation.data(),
                   static_cast<uint64_t>(activation.size()) * sizeof(uint16_t));
    std::array<std::vector<uint16_t>, 2> changed_direct{};
    const std::array<std::vector<uint16_t>, 2> expected_changed = {
        expected_rows(kWidths[0], 2.25F, UINT8_C(0x38), 0.5F, 1.0F),
        expected_rows(kWidths[1], 2.25F, UINT8_C(0xb8), 1.0F, 0.5F)};
    for (std::size_t index = 0U; index != kWidths.size() && valid; ++index) {
      sllm_matmul_dispatch_info_t dispatch{};
      dispatch.struct_size = sizeof(dispatch);
      dispatch.abi_version = SLLM_HIP_ABI_VERSION;
      dispatch.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
      sllm_completion_t *completion = nullptr;
      valid =
          expect(sllm_matmul_execute(direct_plans[index], queue, &completion,
                                     &dispatch, &error.sink),
                 SLLM_STATUS_OK, "FP8 GDN changed direct execute", error) &&
          completion != nullptr &&
          wait_and_release(&completion, "FP8 GDN changed direct completion");
      changed_direct[index].resize(
          static_cast<std::size_t>(m * kWidths[index]));
      valid = download(queue, direct_outputs[index], &changed_direct[index]) &&
              valid;
      valid = changed_direct[index] == expected_changed[index] && valid;
    }
    sllm_qwen38_projection_pack2_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version =
        SLLM_HIP_QWEN38_PROJECTION_PACK2_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    valid =
        expect(sllm_qwen38_projection_pack2_execute(
                   shared_plan, queue, &completion, &dispatch, &error.sink),
               SLLM_STATUS_OK, "FP8 GDN changed shared execute", error) &&
        completion != nullptr &&
        wait_and_release(&completion, "FP8 GDN changed shared completion") &&
        valid;
    for (std::size_t index = 0U; index != kWidths.size() && valid; ++index) {
      std::vector<uint16_t> observed(
          static_cast<std::size_t>(m * kWidths[index]));
      valid = download(queue, shared_outputs[index], &observed) && valid;
      valid = observed == changed_direct[index] &&
              observed != shared_first[index] &&
              observed == expected_changed[index] && valid;
    }
  }

  /* Capture the single fused pack on the deferred request queue.  The graph
   * must contain the shared quantizer plus both member matmuls, and replay
   * must observe a new activation while retaining bitwise output stability. */
  sllm_graph_span_t *graph = nullptr;
  uint64_t graph_node_count = 0U;
  bool graph_valid = true;
  if (valid && m == 1U) {
    graph_valid =
        expect(sllm_queue_set_completion_mode(
                   queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &error.sink),
               SLLM_STATUS_OK, "FP8 GDN graph deferred mode", error);
    const std::array<const void *, 1U> handles = {
        reinterpret_cast<const void *>(shared_plan)};
    graph_valid =
        graph_valid &&
        expect(sllm_graph_span_create(queue, handles.data(), handles.size(),
                                      &graph, &error.sink),
               SLLM_STATUS_OK, "FP8 GDN graph create", error);
    graph_valid = graph_valid && graph != nullptr &&
                  expect(sllm_graph_span_node_count(graph, &graph_node_count,
                                                    &error.sink),
                         SLLM_STATUS_OK, "FP8 GDN graph node count", error) &&
                  graph_node_count == 3U;
    if (graph_node_count != 3U) {
      std::cerr << "FP8 GDN graph node count expected 3, got "
                << graph_node_count << '\n';
    }
    std::vector<uint16_t> graph_activation(static_cast<std::size_t>(kK));
    std::fill(graph_activation.begin(), graph_activation.end(),
              f32_to_bf16_rne(1.5F));
    const std::array<std::vector<uint16_t>, 2> graph_expected = {
        fp8_gdn_expected_output(kK, kWidths[0], 1.5F, UINT8_C(0x38), 0.5F,
                                1.0F),
        fp8_gdn_expected_output(kK, kWidths[1], 1.5F, UINT8_C(0xb8), 1.0F,
                                0.5F)};
    std::array<std::vector<uint16_t>, 2> graph_first{};
    std::array<std::vector<uint16_t>, 2> graph_second{};
    graph_valid = graph_valid &&
                  execute_fp8_gdn_graph_once(queue, graph, activation_buffer,
                                             graph_activation, shared_outputs,
                                             &graph_first);
    graph_valid = graph_valid &&
                  execute_fp8_gdn_graph_once(queue, graph, activation_buffer,
                                             graph_activation, shared_outputs,
                                             &graph_second);
    for (std::size_t index = 0U; index != kWidths.size(); ++index) {
      graph_valid = graph_valid &&
                    graph_first[index] == graph_expected[index] &&
                    graph_second[index] == graph_expected[index] &&
                    graph_first[index] == graph_second[index];
    }
    valid = graph_valid && valid;
  }
  if (graph != nullptr) {
    valid = expect(sllm_graph_span_release(&graph, &error.sink), SLLM_STATUS_OK,
                   "FP8 GDN graph release", error) &&
            valid;
  }

  if (shared_plan != nullptr)
    valid = expect(sllm_qwen38_projection_pack2_plan_release(&shared_plan,
                                                             &error.sink),
                   SLLM_STATUS_OK, "FP8 GDN shared plan release", error) &&
            valid;
  for (sllm_matmul_plan_t *&plan : direct_plans) {
    if (plan != nullptr)
      valid = expect(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "FP8 GDN direct plan release", error) &&
              valid;
  }
  for (sllm_buffer_t *&buffer : shared_outputs)
    valid = release_buffer(&buffer) && valid;
  for (sllm_buffer_t *&buffer : direct_outputs)
    valid = release_buffer(&buffer) && valid;
  for (sllm_buffer_t *&buffer : weight_buffers)
    valid = release_buffer(&buffer) && valid;
  valid = release_buffer(&activation_buffer) && valid;
  if (queue != nullptr)
    valid = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "FP8 GDN queue release", error) &&
            valid;
  if (context != nullptr)
    valid = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "FP8 GDN context release", error) &&
            valid;
  if (old_lds_lut != nullptr)
    setenv(kLdsLutEnvironment, old_lds_lut_value.c_str(), 1);
  else
    unsetenv(kLdsLutEnvironment);
  if (!valid)
    return false;
  std::cout << "Qwen3.8 FP8 GDN shared projection-pack oracle: PASS target="
            << last_shared_dispatch.gcn_arch_name << " M=" << m
            << " dispatch_count=" << last_shared_dispatch.dispatch_count
            << " direct_dispatch_count_total=4"
            << " deterministic=" << shared_deterministic
            << " max_bf16_ulp=" << shared_max_ulp
            << " workspace=" << last_shared_dispatch.workspace_bytes
            << " cleanup=0\n";
  return true;
}

bool run_activation_shared_matmul_oracle(const bool default_lut = false) {
  if (!default_lut && std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") != 0)
    return true;
  constexpr const char *kBaselineEnvironment = "SLLM_NVFP4_W4A4_FORCE_BASELINE";
  constexpr const char *kControlEnvironment =
      "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4";
  constexpr const char *kCandidateEnvironment =
      "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_ACTIVATION_SHARED";
  constexpr const char *kLutEnvironment =
      "SLLM_NVFP4_W4A4_DECODE_FORCE_LDS_F32_LUT";
  constexpr std::array<const char *, 4> kEnvironments = {
      kBaselineEnvironment, kControlEnvironment, kCandidateEnvironment,
      kLutEnvironment};
  std::array<bool, kEnvironments.size()> was_present{};
  std::array<std::string, kEnvironments.size()> old_values{};
  for (std::size_t index = 0U; index != kEnvironments.size(); ++index) {
    const char *const value = std::getenv(kEnvironments[index]);
    was_present[index] = value != nullptr;
    old_values[index] = value != nullptr ? value : "";
    unsetenv(kEnvironments[index]);
  }
  const auto restore = [&]() {
    for (std::size_t index = 0U; index != kEnvironments.size(); ++index) {
      if (was_present[index]) {
        setenv(kEnvironments[index], old_values[index].c_str(), 1);
      } else {
        unsetenv(kEnvironments[index]);
      }
    }
  };

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  bool valid = true;
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  valid = expect(sllm_context_create(&context_info, &context, &error.sink),
                 SLLM_STATUS_OK, "ID73 context create", error);
  if (valid) {
    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "ID73 queue create", error);
  }

  constexpr std::array<std::array<uint64_t, 2>, 2> kShapes = {
      std::array<uint64_t, 2>{5120U, 17408U},
      std::array<uint64_t, 2>{17408U, 5120U}};
  for (const auto &shape : kShapes) {
    if (!valid)
      break;
    const uint64_t k = shape[0];
    const uint64_t n = shape[1];
    const uint64_t value_bytes = n * k / UINT64_C(2);
    const uint64_t scale_bytes = n * (k / UINT64_C(16));
    const uint64_t tensor_scale_offset =
        (value_bytes + scale_bytes + UINT64_C(3)) & ~UINT64_C(3);
    const uint64_t weight_bytes = tensor_scale_offset + UINT64_C(8);
    if (weight_bytes != kWeightBytes) {
      std::cerr << "unexpected symmetric Qwen3.8 weight size\n";
      valid = false;
      break;
    }
    std::vector<uint16_t> activation(static_cast<std::size_t>(k),
                                     f32_to_bf16_rne(6.0F));
    std::vector<uint8_t> weight = make_weight(UINT8_C(0x2));
    sllm_buffer_t *activation_buffer = nullptr;
    sllm_buffer_t *weight_buffer = nullptr;
    sllm_buffer_t *output_buffer = nullptr;
    sllm_matmul_plan_t *control_plan = nullptr;
    sllm_matmul_plan_t *candidate_plan = nullptr;
    valid =
        create_buffer(context, k * sizeof(uint16_t), &activation_buffer) &&
        create_buffer(context, weight_bytes, &weight_buffer) &&
        create_buffer(context, n * sizeof(uint16_t), &output_buffer) &&
        upload(queue, activation_buffer, activation.data(),
               static_cast<uint64_t>(activation.size()) * sizeof(uint16_t)) &&
        upload(queue, weight_buffer, weight.data(), weight_bytes);
    const sllm_matmul_desc_t descriptor = nvfp4_matmul_descriptor(
        activation_buffer, weight_buffer, output_buffer, k, n);
    if (valid) {
      setenv(kControlEnvironment, "1", 1);
      unsetenv(kCandidateEnvironment);
      valid = expect(
          sllm_matmul_prepare(context, &descriptor, &control_plan, &error.sink),
          SLLM_STATUS_OK, "ID67 control prepare", error);
    }
    if (valid) {
      if (default_lut) {
        unsetenv(kControlEnvironment);
        unsetenv(kCandidateEnvironment);
      } else {
        setenv(kCandidateEnvironment, "1", 1);
      }
      valid = expect(sllm_matmul_prepare(context, &descriptor, &candidate_plan,
                                         &error.sink),
                     SLLM_STATUS_OK, "ID73 candidate prepare", error);
    }

    MatmulRunResult control{};
    MatmulRunResult candidate{};
    if (valid) {
      control = run_matmul_plan(
          control_plan, queue, output_buffer, k, n, 67U,
          "matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1",
          "sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_v1",
          static_cast<uint32_t>((n + UINT64_C(31)) / UINT64_C(32)));
      candidate = run_matmul_plan(
          candidate_plan, queue, output_buffer, k, n, default_lut ? 84U : 73U,
          default_lut
              ? "matmul.nvfp4.w4a4.decode.scale_lut.v1"
              : "matmul.nvfp4.w4a4.decode.dp4a.activation_shared.wave4col32.v1",
          default_lut
              ? (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") == 0
                     ? "sllm_nvfp4_w4a4_decode_scale_lut_gfx1201_actshared_v1"
                     : "sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1")
              : "sllm_matmul_nvfp4_w4a4_decode_dp4a_activation_shared_v1",
          static_cast<uint32_t>((n + UINT64_C(31)) / UINT64_C(32)));
      valid = control.valid && candidate.valid && control.deterministic &&
              candidate.deterministic && control.max_bf16_ulp == 0U &&
              candidate.max_bf16_ulp == 0U &&
              control.output == candidate.output;
      if (control.output != candidate.output) {
        std::cerr << "ID73 output is not bitwise equal to ID67: K=" << k
                  << " N=" << n << '\n';
      }
      std::cout << std::fixed << std::setprecision(6)
                << (default_lut ? "ID84 default" : "ID73 forced")
                << " public Matmul oracle shape K=" << k << " N=" << n
                << " warmups=3 measured=10 control_id67_median_ms="
                << control.median_ms
                << " candidate_median_ms=" << candidate.median_ms << " speedup="
                << (candidate.median_ms > 0.0
                        ? control.median_ms / candidate.median_ms
                        : 0.0)
                << " bitwise=PASS deterministic=PASS max_bf16_ulp=0"
                << " dispatch_count=2 lds_bytes=" << (k * UINT64_C(5) / 4U)
                << '\n';
    }
    if (candidate_plan != nullptr) {
      valid = expect(sllm_matmul_plan_release(&candidate_plan, &error.sink),
                     SLLM_STATUS_OK, "ID73 candidate plan release", error) &&
              valid;
    }
    if (control_plan != nullptr) {
      valid = expect(sllm_matmul_plan_release(&control_plan, &error.sink),
                     SLLM_STATUS_OK, "ID67 control plan release", error) &&
              valid;
    }
    valid = release_buffer(&output_buffer) && valid;
    valid = release_buffer(&weight_buffer) && valid;
    valid = release_buffer(&activation_buffer) && valid;
  }
  if (queue != nullptr) {
    valid = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "ID73 queue release", error) &&
            valid;
  }
  if (context != nullptr) {
    valid = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "ID73 context release", error) &&
            valid;
  }
  restore();
  return valid;
}

bool run_nvfp4_projection_pack_queue_public_gpu_oracle() {
  constexpr uint64_t k = UINT64_C(5120);
  constexpr uint64_t n = UINT64_C(17408);
  constexpr uint64_t max_m = UINT64_C(512);
  constexpr uint64_t per_row_workspace = UINT64_C(2880);
  constexpr std::array<uint64_t, 10> rows = {
      UINT64_C(2),  UINT64_C(3),   UINT64_C(4),   UINT64_C(5),   UINT64_C(64),
      UINT64_C(65), UINT64_C(127), UINT64_C(128), UINT64_C(129), UINT64_C(512)};
  const char *const compensated_environment =
      "SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED";
  const char *const wmma_environment =
      "SLLM_NVFP4_W4A4_PREFILL_FORCE_WMMA_COMPENSATED";
  const char *const old_compensated = std::getenv(compensated_environment);
  const char *const old_wmma = std::getenv(wmma_environment);
  const std::string old_compensated_value =
      old_compensated != nullptr ? old_compensated : "";
  const std::string old_wmma_value = old_wmma != nullptr ? old_wmma : "";
  const bool had_compensated = old_compensated != nullptr;
  const bool had_wmma = old_wmma != nullptr;
  const auto restore_environment = [&]() {
    if (had_compensated)
      setenv(compensated_environment, old_compensated_value.c_str(), 1);
    else
      unsetenv(compensated_environment);
    if (had_wmma)
      setenv(wmma_environment, old_wmma_value.c_str(), 1);
    else
      unsetenv(wmma_environment);
  };
  unsetenv(compensated_environment);
  unsetenv(wmma_environment);
  if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0)
    setenv(compensated_environment, "1", 1);
  else
    setenv(wmma_environment, "1", 1);

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation_buffer = nullptr;
  std::array<sllm_buffer_t *, 2> weight_buffers{};
  std::array<sllm_buffer_t *, 2> direct_outputs{};
  std::array<sllm_buffer_t *, 2> shared_outputs{};
  bool valid = true;
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  valid = expect(sllm_context_create(&context_info, &context, &error.sink),
                 SLLM_STATUS_OK, "NVFP4 pair context create", error);
  if (valid) {
    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "NVFP4 pair queue create", error);
  }
  valid = valid && create_buffer(context, max_m * k * sizeof(uint16_t),
                                 &activation_buffer);
  valid = valid && create_buffer(context, kWeightBytes, &weight_buffers[0]);
  valid = valid && create_buffer(context, kWeightBytes, &weight_buffers[1]);
  valid = valid && create_buffer(context, max_m * n * sizeof(uint16_t),
                                 &direct_outputs[0]);
  valid = valid && create_buffer(context, max_m * n * sizeof(uint16_t),
                                 &direct_outputs[1]);
  valid = valid && create_buffer(context, max_m * n * sizeof(uint16_t),
                                 &shared_outputs[0]);
  valid = valid && create_buffer(context, max_m * n * sizeof(uint16_t),
                                 &shared_outputs[1]);
  std::vector<uint16_t> activation(static_cast<std::size_t>(max_m * k),
                                   f32_to_bf16_rne(6.0F));
  const std::array<std::vector<uint8_t>, 2> weights = {
      make_weight(UINT8_C(0x2)), make_weight(UINT8_C(0x1))};
  if (valid) {
    valid =
        upload(queue, activation_buffer, activation.data(),
               static_cast<uint64_t>(activation.size()) * sizeof(uint16_t)) &&
        upload(queue, weight_buffers[0], weights[0].data(),
               weights[0].size()) &&
        upload(queue, weight_buffers[1], weights[1].data(), weights[1].size());
  }
  if (!valid)
    std::cerr << "NVFP4 pair setup/upload failed target="
              << SLLM_TEST_EXPECTED_TARGET << '\n';

  const auto make_pair_descriptor = [&](const uint64_t rows_value) {
    sllm_qwen38_projection_pack2_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_QWEN38_PROJECTION_PACK2_VERSION;
    descriptor.role = SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_NVFP4_MLP_GATE_UP;
    descriptor.input_global_scale_f32_bits = UINT32_C(0x3f800000);
    descriptor.activation =
        binding(activation_buffer, SLLM_TENSOR_DTYPE_BF16,
                SLLM_TENSOR_ENCODING_UNQUANTIZED, rows_value, k);
    descriptor.gate_weight =
        binding(weight_buffers[0], SLLM_TENSOR_DTYPE_U8,
                SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32, n, k);
    descriptor.up_weight =
        binding(weight_buffers[1], SLLM_TENSOR_DTYPE_U8,
                SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32, n, k);
    descriptor.gate_output =
        binding(shared_outputs[0], SLLM_TENSOR_DTYPE_BF16,
                SLLM_TENSOR_ENCODING_UNQUANTIZED, rows_value, n);
    descriptor.up_output =
        binding(shared_outputs[1], SLLM_TENSOR_DTYPE_BF16,
                SLLM_TENSOR_ENCODING_UNQUANTIZED, rows_value, n);
    return descriptor;
  };
  const auto nvfp4_row_activation = [](const uint64_t rows_value,
                                       const uint64_t row) {
    constexpr std::array<float, 4> small_row_values = {6.0F, 3.0F, 1.5F, 0.0F};
    return rows_value >= UINT64_C(2) && rows_value <= UINT64_C(4)
               ? small_row_values[row]
               : 6.0F;
  };
  const auto upload_nvfp4_activation = [&](const uint64_t rows_value) {
    for (uint64_t row = 0U; row != rows_value; ++row)
      std::fill_n(activation.data() + row * k, k,
                  f32_to_bf16_rne(nvfp4_row_activation(rows_value, row)));
    return upload(queue, activation_buffer, activation.data(),
                  rows_value * k * sizeof(uint16_t));
  };

  for (const uint64_t m : rows) {
    if (!valid)
      break;
    const bool small_m = m >= UINT64_C(2) && m <= UINT64_C(4);
    /* M=2..4 use the adopted default ID94 provider.  Large rows retain the
     * per-target adopted controls already covered by this queue oracle. */
    if (small_m) {
      unsetenv(compensated_environment);
      unsetenv(wmma_environment);
    } else if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0) {
      setenv(compensated_environment, "1", 1);
      unsetenv(wmma_environment);
    } else {
      unsetenv(compensated_environment);
      setenv(wmma_environment, "1", 1);
    }

    if (m == UINT64_C(5)) {
      sllm_qwen38_projection_pack2_plan_t *unsupported_plan = nullptr;
      const auto descriptor = make_pair_descriptor(m);
      const sllm_status_t status = sllm_qwen38_projection_pack2_prepare(
          context, &descriptor, &unsupported_plan, &error.sink);
      if (status != SLLM_STATUS_UNSUPPORTED || unsupported_plan != nullptr) {
        std::cerr << "NVFP4 pair M=5 prepare contract failed target="
                  << SLLM_TEST_EXPECTED_TARGET << " status=" << status
                  << " plan=" << (unsupported_plan != nullptr) << '\n';
        if (unsupported_plan != nullptr)
          (void)sllm_qwen38_projection_pack2_plan_release(&unsupported_plan,
                                                          &error.sink);
        valid = false;
      } else {
        std::cout << "NVFP4 pair prepare M=5: UNSUPPORTED target="
                  << SLLM_TEST_EXPECTED_TARGET << '\n';
      }
      continue;
    }
    valid = upload_nvfp4_activation(m) && valid;
    if (!valid)
      break;

    std::array<sllm_matmul_plan_t *, 2> direct_plans{};
    sllm_qwen38_projection_pack2_plan_t *shared_plan = nullptr;
    std::array<std::vector<uint16_t>, 2> direct_first{};
    std::array<std::vector<uint16_t>, 2> expected_output{};
    for (std::size_t member = 0U; member != expected_output.size(); ++member) {
      expected_output[member].resize(static_cast<std::size_t>(m * n));
      const float base = member == 0U ? 30720.0F : 15360.0F;
      for (uint64_t row = 0U; row != m; ++row) {
        const float row_scale = nvfp4_row_activation(m, row) / 6.0F;
        std::fill_n(expected_output[member].data() + row * n, n,
                    f32_to_bf16_rne(base * row_scale));
      }
    }
    for (std::size_t member = 0U; member != direct_plans.size() && valid;
         ++member) {
      const auto descriptor =
          nvfp4_matmul_descriptor(activation_buffer, weight_buffers[member],
                                  direct_outputs[member], k, n, m);
      valid = expect(sllm_matmul_prepare(context, &descriptor,
                                         &direct_plans[member], &error.sink),
                     SLLM_STATUS_OK, "NVFP4 pair direct matmul prepare", error);
      if (!valid)
        std::cerr << "NVFP4 pair direct prepare failed target="
                  << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                  << " member=" << member << '\n';
    }
    if (valid) {
      const auto descriptor = make_pair_descriptor(m);
      valid = expect(sllm_qwen38_projection_pack2_prepare(
                         context, &descriptor, &shared_plan, &error.sink),
                     SLLM_STATUS_OK,
                     "NVFP4 pair shared projection-pack prepare", error);
      if (!valid)
        std::cerr << "NVFP4 pair shared prepare failed target="
                  << SLLM_TEST_EXPECTED_TARGET << " row=" << m << '\n';
    }
    for (std::size_t member = 0U; member != direct_plans.size() && valid;
         ++member) {
      for (std::size_t repeat = 0U; repeat != 2U && valid; ++repeat) {
        sllm_matmul_dispatch_info_t dispatch{};
        dispatch.struct_size = sizeof(dispatch);
        dispatch.abi_version = SLLM_HIP_ABI_VERSION;
        dispatch.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
        sllm_completion_t *completion = nullptr;
        valid =
            expect(sllm_matmul_execute(direct_plans[member], queue, &completion,
                                       &dispatch, &error.sink),
                   SLLM_STATUS_OK, "NVFP4 pair direct matmul execute", error) &&
            completion != nullptr &&
            wait_and_release(&completion, "NVFP4 pair direct matmul wait") &&
            dispatch.m == m && dispatch.k == k && dispatch.n == n &&
            dispatch.output_elements == m * n &&
            dispatch.workgroup_size_x == 256U &&
            (small_m ? dispatch.grid_size_x == ((n + 31U) / 32U) : true) &&
            (small_m
                 ? dispatch.kernel_id == 94U
                 : dispatch.kernel_id ==
                       (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0
                            ? 87U
                            : 89U)) &&
            dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U;
        if (!valid)
          std::cerr << "NVFP4 pair direct execute failed target="
                    << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                    << " member=" << member << " repeat=" << repeat
                    << " dispatch(id,m,k,n,out,fallback,used)="
                    << dispatch.kernel_id << ',' << dispatch.m << ','
                    << dispatch.k << ',' << dispatch.n << ','
                    << dispatch.output_elements << ','
                    << dispatch.fallback_allowed << ','
                    << dispatch.fallback_used << '\n';
        std::vector<uint16_t> observed(static_cast<std::size_t>(m * n));
        if (valid && !download(queue, direct_outputs[member], &observed)) {
          valid = false;
          std::cerr << "NVFP4 pair direct download failed target="
                    << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                    << " member=" << member << " repeat=" << repeat << '\n';
        }
        if (valid) {
          for (std::size_t index = 0U; index != observed.size(); ++index) {
            if (observed[index] != expected_output[member][index]) {
              std::cerr << "NVFP4 pair direct oracle mismatch target="
                        << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                        << " member=" << member << " repeat=" << repeat
                        << " index=" << index << " actual=0x" << std::hex
                        << observed[index] << " expected=0x"
                        << expected_output[member][index] << std::dec << '\n';
              valid = false;
              break;
            }
          }
        }
        if (repeat == 0U)
          direct_first[member] = std::move(observed);
        else if (valid && direct_first[member] != observed) {
          std::cerr << "NVFP4 pair direct repeat mismatch target="
                    << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                    << " member=" << member << '\n';
          valid = false;
        }
      }
    }
    for (std::size_t repeat = 0U; repeat != 2U && valid; ++repeat) {
      sllm_qwen38_projection_pack2_dispatch_info_t dispatch{};
      dispatch.struct_size = sizeof(dispatch);
      dispatch.abi_version = SLLM_HIP_ABI_VERSION;
      dispatch.info_version =
          SLLM_HIP_QWEN38_PROJECTION_PACK2_DISPATCH_INFO_VERSION;
      sllm_completion_t *completion = nullptr;
      valid =
          expect(sllm_qwen38_projection_pack2_execute(
                     shared_plan, queue, &completion, &dispatch, &error.sink),
                 SLLM_STATUS_OK, "NVFP4 pair shared projection-pack execute",
                 error) &&
          completion != nullptr &&
          wait_and_release(&completion,
                           "NVFP4 pair shared projection-pack wait") &&
          dispatch.dispatch_count == 3U && dispatch.m == m && dispatch.k == k &&
          dispatch.n == n && dispatch.output_elements == 2U * m * n &&
          dispatch.workspace_bytes == m * per_row_workspace &&
          dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U;
      const uint64_t quantize_grid = (m * (k / UINT64_C(16)) + 7U) / 8U;
      const uint64_t row_tile =
          std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0 &&
                  m >= UINT64_C(512) && (m % UINT64_C(128)) == 0U
              ? UINT64_C(128)
              : (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1030") == 0
                     ? UINT64_C(64)
                     : UINT64_C(128));
      const uint64_t projection_grid =
          small_m ? ((n + 31U) / 32U)
                  : ((m + row_tile - 1U) / row_tile) * ((n + 63U) / 64U);
      const uint64_t expected_grid = quantize_grid + 2U * projection_grid;
      if (valid && (quantize_grid != m * UINT64_C(40) ||
                    dispatch.grid_size_x != expected_grid)) {
        std::cerr << "NVFP4 pair shared grid mismatch target="
                  << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                  << " repeat=" << repeat << " actual=" << dispatch.grid_size_x
                  << " expected=" << expected_grid
                  << " quantize=" << quantize_grid
                  << " projection=" << projection_grid << '\n';
        valid = false;
      }
      if (!valid)
        std::cerr << "NVFP4 pair shared execute failed target="
                  << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                  << " repeat=" << repeat
                  << " dispatch(count,m,k,n,out,w)=" << dispatch.dispatch_count
                  << ',' << dispatch.m << ',' << dispatch.k << ',' << dispatch.n
                  << ',' << dispatch.output_elements << ','
                  << dispatch.workspace_bytes
                  << " fallback=" << dispatch.fallback_allowed << ','
                  << dispatch.fallback_used << '\n';
      for (std::size_t member = 0U; member != shared_outputs.size() && valid;
           ++member) {
        std::vector<uint16_t> observed(static_cast<std::size_t>(m * n));
        if (!download(queue, shared_outputs[member], &observed)) {
          std::cerr << "NVFP4 pair shared download failed target="
                    << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                    << " repeat=" << repeat << " member=" << member << '\n';
          valid = false;
          break;
        }
        if (observed != direct_first[member]) {
          std::size_t mismatch = 0U;
          while (mismatch != observed.size() &&
                 observed[mismatch] == direct_first[member][mismatch])
            ++mismatch;
          std::cerr << "NVFP4 pair shared/control mismatch target="
                    << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                    << " repeat=" << repeat << " member=" << member
                    << " index=" << mismatch;
          if (mismatch != observed.size())
            std::cerr << " actual=0x" << std::hex << observed[mismatch]
                      << " control=0x" << direct_first[member][mismatch]
                      << std::dec;
          std::cerr << '\n';
          valid = false;
        }
        if (valid) {
          for (std::size_t index = 0U; index != observed.size(); ++index) {
            if (observed[index] != expected_output[member][index]) {
              std::cerr << "NVFP4 pair shared oracle mismatch target="
                        << SLLM_TEST_EXPECTED_TARGET << " row=" << m
                        << " repeat=" << repeat << " member=" << member
                        << " index=" << index << " actual=0x" << std::hex
                        << observed[index] << " expected=0x"
                        << expected_output[member][index] << std::dec << '\n';
              valid = false;
              break;
            }
          }
        }
      }
    }
    if (shared_plan != nullptr)
      valid = expect(sllm_qwen38_projection_pack2_plan_release(&shared_plan,
                                                               &error.sink),
                     SLLM_STATUS_OK,
                     "NVFP4 pair shared projection-pack release", error) &&
              valid;
    for (sllm_matmul_plan_t *&plan : direct_plans) {
      if (plan != nullptr)
        valid =
            expect(sllm_matmul_plan_release(&plan, &error.sink), SLLM_STATUS_OK,
                   "NVFP4 pair direct matmul release", error) &&
            valid;
    }
  }

  for (sllm_buffer_t *&buffer : shared_outputs)
    valid = release_buffer(&buffer) && valid;
  for (sllm_buffer_t *&buffer : direct_outputs)
    valid = release_buffer(&buffer) && valid;
  for (sllm_buffer_t *&buffer : weight_buffers)
    valid = release_buffer(&buffer) && valid;
  valid = release_buffer(&activation_buffer) && valid;
  if (queue != nullptr)
    valid = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "NVFP4 pair queue release", error) &&
            valid;
  if (context != nullptr)
    valid = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "NVFP4 pair context release", error) &&
            valid;
  restore_environment();
  if (!valid)
    return false;
  std::cout << "Qwen3.8 NVFP4 projection-pack queue oracle: PASS target="
            << SLLM_TEST_EXPECTED_TARGET
            << " rows=2,3,4,5,64,65,127,128,129,512"
            << " shared_quantize=1 deterministic=PASS cleanup=0\n";
  return true;
}

} // namespace

int main() {
  for (const uint64_t m : {1U, 2U, 3U, 4U, 5U, 65U, 2048U}) {
    if (!run_fp8_gdn_shared_public_gpu_oracle(m)) {
      std::cerr << "FP8 GDN pair failed M=" << m << '\n';
      return 1;
    }
  }
  unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  setenv("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4", "1", 1);
  setenv("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_ACTIVATION_SHARED", "1", 1);
  std::vector<uint16_t> activation(static_cast<std::size_t>(kK),
                                   f32_to_bf16_rne(6.0F));
  // Positive E2M1 code 2 is 1.0 and code 1 is 0.5.  With E4M3 scale 0x38
  // (1.0), tensor scales 1.0, and 320 blocks, every output has an exact
  // integer-valued reference: 16*320*6 for gate and half that for up.
  std::vector<uint8_t> gate_weight = make_weight(UINT8_C(0x2));
  std::vector<uint8_t> up_weight = make_weight(UINT8_C(0x1));

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation_buffer = nullptr;
  std::array<sllm_buffer_t *, 2> weight_buffers{};
  std::array<sllm_buffer_t *, 2> output_buffers{};
  sllm_qwen38_projection_pack2_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  bool valid = true;
  Error error;

  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  valid = expect(sllm_context_create(&context_info, &context, &error.sink),
                 SLLM_STATUS_OK, "sllm_context_create", error);

  if (valid) {
    sllm_queue_create_info_t queue_info{};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.abi_version = SLLM_HIP_ABI_VERSION;
    valid = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "sllm_queue_create", error);
  }
  valid = valid &&
          create_buffer(context, kK * sizeof(uint16_t), &activation_buffer);
  valid = valid && create_buffer(context, kWeightBytes, &weight_buffers[0]);
  valid = valid && create_buffer(context, kWeightBytes, &weight_buffers[1]);
  valid = valid &&
          create_buffer(context, kN * sizeof(uint16_t), &output_buffers[0]);
  valid = valid &&
          create_buffer(context, kN * sizeof(uint16_t), &output_buffers[1]);
  valid = valid &&
          upload(queue, activation_buffer, activation.data(),
                 static_cast<uint64_t>(activation.size()) * sizeof(uint16_t));
  valid = valid && upload(queue, weight_buffers[0], gate_weight.data(),
                          static_cast<uint64_t>(gate_weight.size()));
  valid = valid && upload(queue, weight_buffers[1], up_weight.data(),
                          static_cast<uint64_t>(up_weight.size()));

  if (valid) {
    sllm_qwen38_projection_pack2_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_QWEN38_PROJECTION_PACK2_VERSION;
    descriptor.role = SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_NVFP4_MLP_GATE_UP;
    descriptor.input_global_scale_f32_bits = UINT32_C(0x3f800000);
    descriptor.activation = binding(activation_buffer, SLLM_TENSOR_DTYPE_BF16,
                                    SLLM_TENSOR_ENCODING_UNQUANTIZED, kM, kK);
    descriptor.gate_weight =
        binding(weight_buffers[0], SLLM_TENSOR_DTYPE_U8,
                SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32, kN, kK);
    descriptor.up_weight =
        binding(weight_buffers[1], SLLM_TENSOR_DTYPE_U8,
                SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32, kN, kK);
    descriptor.gate_output = binding(output_buffers[0], SLLM_TENSOR_DTYPE_BF16,
                                     SLLM_TENSOR_ENCODING_UNQUANTIZED, kM, kN);
    descriptor.up_output = binding(output_buffers[1], SLLM_TENSOR_DTYPE_BF16,
                                   SLLM_TENSOR_ENCODING_UNQUANTIZED, kM, kN);
    valid =
        expect(sllm_qwen38_projection_pack2_prepare(context, &descriptor, &plan,
                                                    &error.sink),
               SLLM_STATUS_OK, "sllm_qwen38_projection_pack2_prepare", error);
  }

  const std::array<uint16_t, 2> expected = {f32_to_bf16_rne(30720.0F),
                                            f32_to_bf16_rne(15360.0F)};
  std::array<sllm_qwen38_projection_pack2_dispatch_info_t, 2> dispatches{};
  std::array<std::array<std::vector<uint16_t>, 2>, 2> observed{};
  uint32_t max_bf16_ulp = 0U;
  bool deterministic = true;
  for (std::size_t repeat = 0U; repeat != dispatches.size() && valid;
       ++repeat) {
    auto &dispatch = dispatches[repeat];
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version =
        SLLM_HIP_QWEN38_PROJECTION_PACK2_DISPATCH_INFO_VERSION;
    const bool submitted =
        expect(sllm_qwen38_projection_pack2_execute(plan, queue, &completion,
                                                    &dispatch, &error.sink),
               SLLM_STATUS_OK, "sllm_qwen38_projection_pack2_execute", error);
    const bool evidence =
        submitted && completion != nullptr && dispatch.dispatch_id != 0U &&
        dispatch.dispatch_count == 3U && dispatch.grid_size_x == 1128U &&
        dispatch.workspace_bytes ==
            SLLM_HIP_QWEN38_PROJECTION_PACK2_WORKSPACE_BYTES &&
        dispatch.m == kM && dispatch.k == kK && dispatch.n == kN &&
        dispatch.output_elements == 2U * kN &&
        dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
        std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0 &&
        (repeat == 0U ||
         dispatch.dispatch_id != dispatches[repeat - 1U].dispatch_id);
    if (completion != nullptr) {
      valid = wait_and_release(&completion, "projection-pack wait") && valid;
    }
    valid = evidence && valid;
    if (!valid)
      break;

    for (std::size_t member = 0U; member != observed[repeat].size(); ++member) {
      observed[repeat][member].resize(static_cast<std::size_t>(kN));
      valid =
          download(queue, output_buffers[member], &observed[repeat][member]) &&
          valid;
      for (std::size_t index = 0U; index != observed[repeat][member].size();
           ++index) {
        const uint16_t value = observed[repeat][member][index];
        const uint32_t ulp =
            value >= expected[member]
                ? static_cast<uint32_t>(value - expected[member])
                : static_cast<uint32_t>(expected[member] - value);
        max_bf16_ulp = std::max(max_bf16_ulp, ulp);
        if (value != expected[member]) {
          std::cerr << "projection-pack numerical mismatch repeat=" << repeat
                    << " member=" << member << " index=" << index
                    << " actual=0x" << std::hex << value << " expected=0x"
                    << expected[member] << std::dec << " ulp=" << ulp << '\n';
          valid = false;
          break;
        }
      }
      if (repeat != 0U && observed[repeat][member] != observed[0][member]) {
        std::cerr << "projection-pack repeat is not bitwise deterministic: "
                  << "member=" << member << '\n';
        deterministic = false;
        valid = false;
      }
    }
  }

  if (plan != nullptr) {
    valid =
        expect(sllm_qwen38_projection_pack2_plan_release(&plan, &error.sink),
               SLLM_STATUS_OK, "sllm_qwen38_projection_pack2_plan_release",
               error) &&
        valid;
  }
  for (sllm_buffer_t *&output : output_buffers) {
    valid = release_buffer(&output) && valid;
  }
  for (sllm_buffer_t *&weight : weight_buffers) {
    valid = release_buffer(&weight) && valid;
  }
  valid = release_buffer(&activation_buffer) && valid;
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
  valid = run_nvfp4_projection_pack_queue_public_gpu_oracle() && valid;
  valid = run_activation_shared_matmul_oracle() && valid;
  valid = run_activation_shared_matmul_oracle(true) && valid;
  if (!valid)
    return 1;
  std::cout << "Qwen3.8 projection-pack public GPU oracle: PASS target="
            << dispatches.back().gcn_arch_name
            << " dispatch_count=" << dispatches.back().dispatch_count
            << " repeats=" << dispatches.size()
            << " deterministic=" << deterministic
            << " max_bf16_ulp=" << max_bf16_ulp
            << " workspace=" << dispatches.back().workspace_bytes
            << " cleanup=0\n";
  return 0;
}
