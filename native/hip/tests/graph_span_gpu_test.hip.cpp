#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1030"
#endif

#ifndef SLLM_TEST_EXPECTED_PCI
#define SLLM_TEST_EXPECTED_PCI "0000:03:00.0"
#endif

namespace {

constexpr uint64_t kM = 1U;
constexpr uint64_t kK = 35U;
constexpr uint64_t kN = 37U;
constexpr uint32_t kPatternCount = 5U;
constexpr uint32_t kStressReplays = 1000U;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
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

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool exact_equal(const std::vector<uint16_t> &left,
                 const std::vector<uint16_t> &right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
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
  if (rank == 1U) {
    result.stride_elements[0] = 1U;
  } else {
    result.shape[1] = second;
    result.stride_elements[0] = second;
    result.stride_elements[1] = 1U;
  }
  return result;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " received a null completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(
          sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
          SLLM_STATUS_OK, operation, error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    std::cerr << operation << " did not reach success state: " << result.state
              << '\n';
    return false;
  }
  const sllm_status_t release_status =
      sllm_completion_release(completion, &error.sink);
  return expect_status(release_status, SLLM_STATUS_OK,
                       "sllm_completion_release", error);
}

bool wait_completion(sllm_completion_t *const completion,
                     const char *const operation) {
  if (completion == nullptr) {
    std::cerr << operation << " received a null completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  return expect_status(
             sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
             SLLM_STATUS_OK, operation, error) &&
         result.state == SLLM_COMPLETION_STATE_SUCCESS;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const data, const uint64_t bytes,
            const char *const label) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(data);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                          &error.sink),
                     SLLM_STATUS_OK, label, error)) {
    return false;
  }
  return wait_release(&completion, "sllm_completion_wait(h2d)");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer,
              std::vector<uint16_t> *const output, const char *const label) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = output->size() * sizeof(uint16_t);
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                          &error.sink),
                     SLLM_STATUS_OK, label, error) ||
      !wait_completion(completion, "sllm_completion_wait(d2h)")) {
    return false;
  }
  uint64_t bytes_written = 0U;
  const bool read_ok =
      expect_status(sllm_completion_read(completion, output->data(),
                                         transfer.size_bytes, &bytes_written,
                                         &error.sink),
                    SLLM_STATUS_OK, "sllm_completion_read", error) &&
      bytes_written == transfer.size_bytes;
  const bool release_ok =
      expect_status(sllm_completion_release(&completion, &error.sink),
                    SLLM_STATUS_OK, "sllm_completion_release(d2h)", error);
  return read_ok && release_ok;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, buffer, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_create", error);
}

bool create_context(const char *const target, sllm_context_t **const context) {
  sllm_context_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.device_index = 0U;
  std::strncpy(info.expected_gcn_arch_name, target,
               sizeof(info.expected_gcn_arch_name) - 1U);
  Error error;
  return expect_status(sllm_context_create(&info, context, &error.sink),
                       SLLM_STATUS_OK, "sllm_context_create", error);
}

bool create_queue(const sllm_context_t *const context,
                  sllm_queue_t **const queue) {
  sllm_queue_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  return expect_status(sllm_queue_create(context, &info, queue, &error.sink),
                       SLLM_STATUS_OK, "sllm_queue_create", error);
}

std::vector<uint16_t> make_activation(const uint32_t pattern) {
  std::vector<uint16_t> values(kK);
  for (uint64_t index = 0U; index != kK; ++index) {
    const int64_t mixed =
        static_cast<int64_t>((index * 29U + pattern * 17U + 5U) % 97U) - 48;
    const float value = static_cast<float>(mixed) / 11.0F +
                        static_cast<float>(pattern) * 0.03125F;
    values[static_cast<std::size_t>(index)] = f32_to_bf16_rne(value);
  }
  return values;
}

std::vector<uint16_t> make_scale() {
  std::vector<uint16_t> values(kK);
  constexpr std::array<float, 7> pattern = {0.5F,  1.0F, 1.5F, -0.25F,
                                            0.75F, 2.0F, -1.0F};
  for (uint64_t index = 0U; index != kK; ++index) {
    values[static_cast<std::size_t>(index)] = f32_to_bf16_rne(
        pattern[static_cast<std::size_t>(index) % pattern.size()]);
  }
  return values;
}

std::vector<uint16_t> make_weight() {
  std::vector<uint16_t> values(kN * kK);
  for (uint64_t row = 0U; row != kN; ++row) {
    for (uint64_t column = 0U; column != kK; ++column) {
      const int64_t mixed =
          static_cast<int64_t>((row * 31U + column * 13U + 7U) % 89U) - 44;
      values[static_cast<std::size_t>(row * kK + column)] =
          f32_to_bf16_rne(static_cast<float>(mixed) / 17.0F);
    }
  }
  return values;
}

std::vector<uint16_t> oracle(const std::vector<uint16_t> &activation,
                             const std::vector<uint16_t> &scale,
                             const std::vector<uint16_t> &weight) {
  std::vector<uint16_t> normalized(kK);
  float sum = 0.0F;
  for (uint64_t column = 0U; column != kK; ++column) {
    const float value =
        bf16_to_f32(activation[static_cast<std::size_t>(column)]);
    sum += value * value;
  }
  const float inverse_rms =
      1.0F / std::sqrt(sum / static_cast<float>(kK) + 1.0e-6F);
  for (uint64_t column = 0U; column != kK; ++column) {
    const float value =
        bf16_to_f32(activation[static_cast<std::size_t>(column)]);
    const float scaled = value * inverse_rms *
                         bf16_to_f32(scale[static_cast<std::size_t>(column)]);
    normalized[static_cast<std::size_t>(column)] = f32_to_bf16_rne(scaled);
  }

  std::vector<uint16_t> output(kN);
  for (uint64_t row = 0U; row != kN; ++row) {
    float sum_row = 0.0F;
    for (uint64_t column = 0U; column != kK; ++column) {
      sum_row +=
          bf16_to_f32(normalized[static_cast<std::size_t>(column)]) *
          bf16_to_f32(weight[static_cast<std::size_t>(row * kK + column)]);
    }
    output[static_cast<std::size_t>(row)] = f32_to_bf16_rne(sum_row);
  }
  return output;
}

bool execute_eager(const sllm_queue_t *const queue,
                   const sllm_rmsnorm_plan_t *const rms_plan,
                   const sllm_matmul_plan_t *const matmul_plan,
                   const sllm_buffer_t *const activation_buffer,
                   const std::vector<uint16_t> &activation,
                   const uint64_t activation_bytes,
                   const sllm_buffer_t *const output_buffer,
                   std::vector<uint16_t> *const output) {
  if (!upload(queue, activation_buffer, activation.data(), activation_bytes,
              "eager activation upload")) {
    return false;
  }
  Error error;
  sllm_rmsnorm_dispatch_info_t rms_info{};
  rms_info.struct_size = sizeof(rms_info);
  rms_info.abi_version = SLLM_HIP_ABI_VERSION;
  rms_info.info_version = SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION;
  sllm_completion_t *rms_completion = nullptr;
  if (!expect_status(sllm_rmsnorm_execute(rms_plan, queue, &rms_completion,
                                          &rms_info, &error.sink),
                     SLLM_STATUS_OK, "eager RMSNorm execute", error) ||
      !wait_release(&rms_completion, "eager RMSNorm wait")) {
    return false;
  }
  sllm_matmul_dispatch_info_t matmul_info{};
  matmul_info.struct_size = sizeof(matmul_info);
  matmul_info.abi_version = SLLM_HIP_ABI_VERSION;
  matmul_info.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
  sllm_completion_t *matmul_completion = nullptr;
  if (!expect_status(sllm_matmul_execute(matmul_plan, queue, &matmul_completion,
                                         &matmul_info, &error.sink),
                     SLLM_STATUS_OK, "eager matmul execute", error) ||
      !wait_release(&matmul_completion, "eager matmul wait")) {
    return false;
  }
  return download(queue, output_buffer, output, "eager output download");
}

bool execute_graph_once(const sllm_queue_t *const queue,
                        const sllm_graph_span_t *const graph,
                        const sllm_buffer_t *const activation_buffer,
                        const std::vector<uint16_t> &activation,
                        const uint64_t activation_bytes,
                        const sllm_buffer_t *const output_buffer,
                        std::vector<uint16_t> *const output,
                        const bool read_output) {
  if (!upload(queue, activation_buffer, activation.data(), activation_bytes,
              "graph activation upload")) {
    return false;
  }
  Error error;
  sllm_completion_t *graph_completion = nullptr;
  if (!expect_status(
          sllm_graph_span_execute(graph, &graph_completion, &error.sink),
          SLLM_STATUS_OK, "graph span execute", error)) {
    return false;
  }
  sllm_completion_t *fence = nullptr;
  if (!expect_status(sllm_queue_fence(queue, &fence, &error.sink),
                     SLLM_STATUS_OK, "graph queue fence", error)) {
    (void)sllm_completion_release(&graph_completion, &error.sink);
    return false;
  }
  sllm_completion_result_t fence_result{};
  fence_result.struct_size = sizeof(fence_result);
  fence_result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(
          sllm_completion_wait(fence, UINT32_MAX, &fence_result, &error.sink),
          SLLM_STATUS_OK, "graph fence wait", error)) {
    (void)sllm_completion_release(&graph_completion, &error.sink);
    (void)sllm_completion_release(&fence, &error.sink);
    return false;
  }
  sllm_completion_result_t graph_result{};
  graph_result.struct_size = sizeof(graph_result);
  graph_result.abi_version = SLLM_HIP_ABI_VERSION;
  const sllm_status_t finalize_status = sllm_completion_finalize_after(
      graph_completion, fence, &graph_result, &error.sink);
  const bool finalized =
      expect_status(finalize_status, SLLM_STATUS_OK,
                    "graph completion finalize_after", error) &&
      graph_result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released_graph =
      expect_status(sllm_completion_release(&graph_completion, &error.sink),
                    SLLM_STATUS_OK, "graph completion release", error);
  const bool released_fence =
      expect_status(sllm_completion_release(&fence, &error.sink),
                    SLLM_STATUS_OK, "graph fence release", error);
  if (!finalized || !released_graph || !released_fence) {
    return false;
  }
  return !read_output ||
         download(queue, output_buffer, output, "graph output download");
}

bool release_buffer(sllm_buffer_t **const buffer, const char *const label,
                    uint32_t *const failures) {
  if (*buffer == nullptr) {
    return true;
  }
  Error error;
  const bool ok = expect_status(sllm_buffer_release(buffer, &error.sink),
                                SLLM_STATUS_OK, label, error);
  if (!ok) {
    ++(*failures);
  }
  return ok;
}

bool release_plan(sllm_rmsnorm_plan_t **const plan, const char *const label,
                  uint32_t *const failures) {
  if (*plan == nullptr) {
    return true;
  }
  Error error;
  const bool ok = expect_status(sllm_rmsnorm_plan_release(plan, &error.sink),
                                SLLM_STATUS_OK, label, error);
  if (!ok) {
    ++(*failures);
  }
  return ok;
}

bool release_matmul_plan(sllm_matmul_plan_t **const plan,
                         const char *const label, uint32_t *const failures) {
  if (*plan == nullptr) {
    return true;
  }
  Error error;
  const bool ok = expect_status(sllm_matmul_plan_release(plan, &error.sink),
                                SLLM_STATUS_OK, label, error);
  if (!ok) {
    ++(*failures);
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  uint32_t cleanup_failures = 0U;
  sllm_context_t *context = nullptr;
  sllm_context_t *cross_context = nullptr;
  sllm_queue_t *eager_queue = nullptr;
  sllm_queue_t *graph_queue = nullptr;
  sllm_queue_t *cross_queue = nullptr;
  sllm_buffer_t *activation_buffer = nullptr;
  sllm_buffer_t *scale_buffer = nullptr;
  sllm_buffer_t *normalized_buffer = nullptr;
  sllm_buffer_t *weight_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  sllm_rmsnorm_plan_t *rms_plan = nullptr;
  sllm_matmul_plan_t *matmul_plan = nullptr;
  sllm_graph_span_t *graph = nullptr;

  char pci_bus_id[64]{};
  const hipError_t pci_status =
      hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0);
  const bool pci_ok = pci_status == hipSuccess &&
                      std::strcmp(pci_bus_id, SLLM_TEST_EXPECTED_PCI) == 0;
  std::cout << "identity pci=" << pci_bus_id
            << " expected_pci=" << SLLM_TEST_EXPECTED_PCI
            << " pci_match=" << (pci_ok ? 1 : 0) << '\n';
  ok = pci_ok && ok;

  const std::vector<uint16_t> scale = make_scale();
  const std::vector<uint16_t> weight = make_weight();
  const uint64_t activation_bytes = kM * kK * sizeof(uint16_t);
  const uint64_t scale_bytes = kK * sizeof(uint16_t);
  const uint64_t normalized_bytes = kM * kK * sizeof(uint16_t);
  const uint64_t weight_bytes = kN * kK * sizeof(uint16_t);
  const uint64_t output_bytes = kM * kN * sizeof(uint16_t);

  ok = create_context(SLLM_TEST_EXPECTED_TARGET, &context) && ok;
  ok = create_queue(context, &eager_queue) && ok;
  ok = create_queue(context, &graph_queue) && ok;
  if (graph_queue != nullptr) {
    Error error;
    ok = expect_status(
             sllm_queue_set_completion_mode(
                 graph_queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &error.sink),
             SLLM_STATUS_OK, "graph queue deferred mode", error) &&
         ok;
  }
  ok = create_buffer(context, activation_bytes, &activation_buffer) && ok;
  ok = create_buffer(context, scale_bytes, &scale_buffer) && ok;
  ok = create_buffer(context, normalized_bytes, &normalized_buffer) && ok;
  ok = create_buffer(context, weight_bytes, &weight_buffer) && ok;
  ok = create_buffer(context, output_bytes, &output_buffer) && ok;

  if (ok) {
    ok = upload(eager_queue, scale_buffer, scale.data(), scale_bytes,
                "scale upload") &&
         upload(eager_queue, weight_buffer, weight.data(), weight_bytes,
                "weight upload") &&
         ok;
  }

  if (ok) {
    sllm_rmsnorm_desc_t rms_desc{};
    rms_desc.struct_size = sizeof(rms_desc);
    rms_desc.abi_version = SLLM_HIP_ABI_VERSION;
    rms_desc.op_version = SLLM_HIP_RMSNORM_VERSION;
    rms_desc.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
    rms_desc.scale_mode = SLLM_RMSNORM_SCALE_MODE_DIRECT;
    rms_desc.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
    const float epsilon = 1.0e-6F;
    std::memcpy(&rms_desc.epsilon_bits, &epsilon, sizeof(epsilon));
    rms_desc.activation =
        binding(activation_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, kM, kK);
    rms_desc.raw_scale = binding(scale_buffer, SLLM_TENSOR_DTYPE_BF16, 1U, kK);
    rms_desc.output =
        binding(normalized_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, kM, kK);
    Error error;
    ok = expect_status(
             sllm_rmsnorm_prepare(context, &rms_desc, &rms_plan, &error.sink),
             SLLM_STATUS_OK, "RMSNorm prepare", error) &&
         ok;

    sllm_matmul_desc_t matmul_desc{};
    matmul_desc.struct_size = sizeof(matmul_desc);
    matmul_desc.abi_version = SLLM_HIP_ABI_VERSION;
    matmul_desc.op_version = SLLM_HIP_MATMUL_VERSION;
    matmul_desc.activation =
        binding(normalized_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, kM, kK);
    matmul_desc.weight =
        binding(weight_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, kN, kK);
    matmul_desc.output =
        binding(output_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, kM, kN);
    ok = expect_status(sllm_matmul_prepare(context, &matmul_desc, &matmul_plan,
                                           &error.sink),
                       SLLM_STATUS_OK, "matmul prepare", error) &&
         ok;
  }

  bool empty_rejected = false;
  bool cross_context_rejected = false;
  if (ok) {
    Error error;
    sllm_graph_span_t *empty_graph = nullptr;
    const sllm_status_t empty_status = sllm_graph_span_create(
        graph_queue, nullptr, 0U, &empty_graph, &error.sink);
    empty_rejected = empty_status != SLLM_STATUS_OK && empty_graph == nullptr;
    std::cout << "invalid_empty status=" << empty_status
              << " rejected=" << (empty_rejected ? 1 : 0) << '\n';
    if (empty_graph != nullptr) {
      (void)sllm_graph_span_release(&empty_graph, &error.sink);
    }
  }

  const std::array<const void *, 2U> plan_handles = {
      reinterpret_cast<const void *>(rms_plan),
      reinterpret_cast<const void *>(matmul_plan)};
  if (ok && create_context(SLLM_TEST_EXPECTED_TARGET, &cross_context) &&
      create_queue(cross_context, &cross_queue)) {
    Error error;
    ok = expect_status(
             sllm_queue_set_completion_mode(
                 cross_queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &error.sink),
             SLLM_STATUS_OK, "cross queue deferred mode", error) &&
         ok;
    sllm_graph_span_t *cross_graph = nullptr;
    const sllm_status_t cross_status =
        sllm_graph_span_create(cross_queue, plan_handles.data(),
                               plan_handles.size(), &cross_graph, &error.sink);
    cross_context_rejected =
        cross_status != SLLM_STATUS_OK && cross_graph == nullptr;
    std::cout << "invalid_cross_context status=" << cross_status
              << " rejected=" << (cross_context_rejected ? 1 : 0) << '\n';
    if (cross_graph != nullptr) {
      (void)sllm_graph_span_release(&cross_graph, &error.sink);
    }
  } else {
    std::cerr << "cross-context setup unavailable\n";
    ok = false;
  }

  const std::vector<uint16_t> first_activation = make_activation(0U);
  std::vector<uint16_t> eager_output(kN);
  std::vector<uint16_t> graph_output(kN);
  const std::vector<uint16_t> first_oracle =
      oracle(first_activation, scale, weight);
  if (ok) {
    /* Eager warmup completes before capture. */
    ok = execute_eager(eager_queue, rms_plan, matmul_plan, activation_buffer,
                       first_activation, activation_bytes, output_buffer,
                       &eager_output) &&
         ok;
    ok = exact_equal(eager_output, first_oracle) && ok;
    if (!exact_equal(eager_output, first_oracle)) {
      std::cerr << "eager warmup oracle mismatch\n";
    }
  }

  const std::vector<uint16_t> sentinel(kN, UINT16_C(0x3e11));
  bool output_unchanged = false;
  uint64_t node_count = 0U;
  if (ok) {
    ok = upload(graph_queue, output_buffer, sentinel.data(), output_bytes,
                "capture sentinel upload") &&
         ok;
    std::vector<uint16_t> before_capture(kN);
    std::vector<uint16_t> after_capture(kN);
    ok = download(graph_queue, output_buffer, &before_capture,
                  "capture sentinel before read") &&
         ok;
    Error error;
    const sllm_status_t create_status =
        sllm_graph_span_create(graph_queue, plan_handles.data(),
                               plan_handles.size(), &graph, &error.sink);
    ok = expect_status(create_status, SLLM_STATUS_OK, "graph span create",
                       error) &&
         ok;
    output_unchanged = download(graph_queue, output_buffer, &after_capture,
                                "capture sentinel after read") &&
                       exact_equal(before_capture, sentinel) &&
                       exact_equal(after_capture, sentinel);
    ok = output_unchanged && ok;
    if (!output_unchanged) {
      std::cerr << "graph span create mutated output\n";
    }
    if (graph != nullptr) {
      ok = expect_status(
               sllm_graph_span_node_count(graph, &node_count, &error.sink),
               SLLM_STATUS_OK, "graph span node count", error) &&
           node_count != 0U && ok;
    }
  }

  bool plan_busy = false;
  bool buffer_busy = false;
  bool queue_busy = false;
  if (ok && graph != nullptr) {
    Error error;
    const sllm_status_t rms_busy =
        sllm_rmsnorm_plan_release(&rms_plan, &error.sink);
    const sllm_status_t matmul_busy =
        sllm_matmul_plan_release(&matmul_plan, &error.sink);
    plan_busy = rms_busy == SLLM_STATUS_PUBLIC_BUSY &&
                matmul_busy == SLLM_STATUS_PUBLIC_BUSY && rms_plan != nullptr &&
                matmul_plan != nullptr;
    const sllm_status_t buffer_busy_status =
        sllm_buffer_release(&activation_buffer, &error.sink);
    buffer_busy = buffer_busy_status == SLLM_STATUS_PUBLIC_BUSY &&
                  activation_buffer != nullptr;
    const sllm_status_t queue_busy_status =
        sllm_queue_release(&graph_queue, &error.sink);
    queue_busy =
        queue_busy_status == SLLM_STATUS_PUBLIC_BUSY && graph_queue != nullptr;
    std::cout << "early_release plan_busy=" << (plan_busy ? 1 : 0)
              << " buffer_busy=" << (buffer_busy ? 1 : 0)
              << " queue_busy=" << (queue_busy ? 1 : 0) << '\n';
    ok = plan_busy && buffer_busy && queue_busy && ok;
  }

  bool inflight_graph_busy = false;
  if (ok && graph != nullptr) {
    ok = upload(graph_queue, activation_buffer, first_activation.data(),
                activation_bytes, "inflight activation upload") &&
         ok;
    Error error;
    sllm_completion_t *inflight_completion = nullptr;
    ok = expect_status(
             sllm_graph_span_execute(graph, &inflight_completion, &error.sink),
             SLLM_STATUS_OK, "inflight graph execute", error) &&
         ok;
    const sllm_status_t release_status =
        sllm_graph_span_release(&graph, &error.sink);
    inflight_graph_busy =
        release_status == SLLM_STATUS_PUBLIC_BUSY && graph != nullptr;
    std::cout << "inflight_graph_release status=" << release_status
              << " busy=" << (inflight_graph_busy ? 1 : 0) << '\n';
    ok = inflight_graph_busy && ok;
    if (inflight_completion != nullptr) {
      sllm_completion_t *fence = nullptr;
      ok = expect_status(sllm_queue_fence(graph_queue, &fence, &error.sink),
                         SLLM_STATUS_OK, "inflight outer fence", error) &&
           ok;
      sllm_completion_result_t fence_result{};
      fence_result.struct_size = sizeof(fence_result);
      fence_result.abi_version = SLLM_HIP_ABI_VERSION;
      ok = expect_status(sllm_completion_wait(fence, UINT32_MAX, &fence_result,
                                              &error.sink),
                         SLLM_STATUS_OK, "inflight outer fence wait", error) &&
           ok;
      sllm_completion_result_t result{};
      result.struct_size = sizeof(result);
      result.abi_version = SLLM_HIP_ABI_VERSION;
      ok = expect_status(sllm_completion_finalize_after(
                             inflight_completion, fence, &result, &error.sink),
                         SLLM_STATUS_OK, "inflight completion finalize",
                         error) &&
           result.state == SLLM_COMPLETION_STATE_SUCCESS && ok;
      ok = expect_status(
               sllm_completion_release(&inflight_completion, &error.sink),
               SLLM_STATUS_OK, "inflight completion release", error) &&
           ok;
      ok = expect_status(sllm_completion_release(&fence, &error.sink),
                         SLLM_STATUS_OK, "inflight fence release", error) &&
           ok;
    }
  }

  uint32_t compared_patterns = 0U;
  uint32_t eager_mismatches = 0U;
  uint32_t graph_mismatches = 0U;
  uint32_t oracle_mismatches = 0U;
  if (ok && graph != nullptr) {
    for (uint32_t pattern = 0U; pattern != kPatternCount; ++pattern) {
      const std::vector<uint16_t> activation = make_activation(pattern);
      const std::vector<uint16_t> expected = oracle(activation, scale, weight);
      if (!execute_eager(eager_queue, rms_plan, matmul_plan, activation_buffer,
                         activation, activation_bytes, output_buffer,
                         &eager_output)) {
        ok = false;
        break;
      }
      const bool eager_matches = exact_equal(eager_output, expected);
      eager_mismatches += eager_matches ? 0U : 1U;
      if (!execute_graph_once(graph_queue, graph, activation_buffer, activation,
                              activation_bytes, output_buffer, &graph_output,
                              true)) {
        ok = false;
        break;
      }
      const bool graph_matches = exact_equal(graph_output, eager_output);
      const bool graph_oracle_matches = exact_equal(graph_output, expected);
      graph_mismatches += graph_matches ? 0U : 1U;
      oracle_mismatches += graph_oracle_matches ? 0U : 1U;
      ++compared_patterns;
      std::cout << "pattern=" << pattern
                << " eager_bit_match=" << (eager_matches ? 1 : 0)
                << " graph_eager_bit_match=" << (graph_matches ? 1 : 0)
                << " graph_oracle_bit_match=" << (graph_oracle_matches ? 1 : 0)
                << '\n';
      ok = eager_matches && graph_matches && graph_oracle_matches && ok;
    }
  }

  uint32_t stress_completed = 0U;
  if (ok && graph != nullptr) {
    const auto stress_start = std::chrono::steady_clock::now();
    for (uint32_t replay = 0U; replay != kStressReplays; ++replay) {
      const uint32_t pattern = replay % kPatternCount;
      if (!execute_graph_once(graph_queue, graph, activation_buffer,
                              make_activation(pattern), activation_bytes,
                              output_buffer, &graph_output, false)) {
        ok = false;
        break;
      }
      ++stress_completed;
    }
    const auto stress_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stress_end - stress_start)
            .count();
    std::cout << "stress_replays=" << stress_completed
              << " requested=" << kStressReplays << " elapsed_ms=" << elapsed_ms
              << '\n';
  }

  if (graph != nullptr) {
    Error error;
    cleanup_failures +=
        sllm_graph_span_release(&graph, &error.sink) == SLLM_STATUS_OK ? 0U
                                                                       : 1U;
  }
  release_plan(&rms_plan, "RMSNorm plan release", &cleanup_failures);
  release_matmul_plan(&matmul_plan, "matmul plan release", &cleanup_failures);
  release_buffer(&output_buffer, "output buffer release", &cleanup_failures);
  release_buffer(&weight_buffer, "weight buffer release", &cleanup_failures);
  release_buffer(&normalized_buffer, "normalized buffer release",
                 &cleanup_failures);
  release_buffer(&scale_buffer, "scale buffer release", &cleanup_failures);
  release_buffer(&activation_buffer, "activation buffer release",
                 &cleanup_failures);
  if (cross_queue != nullptr) {
    Error error;
    cleanup_failures +=
        sllm_queue_release(&cross_queue, &error.sink) == SLLM_STATUS_OK ? 0U
                                                                        : 1U;
  }
  if (cross_context != nullptr) {
    Error error;
    cleanup_failures +=
        sllm_context_release(&cross_context, &error.sink) == SLLM_STATUS_OK
            ? 0U
            : 1U;
  }
  if (eager_queue != nullptr) {
    Error error;
    cleanup_failures +=
        sllm_queue_release(&eager_queue, &error.sink) == SLLM_STATUS_OK ? 0U
                                                                        : 1U;
  }
  if (graph_queue != nullptr) {
    Error error;
    cleanup_failures +=
        sllm_queue_release(&graph_queue, &error.sink) == SLLM_STATUS_OK ? 0U
                                                                        : 1U;
  }
  if (context != nullptr) {
    Error error;
    cleanup_failures +=
        sllm_context_release(&context, &error.sink) == SLLM_STATUS_OK ? 0U : 1U;
  }

  const bool invalid_cases = empty_rejected && cross_context_rejected;
  const bool evidence_ok =
      ok && invalid_cases && output_unchanged && node_count != 0U &&
      plan_busy && buffer_busy && queue_busy && inflight_graph_busy &&
      compared_patterns == kPatternCount && eager_mismatches == 0U &&
      graph_mismatches == 0U && oracle_mismatches == 0U &&
      stress_completed == kStressReplays && cleanup_failures == 0U;
  std::cout << "identity target=" << SLLM_TEST_EXPECTED_TARGET
            << " pci=" << pci_bus_id << " shape=M" << kM << "K" << kK << "N"
            << kN << " queue_mode=DEFERRED graph_plans=RMSNorm+BF16Matmul\n";
  std::cout << "node_count=" << node_count
            << " output_unchanged_on_create=" << (output_unchanged ? 1 : 0)
            << " patterns=" << compared_patterns
            << " eager_mismatches=" << eager_mismatches
            << " graph_mismatches=" << graph_mismatches
            << " oracle_mismatches=" << oracle_mismatches << '\n';
  std::cout << "early_busy plan=" << (plan_busy ? 1 : 0)
            << " buffer=" << (buffer_busy ? 1 : 0)
            << " queue=" << (queue_busy ? 1 : 0)
            << " inflight_graph=" << (inflight_graph_busy ? 1 : 0) << '\n';
  std::cout << "stateful_plan_case=SKIP reason=no_bounded_stateful_fixture\n";
  std::cout << "cleanup_failures=" << cleanup_failures
            << " cleanup=" << (cleanup_failures == 0U ? 0 : 1)
            << " status=" << (evidence_ok ? "PASS" : "FAIL") << '\n';
  return evidence_ok ? 0 : 1;
}
