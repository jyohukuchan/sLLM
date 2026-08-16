#include "sllm/hip.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
  if (actual == expected)
    return true;
  std::cerr << operation << " returned " << actual << ": " << error.message
            << '\n';
  return false;
}

std::vector<uint8_t> read_file(const char *const path,
                               const uint64_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<uint64_t>(stream.tellg()) != expected)
    return {};
  std::vector<uint8_t> result(static_cast<std::size_t>(expected));
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(result.data()),
              static_cast<std::streamsize>(result.size()));
  return stream ? result : std::vector<uint8_t>{};
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

bool wait_release(sllm_completion_t **const completion) {
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  return expect(sllm_completion_wait(*completion, UINT32_MAX, &result,
                                     &error.sink),
                SLLM_STATUS_OK, "wait", error) &&
         expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "completion release", error);
}

sllm_buffer_t *create(const sllm_context_t *const context,
                      const uint64_t bytes) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  sllm_buffer_t *buffer = nullptr;
  Error error;
  return expect(sllm_buffer_create(context, &info, &buffer, &error.sink),
                SLLM_STATUS_OK, "buffer create", error)
             ? buffer
             : nullptr;
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
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "upload", error) &&
         wait_release(&completion);
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "download", error))
    return false;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  uint64_t written = 0U;
  const bool ok =
      expect(sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
             SLLM_STATUS_OK, "download wait", error) &&
      expect(sllm_completion_read(completion, destination, bytes, &written,
                                  &error.sink),
             SLLM_STATUS_OK, "download read", error) &&
      written == bytes;
  return expect(sllm_completion_release(&completion, &error.sink),
                SLLM_STATUS_OK, "download release", error) &&
         ok;
}

float bf16(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint64_t environment_u64(const char *const name, const uint64_t fallback) {
  const char *const value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    return fallback;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  return end != value && *end == '\0' ? static_cast<uint64_t>(parsed) : 0U;
}

} // namespace

int main() {
  const char *const blob_path = std::getenv("SLLM_MOE_LAYER_BLOB");
  const char *const hidden_path = std::getenv("SLLM_MOE_HIDDEN");
  const char *const expected_path = std::getenv("SLLM_MOE_EXPECTED");
  if (blob_path == nullptr || hidden_path == nullptr ||
      expected_path == nullptr) {
    std::cerr << "MoE fixture environment is absent\n";
    return 2;
  }
  const uint64_t token_count = environment_u64("SLLM_MOE_TOKENS", 1U);
  const uint64_t expert_start = environment_u64("SLLM_MOE_EXPERT_START", 0U);
  if (token_count == 0U || expert_start + 8U > 256U) {
    std::cerr << "MoE fixture dimensions are invalid\n";
    return 2;
  }
  const auto blob = read_file(blob_path, SLLM_HIP_MOE_EXPERT_LAYER_BLOB_BYTES);
  const auto hidden = read_file(
      hidden_path, token_count * SLLM_HIP_MOE_EXPERT_HIDDEN_SIZE * 2U);
  const auto expected_bytes = read_file(
      expected_path, token_count * SLLM_HIP_MOE_EXPERT_HIDDEN_SIZE * 4U);
  if (blob.empty() || hidden.empty() || expected_bytes.empty()) {
    std::cerr << "MoE fixture file differs\n";
    return 2;
  }
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  sllm_context_t *context = nullptr;
  Error error;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "context", error))
    return 1;
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  if (!expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
              SLLM_STATUS_OK, "queue", error))
    return 1;
  const uint64_t pair_count = token_count * 8U;
  const uint64_t route_bytes = pair_count * 16U + 256U * 4U + 257U * 4U + 4U;
  const uint64_t workspace_bytes = token_count * (2048U / 2U + 2048U / 32U) +
                                   pair_count * 512U * 2U +
                                   pair_count * (512U / 2U + 512U / 32U) +
                                   token_count * 512U * 2U + token_count * 4U;
  sllm_buffer_t *hidden_buffer = create(context, hidden.size());
  sllm_buffer_t *logits_buffer = create(context, token_count * 256U * 2U);
  sllm_buffer_t *route_buffer = create(context, route_bytes);
  sllm_buffer_t *blob_buffer = create(context, blob.size());
  sllm_buffer_t *workspace_buffer = create(context, workspace_bytes);
  sllm_buffer_t *output_buffer = create(context, token_count * 2048U * 2U);
  std::vector<uint16_t> logits(static_cast<std::size_t>(token_count * 256U),
                               UINT16_C(0xc120));
  for (uint64_t token = 0U; token < token_count; ++token) {
    for (uint64_t expert = expert_start; expert < expert_start + 8U; ++expert) {
      logits[static_cast<std::size_t>(token * 256U + expert)] = 0U;
    }
  }
  bool ok = hidden_buffer != nullptr && logits_buffer != nullptr &&
            route_buffer != nullptr && blob_buffer != nullptr &&
            workspace_buffer != nullptr && output_buffer != nullptr &&
            upload(queue, hidden_buffer, hidden.data(), hidden.size()) &&
            upload(queue, logits_buffer, logits.data(), logits.size() * 2U) &&
            upload(queue, blob_buffer, blob.data(), blob.size());
  sllm_moe_route_desc_t route_desc{};
  route_desc.struct_size = sizeof(route_desc);
  route_desc.abi_version = SLLM_HIP_ABI_VERSION;
  route_desc.op_version = SLLM_HIP_MOE_ROUTE_VERSION;
  route_desc.selected_expert_count = 8U;
  route_desc.logits =
      binding(logits_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, token_count, 256U);
  route_desc.metadata =
      binding(route_buffer, SLLM_TENSOR_DTYPE_U8, 1U, route_bytes);
  sllm_moe_route_plan_t *route_plan = nullptr;
  ok = ok && expect(sllm_moe_route_prepare(context, &route_desc, &route_plan,
                                           &error.sink),
                    SLLM_STATUS_OK, "route prepare", error);
  sllm_moe_route_dispatch_info_t route_info{};
  route_info.struct_size = sizeof(route_info);
  route_info.abi_version = SLLM_HIP_ABI_VERSION;
  route_info.info_version = SLLM_HIP_MOE_ROUTE_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok &&
       expect(sllm_moe_route_execute(route_plan, queue, &completion,
                                     &route_info, &error.sink),
              SLLM_STATUS_OK, "route execute", error) &&
       wait_release(&completion);
  sllm_moe_expert_desc_t expert_desc{};
  expert_desc.struct_size = sizeof(expert_desc);
  expert_desc.abi_version = SLLM_HIP_ABI_VERSION;
  expert_desc.op_version = SLLM_HIP_MOE_EXPERT_VERSION;
  expert_desc.hidden =
      binding(hidden_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, token_count, 2048U);
  expert_desc.routing_metadata =
      binding(route_buffer, SLLM_TENSOR_DTYPE_U8, 1U, route_bytes);
  expert_desc.layer_blob =
      binding(blob_buffer, SLLM_TENSOR_DTYPE_U8, 1U, blob.size());
  expert_desc.workspace =
      binding(workspace_buffer, SLLM_TENSOR_DTYPE_U8, 1U, workspace_bytes);
  expert_desc.output =
      binding(output_buffer, SLLM_TENSOR_DTYPE_BF16, 2U, token_count, 2048U);
  sllm_moe_expert_plan_t *expert_plan = nullptr;
  ok = ok && expect(sllm_moe_expert_prepare(context, &expert_desc, &expert_plan,
                                            &error.sink),
                    SLLM_STATUS_OK, "expert prepare", error);
  sllm_moe_expert_dispatch_info_t expert_info{};
  expert_info.struct_size = sizeof(expert_info);
  expert_info.abi_version = SLLM_HIP_ABI_VERSION;
  expert_info.info_version = SLLM_HIP_MOE_EXPERT_DISPATCH_INFO_VERSION;
  ok = ok &&
       expect(sllm_moe_expert_execute(expert_plan, queue, &completion,
                                      &expert_info, &error.sink),
              SLLM_STATUS_OK, "expert execute", error) &&
       wait_release(&completion);
  std::vector<uint16_t> output(static_cast<std::size_t>(token_count * 2048U));
  ok = ok && download(queue, output_buffer, output.data(), output.size() * 2U);
  const auto *const expected =
      reinterpret_cast<const float *>(expected_bytes.data());
  float maximum_absolute = 0.0F;
  float maximum_relative = 0.0F;
  for (std::size_t index = 0U; ok && index < output.size(); ++index) {
    const float actual = bf16(output[index]);
    const float absolute = std::abs(actual - expected[index]);
    const float relative = absolute / std::max(1.0F, std::abs(expected[index]));
    maximum_absolute = std::max(maximum_absolute, absolute);
    maximum_relative = std::max(maximum_relative, relative);
    ok = std::isfinite(actual) && relative <= 0.025F;
  }
  ok = ok && route_info.fallback_used == 0U &&
       expert_info.fallback_used == 0U &&
       expert_info.active_pair_count == pair_count &&
       expert_info.shared_expert_count == 1U &&
       expert_info.kernel_id ==
           (token_count == 1U ? SLLM_HIP_MOE_EXPERT_KERNEL_ID_DECODE_V1
                              : SLLM_HIP_MOE_EXPERT_KERNEL_ID_PREFILL_V1);
  std::cout << (ok ? "PASS" : "FAIL") << " target=" << SLLM_TEST_EXPECTED_TARGET
            << " tokens=" << token_count << " expert_start=" << expert_start
            << " active_pairs=" << pair_count
            << " shared=1 max_abs=" << maximum_absolute
            << " max_rel=" << maximum_relative << " fallback=0\n";
  if (expert_plan != nullptr)
    sllm_moe_expert_plan_release(&expert_plan, &error.sink);
  if (route_plan != nullptr)
    sllm_moe_route_plan_release(&route_plan, &error.sink);
  for (sllm_buffer_t **buffer :
       {&output_buffer, &workspace_buffer, &blob_buffer, &route_buffer,
        &logits_buffer, &hidden_buffer}) {
    if (*buffer != nullptr)
      sllm_buffer_release(buffer, &error.sink);
  }
  sllm_queue_release(&queue, &error.sink);
  sllm_context_release(&context, &error.sink);
  return ok ? 0 : 1;
}
