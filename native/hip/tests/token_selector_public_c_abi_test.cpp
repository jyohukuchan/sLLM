#include "sllm/hip.h"
#include "token_selector_kernel_internal.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t), SLLM_HIP_ABI_VERSION,
                         message, sizeof(message), 0U, {0U, 0U}};
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

bool verify_rng_contract() {
  // This tiny host-kernel oracle fixes the counter convention: counter zero
  // consumes seed + gamma, and counter one consumes seed + 2*gamma, exactly
  // as OsSamplingRandom does for its first two draws.
  constexpr uint16_t logits[3] = {UINT16_C(0x3f80), UINT16_C(0x3fc0),
                                  UINT16_C(0x3f00)};
  constexpr float additive[3] = {0.0F, 0.0F, 0.0F};
  constexpr uint8_t mask[3] = {1U, 1U, 1U};
  sllm_token_selector_record_t first{};
  sllm_token_selector_record_t second{};
  if (sllm_token_selector_kernel::launch(
          logits, additive, mask, 3U, 1.0F, 7U, 0U, &first, nullptr) !=
      hipSuccess) {
    return false;
  }
  if (sllm_token_selector_kernel::launch(
          logits, additive, mask, 3U, 1.0F, 7U, 1U, &second, nullptr) !=
      hipSuccess) {
    return false;
  }
  // Both draws land in the highest-logit token once the selector uses the
  // legacy categorical order (effective logit descending, token ID tie-break).
  return first.status == SLLM_STATUS_OK && first.token_id == 1 &&
         second.status == SLLM_STATUS_OK && second.token_id == 1;
}

bool wait_release(sllm_completion_t **const completion, const char *const label) {
  Error error;
  sllm_completion_result_t result{sizeof(result), SLLM_HIP_ABI_VERSION,
                                  SLLM_COMPLETION_STATE_PENDING, 0U, 0U, 0U,
                                  {0U, 0U, 0U, 0U}};
  if (!expect(sllm_completion_wait(*completion, UINT32_MAX, &result, &error.sink),
              SLLM_STATUS_OK, label, error)) {
    return false;
  }
  return expect(sllm_completion_release(completion, &error.sink),
                SLLM_STATUS_OK, "sllm_completion_release", error);
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const data, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{sizeof(transfer), SLLM_HIP_ABI_VERSION,
                                const_cast<void *>(data), 0U, bytes,
                                {0U, 0U, 0U, 0U}};
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error) &&
         wait_release(&completion, "sllm_completion_wait(h2d)");
}

bool download(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
              std::vector<uint8_t> *const bytes) {
  sllm_transfer_desc_t transfer{sizeof(transfer), SLLM_HIP_ABI_VERSION, nullptr,
                                0U, bytes->size(), {0U, 0U, 0U, 0U}};
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error)) {
    return false;
  }
  sllm_completion_result_t result{sizeof(result), SLLM_HIP_ABI_VERSION,
                                  SLLM_COMPLETION_STATE_PENDING, 0U, 0U, 0U,
                                  {0U, 0U, 0U, 0U}};
  if (!expect(sllm_completion_wait(completion, UINT32_MAX, &result, &error.sink),
              SLLM_STATUS_OK, "sllm_completion_wait(d2h)", error)) {
    return false;
  }
  uint64_t written = 0U;
  const bool read_ok = expect(sllm_completion_read(
                                  completion, bytes->data(), bytes->size(),
                                  &written, &error.sink),
                              SLLM_STATUS_OK, "sllm_completion_read", error) &&
                       written == bytes->size();
  const bool release_ok = expect(sllm_completion_release(&completion, &error.sink),
                                 SLLM_STATUS_OK, "sllm_completion_release(d2h)",
                                 error);
  return read_ok && release_ok;
}

bool prepare_invalid(const sllm_context_t *const context,
                     sllm_token_selector_desc_t descriptor,
                     const sllm_status_t expected, const char *const label) {
  sllm_token_selector_plan_t *plan = nullptr;
  Error error;
  const bool ok = expect(sllm_token_selector_prepare(context, &descriptor, &plan,
                                                     &error.sink),
                         expected, label, error) &&
                  plan == nullptr;
  if (plan != nullptr) {
    (void)sllm_token_selector_plan_release(&plan, &error.sink);
  }
  return ok;
}

bool run_selector(const sllm_context_t *const context,
                  const sllm_queue_t *const queue) {
  constexpr uint64_t vocab = 5U;
  const uint16_t logits_host[vocab] = {UINT16_C(0x3f80), UINT16_C(0x40c0),
                                       UINT16_C(0x3f00), UINT16_C(0x3f80),
                                       UINT16_C(0x3f80)};
  const float additive_host[vocab] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const uint8_t mask_host[vocab] = {1U, 1U, 1U, 1U, 1U};
  sllm_buffer_create_info_t info{sizeof(info), SLLM_HIP_ABI_VERSION,
                                 sizeof(logits_host), 0U, 0U, {0U, 0U, 0U, 0U, 0U}};
  sllm_buffer_t *logits = nullptr;
  sllm_buffer_t *additive = nullptr;
  sllm_buffer_t *mask = nullptr;
  sllm_buffer_t *output = nullptr;
  Error error;
  info.size_bytes = sizeof(logits_host);
  if (!expect(sllm_buffer_create(context, &info, &logits, &error.sink),
              SLLM_STATUS_OK, "create logits", error)) {
    return false;
  }
  info.size_bytes = sizeof(additive_host);
  if (!expect(sllm_buffer_create(context, &info, &additive, &error.sink),
              SLLM_STATUS_OK, "create additive", error)) {
    return false;
  }
  info.size_bytes = sizeof(mask_host);
  if (!expect(sllm_buffer_create(context, &info, &mask, &error.sink),
              SLLM_STATUS_OK, "create mask", error)) {
    return false;
  }
  info.size_bytes = SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES;
  if (!expect(sllm_buffer_create(context, &info, &output, &error.sink),
              SLLM_STATUS_OK, "create output", error)) {
    return false;
  }
  bool ok = upload(queue, logits, logits_host, sizeof(logits_host)) &&
            upload(queue, additive, additive_host, sizeof(additive_host)) &&
            upload(queue, mask, mask_host, sizeof(mask_host));
  sllm_token_selector_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_TOKEN_SELECTOR_VERSION;
  descriptor.logits = binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab);
  descriptor.additive_logits =
      binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab);
  descriptor.valid_mask = binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                              SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  descriptor.vocab_size = vocab;
  descriptor.temperature = 1.0F;
  descriptor.seed = 7U;
  descriptor.counter = 3U;
  sllm_token_selector_plan_t *plan = nullptr;
  ok = ok && expect(sllm_token_selector_prepare(context, &descriptor, &plan,
                                                 &error.sink),
                    SLLM_STATUS_OK, "sllm_token_selector_prepare", error);
  sllm_token_selector_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  ok = ok && expect(sllm_token_selector_execute(plan, queue, &completion,
                                                &dispatch, &error.sink),
                    SLLM_STATUS_OK, "sllm_token_selector_execute", error);
  ok = ok && wait_release(&completion, "sllm_completion_wait(selector)");
  std::vector<uint8_t> bytes(SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  ok = ok && download(queue, output, &bytes);
  sllm_token_selector_record_t record{};
  std::memcpy(&record, bytes.data(), sizeof(record));
  ok = ok && record.status == SLLM_STATUS_OK && record.token_id >= 0 &&
       record.token_id < static_cast<int32_t>(vocab) && dispatch.dispatch_id != 0U &&
       dispatch.dispatch_count == 1U && dispatch.fallback_allowed == 0U &&
       dispatch.fallback_used == 0U &&
       std::strcmp(dispatch.kernel_symbol, "token_selector.bf16_f32_mask.v1") == 0 &&
       std::strcmp(dispatch.device_symbol, "sllm_token_selector_bf16_f32_mask_v1") == 0;
  if (plan != nullptr) {
    ok = expect(sllm_token_selector_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "sllm_token_selector_plan_release", error) &&
         ok;
  }
  ok = expect(sllm_buffer_release(&output, &error.sink), SLLM_STATUS_OK,
              "release output", error) && ok;
  ok = expect(sllm_buffer_release(&mask, &error.sink), SLLM_STATUS_OK,
              "release mask", error) && ok;
  ok = expect(sllm_buffer_release(&additive, &error.sink), SLLM_STATUS_OK,
              "release additive", error) && ok;
  ok = expect(sllm_buffer_release(&logits, &error.sink), SLLM_STATUS_OK,
              "release logits", error) && ok;
  return ok;
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
  sllm_queue_create_info_t queue_info{sizeof(queue_info), SLLM_HIP_ABI_VERSION,
                                      0U, {0U, 0U, 0U, 0U, 0U}};
  sllm_queue_t *queue = nullptr;
  if (!expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
              SLLM_STATUS_OK, "sllm_queue_create", error)) {
    return 1;
  }
  bool success = verify_rng_contract() && run_selector(context, queue);
  sllm_buffer_t *dummy = nullptr;
  sllm_token_selector_desc_t invalid{};
  invalid.struct_size = sizeof(invalid);
  invalid.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  success = prepare_invalid(context, invalid, SLLM_STATUS_INVALID_ABI_VERSION,
                            "selector ABI negative") &&
            success;
  success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                   "sllm_queue_release", error) &&
            success;
  success = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "sllm_context_release", error) &&
            success;
  (void)dummy;
  return success ? 0 : 1;
}
