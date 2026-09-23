// Phase 87 Stage 9 focused probe for the mapped fixed-K20 selector ABI.
//
// The v3 descriptor stores a sorted local-row -> global-vocabulary map after
// the two fixed-K20 scratch regions in workspace.  This probe intentionally
// uses the public C ABI so that descriptor validation, buffer ownership,
// asynchronous execution, output readback, and the private support record
// all follow the production path.

#include "sllm/hip.h"
#include "token_selector_support_internal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

constexpr uint32_t kTimeoutMs = 60'000U;
constexpr uint64_t kFirstVocab = 98'303U;
constexpr uint64_t kAlignedVocab = 98'304U;
constexpr uint64_t kLastVocab = 98'305U;

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
            const char *const where, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << "FAIL: " << where << " status=" << actual
            << " expected=" << expected << " message=" << error.message << '\n';
  return false;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const where) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << "FAIL: " << where << " returned no completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(sllm_completion_wait(*completion, kTimeoutMs,
                                                  &result, &error.sink),
                             SLLM_STATUS_OK, where, error) &&
                      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  if (!waited) {
    std::cerr << "FAIL: " << where << " completion state=" << result.state
              << '\n';
  }
  const bool released =
      expect(sllm_completion_release(completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release", error) &&
      *completion == nullptr;
  return waited && released;
}

bool copy_h2d(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, const void *const source,
              const uint64_t offset, const uint64_t bytes,
              const char *const where) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  Error error;
  sllm_completion_t *completion = nullptr;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, where, error) &&
         wait_release(&completion, where);
}

bool copy_d2h(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t offset, const uint64_t bytes,
              const char *const where) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = nullptr;
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  Error error;
  sllm_completion_t *completion = nullptr;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, where, error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited =
      expect(sllm_completion_wait(completion, kTimeoutMs, &result, &error.sink),
             SLLM_STATUS_OK, where, error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  uint64_t written = 0U;
  const bool read = waited &&
                    expect(sllm_completion_read(completion, destination, bytes,
                                                &written, &error.sink),
                           SLLM_STATUS_OK, where, error) &&
                    written == bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release(d2h)", error);
  return read && released;
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
  result.stride_elements[0] = rank == 1U ? 1U : second;
  if (rank == 2U) {
    result.shape[1] = second;
    result.stride_elements[1] = 1U;
  }
  return result;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const buffer, const char *const where) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, buffer, &error.sink),
                SLLM_STATUS_OK, where, error);
}

bool release_buffer(sllm_buffer_t **const buffer, const char *const where) {
  if (buffer == nullptr || *buffer == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK, where,
                error) &&
         *buffer == nullptr;
}

uint64_t scratch_bytes(const uint64_t vocab_size) {
  const uint64_t block_count =
      (vocab_size + SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE - 1U) /
      SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
  return block_count * UINT64_C(20) * UINT64_C(16);
}

uint64_t final_input_offset(const uint64_t vocab_size) {
  const uint64_t block_count =
      (vocab_size + SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE - 1U) /
      SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
  const uint64_t region_bytes = block_count * UINT64_C(20) * UINT64_C(8);
  uint64_t input_records = block_count * UINT64_C(20);
  uint64_t input_offset = 0U;
  uint64_t output_offset = region_bytes;
  while (input_records > SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE) {
    const uint64_t reduce_blocks =
        (input_records + SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE - 1U) /
        SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
    input_records = reduce_blocks * UINT64_C(20);
    const uint64_t old_input_offset = input_offset;
    input_offset = output_offset;
    output_offset = old_input_offset;
  }
  return input_offset;
}

uint32_t mapped_id(const uint32_t local_id) {
  // The odd IDs make local and global IDs visibly different while preserving
  // the required sorted map order and staying below Qwen3.8's full vocab.
  return local_id * 2U + 1U;
}

struct RunResult final {
  bool ok = false;
  int32_t token_id = -1;
  std::array<uint32_t, sllm_token_selector_support::kMaxCountV1>
      candidate_ids{};
  sllm_token_selector_support::RecordV1 support{};
};

RunResult
run_selector(const sllm_context_t *const context,
             const sllm_queue_t *const queue, const uint64_t vocab_size,
             const bool mapped, const std::vector<uint16_t> &logits_host,
             const std::vector<float> &additive_host,
             const std::vector<uint8_t> &mask_host, sllm_buffer_t *const logits,
             sllm_buffer_t *const additive, sllm_buffer_t *const mask,
             sllm_buffer_t *const output, sllm_buffer_t *const workspace,
             const uint64_t workspace_size) {
  RunResult result;
  Error error;
  bool ok = copy_h2d(queue, logits, logits_host.data(), 0U,
                     logits_host.size() * sizeof(uint16_t), "upload logits") &&
            copy_h2d(queue, additive, additive_host.data(), 0U,
                     additive_host.size() * sizeof(float), "upload additive") &&
            copy_h2d(queue, mask, mask_host.data(), 0U, mask_host.size(),
                     "upload mask");

  std::vector<uint8_t> workspace_host(workspace_size, 0U);
  if (mapped) {
    const uint64_t map_offset = scratch_bytes(vocab_size);
    for (uint64_t index = 0U; index != vocab_size; ++index) {
      const uint32_t id = mapped_id(static_cast<uint32_t>(index));
      std::memcpy(workspace_host.data() + map_offset + index * sizeof(id), &id,
                  sizeof(id));
    }
  }
  ok = copy_h2d(queue, workspace, workspace_host.data(), 0U,
                workspace_host.size(), "upload selector workspace") &&
       ok;
  uint8_t mask_readback = 0U;
  ok = copy_d2h(queue, mask, &mask_readback, 0U, 1U,
                "roundtrip selector mask") &&
       ok;
  if (mask_readback != 1U) {
    std::cerr << "DIAG: selector mask roundtrip=" << unsigned(mask_readback)
              << '\n';
  }

  sllm_token_selector_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version =
      mapped ? SLLM_HIP_TOKEN_SELECTOR_VERSION_FIXED_TOPK_TOPP_MAPPED
             : SLLM_HIP_TOKEN_SELECTOR_VERSION_FIXED_TOPK_TOPP;
  descriptor.logits =
      binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab_size);
  descriptor.additive_logits =
      binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab_size);
  descriptor.valid_mask =
      binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab_size);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                              SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
  descriptor.vocab_size = vocab_size;
  descriptor.temperature = 1.0F;
  descriptor.seed = UINT64_C(0x8f6e37a1);
  descriptor.counter = UINT64_C(17);
  descriptor.top_k = 20U;
  descriptor.flags = SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                     SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT;
  if (mapped) {
    descriptor.flags |= SLLM_HIP_TOKEN_SELECTOR_FLAG_VOCAB_MAP_PRESENT;
  }
  descriptor.top_p = 0.95F;
  const uint64_t required_workspace =
      mapped ? scratch_bytes(vocab_size) + vocab_size * sizeof(uint32_t)
             : scratch_bytes(vocab_size);
  descriptor.workspace =
      binding(workspace, SLLM_TENSOR_DTYPE_U8, 1U, required_workspace);

  sllm_token_selector_plan_t *plan = nullptr;
  ok = expect(sllm_token_selector_prepare(context, &descriptor, &plan,
                                          &error.sink),
              SLLM_STATUS_OK, "sllm_token_selector_prepare", error) &&
       ok;
  if (ok) {
    sllm_token_selector_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    ok = expect(sllm_token_selector_execute(plan, queue, &completion, &dispatch,
                                            &error.sink),
                SLLM_STATUS_OK, "sllm_token_selector_execute", error) &&
         ok;
    ok = wait_release(&completion, "token selector execute") && ok;

    std::vector<uint8_t> output_host(SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
    std::vector<uint8_t> workspace_after(workspace_size, 0U);
    ok = copy_d2h(queue, output, output_host.data(), 0U, output_host.size(),
                  "download selector output") &&
         ok;
    ok = copy_d2h(queue, workspace, workspace_after.data(), 0U,
                  workspace_after.size(), "download selector workspace") &&
         ok;
    sllm_token_selector_record_t record{};
    std::memcpy(&record, output_host.data(), sizeof(record));
    std::memcpy(&result.support, workspace_after.data(),
                sizeof(result.support));
    const uint64_t candidate_offset = final_input_offset(vocab_size);
    for (uint32_t index = 0U; index != result.candidate_ids.size(); ++index) {
      std::memcpy(&result.candidate_ids[index],
                  workspace_after.data() + candidate_offset + index * 8U,
                  sizeof(result.candidate_ids[index]));
    }
    result.token_id = record.token_id;
    if (record.status != SLLM_STATUS_OK ||
        result.support.version != sllm_token_selector_support::kVersionV1 ||
        result.support.status != SLLM_STATUS_OK ||
        result.support.count != sllm_token_selector_support::kMaxCountV1 - 1U) {
      std::cerr << "DIAG: mapped=" << mapped
                << " record_status=" << record.status
                << " support_version=" << result.support.version
                << " support_status=" << result.support.status
                << " support_count=" << result.support.count
                << " dispatch_kernel_id=" << dispatch.kernel_id;
      if (mapped) {
        uint32_t map_first = 0U;
        std::memcpy(&map_first,
                    workspace_after.data() + scratch_bytes(vocab_size),
                    sizeof(map_first));
        std::cerr << " map_first=" << map_first;
      }
      std::cerr << '\n';
    }
    ok =
        record.status == SLLM_STATUS_OK &&
        result.support.version == sllm_token_selector_support::kVersionV1 &&
        result.support.status == SLLM_STATUS_OK &&
        // top-p=0.95 necessarily leaves one of the 20 equal-mass candidates
        // outside the support record.  The complete 20-row candidate set is
        // checked below from the fixed-K20 scratch region.
        result.support.count == sllm_token_selector_support::kMaxCountV1 - 1U &&
        dispatch.kernel_id ==
            SLLM_HIP_TOKEN_SELECTOR_KERNEL_ID_FIXED_TOPK_TOPP_V1 &&
        dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U && ok;
  }
  if (plan != nullptr) {
    ok = expect(sllm_token_selector_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "sllm_token_selector_plan_release", error) &&
         ok;
  }
  result.ok = ok;
  return result;
}

bool verify_vocab_case(const sllm_context_t *const context,
                       const sllm_queue_t *const queue,
                       const uint64_t vocab_size) {
  std::vector<uint16_t> logits(vocab_size, UINT16_C(0x0000));
  std::vector<float> additive(vocab_size, 0.0F);
  std::vector<uint8_t> mask(vocab_size, 1U);
  const uint64_t mapped_workspace_size =
      scratch_bytes(vocab_size) + vocab_size * sizeof(uint32_t);
  const uint64_t logits_bytes = vocab_size * sizeof(uint16_t);
  const uint64_t additive_bytes = vocab_size * sizeof(float);
  const uint64_t mask_bytes = vocab_size;
  sllm_buffer_t *logits_buffer = nullptr;
  sllm_buffer_t *additive_buffer = nullptr;
  sllm_buffer_t *mask_buffer = nullptr;
  sllm_buffer_t *output_buffer = nullptr;
  sllm_buffer_t *workspace_buffer = nullptr;
  bool ok = create_buffer(context, logits_bytes, &logits_buffer,
                          "create selector logits") &&
            create_buffer(context, additive_bytes, &additive_buffer,
                          "create selector additive") &&
            create_buffer(context, mask_bytes, &mask_buffer,
                          "create selector mask") &&
            create_buffer(context, SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES,
                          &output_buffer, "create selector output") &&
            create_buffer(context, mapped_workspace_size, &workspace_buffer,
                          "create selector workspace");
  if (ok) {
    const RunResult baseline =
        run_selector(context, queue, vocab_size, false, logits, additive, mask,
                     logits_buffer, additive_buffer, mask_buffer, output_buffer,
                     workspace_buffer, mapped_workspace_size);
    const RunResult mapped =
        run_selector(context, queue, vocab_size, true, logits, additive, mask,
                     logits_buffer, additive_buffer, mask_buffer, output_buffer,
                     workspace_buffer, mapped_workspace_size);
    ok = baseline.ok && mapped.ok && baseline.token_id >= 0 &&
         baseline.token_id < 20 &&
         mapped.token_id == static_cast<int32_t>(mapped_id(
                                static_cast<uint32_t>(baseline.token_id)));
    for (uint32_t index = 0U; index != 20U && ok; ++index) {
      ok = baseline.candidate_ids[index] == index &&
           mapped.candidate_ids[index] == index;
      if (index < mapped.support.count) {
        ok = mapped.support.ids[index] == mapped_id(index) &&
             baseline.support.ids[index] == index;
      }
    }
    if (!ok) {
      std::cerr << "FAIL: mapped selector vocab=" << vocab_size
                << " local_token=" << baseline.token_id
                << " mapped_token=" << mapped.token_id
                << " baseline_ok=" << baseline.ok << " mapped_ok=" << mapped.ok
                << " baseline_candidate0=" << baseline.candidate_ids[0]
                << " mapped_candidate0=" << mapped.candidate_ids[0]
                << " baseline_support0=" << baseline.support.ids[0]
                << " mapped_support0=" << mapped.support.ids[0]
                << " mapped_support_count=" << mapped.support.count << '\n';
    }
  }
  ok = release_buffer(&workspace_buffer, "release selector workspace") && ok;
  ok = release_buffer(&output_buffer, "release selector output") && ok;
  ok = release_buffer(&mask_buffer, "release selector mask") && ok;
  ok = release_buffer(&additive_buffer, "release selector additive") && ok;
  ok = release_buffer(&logits_buffer, "release selector logits") && ok;
  return ok;
}

bool verify_invalid_contract(const sllm_context_t *const context,
                             const sllm_queue_t *const queue) {
  constexpr uint64_t vocab_size = 33U;
  const uint64_t mapped_workspace_size =
      scratch_bytes(vocab_size) + vocab_size * sizeof(uint32_t);
  sllm_buffer_t *logits = nullptr;
  sllm_buffer_t *additive = nullptr;
  sllm_buffer_t *mask = nullptr;
  sllm_buffer_t *output = nullptr;
  sllm_buffer_t *workspace = nullptr;
  bool ok = create_buffer(context, vocab_size * sizeof(uint16_t), &logits,
                          "create negative logits") &&
            create_buffer(context, vocab_size * sizeof(float), &additive,
                          "create negative additive") &&
            create_buffer(context, vocab_size, &mask, "create negative mask") &&
            create_buffer(context, SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES,
                          &output, "create negative output") &&
            create_buffer(context, mapped_workspace_size, &workspace,
                          "create negative workspace");
  if (ok) {
    auto descriptor = sllm_token_selector_desc_t{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version =
        SLLM_HIP_TOKEN_SELECTOR_VERSION_FIXED_TOPK_TOPP_MAPPED;
    descriptor.logits =
        binding(logits, SLLM_TENSOR_DTYPE_BF16, 2U, 1U, vocab_size);
    descriptor.additive_logits =
        binding(additive, SLLM_TENSOR_DTYPE_F32, 2U, 1U, vocab_size);
    descriptor.valid_mask =
        binding(mask, SLLM_TENSOR_DTYPE_U8, 2U, 1U, vocab_size);
    descriptor.output = binding(output, SLLM_TENSOR_DTYPE_U8, 1U,
                                SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES);
    descriptor.vocab_size = vocab_size;
    descriptor.temperature = 1.0F;
    descriptor.top_k = 20U;
    descriptor.top_p = 0.95F;
    descriptor.workspace =
        binding(workspace, SLLM_TENSOR_DTYPE_U8, 1U, mapped_workspace_size);
    sllm_token_selector_plan_t *plan = nullptr;
    Error error;

    descriptor.flags = SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                       SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT;
    ok = expect(sllm_token_selector_prepare(context, &descriptor, &plan,
                                            &error.sink),
                SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR,
                "mapped selector missing map flag", error) &&
         ok;
    descriptor.flags |= SLLM_HIP_TOKEN_SELECTOR_FLAG_VOCAB_MAP_PRESENT;
    descriptor.top_k = 64U;
    ok = expect(sllm_token_selector_prepare(context, &descriptor, &plan,
                                            &error.sink),
                SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR,
                "mapped selector top-k 64", error) &&
         ok;
    descriptor.top_k = 20U;
    descriptor.struct_size =
        static_cast<uint32_t>(offsetof(sllm_token_selector_desc_t, top_k));
    ok = expect(sllm_token_selector_prepare(context, &descriptor, &plan,
                                            &error.sink),
                SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR,
                "mapped selector legacy struct size", error) &&
         ok;
    descriptor.struct_size = sizeof(descriptor);
    descriptor.workspace =
        binding(workspace, SLLM_TENSOR_DTYPE_U8, 1U, scratch_bytes(vocab_size));
    ok = expect(sllm_token_selector_prepare(context, &descriptor, &plan,
                                            &error.sink),
                SLLM_STATUS_SHAPE_MISMATCH, "mapped selector short workspace",
                error) &&
         ok;
    (void)queue;
  }
  ok = release_buffer(&workspace, "release negative workspace") && ok;
  ok = release_buffer(&output, "release negative output") && ok;
  ok = release_buffer(&mask, "release negative mask") && ok;
  ok = release_buffer(&additive, "release negative additive") && ok;
  ok = release_buffer(&logits, "release negative logits") && ok;
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
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  sllm_queue_t *queue = nullptr;
  bool ok = expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "sllm_queue_create", error);
  if (ok) {
    ok = verify_invalid_contract(context, queue) && ok;
    ok = verify_vocab_case(context, queue, kFirstVocab) && ok;
    ok = verify_vocab_case(context, queue, kAlignedVocab) && ok;
    ok = verify_vocab_case(context, queue, kLastVocab) && ok;
  }
  ok = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
              "sllm_queue_release", error) &&
       ok;
  ok = expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
              "sllm_context_release", error) &&
       ok;
  std::cout << (ok ? "PASS" : "FAIL")
            << " phase87_s9_mapped_selector_probe target="
            << SLLM_TEST_EXPECTED_TARGET << '\n';
  return ok ? 0 : 1;
}
