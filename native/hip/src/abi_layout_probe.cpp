#include "public_runtime_internal.hpp"
#include "sllm/hip.h"

#include <cstddef>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

bool host_fault_state_tests() {
  using sllm_public_runtime::AccountingState;
  using sllm_public_runtime::CompletionSafetyState;

  CompletionSafetyState fatal_completion;
  if (fatal_completion.can_release_graph() ||
      fatal_completion.observe_event_destroy_success()) {
    return false;
  }
  fatal_completion.quarantine();
  fatal_completion.observe_positive_completion();
  if (fatal_completion.can_release_graph() ||
      fatal_completion.observe_event_destroy_success()) {
    return false;
  }
  CompletionSafetyState completed;
  completed.observe_positive_completion();
  if (!completed.can_release_graph() ||
      !completed.observe_event_destroy_success() ||
      !completed.event_destroyed() ||
      completed.observe_event_destroy_success()) {
    return false;
  }

  CompletionSafetyState concurrent_completion;
  std::vector<std::thread> safety_workers;
  for (int worker = 0; worker != 4; ++worker) {
    safety_workers.emplace_back([&concurrent_completion]() {
      concurrent_completion.observe_positive_completion();
      concurrent_completion.observe_event_destroy_success();
    });
  }
  for (int worker = 0; worker != 4; ++worker) {
    safety_workers.emplace_back(
        [&concurrent_completion]() { concurrent_completion.quarantine(); });
  }
  for (std::thread &worker : safety_workers) {
    worker.join();
  }
  if (concurrent_completion.can_release_graph() &&
      !concurrent_completion.event_destroyed()) {
    return false;
  }
  concurrent_completion.quarantine();
  if (concurrent_completion.can_release_graph()) {
    return false;
  }

  struct ProbeOrphanRecord final {
    uintptr_t token;
  };
  sllm_public_runtime::DurableRecordOwner<ProbeOrphanRecord> orphan_owner;
  for (uintptr_t token = 1U; token <= 4096U; ++token) {
    orphan_owner.retain(ProbeOrphanRecord{token});
  }
  if (sllm_public_runtime::DurableRecordOwner<
          ProbeOrphanRecord>::has_bounded_capacity() ||
      orphan_owner.size() != 4096U) {
    return false;
  }

  AccountingState empty_context;
  AccountingState empty_queue;
  AccountingState empty_buffer;
  if (AccountingState::release_child(empty_context) ||
      AccountingState::release_lifetime_guard(empty_context) ||
      AccountingState::release_child_and_lifetime_guard(empty_context) ||
      AccountingState::release_active(empty_queue, empty_buffer) ||
      AccountingState::release_completion(empty_context, empty_queue,
                                          empty_buffer) ||
      AccountingState::rollback_submission(empty_context, empty_queue,
                                           empty_buffer)) {
    return false;
  }

  const uint64_t max = std::numeric_limits<uint64_t>::max();
  for (int dimension = 0; dimension != 6; ++dimension) {
    AccountingState context;
    AccountingState queue;
    AccountingState buffer;
    switch (dimension) {
    case 0:
      queue.active_submissions = max;
      break;
    case 1:
      buffer.active_submissions = max;
      break;
    case 2:
      queue.completion_references = max;
      break;
    case 3:
      buffer.completion_references = max;
      break;
    case 4:
      context.child_count = max;
      break;
    case 5:
      context.lifetime_guards = max;
      break;
    default:
      return false;
    }
    if (AccountingState::reserve_submission(context, queue, buffer) ||
        context.child_count != (dimension == 4 ? max : 0U) ||
        context.lifetime_guards != (dimension == 5 ? max : 0U)) {
      return false;
    }
  }

  AccountingState exhausted_guard;
  exhausted_guard.lifetime_guards = max;
  if (AccountingState::reserve_lifetime_guard(exhausted_guard)) {
    return false;
  }

  AccountingState guarded_context;
  if (!AccountingState::reserve_child(guarded_context) ||
      !AccountingState::reserve_lifetime_guard(guarded_context) ||
      !AccountingState::release_child_and_lifetime_guard(guarded_context) ||
      guarded_context.child_count != 0U ||
      guarded_context.lifetime_guards != 0U) {
    return false;
  }

  sllm_public_runtime::MonotonicTokenSource tokens;
  const uintptr_t consumed = tokens.issue();
  const uintptr_t stale_replacement = tokens.issue();
  if (consumed == 0U || stale_replacement == 0U ||
      consumed == stale_replacement) {
    return false;
  }
  return true;
}

int main() {
  if (!host_fault_state_tests()) {
    return 1;
  }
  sllm_public_runtime::MonotonicTokenSource tokens;
  const uintptr_t first_token = tokens.issue();
  const uintptr_t second_token = tokens.issue();
  const uintptr_t third_token = tokens.issue();
  if (first_token != 1U || second_token != 2U || third_token != 3U ||
      first_token == second_token || second_token == third_token ||
      first_token == third_token) {
    return 1;
  }
  sllm_public_runtime::MonotonicTokenSource exhaustion(
      std::numeric_limits<uintptr_t>::max() - 1U);
  if (exhaustion.issue() != std::numeric_limits<uintptr_t>::max() - 1U ||
      exhaustion.issue() != std::numeric_limits<uintptr_t>::max() ||
      exhaustion.issue() != 0U || exhaustion.issue() != 0U) {
    return 1;
  }
  sllm_public_runtime::AccountingState context_accounting;
  sllm_public_runtime::AccountingState queue_accounting;
  sllm_public_runtime::AccountingState buffer_accounting;
  if (!sllm_public_runtime::AccountingState::reserve_child(
          context_accounting) ||
      !sllm_public_runtime::AccountingState::release_child(
          context_accounting) ||
      context_accounting.child_count != 0U ||
      !sllm_public_runtime::AccountingState::reserve_submission(
          context_accounting, queue_accounting, buffer_accounting) ||
      queue_accounting.active_submissions != 1U ||
      buffer_accounting.active_submissions != 1U ||
      queue_accounting.completion_references != 1U ||
      buffer_accounting.completion_references != 1U ||
      context_accounting.child_count != 1U ||
      context_accounting.lifetime_guards != 1U ||
      !sllm_public_runtime::AccountingState::release_active(
          queue_accounting, buffer_accounting) ||
      !sllm_public_runtime::AccountingState::
          release_completion_and_lifetime_guard(
              context_accounting, queue_accounting, buffer_accounting) ||
      queue_accounting.active_submissions != 0U ||
      buffer_accounting.active_submissions != 0U ||
      queue_accounting.completion_references != 0U ||
      buffer_accounting.completion_references != 0U ||
      context_accounting.child_count != 0U ||
      context_accounting.lifetime_guards != 0U) {
    return 1;
  }
  context_accounting.child_count = std::numeric_limits<uint64_t>::max();
  if (sllm_public_runtime::AccountingState::reserve_child(context_accounting)) {
    return 1;
  }
  context_accounting.child_count = 0U;
  queue_accounting.active_submissions = std::numeric_limits<uint64_t>::max();
  if (sllm_public_runtime::AccountingState::reserve_submission(
          context_accounting, queue_accounting, buffer_accounting)) {
    return 1;
  }
  char short_message[4] = {};
  sllm_error_sink_t short_sink = SLLM_ERROR_SINK_INIT(short_message);
  const char bounded_source[] = "abc";
  const sllm_status_t bounded_status =
      sllm_public_runtime::write_error_n_bounded(
          &short_sink, SLLM_STATUS_INVALID_ARGUMENT, bounded_source, 300U,
          sizeof(bounded_source) - 1U);
  if (bounded_status != SLLM_STATUS_BUFFER_TOO_SMALL ||
      short_sink.message_length != 300U || short_message[0] != 'a' ||
      short_message[1] != 'b' || short_message[2] != 'c' ||
      short_message[3] != '\0') {
    return 1;
  }
#define SLLM_PRINT_CONSTANT(name)                                              \
  std::cout << "const " #name "=" << name << '\n'
  SLLM_PRINT_CONSTANT(SLLM_HIP_ABI_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_LIBRARY_VERSION_MAJOR);
  SLLM_PRINT_CONSTANT(SLLM_HIP_LIBRARY_VERSION_MINOR);
  SLLM_PRINT_CONSTANT(SLLM_HIP_LIBRARY_VERSION_PATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_OK);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ARGUMENT);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_BUFFER_TOO_SMALL);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_UNSUPPORTED);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_HIP_UNAVAILABLE);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ABI_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_RESERVED_NONZERO);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INTERNAL_ERROR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_PUBLIC_PENDING);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_PUBLIC_TIMEOUT);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_PUBLIC_INVALID_HANDLE);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_PUBLIC_DEVICE_MISMATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_PUBLIC_BUSY);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_PUBLIC_NOT_READY);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_TENSOR_BINDING);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_ZERO_EXTENT);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_SHAPE_MISMATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_STRIDE_MISMATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_METADATA_OVERFLOW);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_BUFFER_OUT_OF_BOUNDS);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_MISALIGNED_OFFSET);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_UNSUPPORTED_DTYPE);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_UNSUPPORTED_ENCODING);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_EPSILON);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_UNSUPPORTED_SCALE_MODE);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_ALIAS_OVERLAP);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_EMBEDDING_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_TOKEN_ID_OUT_OF_RANGE);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ARGMAX_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ROTARY_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_WINDOWED_ATTENTION_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_POSITION_PAYLOAD_MISMATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_KV_LENGTH_MISMATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_KV_CAPACITY_EXCEEDED);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_BF16);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F16);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F32);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F8_E4M3_FN);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_HEAD_COUNT);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_HEAD_DIM);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_MAX_HEAD_DIM);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_MAX_CAPACITY);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_STATE_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_TOKEN_MAJOR_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_STATIC_TOKEN_MAJOR_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_KERNEL_ID_BF16_TO_NVFP4_TOKEN_MAJOR_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_ENCODING_FP16_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_ENCODING_FP8_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_ENCODING_FP8_STATIC_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_KV_ENCODING_NVFP4_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3);
  SLLM_PRINT_CONSTANT(SLLM_BACKEND_HIP);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_READ);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_WRITE);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_READ_WRITE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MAX_DEVICE_NAME);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MAX_GCN_ARCH_NAME);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MAX_TRANSFER_BYTES);
  SLLM_PRINT_CONSTANT(SLLM_HIP_RMSNORM_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_ID_COPY_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_ID_ADD_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_ID_SILU_MUL_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_ID_SIGMOID_MUL_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_ID_SCALAR_MUL_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_ID_GELU_TANH_MUL_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_ID_TANH_SOFTCAP_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ELEMENTWISE_MAX_ELEMENTS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_KERNEL_ID_GATHER_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_MAX_VOCAB);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_MAX_HIDDEN);
  SLLM_PRINT_CONSTANT(SLLM_HIP_EMBEDDING_MAX_TOKENS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_FP8_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_NVFP4_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_MXFP4_W4A4_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_BASELINE_BF16_FP32_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_TILED16_BF16_FP32_V2);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_DECODE_BF16_FP32_V2);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_HIPBLAS_DECODE_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_HIPBLASLT_FP8_OUTER_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_FP8_BYTE_EMULATION_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_NVFP4_PACKED_DEQUANT_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_NVFP4_W4A4_PACKED_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_SERIAL_ROWS_BF16_FP32_V1);
  SLLM_PRINT_CONSTANT(
      SLLM_HIP_MATMUL_KERNEL_ID_SERIAL_ROWS_WAVE64_BF16_FP32_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_MXFP4_W4A4_DECODE_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_ID_MXFP4_W4A4_PREFILL_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_MAX_M);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_MAX_K);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_MAX_N);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MATMUL_MAX_OUTPUT_ELEMENTS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_KERNEL_ID_BASELINE_BF16_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_MAX_V);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ARGMAX_MAX_M);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_KERNEL_ID_STABLE_TOPK_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_MAX_TOKENS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_MAX_EXPERTS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_ROUTE_MAX_SELECTED);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_KERNEL_ID_DECODE_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_KERNEL_ID_PREFILL_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_HIDDEN_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_INTERMEDIATE_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_COUNT);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_TOPK);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_LAYER_BLOB_BYTES);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MOE_EXPERT_MAX_TOKENS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_ID_BASELINE_BF16_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_K_HEAD_DIM);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_QGATE_HEAD_DIM);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_ROTARY_DIM);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_MAX_POSITION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ATTENTION_PREPROCESS_MAX_M);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_KERNEL_ID_SPLIT_HALF_BF16_FP32_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_MAX_M);
  SLLM_PRINT_CONSTANT(SLLM_HIP_ROTARY_MAX_POSITION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_DISPATCH_INFO_VERSION);
  SLLM_PRINT_CONSTANT(
      SLLM_HIP_WINDOWED_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_GQA_BF16_V1);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_KERNEL_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_DEVICE_SYMBOL_MAX);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_WORKGROUP_SIZE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_MAX_M);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_MAX_KV);
  SLLM_PRINT_CONSTANT(SLLM_HIP_WINDOWED_ATTENTION_MAX_HEAD_DIM);
  SLLM_PRINT_CONSTANT(SLLM_HIP_TENSOR_MAX_RANK);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_BF16);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F32);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F8_E4M3_FN);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_U8);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_I32);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_ENCODING_UNQUANTIZED);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_ENCODING_FP8_OUTER_F32);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_ENCODING_MXFP4_W4A4_BLOCK32_E8M0);
  SLLM_PRINT_CONSTANT(SLLM_RMSNORM_ACCUMULATION_F32);
  SLLM_PRINT_CONSTANT(SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE);
  SLLM_PRINT_CONSTANT(SLLM_RMSNORM_SCALE_MODE_DIRECT);
  SLLM_PRINT_CONSTANT(SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP);
  SLLM_PRINT_CONSTANT(SLLM_ELEMENTWISE_OPERATION_COPY);
  SLLM_PRINT_CONSTANT(SLLM_ELEMENTWISE_OPERATION_ADD);
  SLLM_PRINT_CONSTANT(SLLM_ELEMENTWISE_OPERATION_SILU_MUL);
  SLLM_PRINT_CONSTANT(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL);
  SLLM_PRINT_CONSTANT(SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL);
  SLLM_PRINT_CONSTANT(SLLM_ELEMENTWISE_OPERATION_GELU_TANH_MUL);
  SLLM_PRINT_CONSTANT(SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP);
  SLLM_PRINT_CONSTANT(SLLM_COMPLETION_STATE_PENDING);
  SLLM_PRINT_CONSTANT(SLLM_COMPLETION_STATE_SUCCESS);
  SLLM_PRINT_CONSTANT(SLLM_COMPLETION_STATE_FAILURE);
#undef SLLM_PRINT_CONSTANT

  std::cout << "layout sllm_error_sink_t size=" << sizeof(sllm_error_sink_t)
            << " align=" << alignof(sllm_error_sink_t)
            << " struct_size=" << offsetof(sllm_error_sink_t, struct_size)
            << " abi_version=" << offsetof(sllm_error_sink_t, abi_version)
            << " message=" << offsetof(sllm_error_sink_t, message)
            << " message_capacity="
            << offsetof(sllm_error_sink_t, message_capacity)
            << " message_length=" << offsetof(sllm_error_sink_t, message_length)
            << " reserved=" << offsetof(sllm_error_sink_t, reserved) << '\n';
  std::cout << "layout sllm_version_info_t size=" << sizeof(sllm_version_info_t)
            << " align=" << alignof(sllm_version_info_t)
            << " struct_size=" << offsetof(sllm_version_info_t, struct_size)
            << " abi_version=" << offsetof(sllm_version_info_t, abi_version)
            << " major=" << offsetof(sllm_version_info_t, major)
            << " minor=" << offsetof(sllm_version_info_t, minor)
            << " patch=" << offsetof(sllm_version_info_t, patch)
            << " reserved=" << offsetof(sllm_version_info_t, reserved) << '\n';
  std::cout << "layout sllm_backend_probe_result_t size="
            << sizeof(sllm_backend_probe_result_t)
            << " align=" << alignof(sllm_backend_probe_result_t)
            << " struct_size="
            << offsetof(sllm_backend_probe_result_t, struct_size)
            << " abi_version="
            << offsetof(sllm_backend_probe_result_t, abi_version)
            << " backend=" << offsetof(sllm_backend_probe_result_t, backend)
            << " available=" << offsetof(sllm_backend_probe_result_t, available)
            << " hip_runtime_present="
            << offsetof(sllm_backend_probe_result_t, hip_runtime_present)
            << " reserved=" << offsetof(sllm_backend_probe_result_t, reserved)
            << '\n';
  std::cout << "layout sllm_context_probe_result_t size="
            << sizeof(sllm_context_probe_result_t)
            << " align=" << alignof(sllm_context_probe_result_t)
            << " struct_size="
            << offsetof(sllm_context_probe_result_t, struct_size)
            << " abi_version="
            << offsetof(sllm_context_probe_result_t, abi_version)
            << " context_present="
            << offsetof(sllm_context_probe_result_t, context_present)
            << " hip_available="
            << offsetof(sllm_context_probe_result_t, hip_available)
            << " reserved=" << offsetof(sllm_context_probe_result_t, reserved)
            << '\n';
  std::cout << "layout sllm_device_info_t size=" << sizeof(sllm_device_info_t)
            << " align=" << alignof(sllm_device_info_t)
            << " struct_size=" << offsetof(sllm_device_info_t, struct_size)
            << " abi_version=" << offsetof(sllm_device_info_t, abi_version)
            << " device_index=" << offsetof(sllm_device_info_t, device_index)
            << " visible_device_count="
            << offsetof(sllm_device_info_t, visible_device_count)
            << " total_memory_bytes="
            << offsetof(sllm_device_info_t, total_memory_bytes)
            << " wavefront_size="
            << offsetof(sllm_device_info_t, wavefront_size)
            << " reserved0=" << offsetof(sllm_device_info_t, reserved0)
            << " name=" << offsetof(sllm_device_info_t, name)
            << " gcn_arch_name=" << offsetof(sllm_device_info_t, gcn_arch_name)
            << " available_memory_bytes="
            << offsetof(sllm_device_info_t, available_memory_bytes)
            << " reserved=" << offsetof(sllm_device_info_t, reserved) << '\n';
  std::cout
      << "layout sllm_context_create_info_t size="
      << sizeof(sllm_context_create_info_t)
      << " align=" << alignof(sllm_context_create_info_t)
      << " struct_size=" << offsetof(sllm_context_create_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_context_create_info_t, abi_version)
      << " device_index=" << offsetof(sllm_context_create_info_t, device_index)
      << " flags=" << offsetof(sllm_context_create_info_t, flags)
      << " expected_gcn_arch_name="
      << offsetof(sllm_context_create_info_t, expected_gcn_arch_name)
      << " reserved=" << offsetof(sllm_context_create_info_t, reserved) << '\n';
  std::cout << "layout sllm_queue_create_info_t size="
            << sizeof(sllm_queue_create_info_t)
            << " align=" << alignof(sllm_queue_create_info_t) << " struct_size="
            << offsetof(sllm_queue_create_info_t, struct_size)
            << " abi_version="
            << offsetof(sllm_queue_create_info_t, abi_version)
            << " flags=" << offsetof(sllm_queue_create_info_t, flags)
            << " reserved=" << offsetof(sllm_queue_create_info_t, reserved)
            << '\n';
  std::cout << "layout sllm_buffer_create_info_t size="
            << sizeof(sllm_buffer_create_info_t)
            << " align=" << alignof(sllm_buffer_create_info_t)
            << " struct_size="
            << offsetof(sllm_buffer_create_info_t, struct_size)
            << " abi_version="
            << offsetof(sllm_buffer_create_info_t, abi_version)
            << " size_bytes=" << offsetof(sllm_buffer_create_info_t, size_bytes)
            << " alignment_bytes="
            << offsetof(sllm_buffer_create_info_t, alignment_bytes)
            << " flags=" << offsetof(sllm_buffer_create_info_t, flags)
            << " reserved=" << offsetof(sllm_buffer_create_info_t, reserved)
            << '\n';
  std::cout << "layout sllm_transfer_desc_t size="
            << sizeof(sllm_transfer_desc_t)
            << " align=" << alignof(sllm_transfer_desc_t)
            << " struct_size=" << offsetof(sllm_transfer_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_transfer_desc_t, abi_version)
            << " host_pointer=" << offsetof(sllm_transfer_desc_t, host_pointer)
            << " buffer_offset_bytes="
            << offsetof(sllm_transfer_desc_t, buffer_offset_bytes)
            << " size_bytes=" << offsetof(sllm_transfer_desc_t, size_bytes)
            << " reserved=" << offsetof(sllm_transfer_desc_t, reserved) << '\n';
  std::cout << "layout sllm_completion_result_t size="
            << sizeof(sllm_completion_result_t)
            << " align=" << alignof(sllm_completion_result_t) << " struct_size="
            << offsetof(sllm_completion_result_t, struct_size)
            << " abi_version="
            << offsetof(sllm_completion_result_t, abi_version)
            << " state=" << offsetof(sllm_completion_result_t, state)
            << " reserved0=" << offsetof(sllm_completion_result_t, reserved0)
            << " transfer_size_bytes="
            << offsetof(sllm_completion_result_t, transfer_size_bytes)
            << " available_bytes="
            << offsetof(sllm_completion_result_t, available_bytes)
            << " reserved=" << offsetof(sllm_completion_result_t, reserved)
            << '\n';
  std::cout << "layout sllm_tensor_binding_t size="
            << sizeof(sllm_tensor_binding_t)
            << " align=" << alignof(sllm_tensor_binding_t)
            << " struct_size=" << offsetof(sllm_tensor_binding_t, struct_size)
            << " abi_version=" << offsetof(sllm_tensor_binding_t, abi_version)
            << " buffer=" << offsetof(sllm_tensor_binding_t, buffer)
            << " byte_offset=" << offsetof(sllm_tensor_binding_t, byte_offset)
            << " dtype=" << offsetof(sllm_tensor_binding_t, dtype)
            << " encoding=" << offsetof(sllm_tensor_binding_t, encoding)
            << " rank=" << offsetof(sllm_tensor_binding_t, rank)
            << " reserved0=" << offsetof(sllm_tensor_binding_t, reserved0)
            << " shape=" << offsetof(sllm_tensor_binding_t, shape)
            << " stride_elements="
            << offsetof(sllm_tensor_binding_t, stride_elements)
            << " reserved=" << offsetof(sllm_tensor_binding_t, reserved)
            << '\n';
  std::cout << "layout sllm_rmsnorm_desc_t size=" << sizeof(sllm_rmsnorm_desc_t)
            << " align=" << alignof(sllm_rmsnorm_desc_t)
            << " struct_size=" << offsetof(sllm_rmsnorm_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_rmsnorm_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_rmsnorm_desc_t, op_version)
            << " accumulation_dtype="
            << offsetof(sllm_rmsnorm_desc_t, accumulation_dtype)
            << " scale_mode=" << offsetof(sllm_rmsnorm_desc_t, scale_mode)
            << " alias_policy=" << offsetof(sllm_rmsnorm_desc_t, alias_policy)
            << " epsilon_bits=" << offsetof(sllm_rmsnorm_desc_t, epsilon_bits)
            << " reserved=" << offsetof(sllm_rmsnorm_desc_t, reserved)
            << " activation=" << offsetof(sllm_rmsnorm_desc_t, activation)
            << " raw_scale=" << offsetof(sllm_rmsnorm_desc_t, raw_scale)
            << " output=" << offsetof(sllm_rmsnorm_desc_t, output) << '\n';
  std::cout
      << "layout sllm_rmsnorm_dispatch_info_t size="
      << sizeof(sllm_rmsnorm_dispatch_info_t)
      << " align=" << alignof(sllm_rmsnorm_dispatch_info_t)
      << " struct_size=" << offsetof(sllm_rmsnorm_dispatch_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_rmsnorm_dispatch_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_rmsnorm_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_rmsnorm_dispatch_info_t, backend)
      << " dispatch_id=" << offsetof(sllm_rmsnorm_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_rmsnorm_dispatch_info_t, dispatch_count)
      << " kernel_id=" << offsetof(sllm_rmsnorm_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_rmsnorm_dispatch_info_t, workgroup_size_x)
      << " grid_size_x=" << offsetof(sllm_rmsnorm_dispatch_info_t, grid_size_x)
      << " row_count=" << offsetof(sllm_rmsnorm_dispatch_info_t, row_count)
      << " normalized_size="
      << offsetof(sllm_rmsnorm_dispatch_info_t, normalized_size)
      << " fallback_allowed="
      << offsetof(sllm_rmsnorm_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_rmsnorm_dispatch_info_t, fallback_used)
      << " kernel_symbol="
      << offsetof(sllm_rmsnorm_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_rmsnorm_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_rmsnorm_dispatch_info_t, gcn_arch_name)
      << " reserved=" << offsetof(sllm_rmsnorm_dispatch_info_t, reserved)
      << '\n';
  std::cout << "layout sllm_elementwise_desc_t size="
            << sizeof(sllm_elementwise_desc_t)
            << " align=" << alignof(sllm_elementwise_desc_t)
            << " struct_size=" << offsetof(sllm_elementwise_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_elementwise_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_elementwise_desc_t, op_version)
            << " operation=" << offsetof(sllm_elementwise_desc_t, operation)
            << " reserved=" << offsetof(sllm_elementwise_desc_t, reserved)
            << " input0=" << offsetof(sllm_elementwise_desc_t, input0)
            << " input1=" << offsetof(sllm_elementwise_desc_t, input1)
            << " output=" << offsetof(sllm_elementwise_desc_t, output) << '\n';
  std::cout
      << "layout sllm_elementwise_dispatch_info_t size="
      << sizeof(sllm_elementwise_dispatch_info_t)
      << " align=" << alignof(sllm_elementwise_dispatch_info_t)
      << " struct_size="
      << offsetof(sllm_elementwise_dispatch_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_elementwise_dispatch_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_elementwise_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_elementwise_dispatch_info_t, backend)
      << " dispatch_id="
      << offsetof(sllm_elementwise_dispatch_info_t, dispatch_id)
      << " operation=" << offsetof(sllm_elementwise_dispatch_info_t, operation)
      << " dispatch_count="
      << offsetof(sllm_elementwise_dispatch_info_t, dispatch_count)
      << " kernel_id=" << offsetof(sllm_elementwise_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_elementwise_dispatch_info_t, workgroup_size_x)
      << " grid_size_x="
      << offsetof(sllm_elementwise_dispatch_info_t, grid_size_x)
      << " fallback_allowed="
      << offsetof(sllm_elementwise_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_elementwise_dispatch_info_t, fallback_used)
      << " element_count="
      << offsetof(sllm_elementwise_dispatch_info_t, element_count)
      << " kernel_symbol="
      << offsetof(sllm_elementwise_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_elementwise_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_elementwise_dispatch_info_t, gcn_arch_name)
      << " reserved=" << offsetof(sllm_elementwise_dispatch_info_t, reserved)
      << '\n';
  std::cout << "layout sllm_embedding_desc_t size="
            << sizeof(sllm_embedding_desc_t)
            << " align=" << alignof(sllm_embedding_desc_t)
            << " struct_size=" << offsetof(sllm_embedding_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_embedding_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_embedding_desc_t, op_version)
            << " reserved=" << offsetof(sllm_embedding_desc_t, reserved)
            << " weight=" << offsetof(sllm_embedding_desc_t, weight)
            << " token_ids=" << offsetof(sllm_embedding_desc_t, token_ids)
            << " output=" << offsetof(sllm_embedding_desc_t, output) << '\n';
  std::cout << "layout sllm_embedding_dispatch_info_t size="
            << sizeof(sllm_embedding_dispatch_info_t)
            << " align=" << alignof(sllm_embedding_dispatch_info_t)
            << " struct_size="
            << offsetof(sllm_embedding_dispatch_info_t, struct_size)
            << " abi_version="
            << offsetof(sllm_embedding_dispatch_info_t, abi_version)
            << " info_version="
            << offsetof(sllm_embedding_dispatch_info_t, info_version)
            << " backend=" << offsetof(sllm_embedding_dispatch_info_t, backend)
            << " dispatch_id="
            << offsetof(sllm_embedding_dispatch_info_t, dispatch_id)
            << " dispatch_count="
            << offsetof(sllm_embedding_dispatch_info_t, dispatch_count)
            << " kernel_id="
            << offsetof(sllm_embedding_dispatch_info_t, kernel_id)
            << " workgroup_size_x="
            << offsetof(sllm_embedding_dispatch_info_t, workgroup_size_x)
            << " grid_size_x="
            << offsetof(sllm_embedding_dispatch_info_t, grid_size_x)
            << " fallback_allowed="
            << offsetof(sllm_embedding_dispatch_info_t, fallback_allowed)
            << " fallback_used="
            << offsetof(sllm_embedding_dispatch_info_t, fallback_used)
            << " token_count="
            << offsetof(sllm_embedding_dispatch_info_t, token_count)
            << " hidden_size="
            << offsetof(sllm_embedding_dispatch_info_t, hidden_size)
            << " vocab_size="
            << offsetof(sllm_embedding_dispatch_info_t, vocab_size)
            << " kernel_symbol="
            << offsetof(sllm_embedding_dispatch_info_t, kernel_symbol)
            << " device_symbol="
            << offsetof(sllm_embedding_dispatch_info_t, device_symbol)
            << " gcn_arch_name="
            << offsetof(sllm_embedding_dispatch_info_t, gcn_arch_name)
            << " reserved="
            << offsetof(sllm_embedding_dispatch_info_t, reserved) << '\n';
  std::cout << "layout sllm_matmul_desc_t size=" << sizeof(sllm_matmul_desc_t)
            << " align=" << alignof(sllm_matmul_desc_t)
            << " struct_size=" << offsetof(sllm_matmul_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_matmul_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_matmul_desc_t, op_version)
            << " reserved=" << offsetof(sllm_matmul_desc_t, reserved)
            << " activation=" << offsetof(sllm_matmul_desc_t, activation)
            << " weight=" << offsetof(sllm_matmul_desc_t, weight)
            << " output=" << offsetof(sllm_matmul_desc_t, output) << '\n';
  std::cout
      << "layout sllm_matmul_dispatch_info_t size="
      << sizeof(sllm_matmul_dispatch_info_t)
      << " align=" << alignof(sllm_matmul_dispatch_info_t)
      << " struct_size=" << offsetof(sllm_matmul_dispatch_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_matmul_dispatch_info_t, abi_version)
      << " info_version=" << offsetof(sllm_matmul_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_matmul_dispatch_info_t, backend)
      << " dispatch_id=" << offsetof(sllm_matmul_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_matmul_dispatch_info_t, dispatch_count)
      << " kernel_id=" << offsetof(sllm_matmul_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_matmul_dispatch_info_t, workgroup_size_x)
      << " grid_size_x=" << offsetof(sllm_matmul_dispatch_info_t, grid_size_x)
      << " fallback_allowed="
      << offsetof(sllm_matmul_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_matmul_dispatch_info_t, fallback_used)
      << " m=" << offsetof(sllm_matmul_dispatch_info_t, m)
      << " k=" << offsetof(sllm_matmul_dispatch_info_t, k)
      << " n=" << offsetof(sllm_matmul_dispatch_info_t, n)
      << " output_elements="
      << offsetof(sllm_matmul_dispatch_info_t, output_elements)
      << " kernel_symbol="
      << offsetof(sllm_matmul_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_matmul_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_matmul_dispatch_info_t, gcn_arch_name)
      << " reserved=" << offsetof(sllm_matmul_dispatch_info_t, reserved)
      << '\n';
  std::cout << "layout sllm_argmax_desc_t size=" << sizeof(sllm_argmax_desc_t)
            << " align=" << alignof(sllm_argmax_desc_t)
            << " struct_size=" << offsetof(sllm_argmax_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_argmax_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_argmax_desc_t, op_version)
            << " reserved=" << offsetof(sllm_argmax_desc_t, reserved)
            << " logits=" << offsetof(sllm_argmax_desc_t, logits)
            << " output=" << offsetof(sllm_argmax_desc_t, output) << '\n';
  std::cout
      << "layout sllm_argmax_dispatch_info_t size="
      << sizeof(sllm_argmax_dispatch_info_t)
      << " align=" << alignof(sllm_argmax_dispatch_info_t)
      << " struct_size=" << offsetof(sllm_argmax_dispatch_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_argmax_dispatch_info_t, abi_version)
      << " info_version=" << offsetof(sllm_argmax_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_argmax_dispatch_info_t, backend)
      << " dispatch_id=" << offsetof(sllm_argmax_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_argmax_dispatch_info_t, dispatch_count)
      << " kernel_id=" << offsetof(sllm_argmax_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_argmax_dispatch_info_t, workgroup_size_x)
      << " grid_size_x=" << offsetof(sllm_argmax_dispatch_info_t, grid_size_x)
      << " row_count=" << offsetof(sllm_argmax_dispatch_info_t, row_count)
      << " vocab_size=" << offsetof(sllm_argmax_dispatch_info_t, vocab_size)
      << " fallback_allowed="
      << offsetof(sllm_argmax_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_argmax_dispatch_info_t, fallback_used)
      << " kernel_symbol="
      << offsetof(sllm_argmax_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_argmax_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_argmax_dispatch_info_t, gcn_arch_name)
      << " reserved=" << offsetof(sllm_argmax_dispatch_info_t, reserved)
      << '\n';
  std::cout << "layout sllm_moe_route_desc_t size="
            << sizeof(sllm_moe_route_desc_t)
            << " align=" << alignof(sllm_moe_route_desc_t)
            << " struct_size=" << offsetof(sllm_moe_route_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_moe_route_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_moe_route_desc_t, op_version)
            << " selected_expert_count="
            << offsetof(sllm_moe_route_desc_t, selected_expert_count)
            << " reserved=" << offsetof(sllm_moe_route_desc_t, reserved)
            << " logits=" << offsetof(sllm_moe_route_desc_t, logits)
            << " metadata=" << offsetof(sllm_moe_route_desc_t, metadata)
            << '\n';
  std::cout
      << "layout sllm_moe_route_dispatch_info_t size="
      << sizeof(sllm_moe_route_dispatch_info_t)
      << " align=" << alignof(sllm_moe_route_dispatch_info_t) << " struct_size="
      << offsetof(sllm_moe_route_dispatch_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_moe_route_dispatch_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_moe_route_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_moe_route_dispatch_info_t, backend)
      << " dispatch_id="
      << offsetof(sllm_moe_route_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_moe_route_dispatch_info_t, dispatch_count)
      << " kernel_id=" << offsetof(sllm_moe_route_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_moe_route_dispatch_info_t, workgroup_size_x)
      << " grid_size_x="
      << offsetof(sllm_moe_route_dispatch_info_t, grid_size_x)
      << " token_count="
      << offsetof(sllm_moe_route_dispatch_info_t, token_count)
      << " expert_count="
      << offsetof(sllm_moe_route_dispatch_info_t, expert_count)
      << " pair_count=" << offsetof(sllm_moe_route_dispatch_info_t, pair_count)
      << " selected_expert_count="
      << offsetof(sllm_moe_route_dispatch_info_t, selected_expert_count)
      << " fallback_allowed="
      << offsetof(sllm_moe_route_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_moe_route_dispatch_info_t, fallback_used)
      << " reserved0=" << offsetof(sllm_moe_route_dispatch_info_t, reserved0)
      << " kernel_symbol="
      << offsetof(sllm_moe_route_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_moe_route_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_moe_route_dispatch_info_t, gcn_arch_name)
      << " reserved=" << offsetof(sllm_moe_route_dispatch_info_t, reserved)
      << '\n';
  std::cout << "layout sllm_moe_expert_desc_t size="
            << sizeof(sllm_moe_expert_desc_t)
            << " align=" << alignof(sllm_moe_expert_desc_t)
            << " struct_size=" << offsetof(sllm_moe_expert_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_moe_expert_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_moe_expert_desc_t, op_version)
            << " reserved0=" << offsetof(sllm_moe_expert_desc_t, reserved0)
            << " reserved=" << offsetof(sllm_moe_expert_desc_t, reserved)
            << " hidden=" << offsetof(sllm_moe_expert_desc_t, hidden)
            << " routing_metadata="
            << offsetof(sllm_moe_expert_desc_t, routing_metadata)
            << " layer_blob=" << offsetof(sllm_moe_expert_desc_t, layer_blob)
            << " workspace=" << offsetof(sllm_moe_expert_desc_t, workspace)
            << " output=" << offsetof(sllm_moe_expert_desc_t, output) << '\n';
  std::cout << "layout sllm_moe_expert_dispatch_info_t size="
            << sizeof(sllm_moe_expert_dispatch_info_t)
            << " align=" << alignof(sllm_moe_expert_dispatch_info_t)
            << " struct_size="
            << offsetof(sllm_moe_expert_dispatch_info_t, struct_size)
            << " abi_version="
            << offsetof(sllm_moe_expert_dispatch_info_t, abi_version)
            << " info_version="
            << offsetof(sllm_moe_expert_dispatch_info_t, info_version)
            << " backend=" << offsetof(sllm_moe_expert_dispatch_info_t, backend)
            << " dispatch_id="
            << offsetof(sllm_moe_expert_dispatch_info_t, dispatch_id)
            << " dispatch_count="
            << offsetof(sllm_moe_expert_dispatch_info_t, dispatch_count)
            << " kernel_id="
            << offsetof(sllm_moe_expert_dispatch_info_t, kernel_id)
            << " workgroup_size_x="
            << offsetof(sllm_moe_expert_dispatch_info_t, workgroup_size_x)
            << " grid_size_x="
            << offsetof(sllm_moe_expert_dispatch_info_t, grid_size_x)
            << " token_count="
            << offsetof(sllm_moe_expert_dispatch_info_t, token_count)
            << " active_pair_count="
            << offsetof(sllm_moe_expert_dispatch_info_t, active_pair_count)
            << " workspace_bytes="
            << offsetof(sllm_moe_expert_dispatch_info_t, workspace_bytes)
            << " selected_expert_count="
            << offsetof(sllm_moe_expert_dispatch_info_t, selected_expert_count)
            << " shared_expert_count="
            << offsetof(sllm_moe_expert_dispatch_info_t, shared_expert_count)
            << " fallback_allowed="
            << offsetof(sllm_moe_expert_dispatch_info_t, fallback_allowed)
            << " fallback_used="
            << offsetof(sllm_moe_expert_dispatch_info_t, fallback_used)
            << " kernel_symbol="
            << offsetof(sllm_moe_expert_dispatch_info_t, kernel_symbol)
            << " device_symbol="
            << offsetof(sllm_moe_expert_dispatch_info_t, device_symbol)
            << " gcn_arch_name="
            << offsetof(sllm_moe_expert_dispatch_info_t, gcn_arch_name)
            << " reserved="
            << offsetof(sllm_moe_expert_dispatch_info_t, reserved) << '\n';
  std::cout
      << "layout sllm_attention_preprocess_desc_t size="
      << sizeof(sllm_attention_preprocess_desc_t)
      << " align=" << alignof(sllm_attention_preprocess_desc_t)
      << " struct_size="
      << offsetof(sllm_attention_preprocess_desc_t, struct_size)
      << " abi_version="
      << offsetof(sllm_attention_preprocess_desc_t, abi_version)
      << " op_version="
      << offsetof(sllm_attention_preprocess_desc_t, op_version)
      << " start_position="
      << offsetof(sllm_attention_preprocess_desc_t, start_position)
      << " reserved=" << offsetof(sllm_attention_preprocess_desc_t, reserved)
      << " packed_q_gate="
      << offsetof(sllm_attention_preprocess_desc_t, packed_q_gate)
      << " k=" << offsetof(sllm_attention_preprocess_desc_t, k)
      << " q_raw_scale="
      << offsetof(sllm_attention_preprocess_desc_t, q_raw_scale)
      << " k_raw_scale="
      << offsetof(sllm_attention_preprocess_desc_t, k_raw_scale)
      << " positions=" << offsetof(sllm_attention_preprocess_desc_t, positions)
      << " q_output=" << offsetof(sllm_attention_preprocess_desc_t, q_output)
      << " gate_output="
      << offsetof(sllm_attention_preprocess_desc_t, gate_output)
      << " k_output=" << offsetof(sllm_attention_preprocess_desc_t, k_output)
      << '\n';
  std::cout
      << "layout sllm_attention_preprocess_dispatch_info_t size="
      << sizeof(sllm_attention_preprocess_dispatch_info_t)
      << " align=" << alignof(sllm_attention_preprocess_dispatch_info_t)
      << " struct_size="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, info_version)
      << " backend="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, backend)
      << " dispatch_id="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, dispatch_count)
      << " kernel_id="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, workgroup_size_x)
      << " grid_size_x="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, grid_size_x)
      << " m=" << offsetof(sllm_attention_preprocess_dispatch_info_t, m)
      << " q_heads="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, q_heads)
      << " k_heads="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, k_heads)
      << " q_head_dim="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, q_head_dim)
      << " k_head_dim="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, k_head_dim)
      << " rotary_dim="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, rotary_dim)
      << " start_position="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, start_position)
      << " fallback_allowed="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, fallback_used)
      << " kernel_symbol="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, gcn_arch_name)
      << " reserved="
      << offsetof(sllm_attention_preprocess_dispatch_info_t, reserved) << '\n';
  std::cout << "layout sllm_rotary_desc_t size=" << sizeof(sllm_rotary_desc_t)
            << " align=" << alignof(sllm_rotary_desc_t)
            << " struct_size=" << offsetof(sllm_rotary_desc_t, struct_size)
            << " abi_version=" << offsetof(sllm_rotary_desc_t, abi_version)
            << " op_version=" << offsetof(sllm_rotary_desc_t, op_version)
            << " reserved0=" << offsetof(sllm_rotary_desc_t, reserved0)
            << " start_position="
            << offsetof(sllm_rotary_desc_t, start_position)
            << " q_heads=" << offsetof(sllm_rotary_desc_t, q_heads)
            << " kv_heads=" << offsetof(sllm_rotary_desc_t, kv_heads)
            << " head_dim=" << offsetof(sllm_rotary_desc_t, head_dim)
            << " rotary_dim=" << offsetof(sllm_rotary_desc_t, rotary_dim)
            << " theta_bits=" << offsetof(sllm_rotary_desc_t, theta_bits)
            << " max_position=" << offsetof(sllm_rotary_desc_t, max_position)
            << " reserved=" << offsetof(sllm_rotary_desc_t, reserved)
            << " query=" << offsetof(sllm_rotary_desc_t, query)
            << " key=" << offsetof(sllm_rotary_desc_t, key)
            << " positions=" << offsetof(sllm_rotary_desc_t, positions)
            << " query_output=" << offsetof(sllm_rotary_desc_t, query_output)
            << " key_output=" << offsetof(sllm_rotary_desc_t, key_output)
            << '\n';
  std::cout
      << "layout sllm_rotary_dispatch_info_t size="
      << sizeof(sllm_rotary_dispatch_info_t)
      << " align=" << alignof(sllm_rotary_dispatch_info_t)
      << " struct_size=" << offsetof(sllm_rotary_dispatch_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_rotary_dispatch_info_t, abi_version)
      << " info_version=" << offsetof(sllm_rotary_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_rotary_dispatch_info_t, backend)
      << " dispatch_id=" << offsetof(sllm_rotary_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_rotary_dispatch_info_t, dispatch_count)
      << " kernel_id=" << offsetof(sllm_rotary_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_rotary_dispatch_info_t, workgroup_size_x)
      << " grid_size_x=" << offsetof(sllm_rotary_dispatch_info_t, grid_size_x)
      << " token_count=" << offsetof(sllm_rotary_dispatch_info_t, token_count)
      << " q_heads=" << offsetof(sllm_rotary_dispatch_info_t, q_heads)
      << " kv_heads=" << offsetof(sllm_rotary_dispatch_info_t, kv_heads)
      << " head_dim=" << offsetof(sllm_rotary_dispatch_info_t, head_dim)
      << " rotary_dim=" << offsetof(sllm_rotary_dispatch_info_t, rotary_dim)
      << " start_position="
      << offsetof(sllm_rotary_dispatch_info_t, start_position)
      << " max_position=" << offsetof(sllm_rotary_dispatch_info_t, max_position)
      << " fallback_allowed="
      << offsetof(sllm_rotary_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_rotary_dispatch_info_t, fallback_used)
      << " kernel_symbol="
      << offsetof(sllm_rotary_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_rotary_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_rotary_dispatch_info_t, gcn_arch_name)
      << " reserved=" << offsetof(sllm_rotary_dispatch_info_t, reserved)
      << '\n';
  std::cout
      << "layout sllm_windowed_attention_desc_t size="
      << sizeof(sllm_windowed_attention_desc_t)
      << " align=" << alignof(sllm_windowed_attention_desc_t) << " struct_size="
      << offsetof(sllm_windowed_attention_desc_t, struct_size)
      << " abi_version="
      << offsetof(sllm_windowed_attention_desc_t, abi_version)
      << " op_version=" << offsetof(sllm_windowed_attention_desc_t, op_version)
      << " reserved0=" << offsetof(sllm_windowed_attention_desc_t, reserved0)
      << " start_position="
      << offsetof(sllm_windowed_attention_desc_t, start_position)
      << " expected_kv_length="
      << offsetof(sllm_windowed_attention_desc_t, expected_kv_length)
      << " sliding_window="
      << offsetof(sllm_windowed_attention_desc_t, sliding_window)
      << " q_heads=" << offsetof(sllm_windowed_attention_desc_t, q_heads)
      << " kv_heads=" << offsetof(sllm_windowed_attention_desc_t, kv_heads)
      << " head_dim=" << offsetof(sllm_windowed_attention_desc_t, head_dim)
      << " scaling_bits="
      << offsetof(sllm_windowed_attention_desc_t, scaling_bits)
      << " reserved=" << offsetof(sllm_windowed_attention_desc_t, reserved)
      << " query=" << offsetof(sllm_windowed_attention_desc_t, query)
      << " key=" << offsetof(sllm_windowed_attention_desc_t, key)
      << " value=" << offsetof(sllm_windowed_attention_desc_t, value)
      << " output=" << offsetof(sllm_windowed_attention_desc_t, output) << '\n';
  std::cout
      << "layout sllm_windowed_attention_dispatch_info_t size="
      << sizeof(sllm_windowed_attention_dispatch_info_t)
      << " align=" << alignof(sllm_windowed_attention_dispatch_info_t)
      << " struct_size="
      << offsetof(sllm_windowed_attention_dispatch_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_windowed_attention_dispatch_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_windowed_attention_dispatch_info_t, info_version)
      << " backend="
      << offsetof(sllm_windowed_attention_dispatch_info_t, backend)
      << " dispatch_id="
      << offsetof(sllm_windowed_attention_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_windowed_attention_dispatch_info_t, dispatch_count)
      << " kernel_id="
      << offsetof(sllm_windowed_attention_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_windowed_attention_dispatch_info_t, workgroup_size_x)
      << " grid_size_x="
      << offsetof(sllm_windowed_attention_dispatch_info_t, grid_size_x)
      << " query_count="
      << offsetof(sllm_windowed_attention_dispatch_info_t, query_count)
      << " start_position="
      << offsetof(sllm_windowed_attention_dispatch_info_t, start_position)
      << " committed_kv_length="
      << offsetof(sllm_windowed_attention_dispatch_info_t, committed_kv_length)
      << " sliding_window="
      << offsetof(sllm_windowed_attention_dispatch_info_t, sliding_window)
      << " q_heads="
      << offsetof(sllm_windowed_attention_dispatch_info_t, q_heads)
      << " kv_heads="
      << offsetof(sllm_windowed_attention_dispatch_info_t, kv_heads)
      << " head_dim="
      << offsetof(sllm_windowed_attention_dispatch_info_t, head_dim)
      << " scaling_bits="
      << offsetof(sllm_windowed_attention_dispatch_info_t, scaling_bits)
      << " fallback_allowed="
      << offsetof(sllm_windowed_attention_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_windowed_attention_dispatch_info_t, fallback_used)
      << " kernel_symbol="
      << offsetof(sllm_windowed_attention_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_windowed_attention_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_windowed_attention_dispatch_info_t, gcn_arch_name)
      << " reserved="
      << offsetof(sllm_windowed_attention_dispatch_info_t, reserved) << '\n';
  std::cout
      << "layout sllm_kv_state_create_info_t size="
      << sizeof(sllm_kv_state_create_info_t)
      << " align=" << alignof(sllm_kv_state_create_info_t)
      << " struct_size=" << offsetof(sllm_kv_state_create_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_kv_state_create_info_t, abi_version)
      << " session_id=" << offsetof(sllm_kv_state_create_info_t, session_id)
      << " layer_id=" << offsetof(sllm_kv_state_create_info_t, layer_id)
      << " flags=" << offsetof(sllm_kv_state_create_info_t, flags)
      << " capacity_tokens="
      << offsetof(sllm_kv_state_create_info_t, capacity_tokens)
      << " head_count=" << offsetof(sllm_kv_state_create_info_t, head_count)
      << " head_dim=" << offsetof(sllm_kv_state_create_info_t, head_dim)
      << " memory_kind=" << offsetof(sllm_kv_state_create_info_t, memory_kind)
      << " layout=" << offsetof(sllm_kv_state_create_info_t, layout) << '\n';
  std::cout
      << "layout sllm_kv_state_create_info_v2_t size="
      << sizeof(sllm_kv_state_create_info_v2_t)
      << " align=" << alignof(sllm_kv_state_create_info_v2_t) << " struct_size="
      << offsetof(sllm_kv_state_create_info_v2_t, struct_size)
      << " abi_version="
      << offsetof(sllm_kv_state_create_info_v2_t, abi_version)
      << " create_info_version="
      << offsetof(sllm_kv_state_create_info_v2_t, create_info_version)
      << " reserved0=" << offsetof(sllm_kv_state_create_info_v2_t, reserved0)
      << " session_id=" << offsetof(sllm_kv_state_create_info_v2_t, session_id)
      << " layer_id=" << offsetof(sllm_kv_state_create_info_v2_t, layer_id)
      << " flags=" << offsetof(sllm_kv_state_create_info_v2_t, flags)
      << " capacity_tokens="
      << offsetof(sllm_kv_state_create_info_v2_t, capacity_tokens)
      << " head_count=" << offsetof(sllm_kv_state_create_info_v2_t, head_count)
      << " head_dim=" << offsetof(sllm_kv_state_create_info_v2_t, head_dim)
      << " memory_kind="
      << offsetof(sllm_kv_state_create_info_v2_t, memory_kind)
      << " layout=" << offsetof(sllm_kv_state_create_info_v2_t, layout)
      << " dtype=" << offsetof(sllm_kv_state_create_info_v2_t, dtype)
      << " encoding=" << offsetof(sllm_kv_state_create_info_v2_t, encoding)
      << " block_size=" << offsetof(sllm_kv_state_create_info_v2_t, block_size)
      << " scale_dtype="
      << offsetof(sllm_kv_state_create_info_v2_t, scale_dtype)
      << " reserved=" << offsetof(sllm_kv_state_create_info_v2_t, reserved)
      << '\n';
  std::cout
      << "layout sllm_kv_view_info_t size=" << sizeof(sllm_kv_view_info_t)
      << " align=" << alignof(sllm_kv_view_info_t)
      << " struct_size=" << offsetof(sllm_kv_view_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_kv_view_info_t, abi_version)
      << " info_version=" << offsetof(sllm_kv_view_info_t, info_version)
      << " session_id=" << offsetof(sllm_kv_view_info_t, session_id)
      << " layer_id=" << offsetof(sllm_kv_view_info_t, layer_id)
      << " dtype=" << offsetof(sllm_kv_view_info_t, dtype)
      << " memory_kind=" << offsetof(sllm_kv_view_info_t, memory_kind)
      << " layout=" << offsetof(sllm_kv_view_info_t, layout)
      << " capacity_tokens=" << offsetof(sllm_kv_view_info_t, capacity_tokens)
      << " observed_length=" << offsetof(sllm_kv_view_info_t, observed_length)
      << " generation=" << offsetof(sllm_kv_view_info_t, generation)
      << " physical_page_bytes="
      << offsetof(sllm_kv_view_info_t, physical_page_bytes)
      << " tokens_per_page=" << offsetof(sllm_kv_view_info_t, tokens_per_page)
      << " mapped_token_capacity="
      << offsetof(sllm_kv_view_info_t, mapped_token_capacity)
      << " committed_bytes_per_plane="
      << offsetof(sllm_kv_view_info_t, committed_bytes_per_plane)
      << " context_identity=" << offsetof(sllm_kv_view_info_t, context_identity)
      << " state_identity=" << offsetof(sllm_kv_view_info_t, state_identity)
      << " k_stride_elements="
      << offsetof(sllm_kv_view_info_t, k_stride_elements)
      << " v_stride_elements="
      << offsetof(sllm_kv_view_info_t, v_stride_elements)
      << " reserved=" << offsetof(sllm_kv_view_info_t, reserved) << '\n';
  std::cout
      << "layout sllm_kv_append_desc_t size=" << sizeof(sllm_kv_append_desc_t)
      << " align=" << alignof(sllm_kv_append_desc_t)
      << " struct_size=" << offsetof(sllm_kv_append_desc_t, struct_size)
      << " abi_version=" << offsetof(sllm_kv_append_desc_t, abi_version)
      << " append_version=" << offsetof(sllm_kv_append_desc_t, append_version)
      << " expected_length=" << offsetof(sllm_kv_append_desc_t, expected_length)
      << " start_position=" << offsetof(sllm_kv_append_desc_t, start_position)
      << " key_input=" << offsetof(sllm_kv_append_desc_t, key_input)
      << " value_input=" << offsetof(sllm_kv_append_desc_t, value_input)
      << " reserved=" << offsetof(sllm_kv_append_desc_t, reserved) << '\n';
  std::cout
      << "layout sllm_kv_append_info_t size=" << sizeof(sllm_kv_append_info_t)
      << " align=" << alignof(sllm_kv_append_info_t)
      << " struct_size=" << offsetof(sllm_kv_append_info_t, struct_size)
      << " abi_version=" << offsetof(sllm_kv_append_info_t, abi_version)
      << " info_version=" << offsetof(sllm_kv_append_info_t, info_version)
      << " dispatch_id=" << offsetof(sllm_kv_append_info_t, dispatch_id)
      << " start_position=" << offsetof(sllm_kv_append_info_t, start_position)
      << " token_count=" << offsetof(sllm_kv_append_info_t, token_count)
      << " end_position=" << offsetof(sllm_kv_append_info_t, end_position)
      << " commit_allowed=" << offsetof(sllm_kv_append_info_t, commit_allowed)
      << " fallback_used=" << offsetof(sllm_kv_append_info_t, fallback_used)
      << " kernel_symbol=" << offsetof(sllm_kv_append_info_t, kernel_symbol)
      << " device_symbol=" << offsetof(sllm_kv_append_info_t, device_symbol)
      << " gcn_arch_name=" << offsetof(sllm_kv_append_info_t, gcn_arch_name)
      << " reserved=" << offsetof(sllm_kv_append_info_t, reserved) << '\n';
  std::cout
      << "layout sllm_causal_attention_desc_t size="
      << sizeof(sllm_causal_attention_desc_t)
      << " align=" << alignof(sllm_causal_attention_desc_t)
      << " struct_size=" << offsetof(sllm_causal_attention_desc_t, struct_size)
      << " abi_version=" << offsetof(sllm_causal_attention_desc_t, abi_version)
      << " op_version=" << offsetof(sllm_causal_attention_desc_t, op_version)
      << " reserved0=" << offsetof(sllm_causal_attention_desc_t, reserved0)
      << " start_position="
      << offsetof(sllm_causal_attention_desc_t, start_position)
      << " expected_kv_length="
      << offsetof(sllm_causal_attention_desc_t, expected_kv_length)
      << " kv_state=" << offsetof(sllm_causal_attention_desc_t, kv_state)
      << " query=" << offsetof(sllm_causal_attention_desc_t, query)
      << " output=" << offsetof(sllm_causal_attention_desc_t, output)
      << " reserved=" << offsetof(sllm_causal_attention_desc_t, reserved)
      << '\n';
  std::cout
      << "layout sllm_causal_attention_dispatch_info_t size="
      << sizeof(sllm_causal_attention_dispatch_info_t)
      << " align=" << alignof(sllm_causal_attention_dispatch_info_t)
      << " struct_size="
      << offsetof(sllm_causal_attention_dispatch_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_causal_attention_dispatch_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_causal_attention_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_causal_attention_dispatch_info_t, backend)
      << " dispatch_id="
      << offsetof(sllm_causal_attention_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_causal_attention_dispatch_info_t, dispatch_count)
      << " kernel_id="
      << offsetof(sllm_causal_attention_dispatch_info_t, kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_causal_attention_dispatch_info_t, workgroup_size_x)
      << " grid_size_x="
      << offsetof(sllm_causal_attention_dispatch_info_t, grid_size_x)
      << " query_count="
      << offsetof(sllm_causal_attention_dispatch_info_t, query_count)
      << " start_position="
      << offsetof(sllm_causal_attention_dispatch_info_t, start_position)
      << " committed_kv_length="
      << offsetof(sllm_causal_attention_dispatch_info_t, committed_kv_length)
      << " q_heads=" << offsetof(sllm_causal_attention_dispatch_info_t, q_heads)
      << " kv_heads="
      << offsetof(sllm_causal_attention_dispatch_info_t, kv_heads)
      << " head_dim="
      << offsetof(sllm_causal_attention_dispatch_info_t, head_dim)
      << " scale_denominator="
      << offsetof(sllm_causal_attention_dispatch_info_t, scale_denominator)
      << " fallback_allowed="
      << offsetof(sllm_causal_attention_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_causal_attention_dispatch_info_t, fallback_used)
      << " kernel_symbol="
      << offsetof(sllm_causal_attention_dispatch_info_t, kernel_symbol)
      << " device_symbol="
      << offsetof(sllm_causal_attention_dispatch_info_t, device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_causal_attention_dispatch_info_t, gcn_arch_name)
      << " reserved="
      << offsetof(sllm_causal_attention_dispatch_info_t, reserved) << '\n';
  std::cout
      << "layout sllm_linear_attention_state_create_info_t size="
      << sizeof(sllm_linear_attention_state_create_info_t)
      << " align=" << alignof(sllm_linear_attention_state_create_info_t)
      << " struct_size="
      << offsetof(sllm_linear_attention_state_create_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_linear_attention_state_create_info_t, abi_version)
      << " session_id="
      << offsetof(sllm_linear_attention_state_create_info_t, session_id)
      << " layer_id="
      << offsetof(sllm_linear_attention_state_create_info_t, layer_id)
      << " flags=" << offsetof(sllm_linear_attention_state_create_info_t, flags)
      << " capacity_tokens="
      << offsetof(sllm_linear_attention_state_create_info_t, capacity_tokens)
      << " qk_heads="
      << offsetof(sllm_linear_attention_state_create_info_t, qk_heads)
      << " value_heads="
      << offsetof(sllm_linear_attention_state_create_info_t, value_heads)
      << " head_dim="
      << offsetof(sllm_linear_attention_state_create_info_t, head_dim)
      << " conv_kernel_size="
      << offsetof(sllm_linear_attention_state_create_info_t, conv_kernel_size)
      << '\n';
  std::cout
      << "layout sllm_linear_attention_view_info_t size="
      << sizeof(sllm_linear_attention_view_info_t)
      << " align=" << alignof(sllm_linear_attention_view_info_t)
      << " struct_size="
      << offsetof(sllm_linear_attention_view_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_linear_attention_view_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_linear_attention_view_info_t, info_version)
      << " reserved0=" << offsetof(sllm_linear_attention_view_info_t, reserved0)
      << " session_id="
      << offsetof(sllm_linear_attention_view_info_t, session_id)
      << " layer_id=" << offsetof(sllm_linear_attention_view_info_t, layer_id)
      << " conv_state_dtype="
      << offsetof(sllm_linear_attention_view_info_t, conv_state_dtype)
      << " recurrent_state_dtype="
      << offsetof(sllm_linear_attention_view_info_t, recurrent_state_dtype)
      << " encoding=" << offsetof(sllm_linear_attention_view_info_t, encoding)
      << " active_slot="
      << offsetof(sllm_linear_attention_view_info_t, active_slot)
      << " capacity_tokens="
      << offsetof(sllm_linear_attention_view_info_t, capacity_tokens)
      << " observed_length="
      << offsetof(sllm_linear_attention_view_info_t, observed_length)
      << " generation="
      << offsetof(sllm_linear_attention_view_info_t, generation)
      << " context_identity="
      << offsetof(sllm_linear_attention_view_info_t, context_identity)
      << " state_identity="
      << offsetof(sllm_linear_attention_view_info_t, state_identity)
      << " conv_state_shape="
      << offsetof(sllm_linear_attention_view_info_t, conv_state_shape)
      << " recurrent_state_shape="
      << offsetof(sllm_linear_attention_view_info_t, recurrent_state_shape)
      << " reserved=" << offsetof(sllm_linear_attention_view_info_t, reserved)
      << '\n';
  std::cout
      << "layout sllm_linear_attention_desc_t size="
      << sizeof(sllm_linear_attention_desc_t)
      << " align=" << alignof(sllm_linear_attention_desc_t)
      << " struct_size=" << offsetof(sllm_linear_attention_desc_t, struct_size)
      << " abi_version=" << offsetof(sllm_linear_attention_desc_t, abi_version)
      << " op_version=" << offsetof(sllm_linear_attention_desc_t, op_version)
      << " reserved0=" << offsetof(sllm_linear_attention_desc_t, reserved0)
      << " start_position="
      << offsetof(sllm_linear_attention_desc_t, start_position)
      << " expected_length="
      << offsetof(sllm_linear_attention_desc_t, expected_length)
      << " state=" << offsetof(sllm_linear_attention_desc_t, state)
      << " qkv=" << offsetof(sllm_linear_attention_desc_t, qkv)
      << " z=" << offsetof(sllm_linear_attention_desc_t, z)
      << " b_input=" << offsetof(sllm_linear_attention_desc_t, b_input)
      << " a_input=" << offsetof(sllm_linear_attention_desc_t, a_input)
      << " conv_weight=" << offsetof(sllm_linear_attention_desc_t, conv_weight)
      << " a_log=" << offsetof(sllm_linear_attention_desc_t, a_log)
      << " dt_bias=" << offsetof(sllm_linear_attention_desc_t, dt_bias)
      << " norm_weight=" << offsetof(sllm_linear_attention_desc_t, norm_weight)
      << " output=" << offsetof(sllm_linear_attention_desc_t, output)
      << " reserved=" << offsetof(sllm_linear_attention_desc_t, reserved)
      << '\n';
  std::cout
      << "layout sllm_linear_attention_dispatch_info_t size="
      << sizeof(sllm_linear_attention_dispatch_info_t)
      << " align=" << alignof(sllm_linear_attention_dispatch_info_t)
      << " struct_size="
      << offsetof(sllm_linear_attention_dispatch_info_t, struct_size)
      << " abi_version="
      << offsetof(sllm_linear_attention_dispatch_info_t, abi_version)
      << " info_version="
      << offsetof(sllm_linear_attention_dispatch_info_t, info_version)
      << " backend=" << offsetof(sllm_linear_attention_dispatch_info_t, backend)
      << " dispatch_id="
      << offsetof(sllm_linear_attention_dispatch_info_t, dispatch_id)
      << " dispatch_count="
      << offsetof(sllm_linear_attention_dispatch_info_t, dispatch_count)
      << " conv_kernel_id="
      << offsetof(sllm_linear_attention_dispatch_info_t, conv_kernel_id)
      << " recurrent_kernel_id="
      << offsetof(sllm_linear_attention_dispatch_info_t, recurrent_kernel_id)
      << " workgroup_size_x="
      << offsetof(sllm_linear_attention_dispatch_info_t, workgroup_size_x)
      << " conv_grid_size_x="
      << offsetof(sllm_linear_attention_dispatch_info_t, conv_grid_size_x)
      << " recurrent_grid_size_x="
      << offsetof(sllm_linear_attention_dispatch_info_t, recurrent_grid_size_x)
      << " token_count="
      << offsetof(sllm_linear_attention_dispatch_info_t, token_count)
      << " start_position="
      << offsetof(sllm_linear_attention_dispatch_info_t, start_position)
      << " expected_length="
      << offsetof(sllm_linear_attention_dispatch_info_t, expected_length)
      << " qk_heads="
      << offsetof(sllm_linear_attention_dispatch_info_t, qk_heads)
      << " value_heads="
      << offsetof(sllm_linear_attention_dispatch_info_t, value_heads)
      << " head_dim="
      << offsetof(sllm_linear_attention_dispatch_info_t, head_dim)
      << " fallback_allowed="
      << offsetof(sllm_linear_attention_dispatch_info_t, fallback_allowed)
      << " fallback_used="
      << offsetof(sllm_linear_attention_dispatch_info_t, fallback_used)
      << " kernel_symbol="
      << offsetof(sllm_linear_attention_dispatch_info_t, kernel_symbol)
      << " conv_device_symbol="
      << offsetof(sllm_linear_attention_dispatch_info_t, conv_device_symbol)
      << " recurrent_device_symbol="
      << offsetof(sllm_linear_attention_dispatch_info_t,
                  recurrent_device_symbol)
      << " gcn_arch_name="
      << offsetof(sllm_linear_attention_dispatch_info_t, gcn_arch_name)
      << " reserved="
      << offsetof(sllm_linear_attention_dispatch_info_t, reserved) << '\n';
  return 0;
}
