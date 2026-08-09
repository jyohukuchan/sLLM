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
  SLLM_PRINT_CONSTANT(SLLM_BACKEND_HIP);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_READ);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_WRITE);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_READ_WRITE);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MAX_DEVICE_NAME);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MAX_GCN_ARCH_NAME);
  SLLM_PRINT_CONSTANT(SLLM_HIP_MAX_TRANSFER_BYTES);
  SLLM_PRINT_CONSTANT(SLLM_HIP_RMSNORM_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_TENSOR_MAX_RANK);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_BF16);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_DTYPE_F32);
  SLLM_PRINT_CONSTANT(SLLM_TENSOR_ENCODING_UNQUANTIZED);
  SLLM_PRINT_CONSTANT(SLLM_RMSNORM_ACCUMULATION_F32);
  SLLM_PRINT_CONSTANT(SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE);
  SLLM_PRINT_CONSTANT(SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP);
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
  return 0;
}
