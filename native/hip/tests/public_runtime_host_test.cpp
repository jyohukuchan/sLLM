#include "public_runtime_internal.hpp"
#include "rmsnorm_kernel_internal.hpp"

#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

extern "C" std::size_t sllm_test_orphan_count() noexcept;
extern "C" std::size_t sllm_test_poison_count() noexcept;
extern "C" void sllm_test_rmsnorm_execute_throw_after_reservation(
    uint32_t occurrences) noexcept;
extern "C" void sllm_test_rmsnorm_execute_throw_after_registration(
    uint32_t occurrences) noexcept;

namespace {

struct Error final {
  char message[256]{};
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

bool create_context(sllm_context_t **const context) {
  sllm_context_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.device_index = 0U;
  std::memcpy(info.expected_gcn_arch_name, "gfx1201", sizeof("gfx1201"));
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

bool create_buffer(const sllm_context_t *const context,
                   sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = 64U;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, buffer, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_create", error);
}

bool create_buffer_sized(const sllm_context_t *const context,
                         const uint64_t size_bytes,
                         sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = size_bytes;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, buffer, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_create", error);
}

bool release_context(sllm_context_t **const context,
                     const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_context_release(context, &error.sink), expected,
                       "sllm_context_release", error);
}

bool release_queue(sllm_queue_t **const queue,
                   const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_queue_release(queue, &error.sink), expected,
                       "sllm_queue_release", error);
}

bool release_buffer(sllm_buffer_t **const buffer,
                    const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_buffer_release(buffer, &error.sink), expected,
                       "sllm_buffer_release", error);
}

bool submit_h2d(const sllm_queue_t *const queue,
                const sllm_buffer_t *const buffer,
                sllm_completion_t **const completion) {
  uint8_t payload[17] = {};
  for (std::size_t index = 0U; index != sizeof(payload); ++index) {
    payload[index] = static_cast<uint8_t>(index + 1U);
  }
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = payload;
  transfer.size_bytes = sizeof(payload);
  Error error;
  return expect_status(
      sllm_buffer_copy_h2d(queue, buffer, &transfer, completion, &error.sink),
      SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error);
}

bool submit_d2h(const sllm_queue_t *const queue,
                const sllm_buffer_t *const buffer, const std::size_t size,
                sllm_completion_t **const completion) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = size;
  Error error;
  return expect_status(
      sllm_buffer_copy_d2h(queue, buffer, &transfer, completion, &error.sink),
      SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error);
}

bool query_completion(sllm_completion_t *const completion,
                      const sllm_status_t expected) {
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  return expect_status(sllm_completion_query(completion, &result, &error.sink),
                       expected, "sllm_completion_query", error);
}

bool release_completion(sllm_completion_t **const completion,
                        const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_completion_release(completion, &error.sink),
                       expected, "sllm_completion_release", error);
}

bool read_completion(sllm_completion_t *const completion,
                     void *const destination, const std::size_t capacity,
                     const uint8_t *const expected,
                     const std::size_t expected_size) {
  uint64_t bytes_written = 0U;
  Error error;
  const sllm_status_t status = sllm_completion_read(
      completion, destination, capacity, &bytes_written, &error.sink);
  if (!expect_status(status, SLLM_STATUS_OK, "sllm_completion_read", error) ||
      bytes_written != expected_size ||
      std::memcmp(destination, expected, expected_size) != 0) {
    std::cerr << "D2H completion read was not byte exact\n";
    return false;
  }
  return true;
}

bool bounded_counter_cas_contention_is_fail_closed() {
  using SafetyState = sllm_public_runtime::CompletionSafetyState;
  using Injector = sllm_public_runtime::FaultInjector;

  SafetyState::reset_quarantine_cas_failures();
  SafetyState::force_quarantine_cas_failures(1U);
  SafetyState::force_quarantine_counter_cas_contention(true);
  if (!SafetyState::consume_forced_quarantine_cas_failure_for_test()) {
    std::cerr << "quarantine counter CAS exhaustion did not fail closed\n";
    return false;
  }
  SafetyState::force_quarantine_counter_cas_contention(false);
  if (!SafetyState::consume_forced_quarantine_cas_failure_for_test() ||
      SafetyState::consume_forced_quarantine_cas_failure_for_test()) {
    std::cerr
        << "quarantine counter did not recover after bounded contention\n";
    return false;
  }
  SafetyState::reset_quarantine_cas_failures();

  Injector::reset();
  Injector::set(sllm_public_runtime::FaultPoint::AccountingFailure, 1U);
  Injector::force_cas_contention(true);
  if (!Injector::consume(sllm_public_runtime::FaultPoint::AccountingFailure)) {
    std::cerr << "fault injector CAS exhaustion did not fail closed\n";
    return false;
  }
  Injector::force_cas_contention(false);
  if (!Injector::consume(sllm_public_runtime::FaultPoint::AccountingFailure) ||
      Injector::consume(sllm_public_runtime::FaultPoint::AccountingFailure)) {
    std::cerr
        << "fault injector counter did not recover after bounded contention\n";
    return false;
  }
  Injector::reset();
  return true;
}

bool completion_safety_quarantine_is_bounded_and_fail_closed() {
  using SafetyState = sllm_public_runtime::CompletionSafetyState;

  SafetyState exhausted;
  exhausted.observe_positive_completion();
  if (!exhausted.can_release_graph()) {
    std::cerr << "positive completion must initially be releasable\n";
    return false;
  }
  SafetyState::force_quarantine_cas_failures(
      static_cast<uint32_t>(SafetyState::quarantine_cas_attempt_bound() + 1U));
  exhausted.quarantine();
  SafetyState::reset_quarantine_cas_failures();
  if (exhausted.can_release_graph() || exhausted.event_destroyed() ||
      exhausted.observe_event_destroy_success()) {
    std::cerr << "bounded quarantine CAS exhaustion was not fail closed\n";
    return false;
  }
  exhausted.quarantine();
  if (exhausted.can_release_graph()) {
    std::cerr << "repeat quarantine re-enabled release\n";
    return false;
  }

  SafetyState destroyed;
  destroyed.observe_positive_completion();
  if (!destroyed.observe_event_destroy_success() ||
      !destroyed.event_destroyed()) {
    std::cerr << "positive completion could not reach EventDestroyed\n";
    return false;
  }
  destroyed.quarantine();
  if (!destroyed.event_destroyed() || destroyed.can_release_graph()) {
    std::cerr << "quarantine overwrote EventDestroyed or enabled release\n";
    return false;
  }

  SafetyState concurrent;
  concurrent.observe_positive_completion();
  std::thread quarantine_thread([&concurrent]() { concurrent.quarantine(); });
  std::thread destroy_thread(
      [&concurrent]() { (void)concurrent.observe_event_destroy_success(); });
  quarantine_thread.join();
  destroy_thread.join();
  if (concurrent.can_release_graph()) {
    std::cerr << "concurrent safety transition enabled release\n";
    return false;
  }
  return true;
}

bool successful_completion_lifecycle() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  sllm_event_t *event = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  Error event_error;
  if (!expect_status(sllm_event_create(context, &event, &event_error.sink),
                     SLLM_STATUS_OK, "sllm_event_create", event_error)) {
    return false;
  }
  if (!expect_status(sllm_event_release(&event, &event_error.sink),
                     SLLM_STATUS_OK, "sllm_event_release", event_error)) {
    return false;
  }

  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion) || completion == nullptr) {
    return false;
  }
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::CompletionQueryPending, 1U);
  if (!query_completion(completion, SLLM_STATUS_PUBLIC_PENDING) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) || completion != nullptr ||
      !release_queue(&queue) || !release_buffer(&buffer) ||
      !release_context(&context)) {
    return false;
  }
  if (fake_hip::live_events() != 0U || fake_hip::live_streams() != 0U ||
      fake_hip::live_allocations() != 0U) {
    std::cerr << "successful lifecycle left fake HIP resources live\n";
    return false;
  }
  return true;
}

bool d2h_staging_and_completion_read_is_byte_exact() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  constexpr std::size_t transfer_size = 37U;
  uint8_t expected[transfer_size] = {};
  for (std::size_t index = 0U; index != transfer_size; ++index) {
    expected[index] = static_cast<uint8_t>((index * 13U) ^ 0x5AU);
  }
  sllm_transfer_desc_t h2d{};
  h2d.struct_size = sizeof(h2d);
  h2d.abi_version = SLLM_HIP_ABI_VERSION;
  h2d.host_pointer = expected;
  h2d.size_bytes = transfer_size;
  sllm_completion_t *h2d_completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_h2d(queue, buffer, &h2d, &h2d_completion,
                                          &error.sink),
                     SLLM_STATUS_OK, "D2H setup H2D", error) ||
      !query_completion(h2d_completion, SLLM_STATUS_OK) ||
      !release_completion(&h2d_completion)) {
    return false;
  }

  sllm_completion_t *d2h_completion = nullptr;
  if (!submit_d2h(queue, buffer, transfer_size, &d2h_completion) ||
      !query_completion(d2h_completion, SLLM_STATUS_OK)) {
    return false;
  }
  uint8_t actual[transfer_size] = {};
  if (!read_completion(d2h_completion, actual, sizeof(actual), expected,
                       transfer_size) ||
      !release_completion(&d2h_completion) || !release_queue(&queue) ||
      !release_buffer(&buffer) || !release_context(&context)) {
    return false;
  }
  return fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool positive_completion_with_deferred_event_destroy_retains_dependencies() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion) ||
      !query_completion(completion, SLLM_STATUS_OK)) {
    return false;
  }
  const std::size_t poison_before = sllm_test_poison_count();
  const std::size_t destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  if (!release_completion(&completion, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR) ||
      completion != nullptr || sllm_test_poison_count() != poison_before + 1U ||
      fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() == 0U) {
    std::cerr << "positive completion did not retain deferred event cleanup"
              << " poison=" << sllm_test_poison_count()
              << " expected_poison=" << poison_before + 1U
              << " destroy=" << fake_hip::event_destroy_calls()
              << " expected_destroy=" << destroy_before
              << " live_events=" << fake_hip::live_events() << '\n';
    return false;
  }
  Error queue_error;
  Error buffer_error;
  Error context_error;
  return expect_status(sllm_queue_release(&queue, &queue_error.sink),
                       SLLM_STATUS_INTERNAL_ERROR,
                       "deferred-destroy queue retention", queue_error) &&
         expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                       SLLM_STATUS_INTERNAL_ERROR,
                       "deferred-destroy buffer retention", buffer_error) &&
         expect_status(sllm_context_release(&context, &context_error.sink),
                       SLLM_STATUS_INTERNAL_ERROR,
                       "deferred-destroy context retention", context_error);
}

bool concurrent_pin_and_release() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion)) {
    return false;
  }
  fake_hip::set_event_query_gate(true);
  sllm_status_t query_status = SLLM_STATUS_INTERNAL_ERROR;
  std::thread query_thread([&]() {
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error error;
    query_status = sllm_completion_query(completion, &result, &error.sink);
  });
  fake_hip::wait_event_query_entered();
  Error release_error;
  const sllm_status_t release_status =
      sllm_completion_release(&completion, &release_error.sink);
  if (!expect_status(release_status, SLLM_STATUS_PUBLIC_BUSY,
                     "concurrent sllm_completion_release", release_error)) {
    fake_hip::release_event_query_gate();
    query_thread.join();
    return false;
  }
  fake_hip::release_event_query_gate();
  query_thread.join();
  if (query_status != SLLM_STATUS_OK || !release_completion(&completion) ||
      !release_queue(&queue) || !release_buffer(&buffer) ||
      !release_context(&context)) {
    return false;
  }
  return true;
}

bool fatal_completion_is_quarantined() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion)) {
    return false;
  }
  const std::size_t destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::CompletionQueryFatal, 1U);
  if (!query_completion(completion, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR) ||
      !release_completion(&completion, SLLM_STATUS_PUBLIC_INVALID_HANDLE) ||
      completion == nullptr ||
      fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() == 0U) {
    return false;
  }
  Error queue_error;
  Error buffer_error;
  Error context_error;
  if (!expect_status(sllm_queue_release(&queue, &queue_error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fatal queue release",
                     queue_error) ||
      !expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fatal buffer release",
                     buffer_error) ||
      !expect_status(sllm_context_release(&context, &context_error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fatal context release",
                     context_error)) {
    return false;
  }
  return true;
}

bool registry_failure_destroys_or_orphans_before_rollback() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  const std::size_t orphan_before = sllm_test_orphan_count();
  const std::size_t event_destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::RegistryInsertionFailure, 1U);
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  fake_hip::set_event_create_gate(true);
  sllm_completion_t *completion = nullptr;
  sllm_status_t submit_status = SLLM_STATUS_OK;
  std::thread submit_thread([&]() {
    uint8_t payload[17] = {};
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = payload;
    transfer.size_bytes = sizeof(payload);
    Error error;
    submit_status = sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                         &error.sink);
  });
  fake_hip::wait_event_create_entered();
  Error queue_error;
  Error buffer_error;
  Error context_error;
  const bool concurrent_releases =
      expect_status(sllm_queue_release(&queue, &queue_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "concurrent queue release",
                    queue_error) &&
      expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "concurrent buffer release",
                    buffer_error) &&
      expect_status(sllm_context_release(&context, &context_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "concurrent context release",
                    context_error);
  fake_hip::release_event_create_gate();
  submit_thread.join();
  if (!concurrent_releases ||
      submit_status != SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR ||
      completion != nullptr || sllm_test_orphan_count() != orphan_before + 1U ||
      fake_hip::event_destroy_calls() != event_destroy_before ||
      fake_hip::live_events() == 0U) {
    std::cerr << "registry zero-token rollback did not retain exactly one "
                 "ambiguous event\n";
    return false;
  }
  return true;
}

bool registry_exception_reaches_real_catch_before_rollback() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  const std::size_t orphan_before = sllm_test_orphan_count();
  const std::size_t event_destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::RegistryInsertionException, 1U);
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  fake_hip::set_event_create_gate(true);
  sllm_completion_t *completion = nullptr;
  sllm_status_t submit_status = SLLM_STATUS_OK;
  std::thread submit_thread([&]() {
    uint8_t payload[17] = {};
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = payload;
    transfer.size_bytes = sizeof(payload);
    Error error;
    submit_status = sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                         &error.sink);
  });
  fake_hip::wait_event_create_entered();
  Error queue_error;
  Error buffer_error;
  Error context_error;
  const bool concurrent_releases =
      expect_status(sllm_queue_release(&queue, &queue_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY,
                    "exception concurrent queue release", queue_error) &&
      expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY,
                    "exception concurrent buffer release", buffer_error) &&
      expect_status(sllm_context_release(&context, &context_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY,
                    "exception concurrent context release", context_error);
  fake_hip::release_event_create_gate();
  submit_thread.join();
  if (!concurrent_releases ||
      submit_status != SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR ||
      completion != nullptr || sllm_test_orphan_count() != orphan_before + 1U ||
      fake_hip::event_destroy_calls() != event_destroy_before ||
      fake_hip::live_events() == 0U) {
    std::cerr << "registry exception did not use guarded event rollback\n";
    return false;
  }
  return true;
}

bool production_orphan_owner_grows_past_128() {
  sllm_public_runtime::FaultInjector::reset();
  const std::size_t before = sllm_test_orphan_count();
  for (std::size_t index = 0U; index != 129U; ++index) {
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    if (!create_context(&context)) {
      return false;
    }
    sllm_public_runtime::FaultInjector::set(
        sllm_public_runtime::FaultPoint::ConstructionCandidateFailure, 1U);
    sllm_public_runtime::FaultInjector::set(
        sllm_public_runtime::FaultPoint::StreamDestroyError, 1U);
    sllm_queue_create_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    Error error;
    const sllm_status_t status =
        sllm_queue_create(context, &info, &queue, &error.sink);
    if (!expect_status(status, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                       "orphan-growth queue create", error) ||
        queue != nullptr) {
      return false;
    }
  }
  if (sllm_test_orphan_count() < before + 129U) {
    std::cerr << "production orphan owner did not grow beyond 128 records\n";
    return false;
  }
  return true;
}

sllm_tensor_binding_t rmsnorm_binding(const sllm_buffer_t *const buffer,
                                      const uint64_t offset,
                                      const uint64_t rows,
                                      const uint64_t columns) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = rows == 1U ? 1U : 2U;
  if (binding.rank == 1U) {
    binding.shape[0] = columns;
    binding.stride_elements[0] = 1U;
  } else {
    binding.shape[0] = rows;
    binding.shape[1] = columns;
    binding.stride_elements[0] = columns;
    binding.stride_elements[1] = 1U;
  }
  return binding;
}

sllm_tensor_binding_t
rmsnorm_binding_rank(const sllm_buffer_t *const buffer, const uint64_t offset,
                     const uint32_t rank, const uint64_t columns,
                     const uint64_t *const shape = nullptr) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t backwards = 0U; backwards != rank; ++backwards) {
    const uint32_t index = rank - 1U - backwards;
    binding.shape[index] =
        shape == nullptr ? (index == rank - 1U ? columns : 1U) : shape[index];
    binding.stride_elements[index] = stride;
    stride *= binding.shape[index];
  }
  return binding;
}

bool rmsnorm_bf16_rne_bit_contract() {
  struct ConversionCase final {
    uint32_t input_bits;
    uint16_t expected_bits;
  };
  constexpr ConversionCase cases[] = {
      {UINT32_C(0x00000000), UINT16_C(0x0000)},
      {UINT32_C(0x80000000), UINT16_C(0x8000)},
      {UINT32_C(0x00008000), UINT16_C(0x0000)},
      {UINT32_C(0x80008000), UINT16_C(0x8000)},
      {UINT32_C(0x3f807fff), UINT16_C(0x3f80)},
      {UINT32_C(0x3f808000), UINT16_C(0x3f80)},
      {UINT32_C(0x3f808001), UINT16_C(0x3f81)},
      {UINT32_C(0x3f818000), UINT16_C(0x3f82)},
      {UINT32_C(0xbf808000), UINT16_C(0xbf80)},
      {UINT32_C(0xbf818000), UINT16_C(0xbf82)},
      {UINT32_C(0x7f800000), UINT16_C(0x7f80)},
      {UINT32_C(0xff800000), UINT16_C(0xff80)},
      {UINT32_C(0x7f800001), UINT16_C(0x7fc0)},
      {UINT32_C(0xff800001), UINT16_C(0xffc0)},
      {UINT32_C(0x7f900001), UINT16_C(0x7fd0)},
      {UINT32_C(0xffa12345), UINT16_C(0xffe1)},
      {UINT32_C(0x7fc12345), UINT16_C(0x7fc1)},
  };

  for (const ConversionCase &test_case : cases) {
    float value = 0.0F;
    std::memcpy(&value, &test_case.input_bits, sizeof(value));
    const uint16_t actual = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(value);
    if (actual != test_case.expected_bits) {
      std::cerr << "BF16 conversion contract mismatch\n";
      return false;
    }
  }

  constexpr uint32_t nan_bits[] = {
      UINT32_C(0x7f800001), UINT32_C(0x7f900001), UINT32_C(0x7fc12345),
      UINT32_C(0xff800001), UINT32_C(0xffa12345), UINT32_C(0xffc12345),
  };
  for (const uint32_t input_bits : nan_bits) {
    float value = 0.0F;
    std::memcpy(&value, &input_bits, sizeof(value));
    const uint16_t actual = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(value);
    if ((actual & UINT16_C(0x7f80)) != UINT16_C(0x7f80) ||
        (actual & UINT16_C(0x0040)) == 0U ||
        (actual & UINT16_C(0x007f)) == 0U) {
      std::cerr << "BF16 NaN was not a quiet nonzero NaN\n";
      return false;
    }
  }
  return true;
}

sllm_rmsnorm_desc_t rmsnorm_descriptor(
    const sllm_buffer_t *const activation, const uint64_t activation_offset,
    const sllm_buffer_t *const scale, const uint64_t scale_offset,
    const sllm_buffer_t *const output, const uint64_t output_offset,
    const uint64_t rows = 2U, const uint64_t columns = 3U) {
  sllm_rmsnorm_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_RMSNORM_VERSION;
  descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
  descriptor.scale_mode = SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE;
  descriptor.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
  float epsilon = 1.0e-6F;
  std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
  descriptor.activation =
      rmsnorm_binding(activation, activation_offset, rows, columns);
  descriptor.raw_scale = rmsnorm_binding(scale, scale_offset, 1U, columns);
  descriptor.output = rmsnorm_binding(output, output_offset, rows, columns);
  return descriptor;
}

bool rmsnorm_prepare_lifecycle_and_negative_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 2U, scale, 4U, output, 6U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "rmsnorm prepare", error) ||
      plan == nullptr ||
      !release_buffer(&activation, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_context(&context, SLLM_STATUS_PUBLIC_BUSY)) {
    return false;
  }
  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "rmsnorm plan release", error) ||
      plan != nullptr ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "rmsnorm double release",
                     error)) {
    return false;
  }
  if (!release_buffer(&activation) || !release_buffer(&scale) ||
      !release_buffer(&output) || !release_context(&context)) {
    return false;
  }

  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  descriptor.activation.struct_size -= 1U;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_INVALID_ARGUMENT, "rmsnorm binding size", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  descriptor.epsilon_bits = 0U;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_INVALID_EPSILON, "rmsnorm epsilon", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 1U, scale, 0U, output, 0U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_MISALIGNED_OFFSET, "rmsnorm alignment", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 60U, scale, 0U, output, 0U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_BUFFER_OUT_OF_BOUNDS, "rmsnorm bounds", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, activation, 2U, output, 0U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "rmsnorm activation-scale alias", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, activation, 2U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "rmsnorm activation-output alias",
          error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, scale, 2U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "rmsnorm scale-output alias", error)) {
    return false;
  }
  descriptor =
      rmsnorm_descriptor(activation, 0U, activation, 16U, activation, 32U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "rmsnorm disjoint alias", error) ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "rmsnorm disjoint release", error)) {
    return false;
  }
  /* Half-open intervals that exactly touch are valid aliases.  The three
   * intervals below are [0,12), [12,18), and [18,30); none overlaps. */
  descriptor =
      rmsnorm_descriptor(activation, 0U, activation, 12U, activation, 18U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "rmsnorm touching alias", error) ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "rmsnorm touching release", error)) {
    return false;
  }
  return release_buffer(&activation) && release_buffer(&scale) &&
         release_buffer(&output) && release_context(&context);
}

bool rmsnorm_plan_accounting_failure_is_consumed_and_quarantined() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "fault plan prepare", error) ||
      plan == nullptr) {
    return false;
  }
  sllm_rmsnorm_plan_t *stale = plan;
  const std::size_t poison_before = sllm_test_poison_count();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::AccountingFailure, 1U);
  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fault plan release", error) ||
      plan != nullptr || sllm_test_poison_count() != poison_before + 1U) {
    std::cerr << "RMSNorm accounting failure did not consume and quarantine "
                 "the plan\n";
    return false;
  }

  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "null plan retry", error) ||
      !expect_status(sllm_rmsnorm_plan_release(&stale, &error.sink),
                     SLLM_STATUS_PUBLIC_INVALID_HANDLE, "stale plan token",
                     error)) {
    return false;
  }
  sllm_rmsnorm_plan_t *forged = reinterpret_cast<sllm_rmsnorm_plan_t *>(
      static_cast<uintptr_t>(0xfeedfaceU));
  if (!expect_status(sllm_rmsnorm_plan_release(&forged, &error.sink),
                     SLLM_STATUS_PUBLIC_INVALID_HANDLE, "forged plan token",
                     error)) {
    return false;
  }
  sllm_rmsnorm_plan_t *wrong_kind =
      reinterpret_cast<sllm_rmsnorm_plan_t *>(context);
  if (!expect_status(sllm_rmsnorm_plan_release(&wrong_kind, &error.sink),
                     SLLM_STATUS_PUBLIC_INVALID_HANDLE, "wrong-kind plan token",
                     error)) {
    return false;
  }

  /* The poison owner retains all three distinct Buffer dependencies and the
   * Context.  Their callers must see INTERNAL_ERROR, never a retryable BUSY
   * caused by the plan's permanently active release flag. */
  if (!expect_status(sllm_buffer_release(&activation, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined activation",
                     error) ||
      !expect_status(sllm_buffer_release(&scale, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined scale", error) ||
      !expect_status(sllm_buffer_release(&output, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined output", error) ||
      !expect_status(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined context",
                     error)) {
    return false;
  }
  sllm_public_runtime::FaultInjector::reset();
  return true;
}

bool rmsnorm_guard_page_prefix_is_fail_closed() {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    std::cerr << "guard-page test could not determine page size\n";
    return false;
  }
  void *const mapping =
      mmap(nullptr, static_cast<std::size_t>(page_size) * 2U,
           PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) {
    std::cerr << "guard-page test mmap failed\n";
    return false;
  }
  void *const guard = static_cast<char *>(mapping) + page_size;
  if (mprotect(guard, static_cast<std::size_t>(page_size), PROT_NONE) != 0) {
    (void)munmap(mapping, static_cast<std::size_t>(page_size) * 2U);
    std::cerr << "guard-page test mprotect failed\n";
    return false;
  }
  auto *const prefix = reinterpret_cast<uint32_t *>(guard) - 2;
  prefix[0] = sizeof(uint32_t) * 2U;
  prefix[1] = SLLM_HIP_ABI_VERSION;
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  const sllm_status_t status = sllm_rmsnorm_prepare(
      reinterpret_cast<const sllm_context_t *>(static_cast<uintptr_t>(1U)),
      reinterpret_cast<const sllm_rmsnorm_desc_t *>(prefix), &plan,
      &error.sink);
  const bool pass =
      expect_status(status, SLLM_STATUS_INVALID_ARGUMENT,
                    "guard-page truncated RMSNorm descriptor", error) &&
      plan == nullptr;
  (void)munmap(mapping, static_cast<std::size_t>(page_size) * 2U);
  return pass;
}

bool rmsnorm_table_driven_negative_contract() {
  constexpr uint32_t ranks[] = {3U, 4U, 5U, 6U, 7U, 8U};
  constexpr uint64_t columns[] = {1U, 3U, 17U, 255U, 256U, 257U, 2560U};
  for (const uint32_t rank : ranks) {
    for (const uint64_t column : columns) {
      fake_hip::reset();
      sllm_context_t *context = nullptr;
      sllm_buffer_t *activation = nullptr;
      sllm_buffer_t *scale = nullptr;
      sllm_buffer_t *output = nullptr;
      const uint64_t bytes = column * 2U + 64U;
      if (!create_context(&context) ||
          !create_buffer_sized(context, bytes, &activation) ||
          !create_buffer_sized(context, bytes, &scale) ||
          !create_buffer_sized(context, bytes, &output)) {
        return false;
      }
      sllm_rmsnorm_desc_t descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = SLLM_HIP_ABI_VERSION;
      descriptor.op_version = SLLM_HIP_RMSNORM_VERSION;
      descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
      descriptor.scale_mode = SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE;
      descriptor.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
      float epsilon = 1.0e-6F;
      std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
      descriptor.activation =
          rmsnorm_binding_rank(activation, 0U, rank, column);
      descriptor.raw_scale = rmsnorm_binding(scale, 0U, 1U, column);
      descriptor.output = rmsnorm_binding_rank(output, 0U, rank, column);
      sllm_rmsnorm_plan_t *plan = nullptr;
      Error error;
      if (!expect_status(
              sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
              SLLM_STATUS_OK, "rank/N RMSNorm prepare", error) ||
          !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                         SLLM_STATUS_OK, "rank/N RMSNorm release", error) ||
          !release_buffer(&activation) || !release_buffer(&scale) ||
          !release_buffer(&output) || !release_context(&context)) {
        return false;
      }
    }
  }

  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  const auto valid = rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  Error error;
  sllm_rmsnorm_plan_t *plan = nullptr;
  const auto expect = [&](sllm_rmsnorm_desc_t descriptor,
                          const sllm_status_t status, const char *name) {
    plan = nullptr;
    return expect_status(
               sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
               status, name, error) &&
           plan == nullptr;
  };
  if (!expect_status(sllm_rmsnorm_prepare(context, nullptr, &plan, &error.sink),
                     SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
                     "null RMSNorm descriptor", error) ||
      plan != nullptr) {
    return false;
  }
  auto descriptor = valid;
  descriptor.activation.dtype = SLLM_TENSOR_DTYPE_F32;
  if (!expect(descriptor, SLLM_STATUS_UNSUPPORTED_DTYPE, "negative dtype")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.encoding = UINT32_C(99);
  if (!expect(descriptor, SLLM_STATUS_UNSUPPORTED_ENCODING,
              "negative encoding")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.shape[1] = 0U;
  if (!expect(descriptor, SLLM_STATUS_ZERO_EXTENT, "negative zero extent")) {
    return false;
  }
  descriptor = valid;
  descriptor.output.shape[1] = 4U;
  descriptor.output.stride_elements[0] = 4U;
  if (!expect(descriptor, SLLM_STATUS_SHAPE_MISMATCH, "negative shape")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.stride_elements[0] = 1U;
  if (!expect(descriptor, SLLM_STATUS_STRIDE_MISMATCH, "negative stride")) {
    return false;
  }
  descriptor = valid;
  descriptor.struct_size -= 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ARGUMENT,
              "negative descriptor size")) {
    return false;
  }
  descriptor = valid;
  descriptor.struct_size = sizeof(uint32_t) * 2U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ARGUMENT,
              "malformed top-level prefix")) {
    return false;
  }
  descriptor = valid;
  descriptor.abi_version += 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ABI_VERSION,
              "negative descriptor ABI")) {
    return false;
  }
  descriptor = valid;
  descriptor.reserved[0] = 1U;
  if (!expect(descriptor, SLLM_STATUS_RESERVED_NONZERO,
              "negative descriptor reserved")) {
    return false;
  }
  descriptor = valid;
  descriptor.op_version += 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
              "negative op version")) {
    return false;
  }
  descriptor = valid;
  descriptor.accumulation_dtype = SLLM_TENSOR_DTYPE_BF16;
  if (!expect(descriptor, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
              "negative accumulation")) {
    return false;
  }
  descriptor = valid;
  descriptor.scale_mode = 0U;
  if (!expect(descriptor, SLLM_STATUS_UNSUPPORTED_SCALE_MODE,
              "negative scale mode")) {
    return false;
  }
  descriptor = valid;
  descriptor.alias_policy = 0U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
              "negative alias policy")) {
    return false;
  }
  for (const float epsilon : {0.0F, -1.0F, NAN, INFINITY}) {
    descriptor = valid;
    std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
    if (!expect(descriptor, SLLM_STATUS_INVALID_EPSILON, "negative epsilon")) {
      return false;
    }
  }
  descriptor = valid;
  descriptor.raw_scale.rank = 2U;
  descriptor.raw_scale.shape[1] = 1U;
  descriptor.raw_scale.stride_elements[0] = 1U;
  descriptor.raw_scale.stride_elements[1] = 1U;
  if (!expect(descriptor, SLLM_STATUS_SHAPE_MISMATCH,
              "negative raw-scale rank")) {
    return false;
  }
  descriptor = valid;
  descriptor.raw_scale.shape[0] = 2U;
  if (!expect(descriptor, SLLM_STATUS_SHAPE_MISMATCH,
              "negative raw-scale shape")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.reserved0 = 1U;
  if (!expect(descriptor, SLLM_STATUS_RESERVED_NONZERO,
              "negative nested reserved")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.abi_version += 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ABI_VERSION,
              "negative nested ABI")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.buffer = nullptr;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative nested null")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.rank = 0U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative rank zero")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.rank = 9U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative rank nine")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.shape[2] = 2U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative unused shape")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.stride_elements[2] = 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative unused stride")) {
    return false;
  }
  return release_buffer(&activation) && release_buffer(&scale) &&
         release_buffer(&output) && release_context(&context);
}

bool rmsnorm_prepare_required_shape_and_context_cases() {
  constexpr uint64_t dimensions[] = {1U, 3U, 17U, 255U, 256U, 257U, 2560U};
  for (const uint64_t columns : dimensions) {
    fake_hip::reset();
    sllm_context_t *context = nullptr;
    sllm_buffer_t *activation = nullptr;
    sllm_buffer_t *scale = nullptr;
    sllm_buffer_t *output = nullptr;
    const uint64_t bytes = columns * 4U;
    if (!create_context(&context) ||
        !create_buffer_sized(context, bytes, &activation) ||
        !create_buffer_sized(context, bytes, &scale) ||
        !create_buffer_sized(context, bytes, &output)) {
      return false;
    }
    sllm_rmsnorm_desc_t descriptor =
        rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 1U, columns);
    sllm_rmsnorm_plan_t *plan = nullptr;
    Error error;
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "rmsnorm rank-one prepare", error) ||
        !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "rmsnorm rank-one release", error)) {
      return false;
    }
    descriptor =
        rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, columns);
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "rmsnorm rank-two prepare", error) ||
        !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "rmsnorm rank-two release", error) ||
        !release_buffer(&activation) || !release_buffer(&scale) ||
        !release_buffer(&output) || !release_context(&context)) {
      return false;
    }
  }

  sllm_context_t *first = nullptr;
  sllm_context_t *second = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&first) || !create_context(&second) ||
      !create_buffer_sized(first, 4096U, &activation) ||
      !create_buffer_sized(first, 4096U, &scale) ||
      !create_buffer_sized(second, 4096U, &output)) {
    return false;
  }
  Error error;
  sllm_rmsnorm_plan_t *plan = nullptr;
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 257U);
  if (!expect_status(
          sllm_rmsnorm_prepare(first, &descriptor, &plan, &error.sink),
          SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH, "rmsnorm context mismatch",
          error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, activation, 0U);
  descriptor.activation.shape[0] = UINT64_MAX;
  descriptor.activation.rank = 1U;
  descriptor.activation.stride_elements[0] = 1U;
  descriptor.activation.shape[1] = 0U;
  descriptor.activation.stride_elements[1] = 0U;
  if (!expect_status(
          sllm_rmsnorm_prepare(first, &descriptor, &plan, &error.sink),
          SLLM_STATUS_METADATA_OVERFLOW, "rmsnorm metadata overflow", error)) {
    return false;
  }
  return release_buffer(&activation) && release_buffer(&scale) &&
         release_buffer(&output) && release_context(&first) &&
         release_context(&second);
}

sllm_rmsnorm_dispatch_info_t rmsnorm_dispatch_info() {
  sllm_rmsnorm_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION;
  return info;
}

bool rmsnorm_execute_metadata_and_reuse() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_context_t *other_context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_queue_t *other_queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_context(&other_context) ||
      !create_queue(context, &queue) ||
      !create_queue(other_context, &other_queue) ||
      !create_buffer_sized(context, 2U * 2U * 257U + 64U, &activation) ||
      !create_buffer_sized(context, 2U * 257U + 64U, &scale) ||
      !create_buffer_sized(context, 2U * 2U * 257U + 64U, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 2U, scale, 4U, output, 6U, 2U, 257U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "execute prepare", error)) {
    return false;
  }
  sllm_completion_t *sentinel_completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x55U));
  sllm_rmsnorm_dispatch_info_t invalid_info = rmsnorm_dispatch_info();
  invalid_info.struct_size = sizeof(invalid_info) - 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          &invalid_info, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "truncated dispatch info",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      invalid_info.struct_size != sizeof(invalid_info) - 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  invalid_info.reserved[0] = 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          &invalid_info, &error.sink),
                     SLLM_STATUS_RESERVED_NONZERO, "reserved dispatch info",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      invalid_info.reserved[0] != 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          nullptr, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "null dispatch info",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U))) {
    return false;
  }
  invalid_info.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, nullptr, &invalid_info,
                                          &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "null completion output",
                     error) ||
      invalid_info.abi_version != SLLM_HIP_ABI_VERSION + 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  invalid_info.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          &invalid_info, &error.sink),
                     SLLM_STATUS_INVALID_ABI_VERSION, "wrong dispatch ABI",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      invalid_info.abi_version != SLLM_HIP_ABI_VERSION + 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, other_queue, &sentinel_completion,
                               &invalid_info, &error.sink),
          SLLM_STATUS_PUBLIC_DEVICE_MISMATCH, "wrong RMSNorm queue", error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U))) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "RMSNorm execute", error) ||
      completion == nullptr || info.dispatch_id == 0U ||
      info.dispatch_count != 1U || info.kernel_id != 1U ||
      info.workgroup_size_x != 256U || info.grid_size_x != 2U ||
      info.row_count != 2U || info.normalized_size != 257U ||
      info.backend != SLLM_BACKEND_HIP || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol, "rmsnorm.baseline.wave32.v1") != 0 ||
      std::strcmp(info.device_symbol, "sllm_rmsnorm_baseline_wave32_v1") != 0 ||
      fake_hip::rmsnorm_launch_calls() != 1U ||
      fake_hip::rmsnorm_last_normalized_size() != 257U ||
      fake_hip::rmsnorm_last_row_count() != 2U) {
    return false;
  }
  sllm_completion_t *second_completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x77U));
  sllm_rmsnorm_dispatch_info_t second_info = rmsnorm_dispatch_info();
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &second_completion,
                                          &second_info, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "second in-flight execute",
                     error) ||
      second_completion != reinterpret_cast<sllm_completion_t *>(
                               static_cast<uintptr_t>(0x77U))) {
    return false;
  }
  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "in-flight plan release",
                     error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(sllm_completion_query(completion, &result, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm completion query", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS ||
      result.transfer_size_bytes != 0U || result.available_bytes != 0U ||
      !expect_status(sllm_completion_release(&completion, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm completion release", error) ||
      completion != nullptr) {
    return false;
  }
  const uint64_t first_dispatch = info.dispatch_id;
  completion = nullptr;
  info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "RMSNorm plan reuse", error) ||
      info.dispatch_id <= first_dispatch ||
      !expect_status(sllm_completion_query(completion, &result, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm reused completion query",
                     error) ||
      !expect_status(sllm_completion_release(&completion, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm reused completion release",
                     error) ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm reused plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&scale) || !release_buffer(&output) ||
      !release_context(&context) || !release_queue(&other_queue) ||
      !release_context(&other_context)) {
    return false;
  }
  return true;
}

bool rmsnorm_execute_boundaries_and_failures() {
  constexpr uint64_t columns[] = {1U,   3U,    17U,   255U,  256U,
                                  257U, 2560U, 4095U, 4096U, 4097U};
  constexpr uint64_t rows[] = {1U, 2U, 3U};
  for (const uint64_t row_count : rows) {
    for (const uint64_t column_count : columns) {
      fake_hip::reset();
      sllm_context_t *context = nullptr;
      sllm_queue_t *queue = nullptr;
      sllm_buffer_t *activation = nullptr;
      sllm_buffer_t *scale = nullptr;
      sllm_buffer_t *output = nullptr;
      const uint64_t row_bytes = row_count * column_count * 2U + 64U;
      const uint64_t scale_bytes = column_count * 2U + 64U;
      if (!create_context(&context) || !create_queue(context, &queue) ||
          !create_buffer_sized(context, row_bytes, &activation) ||
          !create_buffer_sized(context, scale_bytes, &scale) ||
          !create_buffer_sized(context, row_bytes, &output)) {
        return false;
      }
      sllm_rmsnorm_desc_t descriptor = rmsnorm_descriptor(
          activation, 0U, scale, 0U, output, 0U, row_count, column_count);
      sllm_rmsnorm_plan_t *plan = nullptr;
      Error error;
      if (!expect_status(
              sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
              SLLM_STATUS_OK, "boundary prepare", error)) {
        return false;
      }
      sllm_completion_t *completion = nullptr;
      sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
      const sllm_status_t expected =
          column_count == 4097U ? SLLM_STATUS_UNSUPPORTED : SLLM_STATUS_OK;
      if (!expect_status(sllm_rmsnorm_execute(plan, queue, &completion, &info,
                                              &error.sink),
                         expected, "boundary execute", error) ||
          (expected == SLLM_STATUS_OK &&
           (!expect_status(
                sllm_completion_query(completion, nullptr, &error.sink),
                SLLM_STATUS_INVALID_ARGUMENT, "boundary null completion result",
                error) ||
            !expect_status(
                sllm_completion_query(
                    completion,
                    reinterpret_cast<sllm_completion_result_t *>(&info),
                    &error.sink),
                SLLM_STATUS_RESERVED_NONZERO,
                "boundary wrong completion result", error))) ||
          (expected == SLLM_STATUS_OK &&
           !expect_status(sllm_completion_release(&completion, &error.sink),
                          SLLM_STATUS_PUBLIC_BUSY,
                          "boundary unqueried completion release", error))) {
        return false;
      }
      if (expected == SLLM_STATUS_OK) {
        sllm_completion_result_t result{};
        result.struct_size = sizeof(result);
        result.abi_version = SLLM_HIP_ABI_VERSION;
        if (!expect_status(
                sllm_completion_query(completion, &result, &error.sink),
                SLLM_STATUS_OK, "boundary query", error) ||
            !expect_status(sllm_completion_release(&completion, &error.sink),
                           SLLM_STATUS_OK, "boundary release", error)) {
          return false;
        }
      }
      if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                         SLLM_STATUS_OK, "boundary plan release", error) ||
          !release_queue(&queue) || !release_buffer(&activation) ||
          !release_buffer(&scale) || !release_buffer(&output) ||
          !release_context(&context)) {
        return false;
      }
    }
  }

  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 4096U, &activation) ||
      !create_buffer_sized(context, 2048U, &scale) ||
      !create_buffer_sized(context, 4096U, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 256U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "failure prepare", error)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
  const std::size_t events_before_failures = fake_hip::live_events();
  fake_hip::set_rmsnorm_launch_status(hipErrorUnknown);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, "launch failure", error) ||
      completion != nullptr ||
      fake_hip::live_events() != events_before_failures ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "launch rollback plan release", error)) {
    return false;
  }
  fake_hip::set_rmsnorm_launch_status(hipSuccess);
  plan = nullptr;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "event failure prepare", error)) {
    return false;
  }
  fake_hip::set_event_record_status(hipErrorUnknown);
  info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, "event failure", error) ||
      completion != nullptr ||
      fake_hip::live_events() != events_before_failures ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "event rollback plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&scale) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  return true;
}

bool rmsnorm_execute_exception_scope_guards_restore_plan_reuse() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_test_rmsnorm_execute_throw_after_reservation(0U);
  sllm_test_rmsnorm_execute_throw_after_registration(0U);

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 4096U, &activation) ||
      !create_buffer_sized(context, 2048U, &scale) ||
      !create_buffer_sized(context, 4096U, &output)) {
    return false;
  }
  const sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 257U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "scope-guard prepare", error) ||
      plan == nullptr) {
    return false;
  }
  const std::size_t events_before = fake_hip::live_events();
  sllm_completion_t *completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x9U));
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();

  sllm_test_rmsnorm_execute_throw_after_reservation(1U);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_INTERNAL_ERROR, "scope-guard reservation exception",
          error) ||
      completion != nullptr || fake_hip::live_events() != events_before ||
      fake_hip::rmsnorm_launch_calls() != 0U) {
    std::cerr
        << "reservation exception leaked RMSNorm accounting or a handle\n";
    return false;
  }

  const auto execute_success = [&]() {
    completion = nullptr;
    info = rmsnorm_dispatch_info();
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    return expect_status(sllm_rmsnorm_execute(plan, queue, &completion, &info,
                                              &error.sink),
                         SLLM_STATUS_OK, "scope-guard plan reuse", error) &&
           completion != nullptr &&
           expect_status(
               sllm_completion_query(completion, &result, &error.sink),
               SLLM_STATUS_OK, "scope-guard completion query", error) &&
           expect_status(sllm_completion_release(&completion, &error.sink),
                         SLLM_STATUS_OK, "scope-guard completion release",
                         error) &&
           completion == nullptr;
  };
  if (!execute_success()) {
    return false;
  }

  const uint64_t launches_before_registration_fault =
      fake_hip::rmsnorm_launch_calls();
  completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0xaU));
  info = rmsnorm_dispatch_info();
  sllm_test_rmsnorm_execute_throw_after_registration(1U);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_INTERNAL_ERROR, "scope-guard registration exception",
          error) ||
      completion != nullptr || fake_hip::live_events() != events_before ||
      fake_hip::rmsnorm_launch_calls() != launches_before_registration_fault) {
    std::cerr
        << "registration exception leaked RMSNorm accounting or a handle\n";
    return false;
  }
  if (!execute_success() ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "scope-guard plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&scale) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  sllm_test_rmsnorm_execute_throw_after_reservation(0U);
  sllm_test_rmsnorm_execute_throw_after_registration(0U);
  return true;
}

bool rmsnorm_registered_exception_with_event_destroy_failure_is_quarantined() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_test_rmsnorm_execute_throw_after_reservation(0U);
  sllm_test_rmsnorm_execute_throw_after_registration(0U);

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 4096U, &activation) ||
      !create_buffer_sized(context, 2048U, &scale) ||
      !create_buffer_sized(context, 4096U, &output)) {
    return false;
  }
  const sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 257U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "ambiguous-cleanup prepare", error) ||
      plan == nullptr) {
    return false;
  }

  const std::size_t events_before = fake_hip::live_events();
  const std::size_t destroy_before = fake_hip::event_destroy_calls();
  const std::size_t poison_before = sllm_test_poison_count();
  const std::size_t launches_before = fake_hip::rmsnorm_launch_calls();
  sllm_completion_t *completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0xbU));
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();

  /* EventDestroyError models an ownership-ambiguous native cleanup result:
   * the injection returns an error without calling fake hipEventDestroy. */
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  sllm_test_rmsnorm_execute_throw_after_registration(1U);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_INTERNAL_ERROR, "ambiguous-cleanup registered exception",
          error) ||
      completion != nullptr ||
      sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::EventDestroyError) ||
      sllm_test_poison_count() != poison_before + 1U ||
      fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() != events_before + 1U ||
      fake_hip::rmsnorm_launch_calls() != launches_before) {
    std::cerr
        << "registered exception did not fail closed on ambiguous event cleanup"
        << " completion=" << (completion == nullptr ? "null" : "set")
        << " poison=" << sllm_test_poison_count()
        << " expected_poison=" << poison_before + 1U
        << " destroy=" << fake_hip::event_destroy_calls()
        << " expected_destroy=" << destroy_before
        << " live_events=" << fake_hip::live_events()
        << " expected_live_events=" << events_before + 1U << '\n';
    return false;
  }
  sllm_test_rmsnorm_execute_throw_after_registration(0U);

  /* The poison owner retains the completion graph and its live event.  The
   * plan remains in-flight, and the poisoned Context rejects all reuse and
   * cleanup attempts; this test deliberately does not claim safe reuse. */
  completion = nullptr;
  info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_BUSY, "ambiguous-cleanup execute reuse", error) ||
      completion != nullptr ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "ambiguous-cleanup plan release",
                     error) ||
      plan == nullptr ||
      !expect_status(sllm_queue_release(&queue, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup queue release", error) ||
      queue == nullptr ||
      !expect_status(sllm_buffer_release(&activation, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup activation release", error) ||
      activation == nullptr ||
      !expect_status(sllm_buffer_release(&scale, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup scale release", error) ||
      scale == nullptr ||
      !expect_status(sllm_buffer_release(&output, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup output release", error) ||
      output == nullptr ||
      !expect_status(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup context release", error) ||
      context == nullptr || fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() != events_before + 1U ||
      fake_hip::rmsnorm_launch_calls() != launches_before) {
    std::cerr
        << "ambiguous cleanup was retried or the poisoned graph was reusable\n";
    return false;
  }
  sllm_public_runtime::FaultInjector::reset();
  return true;
}

bool rmsnorm_execute_row_limit_and_overflow() {
  constexpr uint64_t rows[] = {UINT64_C(4294967295), UINT64_C(4294967296)};
  for (const uint64_t row_count : rows) {
    fake_hip::reset();
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    sllm_buffer_t *activation = nullptr;
    sllm_buffer_t *scale = nullptr;
    sllm_buffer_t *output = nullptr;
    constexpr uint64_t activation_bytes = UINT64_C(8589934592);
    if (!create_context(&context) || !create_queue(context, &queue) ||
        !create_buffer_sized(context, activation_bytes, &activation) ||
        !create_buffer_sized(context, 2U, &scale) ||
        !create_buffer_sized(context, activation_bytes, &output)) {
      return false;
    }
    sllm_rmsnorm_desc_t descriptor = rmsnorm_descriptor(
        activation, 0U, scale, 0U, output, 0U, row_count, 1U);
    sllm_rmsnorm_plan_t *plan = nullptr;
    Error error;
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "row-limit prepare", error)) {
      return false;
    }
    sllm_completion_t *completion = nullptr;
    sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
    const sllm_status_t expected = row_count == UINT64_C(4294967296)
                                       ? SLLM_STATUS_UNSUPPORTED
                                       : SLLM_STATUS_OK;
    if (!expect_status(
            sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
            expected, "row-limit execute", error) ||
        fake_hip::rmsnorm_last_row_count() !=
            (expected == SLLM_STATUS_OK ? UINT32_C(4294967295) : 0U)) {
      return false;
    }
    if (expected == SLLM_STATUS_OK) {
      sllm_completion_result_t result{};
      result.struct_size = sizeof(result);
      result.abi_version = SLLM_HIP_ABI_VERSION;
      if (!expect_status(
              sllm_completion_query(completion, &result, &error.sink),
              SLLM_STATUS_OK, "row-limit query", error) ||
          !expect_status(sllm_completion_release(&completion, &error.sink),
                         SLLM_STATUS_OK, "row-limit completion release",
                         error)) {
        return false;
      }
    }
    if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "row-limit plan release", error) ||
        !release_queue(&queue) || !release_buffer(&activation) ||
        !release_buffer(&scale) || !release_buffer(&output) ||
        !release_context(&context)) {
      return false;
    }
  }
  return true;
}

bool rmsnorm_execute_flattens_rank_one_through_eight() {
  constexpr std::size_t rank_count = 8U;
  constexpr uint32_t ranks[] = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
  constexpr uint64_t shapes[rank_count][8U] = {
      {17U, 0U, 0U, 0U, 0U, 0U, 0U, 0U}, {3U, 17U, 0U, 0U, 0U, 0U, 0U, 0U},
      {2U, 3U, 17U, 0U, 0U, 0U, 0U, 0U}, {2U, 2U, 3U, 17U, 0U, 0U, 0U, 0U},
      {2U, 3U, 2U, 3U, 17U, 0U, 0U, 0U}, {2U, 2U, 3U, 2U, 3U, 17U, 0U, 0U},
      {2U, 3U, 2U, 2U, 3U, 2U, 17U, 0U}, {2U, 2U, 3U, 2U, 2U, 3U, 2U, 17U},
  };
  constexpr uint64_t expected_rows[] = {1U, 3U, 6U, 12U, 36U, 72U, 144U, 288U};
  for (std::size_t rank_index = 0U; rank_index != rank_count; ++rank_index) {
    fake_hip::reset();
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    sllm_buffer_t *activation = nullptr;
    sllm_buffer_t *scale = nullptr;
    sllm_buffer_t *output = nullptr;
    if (!create_context(&context) || !create_queue(context, &queue) ||
        !create_buffer_sized(context, 32768U, &activation) ||
        !create_buffer_sized(context, 32768U, &scale) ||
        !create_buffer_sized(context, 32768U, &output)) {
      return false;
    }
    sllm_rmsnorm_desc_t descriptor =
        rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 1U, 17U);
    descriptor.activation = rmsnorm_binding_rank(
        activation, 0U, ranks[rank_index], 17U, shapes[rank_index]);
    descriptor.output = rmsnorm_binding_rank(output, 0U, ranks[rank_index], 17U,
                                             shapes[rank_index]);
    sllm_rmsnorm_plan_t *plan = nullptr;
    Error error;
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "rank-flatten prepare", error)) {
      return false;
    }
    sllm_completion_t *completion = nullptr;
    sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
    if (!expect_status(
            sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
            SLLM_STATUS_OK, "rank-flatten execute", error) ||
        info.row_count != expected_rows[rank_index] ||
        info.normalized_size != 17U ||
        fake_hip::rmsnorm_last_row_count() != expected_rows[rank_index]) {
      return false;
    }
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    if (!expect_status(sllm_completion_query(completion, &result, &error.sink),
                       SLLM_STATUS_OK, "rank-flatten query", error) ||
        !expect_status(sllm_completion_release(&completion, &error.sink),
                       SLLM_STATUS_OK, "rank-flatten completion release",
                       error) ||
        !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "rank-flatten plan release", error) ||
        !release_queue(&queue) || !release_buffer(&activation) ||
        !release_buffer(&scale) || !release_buffer(&output) ||
        !release_context(&context)) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  if (!rmsnorm_bf16_rne_bit_contract()) {
    return 1;
  }
  if (!bounded_counter_cas_contention_is_fail_closed() ||
      !completion_safety_quarantine_is_bounded_and_fail_closed() ||
      !successful_completion_lifecycle() ||
      !d2h_staging_and_completion_read_is_byte_exact() ||
      !positive_completion_with_deferred_event_destroy_retains_dependencies()) {
    return 1;
  }
  if (!concurrent_pin_and_release() || !fatal_completion_is_quarantined() ||
      !registry_failure_destroys_or_orphans_before_rollback() ||
      !registry_exception_reaches_real_catch_before_rollback() ||
      !production_orphan_owner_grows_past_128() ||
      !rmsnorm_prepare_lifecycle_and_negative_contract() ||
      !rmsnorm_plan_accounting_failure_is_consumed_and_quarantined() ||
      !rmsnorm_guard_page_prefix_is_fail_closed() ||
      !rmsnorm_table_driven_negative_contract() ||
      !rmsnorm_prepare_required_shape_and_context_cases()) {
    return 1;
  }
  if (!rmsnorm_execute_metadata_and_reuse()) {
    std::cerr << "RMSNorm execute metadata/reuse test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_boundaries_and_failures()) {
    std::cerr << "RMSNorm execute boundary/failure test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_exception_scope_guards_restore_plan_reuse()) {
    std::cerr << "RMSNorm execute exception-scope-guard test failed\n";
    return 1;
  }
  if (!rmsnorm_registered_exception_with_event_destroy_failure_is_quarantined()) {
    std::cerr << "RMSNorm registered-exception ambiguous-cleanup test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_row_limit_and_overflow()) {
    std::cerr << "RMSNorm execute row-limit test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_flattens_rank_one_through_eight()) {
    std::cerr << "RMSNorm execute rank-flatten test failed\n";
    return 1;
  }
  std::cout << "production public runtime host fault test: PASS\n";
  return 0;
}
