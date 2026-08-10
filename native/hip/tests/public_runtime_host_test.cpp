#include "evidence_abi.h"
#include "public_runtime_internal.hpp"
#include "rmsnorm_kernel_internal.hpp"

#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

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

sllm_elementwise_desc_t
elementwise_descriptor(const sllm_elementwise_operation_t operation,
                       const sllm_buffer_t *const input0,
                       const sllm_buffer_t *const input1,
                       const sllm_buffer_t *const output, const uint64_t size) {
  sllm_elementwise_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_ELEMENTWISE_VERSION;
  descriptor.operation = operation;
  const auto binding = [operation](const sllm_buffer_t *const buffer,
                                   const uint64_t logical_size) {
    auto result = rmsnorm_binding(buffer, 0U, 1U, logical_size);
    if (operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL) {
      result.rank = 3U;
      result.shape[0] = logical_size;
      result.shape[1] = 16U;
      result.shape[2] = 256U;
      result.stride_elements[0] = 4096U;
      result.stride_elements[1] = 256U;
      result.stride_elements[2] = 1U;
    }
    return result;
  };
  descriptor.input0 = binding(input0, size);
  if (operation != SLLM_ELEMENTWISE_OPERATION_COPY) {
    descriptor.input1 = binding(input1, size);
  }
  descriptor.output = binding(output, size);
  return descriptor;
}

sllm_elementwise_dispatch_info_t elementwise_dispatch_info() {
  sllm_elementwise_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION;
  return info;
}

bool elementwise_prepare_execute_and_negative_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *input0 = nullptr;
  sllm_buffer_t *input1 = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 32768U, &input0) ||
      !create_buffer_sized(context, 32768U, &input1) ||
      !create_buffer_sized(context, 32768U, &output)) {
    return false;
  }
  Error error;
  const auto run = [&](const sllm_elementwise_operation_t operation,
                       const uint64_t size, const uint32_t kernel_id,
                       const char *const symbol) {
    const uint64_t elements =
        operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL ? size * 4096U
                                                            : size;
    auto descriptor =
        elementwise_descriptor(operation, input0, input1, output, size);
    sllm_elementwise_plan_t *plan = nullptr;
    if (!expect_status(
            sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "elementwise prepare", error) ||
        plan == nullptr) {
      return false;
    }
    sllm_completion_t *completion = nullptr;
    auto info = elementwise_dispatch_info();
    const bool executed =
        expect_status(sllm_elementwise_execute(plan, queue, &completion, &info,
                                               &error.sink),
                      SLLM_STATUS_OK, "elementwise execute", error) &&
        completion != nullptr && info.operation == operation &&
        info.dispatch_count == 1U && info.kernel_id == kernel_id &&
        info.workgroup_size_x == SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE &&
        info.grid_size_x ==
            (elements + SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE - 1U) /
                SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE &&
        info.element_count == elements && info.fallback_allowed == 0U &&
        info.fallback_used == 0U &&
        std::strcmp(info.kernel_symbol, symbol) == 0 &&
        query_completion(completion, SLLM_STATUS_OK) &&
        release_completion(&completion) &&
        expect_status(sllm_elementwise_plan_release(&plan, &error.sink),
                      SLLM_STATUS_OK, "elementwise plan release", error);
    return executed && plan == nullptr && completion == nullptr;
  };
  const auto upload_words = [&](const sllm_buffer_t *const buffer,
                                const std::vector<uint16_t> &words) {
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = const_cast<uint16_t *>(words.data());
    transfer.size_bytes = words.size() * sizeof(uint16_t);
    sllm_completion_t *completion = nullptr;
    return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                              &completion, &error.sink),
                         SLLM_STATUS_OK, "elementwise test upload", error) &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion);
  };
  std::vector<uint16_t> sigmoid_gate(3U * 16U * 256U, UINT16_C(0x4000));
  std::vector<uint16_t> attention_value(sigmoid_gate.size(), UINT16_C(0x3f80));
  if (!upload_words(input0, sigmoid_gate) ||
      !upload_words(input1, attention_value)) {
    return false;
  }
  if (!run(SLLM_ELEMENTWISE_OPERATION_COPY, 257U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_COPY_V1,
           "elementwise.copy.bf16.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_ADD, 17U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_ADD_V1,
           "elementwise.add.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_SILU_MUL, 255U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_SILU_MUL_V1,
           "elementwise.silu_mul.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL, 3U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_SIGMOID_MUL_V1,
           "elementwise.sigmoid_mul.bf16_fp32.v1") ||
      fake_hip::elementwise_copy_launch_calls() != 1U ||
      fake_hip::elementwise_add_launch_calls() != 1U ||
      fake_hip::elementwise_silu_mul_launch_calls() != 1U ||
      fake_hip::elementwise_sigmoid_mul_launch_calls() != 1U ||
      fake_hip::elementwise_last_element_count() != 3U * 16U * 256U) {
    return false;
  }

  sllm_completion_t *readback = nullptr;
  const std::size_t sigmoid_bytes = sigmoid_gate.size() * sizeof(uint16_t);
  if (!submit_d2h(queue, output, sigmoid_bytes, &readback) ||
      !query_completion(readback, SLLM_STATUS_OK)) {
    return false;
  }
  std::vector<uint16_t> sigmoid_output(sigmoid_gate.size());
  uint64_t bytes_written = 0U;
  const float sigmoid = 1.0F / (1.0F + std::exp(-2.0F));
  const uint16_t expected_sigmoid =
      sllm_rmsnorm_kernel::float_to_bf16_rne_bits(sigmoid);
  const uint16_t forbidden_silu =
      sllm_rmsnorm_kernel::float_to_bf16_rne_bits(2.0F * sigmoid);
  if (!expect_status(sllm_completion_read(readback, sigmoid_output.data(),
                                          sigmoid_bytes, &bytes_written,
                                          &error.sink),
                     SLLM_STATUS_OK, "sigmoid output read", error) ||
      bytes_written != sigmoid_bytes ||
      sigmoid_output.front() != expected_sigmoid ||
      sigmoid_output.front() == forbidden_silu ||
      !release_completion(&readback)) {
    return false;
  }

  sllm_elementwise_plan_t *plan = nullptr;
  auto descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_COPY,
                                           input0, input1, output, 3U);
  descriptor.input0.dtype = SLLM_TENSOR_DTYPE_F32;
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_UNSUPPORTED_DTYPE, "elementwise dtype rejection",
          error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_ADD, input0,
                                      input0, output, 3U);
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "elementwise alias rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL,
                                      input0, input1, input0, 3U);
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "sigmoid alias rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL,
                                      input0, input1, output, 3U);
  descriptor.input0.rank = 2U;
  descriptor.input0.shape[0] = 3U;
  descriptor.input0.shape[1] = 4096U;
  descriptor.input0.shape[2] = 0U;
  descriptor.input0.stride_elements[0] = 4096U;
  descriptor.input0.stride_elements[1] = 1U;
  descriptor.input0.stride_elements[2] = 0U;
  descriptor.input1 = descriptor.input0;
  descriptor.input1.buffer = input1;
  descriptor.output = descriptor.input0;
  descriptor.output.buffer = output;
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_SHAPE_MISMATCH, "sigmoid flat shape rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL,
                                      input0, input1, output, 3U);
  descriptor.input0.shape[1] = 4U;
  descriptor.input0.stride_elements[0] = 1024U;
  descriptor.input1 = descriptor.input0;
  descriptor.input1.buffer = input1;
  descriptor.output = descriptor.input0;
  descriptor.output.buffer = output;
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_SHAPE_MISMATCH, "sigmoid GQA head rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_COPY, input0,
                                      input1, output, 3U);
  descriptor.input1 = rmsnorm_binding(input1, 0U, 1U, 3U);
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
          "copy second-input rejection", error) ||
      plan != nullptr || !release_queue(&queue) || !release_buffer(&input0) ||
      !release_buffer(&input1) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  return fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool embedding_prepare_execute_and_token_range_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *token_ids = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 24U, &weight) ||
      !create_buffer_sized(context, 12U, &token_ids) ||
      !create_buffer_sized(context, 18U, &output)) {
    return false;
  }
  uint16_t weight_words[12] = {0U, 1U, 2U, 3U, 4U,  5U,
                               6U, 7U, 8U, 9U, 10U, 11U};
  int32_t ids[3] = {2, 0, 2};
  const auto upload = [&](const sllm_buffer_t *const buffer, void *const bytes,
                          const uint64_t size) {
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = bytes;
    transfer.size_bytes = size;
    sllm_completion_t *completion = nullptr;
    Error error;
    return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                              &completion, &error.sink),
                         SLLM_STATUS_OK, "embedding input upload", error) &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion);
  };
  if (!upload(weight, weight_words, sizeof(weight_words)) ||
      !upload(token_ids, ids, sizeof(ids))) {
    return false;
  }
  sllm_embedding_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_EMBEDDING_VERSION;
  descriptor.weight = rmsnorm_binding(weight, 0U, 4U, 3U);
  descriptor.token_ids = rmsnorm_binding(token_ids, 0U, 1U, 3U);
  descriptor.token_ids.dtype = SLLM_TENSOR_DTYPE_I32;
  descriptor.token_ids.rank = 1U;
  descriptor.token_ids.shape[0] = 3U;
  descriptor.token_ids.shape[1] = 0U;
  descriptor.token_ids.stride_elements[0] = 1U;
  descriptor.token_ids.stride_elements[1] = 0U;
  descriptor.output = rmsnorm_binding(output, 0U, 3U, 3U);
  sllm_embedding_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_embedding_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "embedding prepare", error) ||
      plan == nullptr) {
    return false;
  }
  sllm_embedding_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  if (!expect_status(
          sllm_embedding_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "embedding execute", error) ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_EMBEDDING_KERNEL_ID_GATHER_V1 ||
      info.token_count != 3U || info.hidden_size != 3U ||
      info.vocab_size != 4U || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol, "embedding.gather.bf16_i32.v1") != 0 ||
      fake_hip::embedding_gather_launch_calls() != 1U ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion)) {
    return false;
  }
  ids[1] = -1;
  if (!upload(token_ids, ids, sizeof(ids)) ||
      !expect_status(
          sllm_embedding_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_TOKEN_ID_OUT_OF_RANGE,
          "embedding negative token rejection", error) ||
      completion != nullptr ||
      fake_hip::embedding_gather_launch_calls() != 1U ||
      !expect_status(sllm_embedding_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "embedding plan release", error) ||
      !release_queue(&queue) || !release_buffer(&weight) ||
      !release_buffer(&token_ids) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  return true;
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
      fake_hip::live_events() != events_before + 2U ||
      fake_hip::rmsnorm_launch_calls() != launches_before) {
    std::cerr
        << "registered exception did not fail closed on ambiguous event cleanup"
        << " completion=" << (completion == nullptr ? "null" : "set")
        << " poison=" << sllm_test_poison_count()
        << " expected_poison=" << poison_before + 1U
        << " destroy=" << fake_hip::event_destroy_calls()
        << " expected_destroy=" << destroy_before
        << " live_events=" << fake_hip::live_events()
        << " expected_live_events=" << events_before + 2U << '\n';
    return false;
  }
  sllm_test_rmsnorm_execute_throw_after_registration(0U);

  /* The poison owner retains the completion graph and both live timing events.
   * The plan remains in-flight, and the poisoned Context rejects all reuse and
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
      fake_hip::live_events() != events_before + 2U ||
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

sllm_tensor_binding_t matmul_binding(const sllm_buffer_t *const buffer,
                                     const uint64_t offset, const uint64_t rows,
                                     const uint64_t columns) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = 2U;
  binding.shape[0] = rows;
  binding.shape[1] = columns;
  binding.stride_elements[0] = columns;
  binding.stride_elements[1] = 1U;
  return binding;
}

sllm_matmul_desc_t matmul_descriptor(
    const sllm_buffer_t *const activation, const uint64_t activation_offset,
    const sllm_buffer_t *const weight, const uint64_t weight_offset,
    const sllm_buffer_t *const output, const uint64_t output_offset,
    const uint64_t m = 3U, const uint64_t k = 5U, const uint64_t n = 7U) {
  sllm_matmul_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_MATMUL_VERSION;
  descriptor.activation = matmul_binding(activation, activation_offset, m, k);
  descriptor.weight = matmul_binding(weight, weight_offset, n, k);
  descriptor.output = matmul_binding(output, output_offset, m, n);
  return descriptor;
}

sllm_matmul_dispatch_info_t matmul_dispatch_info() {
  sllm_matmul_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
  return info;
}

bool matmul_prepare_execute_and_negative_contract() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_context_t *other_context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_context(&other_context) ||
      !create_queue(context, &queue) ||
      !create_buffer_sized(context, 1024U, &activation) ||
      !create_buffer_sized(context, 1024U, &weight) ||
      !create_buffer_sized(context, 1024U, &output)) {
    return false;
  }
  Error error;
  auto descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  sllm_matmul_plan_t *plan = nullptr;
  descriptor.reserved[0] = 1U;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_RESERVED_NONZERO, "matmul descriptor reserved rejection",
          error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  descriptor.activation.dtype = SLLM_TENSOR_DTYPE_F32;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_UNSUPPORTED_DTYPE, "matmul dtype rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  descriptor.weight.shape[1] = 4U;
  descriptor.weight.stride_elements[0] = 4U;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_SHAPE_MISMATCH, "matmul shape rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, activation, 0U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "matmul overlap rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 1024U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_BUFFER_OUT_OF_BOUNDS, "matmul bounds rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 1U, weight, 0U, output, 0U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_MISALIGNED_OFFSET, "matmul alignment rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  if (!expect_status(
          sllm_matmul_prepare(other_context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH, "matmul context rejection",
          error) ||
      plan != nullptr) {
    return false;
  }

  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "matmul prepare", error) ||
      plan == nullptr) {
    return false;
  }
  sllm_completion_t *sentinel_completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x55U));
  auto invalid_info = matmul_dispatch_info();
  invalid_info.reserved[0] = 1U;
  if (!expect_status(sllm_matmul_execute(plan, queue, &sentinel_completion,
                                         &invalid_info, &error.sink),
                     SLLM_STATUS_RESERVED_NONZERO,
                     "matmul dispatch reserved rejection", error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      fake_hip::matmul_launch_calls() != 0U) {
    return false;
  }

  sllm_completion_t *completion = nullptr;
  auto info = matmul_dispatch_info();
  if (!expect_status(
          sllm_matmul_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "matmul execute", error) ||
      completion == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_MATMUL_KERNEL_ID_BASELINE_BF16_FP32_V1 ||
      info.workgroup_size_x != SLLM_HIP_MATMUL_WORKGROUP_SIZE ||
      info.grid_size_x != 1U || info.m != 3U || info.k != 5U || info.n != 7U ||
      info.output_elements != 21U || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol, "matmul.bf16_fp32.v1") != 0 ||
      std::strcmp(info.device_symbol, "sllm_matmul_bf16_fp32_v1") != 0 ||
      std::strcmp(info.gcn_arch_name, "gfx1201") != 0 ||
      fake_hip::matmul_launch_calls() != 1U ||
      fake_hip::matmul_last_m() != 3U || fake_hip::matmul_last_k() != 5U ||
      fake_hip::matmul_last_n() != 7U ||
      fake_hip::matmul_last_output_elements() != 21U) {
    return false;
  }
  if (!query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "matmul plan release", error) ||
      plan != nullptr || !release_queue(&queue) ||
      !release_buffer(&activation) || !release_buffer(&weight) ||
      !release_buffer(&output) || !release_context(&other_context) ||
      !release_context(&context)) {
    return false;
  }
  return fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool matmul_async_lifetime_and_cleanup() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 1024U, &activation) ||
      !create_buffer_sized(context, 1024U, &weight) ||
      !create_buffer_sized(context, 1024U, &output)) {
    return false;
  }
  Error error;
  auto descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  sllm_matmul_plan_t *plan = nullptr;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "matmul async prepare", error)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  auto info = matmul_dispatch_info();
  if (!expect_status(
          sllm_matmul_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "matmul async execute", error) ||
      !expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "matmul in-flight plan release",
                     error) ||
      !release_queue(&queue, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_buffer(&activation, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_buffer(&weight, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_buffer(&output, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_context(&context, SLLM_STATUS_PUBLIC_BUSY) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "matmul async plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&weight) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }

  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 1024U, &activation) ||
      !create_buffer_sized(context, 1024U, &weight) ||
      !create_buffer_sized(context, 1024U, &output)) {
    return false;
  }
  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "matmul cleanup prepare", error)) {
    return false;
  }
  fake_hip::set_matmul_launch_status(hipErrorUnknown);
  completion = nullptr;
  info = matmul_dispatch_info();
  const bool cleanup_failed =
      expect_status(
          sllm_matmul_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, "matmul launch cleanup",
          error) &&
      completion == nullptr && fake_hip::matmul_launch_calls() == 2U;
  fake_hip::set_matmul_launch_status(hipSuccess);
  return cleanup_failed &&
         expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "matmul cleanup plan release", error) &&
         release_queue(&queue) && release_buffer(&activation) &&
         release_buffer(&weight) && release_buffer(&output) &&
         release_context(&context) && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

using AttentionBuffers = std::array<sllm_buffer_t *, 8>;

void release_attention_buffers(AttentionBuffers *const buffers) {
  if (buffers == nullptr) {
    return;
  }
  for (sllm_buffer_t *&buffer : *buffers) {
    if (buffer != nullptr) {
      release_buffer(&buffer);
    }
  }
}

bool create_attention_resources(sllm_context_t **const context,
                                sllm_queue_t **const queue,
                                AttentionBuffers *const buffers,
                                const uint64_t m) {
  if (!create_context(context) || !create_queue(*context, queue)) {
    return false;
  }
  const uint64_t sizes[8] = {
      m * 16U * 512U * sizeof(uint16_t),
      m * 4U * 256U * sizeof(uint16_t),
      16U * 256U * sizeof(uint16_t),
      4U * 256U * sizeof(uint16_t),
      m * sizeof(int32_t),
      m * 16U * 256U * sizeof(uint16_t),
      m * 16U * 256U * sizeof(uint16_t),
      m * 4U * 256U * sizeof(uint16_t),
  };
  for (std::size_t index = 0U; index != buffers->size(); ++index) {
    if (!create_buffer_sized(*context, sizes[index], &(*buffers)[index])) {
      return false;
    }
  }
  return true;
}

sllm_tensor_binding_t attention_binding(const sllm_buffer_t *const buffer,
                                        const uint32_t dtype,
                                        const uint32_t rank,
                                        const uint64_t *const shape) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.dtype = dtype;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t backwards = 0U; backwards != rank; ++backwards) {
    const uint32_t index = rank - 1U - backwards;
    binding.shape[index] = shape[index];
    binding.stride_elements[index] = stride;
    stride *= shape[index];
  }
  return binding;
}

sllm_attention_preprocess_desc_t
attention_preprocess_descriptor(const AttentionBuffers &buffers,
                                const uint64_t m,
                                const uint32_t start_position) {
  uint64_t packed_shape[] = {m, 16U, 512U};
  uint64_t k_shape[] = {m, 4U, 256U};
  constexpr uint64_t scale_q_shape[] = {16U, 256U};
  constexpr uint64_t scale_k_shape[] = {4U, 256U};
  uint64_t positions_shape[] = {m};
  uint64_t output_q_shape[] = {m, 16U, 256U};
  uint64_t output_k_shape[] = {m, 4U, 256U};
  sllm_attention_preprocess_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_ATTENTION_PREPROCESS_VERSION;
  descriptor.start_position = start_position;
  descriptor.packed_q_gate =
      attention_binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 3U, packed_shape);
  descriptor.k =
      attention_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 3U, k_shape);
  descriptor.q_raw_scale =
      attention_binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 2U, scale_q_shape);
  descriptor.k_raw_scale =
      attention_binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 2U, scale_k_shape);
  descriptor.positions =
      attention_binding(buffers[4], SLLM_TENSOR_DTYPE_I32, 1U, positions_shape);
  descriptor.q_output =
      attention_binding(buffers[5], SLLM_TENSOR_DTYPE_BF16, 3U, output_q_shape);
  descriptor.gate_output =
      attention_binding(buffers[6], SLLM_TENSOR_DTYPE_BF16, 3U, output_q_shape);
  descriptor.k_output =
      attention_binding(buffers[7], SLLM_TENSOR_DTYPE_BF16, 3U, output_k_shape);
  return descriptor;
}

bool upload_attention_positions(const sllm_queue_t *const queue,
                                const sllm_buffer_t *const buffer,
                                std::vector<int32_t> &positions) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = positions.data();
  transfer.size_bytes =
      static_cast<uint64_t>(positions.size() * sizeof(int32_t));
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                            &completion, &error.sink),
                       SLLM_STATUS_OK, "attention position upload", error) &&
         query_completion(completion, SLLM_STATUS_OK) &&
         release_completion(&completion);
}

sllm_attention_preprocess_dispatch_info_t attention_preprocess_dispatch_info() {
  sllm_attention_preprocess_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION;
  return info;
}

bool attention_preprocess_prepare_validation_and_old_abi() {
  fake_hip::reset();
  uint32_t abi_version = 0U;
  Error error;
  if (!expect_status(sllm_get_abi_version(&abi_version, &error.sink),
                     SLLM_STATUS_OK, "old ABI version query", error) ||
      abi_version != SLLM_HIP_ABI_VERSION) {
    return false;
  }
  constexpr uint64_t m = 2U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  auto descriptor = attention_preprocess_descriptor(buffers, m, 0U);
  sllm_attention_preprocess_plan_t *plan = nullptr;
  auto expect = [&](const sllm_attention_preprocess_desc_t &candidate,
                    const sllm_status_t expected, const char *const name) {
    plan = nullptr;
    return expect_status(sllm_attention_preprocess_prepare(context, &candidate,
                                                           &plan, &error.sink),
                         expected, name, error) &&
           plan == nullptr;
  };
  auto flat = descriptor;
  flat.packed_q_gate.rank = 2U;
  flat.packed_q_gate.shape[1] = 4096U;
  flat.packed_q_gate.shape[2] = 0U;
  flat.packed_q_gate.stride_elements[0] = 4096U;
  flat.packed_q_gate.stride_elements[1] = 1U;
  flat.packed_q_gate.stride_elements[2] = 0U;
  if (!expect(flat, SLLM_STATUS_SHAPE_MISMATCH, "flat Q/gate rejection")) {
    return false;
  }
  auto reserved = descriptor;
  reserved.reserved[0] = 1U;
  if (!expect(reserved, SLLM_STATUS_RESERVED_NONZERO,
              "attention reserved rejection")) {
    return false;
  }
  auto alias = descriptor;
  alias.q_output.buffer = alias.packed_q_gate.buffer;
  if (!expect(alias, SLLM_STATUS_ALIAS_OVERLAP, "attention alias rejection")) {
    return false;
  }
  const uint64_t packed_bytes = m * 16U * 512U * sizeof(uint16_t);
  const uint64_t q_output_bytes = m * 16U * 256U * sizeof(uint16_t);
  if (!release_buffer(&buffers[0]) ||
      !create_buffer_sized(context, packed_bytes + q_output_bytes,
                           &buffers[0])) {
    return false;
  }
  auto shared_nonoverlap = attention_preprocess_descriptor(buffers, m, 0U);
  shared_nonoverlap.q_output.buffer = buffers[0];
  shared_nonoverlap.q_output.byte_offset = packed_bytes;
  if (!expect_status(sllm_attention_preprocess_prepare(
                         context, &shared_nonoverlap, &plan, &error.sink),
                     SLLM_STATUS_OK, "attention shared nonoverlap prepare",
                     error) ||
      plan == nullptr ||
      !expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "attention shared nonoverlap plan release",
                     error)) {
    return false;
  }
  release_attention_buffers(&buffers);
  return release_queue(&queue) && release_context(&context) &&
         fake_hip::attention_preprocess_launch_calls() == 0U;
}

bool attention_preprocess_position_payload_mismatch_is_pre_dispatch() {
  fake_hip::reset();
  constexpr uint64_t m = 2U;
  constexpr uint32_t start_position = 9U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<int32_t> positions = {static_cast<int32_t>(start_position),
                                    static_cast<int32_t>(start_position + 2U)};
  Error error;
  auto descriptor = attention_preprocess_descriptor(buffers, m, start_position);
  sllm_attention_preprocess_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  auto info = attention_preprocess_dispatch_info();
  const bool valid =
      upload_attention_positions(queue, buffers[4], positions) &&
      expect_status(sllm_attention_preprocess_prepare(context, &descriptor,
                                                      &plan, &error.sink),
                    SLLM_STATUS_OK, "attention mismatch prepare", error) &&
      plan != nullptr &&
      expect_status(sllm_attention_preprocess_execute(plan, queue, &completion,
                                                      &info, &error.sink),
                    SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
                    "attention position mismatch", error) &&
      completion == nullptr &&
      fake_hip::attention_preprocess_launch_calls() == 0U &&
      expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "attention mismatch plan release", error);
  release_attention_buffers(&buffers);
  return valid && release_queue(&queue) && release_context(&context);
}

bool attention_preprocess_success_metadata_and_dispatch() {
  fake_hip::reset();
  constexpr uint64_t m = 3U;
  constexpr uint32_t start_position = 17U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<int32_t> positions;
  for (uint64_t index = 0U; index != m; ++index) {
    positions.push_back(static_cast<int32_t>(start_position + index));
  }
  Error error;
  auto descriptor = attention_preprocess_descriptor(buffers, m, start_position);
  sllm_attention_preprocess_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  auto info = attention_preprocess_dispatch_info();
  const bool valid =
      upload_attention_positions(queue, buffers[4], positions) &&
      expect_status(sllm_attention_preprocess_prepare(context, &descriptor,
                                                      &plan, &error.sink),
                    SLLM_STATUS_OK, "attention success prepare", error) &&
      plan != nullptr &&
      expect_status(sllm_attention_preprocess_execute(plan, queue, &completion,
                                                      &info, &error.sink),
                    SLLM_STATUS_OK, "attention success execute", error) &&
      completion != nullptr && info.dispatch_id != 0U &&
      info.dispatch_count == 1U &&
      info.kernel_id ==
          SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_ID_BASELINE_BF16_V1 &&
      info.workgroup_size_x == SLLM_HIP_ATTENTION_PREPROCESS_WORKGROUP_SIZE &&
      info.grid_size_x == m * 20U && info.m == m &&
      info.q_heads == SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS &&
      info.k_heads == SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS &&
      info.q_head_dim == SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM &&
      info.k_head_dim == SLLM_HIP_ATTENTION_PREPROCESS_K_HEAD_DIM &&
      info.rotary_dim == SLLM_HIP_ATTENTION_PREPROCESS_ROTARY_DIM &&
      info.start_position == start_position && info.fallback_allowed == 0U &&
      info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol,
                  "attention_preprocess.headwise_norm_rope.v1") == 0 &&
      std::strcmp(info.device_symbol,
                  "sllm_attention_preprocess_headwise_norm_rope_v1") == 0 &&
      std::strcmp(info.gcn_arch_name, "gfx1201") == 0 &&
      fake_hip::attention_preprocess_launch_calls() == 1U &&
      fake_hip::attention_preprocess_last_m() == m &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) &&
      expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "attention success plan release", error);
  release_attention_buffers(&buffers);
  return valid && release_queue(&queue) && release_context(&context);
}

bool create_kv_state(const sllm_context_t *const context,
                     const uint64_t capacity, sllm_kv_state_t **const state) {
  sllm_kv_state_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.session_id = 0x1234U;
  info.layer_id = 7U;
  info.capacity_tokens = capacity;
  Error error;
  return expect_status(sllm_kv_state_create(context, &info, state, &error.sink),
                       SLLM_STATUS_OK, "sllm_kv_state_create", error);
}

sllm_tensor_binding_t kv_input_binding(const sllm_buffer_t *const buffer,
                                       const uint64_t token_count,
                                       const uint64_t byte_offset = 0U) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = byte_offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = 3U;
  binding.shape[0] = token_count;
  binding.shape[1] = SLLM_HIP_KV_HEAD_COUNT;
  binding.shape[2] = SLLM_HIP_KV_HEAD_DIM;
  binding.stride_elements[0] = SLLM_HIP_KV_HEAD_COUNT * SLLM_HIP_KV_HEAD_DIM;
  binding.stride_elements[1] = SLLM_HIP_KV_HEAD_DIM;
  binding.stride_elements[2] = 1U;
  return binding;
}

sllm_kv_append_desc_t kv_append_descriptor(const sllm_buffer_t *const key,
                                           const sllm_buffer_t *const value,
                                           const uint64_t token_count,
                                           const uint64_t position) {
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = position;
  descriptor.start_position = position;
  descriptor.key_input = kv_input_binding(key, token_count);
  descriptor.value_input = kv_input_binding(value, token_count);
  return descriptor;
}

sllm_kv_append_info_t kv_append_info() {
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  return info;
}

bool upload_kv_words(const sllm_queue_t *const queue,
                     const sllm_buffer_t *const buffer,
                     const std::vector<uint16_t> &words) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<uint16_t *>(words.data());
  transfer.size_bytes = words.size() * sizeof(uint16_t);
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                            &completion, &error.sink),
                       SLLM_STATUS_OK, "KV input upload", error) &&
         completion != nullptr &&
         query_completion(completion, SLLM_STATUS_OK) &&
         release_completion(&completion);
}

bool kv_query(const sllm_kv_state_t *const state, uint64_t length,
              uint64_t generation) {
  sllm_kv_view_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  Error error;
  return expect_status(sllm_kv_state_query(state, &info, &error.sink),
                       SLLM_STATUS_OK, "KV state query", error) &&
         info.dtype == SLLM_TENSOR_DTYPE_F16 &&
         info.encoding == SLLM_TENSOR_ENCODING_UNQUANTIZED &&
         info.head_count == 4U && info.head_dim == 256U &&
         info.observed_length == length && info.generation == generation &&
         info.k_stride_elements[0] == info.capacity_tokens * 256U &&
         info.k_stride_elements[1] == 256U && info.k_stride_elements[2] == 1U &&
         info.v_stride_elements[0] == info.capacity_tokens * 256U &&
         info.v_stride_elements[1] == 256U && info.v_stride_elements[2] == 1U &&
         info.context_identity != 0U && info.state_identity != 0U;
}

sllm_tensor_binding_t
causal_attention_binding(const sllm_buffer_t *const buffer,
                         const uint64_t query_count) {
  const uint64_t shape[] = {query_count, 16U, 256U};
  return attention_binding(buffer, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
}

sllm_causal_attention_dispatch_info_t causal_attention_dispatch_info() {
  sllm_causal_attention_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  return info;
}

sllm_causal_attention_desc_t causal_attention_descriptor(
    const sllm_kv_state_t *const state, const sllm_buffer_t *const query,
    const sllm_buffer_t *const output, const uint64_t query_count,
    const uint64_t start_position, const uint64_t expected_kv_length) {
  sllm_causal_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
  descriptor.start_position = start_position;
  descriptor.expected_kv_length = expected_kv_length;
  descriptor.kv_state = state;
  descriptor.query = causal_attention_binding(query, query_count);
  descriptor.output = causal_attention_binding(output, query_count);
  return descriptor;
}

uint16_t causal_float_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

bool causal_attention_numerical_gqa_and_lifetime_contract() {
  fake_hip::reset();
  if (fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x0000)) !=
          UINT32_C(0x00000000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x8000)) !=
          UINT32_C(0x80000000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x0001)) !=
          UINT32_C(0x33800000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x8001)) !=
          UINT32_C(0xb3800000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x03ff)) !=
          UINT32_C(0x387fc000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x83ff)) !=
          UINT32_C(0xb87fc000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x7c00)) !=
          UINT32_C(0x7f800000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0xfc00)) !=
          UINT32_C(0xff800000) ||
      (fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x7e01)) &
       UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) ||
      (fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x7e01)) &
       UINT32_C(0x007fffff)) == 0U) {
    return false;
  }
  constexpr uint64_t query_count = 3U;
  constexpr uint64_t capacity = 3U;
  const std::size_t kv_elements = query_count * 4U * 256U;
  const std::size_t query_elements = query_count * 16U * 256U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  sllm_buffer_t *query = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, kv_elements * sizeof(uint16_t), &key) ||
      !create_buffer_sized(context, kv_elements * sizeof(uint16_t), &value) ||
      !create_buffer_sized(context, query_elements * sizeof(uint16_t),
                           &query) ||
      !create_buffer_sized(context, query_elements * sizeof(uint16_t),
                           &output)) {
    return false;
  }

  std::vector<uint16_t> key_words(kv_elements, UINT16_C(0));
  std::vector<uint16_t> value_words(kv_elements, UINT16_C(0));
  for (uint64_t token = 0U; token != query_count; ++token) {
    for (uint64_t head = 0U; head != 4U; ++head) {
      const uint16_t word =
          causal_float_to_bf16_rne(static_cast<float>(token + head + 1U));
      for (uint64_t dimension = 0U; dimension != 256U; ++dimension) {
        value_words[(token * 4U + head) * 256U + dimension] = word;
      }
    }
  }
  std::vector<uint16_t> query_words(query_elements, UINT16_C(0));
  if (!upload_kv_words(queue, key, key_words) ||
      !upload_kv_words(queue, value, value_words) ||
      !upload_kv_words(queue, query, query_words)) {
    return false;
  }

  Error error;
  sllm_completion_t *append_completion = nullptr;
  sllm_kv_append_desc_t append =
      kv_append_descriptor(key, value, query_count, 0U);
  sllm_kv_append_info_t append_info = kv_append_info();
  if (!expect_status(sllm_kv_state_append(state, queue, &append,
                                          &append_completion, &append_info,
                                          &error.sink),
                     SLLM_STATUS_OK, "causal KV append", error) ||
      !query_completion(append_completion, SLLM_STATUS_OK) ||
      !release_completion(&append_completion) ||
      !kv_query(state, query_count, 1U)) {
    return false;
  }

  sllm_completion_t *completion = nullptr;
  sllm_causal_attention_dispatch_info_t info = causal_attention_dispatch_info();
  sllm_causal_attention_desc_t descriptor = causal_attention_descriptor(
      state, query, output, query_count, 0U, query_count);
  sllm_causal_attention_desc_t wrong_length = descriptor;
  wrong_length.expected_kv_length = query_count - 1U;
  if (!expect_status(sllm_causal_attention_execute(context, queue,
                                                   &wrong_length, &completion,
                                                   &info, &error.sink),
                     SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH,
                     "causal wrong length", error) ||
      completion != nullptr ||
      fake_hip::causal_attention_launch_calls() != 0U) {
    return false;
  }
  sllm_causal_attention_desc_t alias = descriptor;
  alias.output = alias.query;
  info = causal_attention_dispatch_info();
  if (!expect_status(sllm_causal_attention_execute(context, queue, &alias,
                                                   &completion, &info,
                                                   &error.sink),
                     SLLM_STATUS_ALIAS_OVERLAP, "causal alias", error) ||
      completion != nullptr) {
    return false;
  }

  info = causal_attention_dispatch_info();
  if (!expect_status(sllm_causal_attention_execute(context, queue, &descriptor,
                                                   &completion, &info,
                                                   &error.sink),
                     SLLM_STATUS_OK, "causal execute", error) ||
      completion == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_STABLE_SOFTMAX_V1 ||
      info.workgroup_size_x != 256U || info.grid_size_x != query_count * 16U ||
      info.query_count != query_count || info.start_position != 0U ||
      info.committed_kv_length != query_count || info.q_heads != 16U ||
      info.kv_heads != 4U || info.head_dim != 256U ||
      info.scale_denominator != 16U || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol,
                  "causal_attention.stable_softmax_gqa.v1") != 0 ||
      std::strcmp(info.device_symbol,
                  "sllm_causal_attention_stable_softmax_gqa_v1") != 0 ||
      std::strcmp(info.gcn_arch_name, "gfx1201") != 0 ||
      fake_hip::causal_attention_launch_calls() != 1U) {
    return false;
  }

  fake_hip::set_completion_pending(true);
  sllm_completion_t *blocked_append_completion = nullptr;
  sllm_kv_append_info_t blocked_append_info = kv_append_info();
  if (!expect_status(sllm_kv_state_append(state, queue, &append,
                                          &blocked_append_completion,
                                          &blocked_append_info, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "causal append while active",
                     error) ||
      blocked_append_completion != nullptr ||
      !expect_status(sllm_kv_state_release(&state, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "causal state while active",
                     error) ||
      state == nullptr) {
    return false;
  }
  fake_hip::set_completion_pending(false);
  if (!query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_kv_state_release(&state, &error.sink), SLLM_STATUS_OK,
                     "causal state release", error)) {
    return false;
  }

  sllm_completion_t *readback = nullptr;
  if (!submit_d2h(queue, output, query_elements * sizeof(uint16_t),
                  &readback) ||
      readback == nullptr || !query_completion(readback, SLLM_STATUS_OK)) {
    return false;
  }
  std::vector<uint16_t> expected(query_elements, UINT16_C(0));
  for (uint64_t row = 0U; row != query_count; ++row) {
    for (uint64_t head = 0U; head != 16U; ++head) {
      float sum = 0.0F;
      for (uint64_t token = 0U; token <= row; ++token) {
        sum += static_cast<float>(token + head / 4U + 1U);
      }
      const uint16_t word =
          causal_float_to_bf16_rne(sum / static_cast<float>(row + 1U));
      for (uint64_t dimension = 0U; dimension != 256U; ++dimension) {
        expected[(row * 16U + head) * 256U + dimension] = word;
      }
    }
  }
  std::vector<uint16_t> actual(query_elements, UINT16_C(0));
  const bool output_matches =
      read_completion(readback, actual.data(), actual.size() * sizeof(uint16_t),
                      reinterpret_cast<const uint8_t *>(expected.data()),
                      expected.size() * sizeof(uint16_t));
  const bool readback_released = release_completion(&readback);
  const bool buffers_released =
      release_buffer(&key) && release_buffer(&value) &&
      release_buffer(&query) && release_buffer(&output);
  return output_matches && readback_released && buffers_released &&
         release_queue(&queue) && release_context(&context);
}

bool kv_append_accounting_multiplicity_contract() {
  using sllm_public_runtime::AccountingState;

  const auto reservation_must_fail_without_mutation =
      [](const bool active_exhausted, const bool completion_exhausted) {
        AccountingState context{};
        AccountingState queue{};
        AccountingState state{};
        AccountingState shared_input{};
        AccountingState key_buffer{};
        AccountingState value_buffer{};
        if (active_exhausted) {
          shared_input.active_submissions = UINT64_MAX - 1U;
        }
        if (completion_exhausted) {
          shared_input.completion_references = UINT64_MAX - 1U;
        }
        const bool reserved = AccountingState::reserve_kv_append(
            context, queue, state, shared_input, shared_input, key_buffer,
            value_buffer);
        return !reserved &&
               shared_input.active_submissions ==
                   (active_exhausted ? UINT64_MAX - 1U : 0U) &&
               shared_input.completion_references ==
                   (completion_exhausted ? UINT64_MAX - 1U : 0U) &&
               queue.active_submissions == 0U &&
               queue.completion_references == 0U && context.child_count == 0U &&
               context.lifetime_guards == 0U;
      };
  if (!reservation_must_fail_without_mutation(true, false) ||
      !reservation_must_fail_without_mutation(false, true) ||
      !reservation_must_fail_without_mutation(true, true)) {
    std::cerr
        << "KV duplicate input reservation did not fail closed at max-1\n";
    return false;
  }

  AccountingState context{};
  AccountingState queue{};
  AccountingState state{};
  AccountingState shared_input{};
  AccountingState key_buffer{};
  AccountingState value_buffer{};
  const bool reserved = AccountingState::reserve_kv_append(
      context, queue, state, shared_input, shared_input, key_buffer,
      value_buffer);
  const bool active_released = AccountingState::release_kv_active(
      queue, state, shared_input, shared_input, key_buffer, value_buffer);
  const bool completion_released = AccountingState::release_kv_completion(
      context, queue, state, shared_input, shared_input, key_buffer,
      value_buffer);
  if (!reserved || shared_input.active_submissions != 0U ||
      shared_input.completion_references != 0U ||
      queue.active_submissions != 0U || queue.completion_references != 0U ||
      context.child_count != 0U || context.lifetime_guards != 0U ||
      !active_released || !completion_released) {
    std::cerr << "KV duplicate input active/completion release was asymmetric: "
              << reserved << ", " << active_released << ", "
              << completion_released << "; shared active/completion="
              << shared_input.active_submissions << "/"
              << shared_input.completion_references
              << "; queue active/"
                 "completion="
              << queue.active_submissions << "/" << queue.completion_references
              << "; context child/guard=" << context.child_count << "/"
              << context.lifetime_guards << "\n";
    return false;
  }
  if (!AccountingState::reserve_kv_append(context, queue, state, shared_input,
                                          shared_input, key_buffer,
                                          value_buffer) ||
      !AccountingState::rollback_kv_append(context, queue, state, shared_input,
                                           shared_input, key_buffer,
                                           value_buffer) ||
      shared_input.active_submissions != 0U ||
      shared_input.completion_references != 0U ||
      queue.active_submissions != 0U || queue.completion_references != 0U ||
      context.child_count != 0U || context.lifetime_guards != 0U) {
    std::cerr << "KV duplicate input rollback was asymmetric\n";
    return false;
  }
  return true;
}

bool kv_append_same_buffer_disjoint_lifecycle_contract() {
  fake_hip::reset();
  constexpr uint64_t capacity = 2U;
  constexpr uint64_t elements_per_input = 4U * 256U;
  constexpr uint64_t input_bytes = elements_per_input * sizeof(uint16_t);
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *shared = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, input_bytes * 2U, &shared)) {
    if (shared != nullptr) {
      (void)release_buffer(&shared);
    }
    if (state != nullptr) {
      Error error;
      (void)sllm_kv_state_release(&state, &error.sink);
    }
    if (queue != nullptr) {
      (void)release_queue(&queue);
    }
    if (context != nullptr) {
      (void)release_context(&context);
    }
    return false;
  }

  std::vector<uint16_t> words(elements_per_input * 2U, 0x3f80U);
  for (uint64_t index = elements_per_input; index != words.size(); ++index) {
    words[static_cast<std::size_t>(index)] = 0x4000U;
  }
  Error error;
  bool valid = upload_kv_words(queue, shared, words);
  sllm_completion_t *completion = nullptr;
  auto descriptor = kv_append_descriptor(shared, shared, 1U, 0U);
  descriptor.value_input.byte_offset = input_bytes;
  sllm_kv_append_info_t info = kv_append_info();

  fake_hip::set_completion_pending(true);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV same-buffer append", error) &&
          completion != nullptr &&
          expect_status(sllm_buffer_release(&shared, &error.sink),
                        SLLM_STATUS_PUBLIC_BUSY,
                        "KV same-buffer pending buffer release", error) &&
          shared != nullptr && kv_query(state, 0U, 0U);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 1U, 1U);

  fake_hip::set_kv_state_append_launch_status(hipErrorUnknown);
  descriptor.expected_length = 1U;
  descriptor.start_position = 1U;
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                        "KV same-buffer launch rollback", error) &&
          completion == nullptr && kv_query(state, 1U, 1U);
  fake_hip::set_kv_state_append_launch_status(hipSuccess);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV same-buffer reuse", error) &&
          completion != nullptr &&
          query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 2U, 2U);

  const bool shared_released = release_buffer(&shared);
  const bool state_released =
      expect_status(sllm_kv_state_release(&state, &error.sink), SLLM_STATUS_OK,
                    "KV same-buffer state release", error);
  const bool queue_released = release_queue(&queue);
  const bool context_released = release_context(&context);
  return valid && shared_released && state_released && queue_released &&
         context_released;
}

bool kv_state_create_snapshot_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue)) {
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  sllm_kv_state_create_info_t invalid{};
  invalid.struct_size = sizeof(invalid);
  invalid.abi_version = SLLM_HIP_ABI_VERSION;
  invalid.session_id = 1U;
  invalid.capacity_tokens = 17U;
  invalid.reserved[0] = 1U;
  Error error;
  bool valid = expect_status(
                   sllm_kv_state_create(context, &invalid, &state, &error.sink),
                   SLLM_STATUS_RESERVED_NONZERO, "KV reserved create", error) &&
               state == nullptr;
  invalid.reserved[0] = 0U;
  invalid.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  valid = valid &&
          expect_status(
              sllm_kv_state_create(context, &invalid, &state, &error.sink),
              SLLM_STATUS_INVALID_ABI_VERSION, "KV old ABI create", error) &&
          state == nullptr;
  valid = valid && create_kv_state(context, 257U, &state) &&
          kv_query(state, 0U, 0U);
  sllm_kv_view_t *view = nullptr;
  sllm_kv_view_info_t view_info{};
  view_info.struct_size = sizeof(view_info);
  view_info.abi_version = SLLM_HIP_ABI_VERSION;
  view_info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  valid = valid &&
          expect_status(sllm_kv_state_snapshot(state, &view, &error.sink),
                        SLLM_STATUS_OK, "KV state snapshot", error) &&
          view != nullptr && ([&]() {
            view_info.struct_size -= 1U;
            const bool result = expect_status(
                sllm_kv_view_query(view, &view_info, &error.sink),
                SLLM_STATUS_INVALID_ARGUMENT, "KV view wrong size", error);
            view_info.struct_size = sizeof(view_info);
            return result;
          })() &&
          expect_status(sllm_kv_view_query(view, &view_info, &error.sink),
                        SLLM_STATUS_OK, "KV view query", error) &&
          view_info.observed_length == 0U && view_info.generation == 0U &&
          view_info.capacity_tokens == 257U && view_info.state_identity != 0U &&
          view_info.context_identity != 0U &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_PUBLIC_BUSY, "KV state live view Busy",
                        error) &&
          expect_status(sllm_kv_view_release(&view, &error.sink),
                        SLLM_STATUS_OK, "KV view release", error) &&
          view == nullptr &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_OK, "KV state release", error);
  return valid && release_queue(&queue) && release_context(&context);
}

bool kv_evidence_readback_contract() {
  fake_hip::reset();
  constexpr uint64_t capacity = 3U;
  constexpr std::size_t input_words = 4U * 256U;
  constexpr uint64_t input_bytes = input_words * sizeof(uint16_t);
  constexpr uint64_t head_bytes = capacity * 256U * sizeof(uint16_t);
  constexpr uint64_t plane_bytes = capacity * 4U * 256U * sizeof(uint16_t);
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value)) {
    release_buffer(&key);
    release_buffer(&value);
    if (state != nullptr) {
      Error error;
      (void)sllm_kv_state_release(&state, &error.sink);
    }
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<uint16_t> key_words(input_words, 0x3f80U);
  std::vector<uint16_t> value_words(input_words, 0x4000U);
  Error error;
  bool valid = upload_kv_words(queue, key, key_words) &&
               upload_kv_words(queue, value, value_words);
  auto make_request = [](const sllm_kv_view_t *const view, const uint32_t plane,
                         const uint64_t offset, const uint64_t length,
                         const uint64_t capacity_bytes, uint8_t *const output) {
    sllm_hip_kv_readback_request_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_KV_EVIDENCE_ABI_VERSION;
    result.view = view;
    result.plane = plane;
    result.byte_offset = offset;
    result.byte_length = length;
    result.host_capacity = capacity_bytes;
    result.host_output = output;
    return result;
  };

  sllm_kv_view_t *empty_view = nullptr;
  valid = valid &&
          expect_status(sllm_kv_state_snapshot(state, &empty_view, &error.sink),
                        SLLM_STATUS_OK, "KV evidence empty snapshot", error);
  std::vector<uint8_t> output(16U, 0xa5U);
  auto empty_request =
      make_request(empty_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, 0U, output.size(),
                   output.size(), output.data());
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&empty_request, &error.sink),
                        SLLM_STATUS_OK, "KV evidence empty readback", error);
  auto wrong_kind = empty_request;
  wrong_kind.view = reinterpret_cast<const sllm_kv_view_t *>(state);
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&wrong_kind, &error.sink),
                        SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                        "KV evidence wrong-kind handle", error);
  auto undersized = empty_request;
  undersized.host_capacity = 1U;
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&undersized, &error.sink),
                        SLLM_STATUS_BUFFER_TOO_SMALL,
                        "KV evidence undersized host output", error);
  auto null_output = empty_request;
  null_output.host_output = nullptr;
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&null_output, &error.sink),
                        SLLM_STATUS_INVALID_ARGUMENT,
                        "KV evidence null host output", error);
  auto reserved = empty_request;
  reserved.reserved[0] = 1U;
  valid =
      valid && expect_status(sllm_hip_kv_view_readback(&reserved, &error.sink),
                             SLLM_STATUS_RESERVED_NONZERO,
                             "KV evidence reserved field", error);
  auto out_of_bounds = empty_request;
  out_of_bounds.byte_offset = plane_bytes;
  valid =
      valid &&
      expect_status(sllm_hip_kv_view_readback(&out_of_bounds, &error.sink),
                    SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
                    "KV evidence plane bounds", error) &&
      expect_status(sllm_kv_view_release(&empty_view, &error.sink),
                    SLLM_STATUS_OK, "KV evidence empty view release", error);

  sllm_kv_append_desc_t descriptor = kv_append_descriptor(key, value, 1U, 0U);
  sllm_kv_append_info_t append_info = kv_append_info();
  sllm_completion_t *completion = nullptr;
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &append_info, &error.sink),
                    SLLM_STATUS_OK, "KV evidence append", error) &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion);
  sllm_kv_view_t *live_view = nullptr;
  valid = valid &&
          expect_status(sllm_kv_state_snapshot(state, &live_view, &error.sink),
                        SLLM_STATUS_OK, "KV evidence live snapshot", error);
  for (uint64_t head = 0U; head != 4U; ++head) {
    const uint64_t head_offset = head * head_bytes;
    std::fill(output.begin(), output.end(), 0xa5U);
    auto key_request =
        make_request(live_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, head_offset,
                     output.size(), output.size(), output.data());
    valid = valid &&
            expect_status(sllm_hip_kv_view_readback(&key_request, &error.sink),
                          SLLM_STATUS_OK, "KV evidence K readback", error);
    for (std::size_t index = 0U; index != output.size(); index += 2U) {
      valid = valid && output[index] == 0x00U && output[index + 1U] == 0x3cU;
    }

    std::fill(output.begin(), output.end(), 0xa5U);
    auto value_request =
        make_request(live_view, SLLM_HIP_KV_EVIDENCE_PLANE_V, head_offset,
                     output.size(), output.size(), output.data());
    valid = valid && expect_status(
                         sllm_hip_kv_view_readback(&value_request, &error.sink),
                         SLLM_STATUS_OK, "KV evidence V readback", error);
    for (std::size_t index = 0U; index != output.size(); index += 2U) {
      valid = valid && output[index] == 0x00U && output[index + 1U] == 0x40U;
    }
  }
  const sllm_kv_view_t *const stale_view = live_view;
  valid = valid &&
          expect_status(sllm_kv_view_release(&live_view, &error.sink),
                        SLLM_STATUS_OK, "KV evidence live view release", error);
  auto stale_request =
      make_request(stale_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, 0U, output.size(),
                   output.size(), output.data());
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&stale_request, &error.sink),
                        SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                        "KV evidence stale handle", error);

  fake_hip::set_completion_pending(true);
  descriptor = kv_append_descriptor(key, value, 1U, 1U);
  append_info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &append_info, &error.sink),
                    SLLM_STATUS_OK, "KV evidence pending append", error);
  sllm_kv_view_t *pending_view = nullptr;
  valid =
      valid &&
      expect_status(sllm_kv_state_snapshot(state, &pending_view, &error.sink),
                    SLLM_STATUS_OK, "KV evidence pending snapshot", error);
  auto pending_request =
      make_request(pending_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, 0U,
                   output.size(), output.size(), output.data());
  valid =
      valid &&
      expect_status(sllm_hip_kv_view_readback(&pending_request, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "KV evidence pending readback",
                    error) &&
      expect_status(sllm_kv_view_release(&pending_view, &error.sink),
                    SLLM_STATUS_OK, "KV evidence pending view release", error);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && release_buffer(&key) &&
          release_buffer(&value) &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_OK, "KV evidence state release", error) &&
          release_queue(&queue) && release_context(&context);
  return valid;
}

bool kv_append_layout_and_transaction_contract() {
  fake_hip::reset();
  constexpr uint64_t capacity = 257U;
  constexpr uint64_t max_tokens = 255U;
  const std::size_t input_bytes =
      static_cast<std::size_t>(max_tokens * 4U * 256U * sizeof(uint16_t));
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value)) {
    release_buffer(&key);
    release_buffer(&value);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  const uint16_t bf16_values[4] = {0x3f80U, 0x4000U, 0x4040U, 0x4080U};
  const uint16_t f16_values[4] = {0x3c00U, 0x4000U, 0x4200U, 0x4400U};
  auto make_words = [&](const uint64_t tokens, const uint32_t shift) {
    std::vector<uint16_t> words(tokens * 4U * 256U);
    for (std::size_t index = 0U; index != words.size(); ++index) {
      words[index] = bf16_values[(index + shift) % 4U];
    }
    return words;
  };
  Error error;
  bool valid = true;
  std::vector<uint16_t> first = make_words(1U, 0U);
  const uint16_t special_bf16[] = {0x8000U, 0x7f80U, 0xff80U,
                                   0x7fc1U, 0xffc1U, 0x8001U};
  const uint16_t special_f16[] = {0x8000U, 0x7c00U, 0xfc00U,
                                  0x7e00U, 0xfe00U, 0x8000U};
  constexpr std::size_t special_count =
      sizeof(special_bf16) / sizeof(special_bf16[0]);
  for (std::size_t index = 0U; index != special_count; ++index) {
    first[index] = special_bf16[index];
  }
  std::vector<uint16_t> three_key = make_words(3U, 0U);
  std::vector<uint16_t> three_value = make_words(3U, 1U);
  valid = valid && upload_kv_words(queue, key, first) &&
          upload_kv_words(queue, value, first);
  sllm_completion_t *completion = nullptr;
  sllm_kv_append_info_t info = kv_append_info();
  auto descriptor = kv_append_descriptor(key, value, 1U, 0U);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV first append", error) &&
          completion != nullptr && info.token_count == 1U &&
          info.end_position == 1U && info.commit_allowed == 1U &&
          info.fallback_allowed == 0U && info.fallback_used == 0U &&
          info.grid_size_x == 4U &&
          std::strcmp(info.kernel_symbol,
                      "kv_state.bf16_to_f16_transpose.v1") == 0 &&
          query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 1U, 1U);
  std::vector<uint16_t> first_key_output(4U * capacity * 256U);
  std::vector<uint16_t> first_value_output(4U * capacity * 256U);
  valid = valid &&
          fake_hip::copy_kv_key_output(first_key_output.data(),
                                       first_key_output.size()) &&
          fake_hip::copy_kv_value_output(first_value_output.data(),
                                         first_value_output.size());
  for (std::size_t index = 0U; index != special_count; ++index) {
    const std::size_t offset =
        (index / 256U) * static_cast<std::size_t>(capacity) * 256U +
        (index % 256U);
    if (first_key_output[offset] != special_f16[index] ||
        first_value_output[offset] != special_f16[index]) {
      valid = false;
    }
  }
  valid = valid && upload_kv_words(queue, key, three_key) &&
          upload_kv_words(queue, value, three_value);
  descriptor = kv_append_descriptor(key, value, 3U, 1U);
  info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_OK, "KV non-aligned M/start append", error) &&
      completion != nullptr && info.grid_size_x == 12U &&
      fake_hip::kv_state_last_token_count() == 3U &&
      fake_hip::kv_state_last_capacity_tokens() == capacity &&
      fake_hip::kv_state_last_start_position() == 1U &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion);
  std::vector<uint16_t> key_output(4U * capacity * 256U);
  std::vector<uint16_t> value_output(4U * capacity * 256U);
  valid =
      valid &&
      fake_hip::copy_kv_key_output(key_output.data(), key_output.size()) &&
      fake_hip::copy_kv_value_output(value_output.data(), value_output.size());
  for (uint64_t row = 0U; row != 3U && valid; ++row) {
    for (uint64_t head = 0U; head != 4U && valid; ++head) {
      for (uint64_t dimension = 0U; dimension != 256U; ++dimension) {
        const uint64_t source = row * 1024U + head * 256U + dimension;
        const uint64_t destination =
            head * capacity * 256U + (1U + row) * 256U + dimension;
        if (key_output[destination] != f16_values[source % 4U] ||
            value_output[destination] != f16_values[(source + 1U) % 4U]) {
          valid = false;
        }
      }
    }
  }
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid =
      valid && ([&]() {
        sllm_kv_append_desc_t wrong_size = descriptor;
        wrong_size.struct_size -= 1U;
        sllm_completion_t *wrong_completion = nullptr;
        sllm_kv_append_info_t wrong_info = kv_append_info();
        const bool size_result = expect_status(
            sllm_kv_state_append(state, queue, &wrong_size, &wrong_completion,
                                 &wrong_info, &error.sink),
            SLLM_STATUS_INVALID_ARGUMENT, "KV append wrong size", error);
        wrong_size = descriptor;
        wrong_size.append_version += 1U;
        const bool version_result = expect_status(
            sllm_kv_state_append(state, queue, &wrong_size, &wrong_completion,
                                 &wrong_info, &error.sink),
            SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR, "KV append wrong version",
            error);
        wrong_info = kv_append_info();
        wrong_info.struct_size -= 1U;
        const bool info_size_result = expect_status(
            sllm_kv_state_append(state, queue, &descriptor, &wrong_completion,
                                 &wrong_info, &error.sink),
            SLLM_STATUS_INVALID_ARGUMENT, "KV append info wrong size", error);
        return size_result && version_result && info_size_result &&
               wrong_completion == nullptr;
      })() &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_KV_LENGTH_MISMATCH, "KV stale length", error) &&
      completion == nullptr && kv_query(state, 4U, 2U);

  std::vector<uint16_t> boundary_key = make_words(255U, 0U);
  std::vector<uint16_t> boundary_value = make_words(255U, 1U);
  sllm_context_t *boundary_context = nullptr;
  sllm_queue_t *boundary_queue = nullptr;
  sllm_kv_state_t *boundary_state = nullptr;
  sllm_buffer_t *boundary_key_buffer = nullptr;
  sllm_buffer_t *boundary_value_buffer = nullptr;
  valid =
      valid && create_context(&boundary_context) &&
      create_queue(boundary_context, &boundary_queue) &&
      create_kv_state(boundary_context, capacity, &boundary_state) &&
      create_buffer_sized(boundary_context, input_bytes,
                          &boundary_key_buffer) &&
      create_buffer_sized(boundary_context, input_bytes,
                          &boundary_value_buffer) &&
      upload_kv_words(boundary_queue, boundary_key_buffer, boundary_key) &&
      upload_kv_words(boundary_queue, boundary_value_buffer, boundary_value);
  auto boundary_append = [&](const uint64_t tokens, const uint64_t position,
                             const sllm_status_t expected) {
    sllm_kv_append_desc_t boundary_descriptor = kv_append_descriptor(
        boundary_key_buffer, boundary_value_buffer, tokens, position);
    sllm_kv_append_info_t boundary_info = kv_append_info();
    sllm_completion_t *boundary_completion = nullptr;
    const bool result =
        expect_status(sllm_kv_state_append(
                          boundary_state, boundary_queue, &boundary_descriptor,
                          &boundary_completion, &boundary_info, &error.sink),
                      expected, "KV boundary append", error) &&
        (expected != SLLM_STATUS_OK || boundary_completion != nullptr);
    if (expected == SLLM_STATUS_OK) {
      return result && query_completion(boundary_completion, SLLM_STATUS_OK) &&
             release_completion(&boundary_completion);
    }
    return result && boundary_completion == nullptr;
  };
  valid = valid && boundary_append(255U, 0U, SLLM_STATUS_OK) &&
          boundary_append(1U, 255U, SLLM_STATUS_OK) &&
          boundary_append(1U, 256U, SLLM_STATUS_OK) &&
          boundary_append(1U, 257U, SLLM_STATUS_KV_CAPACITY_EXCEEDED) &&
          kv_query(boundary_state, 257U, 3U);
  valid = valid && release_buffer(&boundary_key_buffer) &&
          release_buffer(&boundary_value_buffer) &&
          expect_status(sllm_kv_state_release(&boundary_state, &error.sink),
                        SLLM_STATUS_OK, "KV boundary state release", error) &&
          release_queue(&boundary_queue) && release_context(&boundary_context);
  valid = valid && release_buffer(&key) && release_buffer(&value) &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_OK, "KV layout state release", error) &&
          release_queue(&queue) && release_context(&context);
  return valid;
}

bool kv_append_lifetime_alias_and_quarantine_contract() {
  fake_hip::reset();
  constexpr uint64_t input_bytes = 17U * 4U * 256U * sizeof(uint16_t);
  sllm_context_t *context = nullptr;
  sllm_context_t *other_context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_queue_t *other_queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  if (!create_context(&context) || !create_context(&other_context) ||
      !create_queue(context, &queue) ||
      !create_queue(other_context, &other_queue) ||
      !create_kv_state(context, 17U, &state) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value)) {
    if (key != nullptr) {
      (void)release_buffer(&key);
    }
    if (value != nullptr) {
      (void)release_buffer(&value);
    }
    if (state != nullptr) {
      Error error;
      (void)sllm_kv_state_release(&state, &error.sink);
    }
    if (queue != nullptr) {
      (void)release_queue(&queue);
    }
    if (other_queue != nullptr) {
      (void)release_queue(&other_queue);
    }
    if (context != nullptr) {
      (void)release_context(&context);
    }
    if (other_context != nullptr) {
      (void)release_context(&other_context);
    }
    return false;
  }
  std::vector<uint16_t> words(4U * 256U, 0x3f80U);
  bool valid = upload_kv_words(queue, key, words) &&
               upload_kv_words(queue, value, words);
  Error error;
  sllm_kv_append_info_t info = kv_append_info();
  sllm_completion_t *completion = nullptr;
  auto descriptor = kv_append_descriptor(key, value, 1U, 0U);
  fake_hip::set_completion_pending(true);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV pending append", error) &&
          completion != nullptr && ([&]() {
            sllm_completion_t *second_completion = nullptr;
            return expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                                      &second_completion, &info,
                                                      &error.sink),
                                 SLLM_STATUS_PUBLIC_BUSY,
                                 "KV double append Busy", error) &&
                   second_completion == nullptr;
          })();
  sllm_completion_result_t pending_result{};
  pending_result.struct_size = sizeof(pending_result);
  pending_result.abi_version = SLLM_HIP_ABI_VERSION;
  valid =
      valid &&
      expect_status(
          sllm_completion_wait(completion, 0U, &pending_result, &error.sink),
          SLLM_STATUS_PUBLIC_TIMEOUT, "KV append timeout", error) &&
      expect_status(
          sllm_completion_query(completion, &pending_result, &error.sink),
          SLLM_STATUS_PUBLIC_PENDING, "KV pending query", error) &&
      expect_status(sllm_completion_release(&completion, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "KV pending release revokes",
                    error) &&
      kv_query(state, 0U, 0U) &&
      release_buffer(&key, SLLM_STATUS_PUBLIC_BUSY) &&
      release_buffer(&value, SLLM_STATUS_PUBLIC_BUSY) &&
      expect_status(sllm_kv_state_release(&state, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "KV pending state Busy", error);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 0U, 0U);

  fake_hip::set_completion_pending(true);
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_OK, "KV explicit cancel append", error) &&
      expect_status(sllm_kv_state_append_cancel(state, completion, &error.sink),
                    SLLM_STATUS_OK, "KV explicit append cancel", error) &&
      kv_query(state, 0U, 0U);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 0U, 0U);

  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, other_queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
                        "KV wrong context queue", error) &&
          completion == nullptr;
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  descriptor.value_input.buffer = key;
  info = kv_append_info();
  const std::size_t calls_before_alias =
      fake_hip::kv_state_append_launch_calls();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_ALIAS_OVERLAP, "KV alias rejection", error) &&
      completion == nullptr &&
      fake_hip::kv_state_append_launch_calls() == calls_before_alias;
  fake_hip::set_kv_state_append_launch_status(hipErrorUnknown);
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                        "KV launch failure rollback", error) &&
          completion == nullptr && kv_query(state, 0U, 0U);
  fake_hip::set_kv_state_append_launch_status(hipSuccess);
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_OK, "KV reuse after launch failure", error) &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) && kv_query(state, 1U, 1U);

  const std::size_t poison_before = sllm_test_poison_count();
  descriptor = kv_append_descriptor(key, value, 1U, 1U);
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV quarantine append", error) &&
          query_completion(completion, SLLM_STATUS_OK);
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  const sllm_status_t quarantine_status =
      sllm_completion_release(&completion, &error.sink);
  sllm_public_runtime::FaultInjector::reset();
  valid = valid &&
          expect_status(quarantine_status, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                        "KV event cleanup quarantine", error) &&
          completion == nullptr && sllm_test_poison_count() > poison_before;
  /* This context is intentionally poisoned by the injected ambiguous event
   * cleanup.  The remaining graph is owned by the process-lifetime quarantine.
   */
  const bool other_queue_released = release_queue(&other_queue);
  const bool other_context_released = release_context(&other_context);
  return valid && other_queue_released && other_context_released;
}

} // namespace

int main() {
  if (!rmsnorm_bf16_rne_bit_contract()) {
    return 1;
  }
  if (!elementwise_prepare_execute_and_negative_contract()) {
    std::cerr << "elementwise prepare/execute contract test failed\n";
    return 1;
  }
  if (!embedding_prepare_execute_and_token_range_contract()) {
    std::cerr << "embedding prepare/execute contract test failed\n";
    return 1;
  }
  if (!matmul_prepare_execute_and_negative_contract() ||
      !matmul_async_lifetime_and_cleanup()) {
    std::cerr << "matmul prepare/execute contract test failed\n";
    return 1;
  }
  if (!attention_preprocess_prepare_validation_and_old_abi() ||
      !attention_preprocess_position_payload_mismatch_is_pre_dispatch() ||
      !attention_preprocess_success_metadata_and_dispatch()) {
    std::cerr << "attention preprocess public ABI contract test failed\n";
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
  if (!kv_append_accounting_multiplicity_contract() ||
      !causal_attention_numerical_gqa_and_lifetime_contract() ||
      !kv_append_same_buffer_disjoint_lifecycle_contract() ||
      !kv_state_create_snapshot_contract() ||
      !kv_evidence_readback_contract() ||
      !kv_append_layout_and_transaction_contract() ||
      !kv_append_lifetime_alias_and_quarantine_contract()) {
    std::cerr << "KV state public contract test failed\n";
    return 1;
  }
  std::cout << "production public runtime host fault test: PASS\n";
  return 0;
}
