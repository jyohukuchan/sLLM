#include "evidence_abi.h"

#include "ullm/hip.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#ifndef ULLM_HIP_COMPILE_TARGET
#error "ULLM_HIP_COMPILE_TARGET must be supplied by CMake"
#endif

namespace {

static_assert(sizeof(std::size_t) <= sizeof(uint64_t),
              "HIP evidence ABI sizes require size_t to fit in uint64_t");

constexpr uint32_t kBackendHip = ULLM_BACKEND_HIP;
constexpr char kCompileTarget[] = ULLM_HIP_COMPILE_TARGET;
constexpr std::size_t kMaxCleanupSlots = 2U;

constexpr bool exact_architecture_matches(std::string_view reported,
                                          std::string_view target) noexcept {
  return !reported.empty() && reported == target;
}

// Compile-time contract tests for the closed runtime check below. The G1
// binary is built for bare exact gfx1030/gfx1201 targets with the code-object
// feature tuple pinned separately; a feature-qualified runtime string must not
// be silently normalized into that contract.
static_assert(exact_architecture_matches("gfx1030", "gfx1030"));
static_assert(exact_architecture_matches("gfx1201", "gfx1201"));
static_assert(!exact_architecture_matches("gfx1030:xnack-", "gfx1030"));
static_assert(!exact_architecture_matches("gfx1201:sramecc+:xnack-",
                                          "gfx1201"));

struct Completion {
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;
  uint8_t *device_input = nullptr;
  uint8_t *device_output = nullptr;
  uint8_t *pinned_input = nullptr;
  uint8_t *pinned_output = nullptr;
  uint64_t input_size = 0U;
  uint64_t allocation_count = 0U;
  uint64_t copy_count = 0U;
  uint64_t dispatch_count = 0U;
  bool async_work_may_be_submitted = false;
  bool event_recorded = false;
  bool cleanup_slot_reserved = false;

  // The reaper owns this link after the completion is retired. It is part of
  // the already-allocated completion, so enqueue cannot allocate.
  Completion *reaper_next = nullptr;
};

std::mutex live_mutex;
std::unordered_map<uintptr_t, std::unique_ptr<Completion>> live_completions;
uintptr_t next_handle = 1U;

bool cleanup_resources(Completion *completion) noexcept {
  if (completion == nullptr) {
    return true;
  }

  // Never release an allocation until HIP has positively reported completion.
  // A timeout is deliberately not a synchronization operation. If both
  // synchronization attempts fail, the Completion and all its resources are
  // abandoned so that a permanently hung or unhealthy runtime cannot turn a
  // failed cleanup into a use-after-free.
  if (completion->async_work_may_be_submitted) {
    bool synchronization_proven = false;
    if (completion->event_recorded && completion->event != nullptr) {
      synchronization_proven =
          hipEventSynchronize(completion->event) == hipSuccess;
      if (!synchronization_proven && completion->stream != nullptr) {
        synchronization_proven =
            hipStreamSynchronize(completion->stream) == hipSuccess;
      }
    } else if (!completion->event_recorded && completion->stream != nullptr) {
      synchronization_proven =
          hipStreamSynchronize(completion->stream) == hipSuccess;
    }
    if (!synchronization_proven) {
      return false;
    }
  }

  bool cleanup_proven = true;
  if (completion->event != nullptr) {
    if (hipEventDestroy(completion->event) == hipSuccess) {
      completion->event = nullptr;
    } else {
      cleanup_proven = false;
    }
  }
  if (completion->device_output != nullptr) {
    if (hipFree(completion->device_output) == hipSuccess) {
      completion->device_output = nullptr;
    } else {
      cleanup_proven = false;
    }
  }
  if (completion->device_input != nullptr) {
    if (hipFree(completion->device_input) == hipSuccess) {
      completion->device_input = nullptr;
    } else {
      cleanup_proven = false;
    }
  }
  if (completion->stream != nullptr) {
    if (hipStreamDestroy(completion->stream) == hipSuccess) {
      completion->stream = nullptr;
    } else {
      cleanup_proven = false;
    }
  }
  if (completion->pinned_output != nullptr) {
    if (hipHostFree(completion->pinned_output) == hipSuccess) {
      completion->pinned_output = nullptr;
    } else {
      cleanup_proven = false;
    }
  }
  if (completion->pinned_input != nullptr) {
    if (hipHostFree(completion->pinned_input) == hipSuccess) {
      completion->pinned_input = nullptr;
    } else {
      cleanup_proven = false;
    }
  }
  return cleanup_proven;
}

class Reaper {
public:
  Reaper() = default;
  Reaper(const Reaper &) = delete;
  Reaper &operator=(const Reaper &) = delete;

  bool start() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_.joinable()) {
      return true;
    }
    try {
      worker_ = std::thread(&Reaper::run, this);
      return true;
    } catch (...) {
      return false;
    }
  }

  // A slot is reserved before any device allocation. It covers the live
  // Completion as well as its eventual cleanup, so a hung reaper cannot be
  // hidden behind an unbounded sequence of new submissions.
  bool try_reserve_cleanup_slot() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!worker_.joinable() || cleanup_blocked_ ||
        cleanup_slots_ >= kMaxCleanupSlots) {
      return false;
    }
    ++cleanup_slots_;
    return true;
  }

  void release_cleanup_slot() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cleanup_slots_ == 0U) {
      std::terminate();
    }
    --cleanup_slots_;
  }

  void mark_cleanup_unproven() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanup_blocked_ = true;
  }

  // Ownership of completion is transferred to this method. The queue is an
  // intrusive singly-linked list; no allocation or fallible operation occurs
  // after asynchronous HIP work has started.
  void enqueue(Completion *completion) noexcept {
    if (completion == nullptr) {
      std::terminate();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // start() is required before the first GPU allocation. The singleton is
    // never stopped, so reaching this branch is an internal invariant failure,
    // not a reason to synchronize HIP work on the caller thread.
    if (!worker_.joinable()) {
      std::terminate();
    }
    completion->reaper_next = nullptr;
    if (tail_ == nullptr) {
      head_ = completion;
    } else {
      tail_->reaper_next = completion;
    }
    tail_ = completion;
    condition_.notify_one();
  }

private:
  void run() noexcept {
    for (;;) {
      Completion *completion = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return head_ != nullptr; });
        completion = head_;
        head_ = completion->reaper_next;
        if (head_ == nullptr) {
          tail_ = nullptr;
        }
        completion->reaper_next = nullptr;
      }
      if (cleanup_resources(completion)) {
        const bool slot_reserved = completion->cleanup_slot_reserved;
        delete completion;
        if (!slot_reserved) {
          std::terminate();
        }
        release_cleanup_slot();
      } else {
        mark_cleanup_unproven();
        // Intentionally leak the Completion and HIP handles. The runtime did
        // not prove that they are no longer in use, so safety takes priority
        // over reclamation; process teardown reclaims the address space.
      }
    }
  }

  std::condition_variable condition_;
  std::mutex mutex_;
  Completion *head_ = nullptr;
  Completion *tail_ = nullptr;
  std::thread worker_;
  std::size_t cleanup_slots_ = 0U;
  bool cleanup_blocked_ = false;
};

Reaper &reaper() {
  // Deliberately leaked so process shutdown never invokes an unbounded join
  // while a driver call in the cleanup thread is wedged. The operating system
  // reclaims the worker and its HIP resources when the evidence process exits.
  static Reaper *const instance = new Reaper();
  return *instance;
}

uintptr_t
handle_key(const ullm_hip_evidence_completion_t *completion) noexcept {
  return reinterpret_cast<uintptr_t>(completion);
}

ullm_hip_evidence_completion_t *opaque_handle(uintptr_t key) noexcept {
  return reinterpret_cast<ullm_hip_evidence_completion_t *>(key);
}

uintptr_t register_completion(std::unique_ptr<Completion> &completion) {
  std::lock_guard<std::mutex> lock(live_mutex);
  if (next_handle == 0U) {
    throw std::overflow_error("HIP evidence completion handle space exhausted");
  }
  const uintptr_t key = next_handle;
  ++next_handle;
  auto [entry, inserted] = live_completions.try_emplace(key);
  if (!inserted) {
    throw std::runtime_error("HIP evidence completion handle collision");
  }
  entry->second = std::move(completion);
  return key;
}

std::unique_ptr<Completion> take_completion(uintptr_t key) noexcept {
  if (key == 0U) {
    return {};
  }
  std::lock_guard<std::mutex> lock(live_mutex);
  const auto entry = live_completions.find(key);
  if (entry == live_completions.end()) {
    return {};
  }
  std::unique_ptr<Completion> completion = std::move(entry->second);
  live_completions.erase(entry);
  return completion;
}

void retire_to_reaper(std::unique_ptr<Completion> completion) noexcept {
  if (completion != nullptr) {
    if (!completion->cleanup_slot_reserved) {
      std::terminate();
    }
    reaper().enqueue(completion.release());
  }
}

void cleanup_without_async_or_leak(
    std::unique_ptr<Completion> completion) noexcept {
  if (completion == nullptr) {
    return;
  }
  Completion *raw = completion.release();
  if (!cleanup_resources(raw)) {
    reaper().mark_cleanup_unproven();
    // Do not delete raw: cleanup was not proven, and its handles may still be
    // owned by the HIP runtime even on this non-asynchronous error path.
    return;
  }
  const bool slot_reserved = raw->cleanup_slot_reserved;
  delete raw;
  if (!slot_reserved) {
    std::terminate();
  }
  reaper().release_cleanup_slot();
}

void release_after_error(std::unique_ptr<Completion> completion) noexcept {
  if (completion == nullptr) {
    return;
  }
  if (completion->async_work_may_be_submitted) {
    retire_to_reaper(std::move(completion));
  } else {
    cleanup_without_async_or_leak(std::move(completion));
  }
}

void clear_error(ullm_error_sink_t *sink) noexcept {
  if (sink != nullptr && sink->message != nullptr &&
      sink->message_capacity != 0U) {
    sink->message[0] = '\0';
  }
}

uint32_t write_error(ullm_error_sink_t *sink, uint32_t status,
                     const char *message) noexcept {
  if (sink == nullptr) {
    return status;
  }
  const std::size_t length = message == nullptr ? 0U : std::strlen(message);
  sink->message_length = static_cast<uint64_t>(length);
  if (sink->message_capacity == 0U) {
    return ULLM_STATUS_BUFFER_TOO_SMALL;
  }
  if (sink->message == nullptr) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t capacity = static_cast<std::size_t>(sink->message_capacity);
  const std::size_t copied = length < capacity - 1U ? length : capacity - 1U;
  if (copied != 0U) {
    std::memcpy(sink->message, message, copied);
  }
  sink->message[copied] = '\0';
  return length <= capacity - 1U ? status : ULLM_STATUS_BUFFER_TOO_SMALL;
}

uint32_t validate_sink(ullm_error_sink_t *sink) noexcept {
  if (sink == nullptr) {
    return ULLM_STATUS_OK;
  }
  if (sink->struct_size < sizeof(*sink)) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->abi_version != ULLM_HIP_ABI_VERSION) {
    return ULLM_STATUS_INVALID_ABI_VERSION;
  }
  if (sink->reserved[0] != 0U || sink->reserved[1] != 0U) {
    return ULLM_STATUS_RESERVED_NONZERO;
  }
  if (sink->message_capacity >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->message_capacity != 0U && sink->message == nullptr) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  sink->message_length = 0U;
  clear_error(sink);
  return ULLM_STATUS_OK;
}

uint32_t validate_request(const ullm_hip_evidence_request_t *request,
                          ullm_error_sink_t *sink) noexcept {
  if (request == nullptr) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT,
                       "evidence request is null");
  }
  if (request->struct_size < sizeof(*request)) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT,
                       "evidence request struct_size is too small");
  }
  if (request->abi_version != ULLM_HIP_EVIDENCE_ABI_VERSION) {
    return write_error(sink, ULLM_STATUS_INVALID_ABI_VERSION,
                       "unsupported evidence ABI version");
  }
  for (uint32_t value : request->reserved) {
    if (value != 0U) {
      return write_error(sink, ULLM_STATUS_RESERVED_NONZERO,
                         "evidence request reserved field is non-zero");
    }
  }
  if (request->input == nullptr || request->input_size == 0U) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT,
                       "evidence input must be non-null and non-empty");
  }
  if (request->input_size >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT,
                       "evidence input is too large");
  }
  return ULLM_STATUS_OK;
}

uint32_t validate_result(const ullm_hip_evidence_result_t *result,
                         ullm_error_sink_t *sink) noexcept {
  if (result == nullptr) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT,
                       "evidence result is null");
  }
  if (result->struct_size < sizeof(*result)) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT,
                       "evidence result struct_size is too small");
  }
  if (result->abi_version != ULLM_HIP_EVIDENCE_ABI_VERSION) {
    return write_error(sink, ULLM_STATUS_INVALID_ABI_VERSION,
                       "unsupported evidence result ABI version");
  }
  for (uint32_t value : result->reserved) {
    if (value != 0U) {
      return write_error(sink, ULLM_STATUS_RESERVED_NONZERO,
                         "evidence result reserved field is non-zero");
    }
  }
  return ULLM_STATUS_OK;
}

uint32_t hip_failure(ullm_error_sink_t *sink, hipError_t error,
                     const char *operation) noexcept {
  const char *name = hipGetErrorString(error);
  char message[192] = {};
  std::snprintf(message, sizeof(message), "%s failed: %s", operation,
                name == nullptr ? "unknown HIP error" : name);
  return write_error(sink, ULLM_STATUS_HIP_RUNTIME_ERROR, message);
}

uint32_t validate_visible_device(ullm_error_sink_t *sink) {
  int device_count = 0;
  hipError_t status = hipGetDeviceCount(&device_count);
  if (status != hipSuccess) {
    return hip_failure(sink, status, "hipGetDeviceCount");
  }
  if (device_count != 1) {
    return write_error(sink, ULLM_STATUS_HIP_RUNTIME_ERROR,
                       "HIP evidence requires exactly one visible device");
  }

  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, 0);
  if (status != hipSuccess) {
    return hip_failure(sink, status, "hipGetDeviceProperties");
  }

  // On the canonical G1 devices HIP reports the bare raw values gfx1030 and
  // gfx1201. Some ROCm paths can expose feature-qualified values such as
  // gfx90a:sramecc+:xnack-. This build pins those feature states in the code
  // object separately and CMake accepts only the bare exact target, so a
  // suffix is a closed mismatch rather than something this check may strip.
  const std::string reported_architecture(properties.gcnArchName);
  if (!exact_architecture_matches(reported_architecture,
                                  std::string_view(kCompileTarget))) {
    char message[256] = {};
    std::snprintf(
        message, sizeof(message),
        "HIP gcnArchName '%s' does not exactly match compile target '%s'",
        reported_architecture.c_str(), kCompileTarget);
    return write_error(sink, ULLM_STATUS_HIP_RUNTIME_ERROR, message);
  }
  return ULLM_STATUS_OK;
}

__global__ void evidence_transform(const uint8_t *input, uint8_t *output,
                                   uint64_t size) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    output[index] =
        static_cast<uint8_t>(input[index] ^ ULLM_HIP_EVIDENCE_TRANSFORM_XOR);
  }
}

uint32_t wait_event(Completion *completion, uint32_t timeout_ms,
                    ullm_error_sink_t *sink) noexcept {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    // hipEventQuery is the only HIP operation on the caller-side success and
    // timeout path. It never synchronizes the stream or event.
    const hipError_t status = hipEventQuery(completion->event);
    if (status == hipSuccess) {
      return ULLM_STATUS_OK;
    }
    if (status != hipErrorNotReady) {
      return hip_failure(sink, status, "hipEventQuery");
    }

    const auto now = std::chrono::steady_clock::now();
    if (timeout_ms == 0U || now >= deadline) {
      return write_error(sink, ULLM_STATUS_HIP_TIMEOUT,
                         "evidence completion timed out");
    }
    auto remaining = deadline - now;
    const auto poll_interval = std::chrono::milliseconds(1);
    if (remaining > poll_interval) {
      remaining = poll_interval;
    }
    std::this_thread::sleep_for(remaining);
  }
}

} // namespace

extern "C" uint32_t
ullm_hip_evidence_submit(const ullm_hip_evidence_request_t *request,
                         ullm_hip_evidence_completion_t **completion,
                         ullm_error_sink_t *error_sink) {
  std::unique_ptr<Completion> value;
  try {
    const uint32_t sink_status = validate_sink(error_sink);
    if (sink_status != ULLM_STATUS_OK) {
      return sink_status;
    }
    const uint32_t request_status = validate_request(request, error_sink);
    if (request_status != ULLM_STATUS_OK) {
      return request_status;
    }
    if (completion == nullptr) {
      return write_error(error_sink, ULLM_STATUS_INVALID_ARGUMENT,
                         "completion output is null");
    }
    *completion = nullptr;

    const uint64_t block_count = request->input_size / 256U +
                                 (request->input_size % 256U == 0U ? 0U : 1U);
    if (block_count == 0U ||
        block_count >
            static_cast<uint64_t>(std::numeric_limits<unsigned int>::max())) {
      return write_error(error_sink, ULLM_STATUS_INVALID_ARGUMENT,
                         "evidence input requires too many HIP grid blocks");
    }

    // This must succeed before any HIP allocation. Reaper enqueue is
    // allocation-free after this point.
    if (!reaper().start()) {
      return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                         "cannot start the HIP evidence cleanup reaper");
    }

    value = std::make_unique<Completion>();
    if (!reaper().try_reserve_cleanup_slot()) {
      return write_error(
          error_sink, ULLM_STATUS_INTERNAL_ERROR,
          "HIP evidence cleanup circuit breaker is open or at capacity");
    }
    value->cleanup_slot_reserved = true;
    value->input_size = request->input_size;
    const std::size_t input_size = static_cast<std::size_t>(value->input_size);

    hipError_t status = hipSetDevice(0);
    if (status == hipSuccess) {
      const uint32_t device_status = validate_visible_device(error_sink);
      if (device_status != ULLM_STATUS_OK) {
        release_after_error(std::move(value));
        return device_status;
      }
    }
    if (status == hipSuccess) {
      status = hipStreamCreateWithFlags(&value->stream, hipStreamNonBlocking);
    }
    if (status == hipSuccess) {
      status = hipMalloc(reinterpret_cast<void **>(&value->device_input),
                         input_size);
      if (status == hipSuccess) {
        value->allocation_count = 1U;
      }
    }
    if (status == hipSuccess) {
      status = hipMalloc(reinterpret_cast<void **>(&value->device_output),
                         input_size);
      if (status == hipSuccess) {
        value->allocation_count = 2U;
      }
    }
    if (status == hipSuccess) {
      status = hipEventCreateWithFlags(&value->event, hipEventDisableTiming);
    }
    if (status == hipSuccess) {
      status = hipHostMalloc(reinterpret_cast<void **>(&value->pinned_input),
                             input_size, hipHostMallocDefault);
    }
    if (status == hipSuccess) {
      // The caller's memory is copied into HIP-owned page-locked storage before
      // either asynchronous host-device transfer is submitted.
      std::memcpy(value->pinned_input, request->input, input_size);
      status = hipHostMalloc(reinterpret_cast<void **>(&value->pinned_output),
                             input_size, hipHostMallocDefault);
    }
    if (status == hipSuccess) {
      // The flag is set before the call because a failed asynchronous API may
      // still have submitted work from the caller's point of view.
      value->async_work_may_be_submitted = true;
      status = hipMemcpyAsync(value->device_input, value->pinned_input,
                              input_size, hipMemcpyHostToDevice, value->stream);
      if (status == hipSuccess) {
        value->copy_count = 1U;
      }
    }
    if (status == hipSuccess) {
      value->async_work_may_be_submitted = true;
      const unsigned int blocks = static_cast<unsigned int>(block_count);
      hipLaunchKernelGGL(evidence_transform, dim3(blocks), dim3(256U), 0U,
                         value->stream, value->device_input,
                         value->device_output, value->input_size);
      status = hipGetLastError();
      if (status == hipSuccess) {
        value->dispatch_count = 1U;
      }
    }
    if (status == hipSuccess) {
      value->async_work_may_be_submitted = true;
      status = hipMemcpyAsync(value->pinned_output, value->device_output,
                              input_size, hipMemcpyDeviceToHost, value->stream);
      if (status == hipSuccess) {
        value->copy_count = 2U;
      }
    }
    if (status == hipSuccess) {
      value->async_work_may_be_submitted = true;
      status = hipEventRecord(value->event, value->stream);
      if (status == hipSuccess) {
        value->event_recorded = true;
      }
    }
    if (status != hipSuccess) {
      const uint32_t result =
          hip_failure(error_sink, status, "HIP evidence submit");
      release_after_error(std::move(value));
      return result;
    }

    const uintptr_t key = register_completion(value);
    *completion = opaque_handle(key);
    return ULLM_STATUS_OK;
  } catch (const std::bad_alloc &) {
    release_after_error(std::move(value));
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "allocation failed in evidence submit");
  } catch (...) {
    release_after_error(std::move(value));
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in evidence submit");
  }
}

extern "C" uint32_t ullm_hip_evidence_wait(
    ullm_hip_evidence_completion_t *opaque_completion, uint32_t timeout_ms,
    uint8_t *output, uint64_t output_capacity,
    ullm_hip_evidence_result_t *result, ullm_error_sink_t *error_sink) {
  std::unique_ptr<Completion> completion;
  try {
    const uint32_t sink_status = validate_sink(error_sink);
    if (sink_status != ULLM_STATUS_OK) {
      return sink_status;
    }
    completion = take_completion(handle_key(opaque_completion));
    if (completion == nullptr) {
      return write_error(error_sink, ULLM_STATUS_HIP_INVALID_HANDLE,
                         "evidence completion handle is not live");
    }

    // Taking the unique_ptr above consumes the handle. Every subsequent error
    // retires it, so a caller cannot retry or accidentally double-destroy it.
    const uint32_t result_status = validate_result(result, error_sink);
    if (result_status != ULLM_STATUS_OK) {
      release_after_error(std::move(completion));
      return result_status;
    }
    if (output == nullptr || output_capacity < completion->input_size) {
      const uint32_t result_code =
          write_error(error_sink, ULLM_STATUS_INVALID_ARGUMENT,
                      "evidence output is null or too small");
      release_after_error(std::move(completion));
      return result_code;
    }

    const uint32_t wait_status =
        wait_event(completion.get(), timeout_ms, error_sink);
    if (wait_status != ULLM_STATUS_OK) {
      release_after_error(std::move(completion));
      return wait_status;
    }
    if (completion->allocation_count != 2U || completion->copy_count != 2U) {
      const uint32_t result_code =
          write_error(error_sink, ULLM_STATUS_HIP_DISPATCH_CONTRACT,
                      "HIP evidence requires exactly two allocations, two "
                      "copies, and one dispatch");
      release_after_error(std::move(completion));
      return result_code;
    }
    if (completion->dispatch_count == 0U) {
      const uint32_t result_code =
          write_error(error_sink, ULLM_STATUS_HIP_ZERO_DISPATCH,
                      "HIP evidence completed with zero kernel dispatches");
      release_after_error(std::move(completion));
      return result_code;
    }
    if (completion->dispatch_count != 1U) {
      const uint32_t result_code =
          write_error(error_sink, ULLM_STATUS_HIP_DISPATCH_CONTRACT,
                      "HIP evidence requires exactly two allocations, two "
                      "copies, and one dispatch");
      release_after_error(std::move(completion));
      return result_code;
    }

    // D2H was enqueued by submit. A successful wait only polls the event and
    // copies the completed owned host storage; it issues no synchronous HIP
    // work and no second device copy.
    std::memcpy(output, completion->pinned_output,
                static_cast<std::size_t>(completion->input_size));
    result->output_size = completion->input_size;
    result->allocation_count = completion->allocation_count;
    result->copy_count = completion->copy_count;
    result->dispatch_count = completion->dispatch_count;
    result->selected_backend = kBackendHip;
    result->fallback_used = 0U;
    result->terminal = 1U;
    retire_to_reaper(std::move(completion));
    return ULLM_STATUS_OK;
  } catch (...) {
    release_after_error(std::move(completion));
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in evidence wait");
  }
}

extern "C" uint32_t
ullm_hip_evidence_destroy(ullm_hip_evidence_completion_t **opaque_completion,
                          ullm_error_sink_t *error_sink) {
  std::unique_ptr<Completion> completion;
  try {
    const uint32_t sink_status = validate_sink(error_sink);
    if (sink_status != ULLM_STATUS_OK) {
      return sink_status;
    }
    if (opaque_completion == nullptr) {
      return write_error(error_sink, ULLM_STATUS_INVALID_ARGUMENT,
                         "completion pointer is null");
    }
    completion = take_completion(handle_key(*opaque_completion));
    if (completion == nullptr) {
      return write_error(error_sink, ULLM_STATUS_HIP_INVALID_HANDLE,
                         "evidence completion handle is not live");
    }
    *opaque_completion = nullptr;
    retire_to_reaper(std::move(completion));
    return ULLM_STATUS_OK;
  } catch (...) {
    release_after_error(std::move(completion));
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in evidence destroy");
  }
}
