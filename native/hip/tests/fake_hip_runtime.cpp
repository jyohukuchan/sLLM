#include <hip/hip_runtime.h>

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_set>

struct FakeHipStream {};

namespace {

struct State final {
  std::mutex mutex;
  std::condition_variable condition;
  bool event_create_gate = false;
  bool event_create_entered = false;
  bool event_query_gate = false;
  bool event_query_entered = false;
  hipError_t rmsnorm_launch_status = hipSuccess;
  hipError_t event_record_status = hipSuccess;
  std::size_t rmsnorm_launch_calls = 0U;
  uint32_t rmsnorm_last_normalized_size = 0U;
  uint32_t rmsnorm_last_row_count = 0U;
  std::size_t event_destroy_calls = 0U;
  std::size_t stream_destroy_calls = 0U;
  std::size_t allocation_free_calls = 0U;
  std::unordered_set<hipEvent_t> events;
  std::unordered_set<hipStream_t> streams;
  std::unordered_set<void *> allocations;
};

State state;

} // namespace

namespace fake_hip {

void reset() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_create_gate = false;
  state.event_create_entered = false;
  state.event_query_gate = false;
  state.event_query_entered = false;
  state.rmsnorm_launch_status = hipSuccess;
  state.event_record_status = hipSuccess;
  state.rmsnorm_launch_calls = 0U;
  state.rmsnorm_last_normalized_size = 0U;
  state.rmsnorm_last_row_count = 0U;
  state.event_destroy_calls = 0U;
  state.stream_destroy_calls = 0U;
  state.allocation_free_calls = 0U;
}

hipError_t rmsnorm_launch(const uint16_t *const /*activation*/,
                          const uint16_t *const /*raw_scale*/,
                          uint16_t *const /*output*/,
                          const uint32_t normalized_size,
                          const uint32_t row_count, const float /*epsilon*/,
                          const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.rmsnorm_launch_calls;
  state.rmsnorm_last_normalized_size = normalized_size;
  state.rmsnorm_last_row_count = row_count;
  return state.rmsnorm_launch_status;
}

void set_rmsnorm_launch_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.rmsnorm_launch_status = status;
}

std::size_t rmsnorm_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.rmsnorm_launch_calls;
}

uint32_t rmsnorm_last_normalized_size() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.rmsnorm_last_normalized_size;
}

uint32_t rmsnorm_last_row_count() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.rmsnorm_last_row_count;
}

void set_event_record_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_record_status = status;
}

void set_event_create_gate(const bool enabled) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_create_gate = enabled;
  state.event_create_entered = false;
  state.condition.notify_all();
}

void wait_event_create_entered() {
  std::unique_lock<std::mutex> lock(state.mutex);
  state.condition.wait(lock, [] { return state.event_create_entered; });
}

void release_event_create_gate() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_create_gate = false;
  state.condition.notify_all();
}

void set_event_query_gate(const bool enabled) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_query_gate = enabled;
  state.event_query_entered = false;
  state.condition.notify_all();
}

void wait_event_query_entered() {
  std::unique_lock<std::mutex> lock(state.mutex);
  state.condition.wait(lock, [] { return state.event_query_entered; });
}

void release_event_query_gate() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_query_gate = false;
  state.condition.notify_all();
}

std::size_t event_destroy_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.event_destroy_calls;
}

std::size_t stream_destroy_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.stream_destroy_calls;
}

std::size_t allocation_free_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.allocation_free_calls;
}

std::size_t live_events() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.events.size();
}

std::size_t live_streams() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.streams.size();
}

std::size_t live_allocations() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.allocations.size();
}

} // namespace fake_hip

const char *hipGetErrorString(const hipError_t error) noexcept {
  switch (error) {
  case hipSuccess:
    return "success";
  case hipErrorInvalidValue:
    return "invalid value";
  case hipErrorNotReady:
    return "not ready";
  case hipErrorUnknown:
    return "unknown";
  }
  return "unknown";
}

hipError_t hipGetDeviceCount(int *const count) noexcept {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = 1;
  return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *const properties,
                                  const unsigned int device) noexcept {
  if (properties == nullptr || device != 0U) {
    return hipErrorInvalidValue;
  }
  std::memset(properties, 0, sizeof(*properties));
  std::strncpy(properties->name, "fake-host-device",
               sizeof(properties->name) - 1U);
  std::strncpy(properties->gcnArchName, "gfx1201",
               sizeof(properties->gcnArchName) - 1U);
  properties->totalGlobalMem = static_cast<std::size_t>(16U) * 1024U * 1024U;
  properties->warpSize = 32;
  return hipSuccess;
}

hipError_t hipSetDevice(const int device) noexcept {
  return device == 0 ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipStreamCreateWithFlags(hipStream_t *const stream,
                                    const unsigned int /*flags*/) noexcept {
  if (stream == nullptr) {
    return hipErrorInvalidValue;
  }
  *stream = new (std::nothrow) FakeHipStream;
  if (*stream == nullptr) {
    return hipErrorUnknown;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.streams.insert(*stream);
  return hipSuccess;
}

hipError_t hipStreamDestroy(const hipStream_t stream) noexcept {
  if (stream == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.streams.erase(stream) == 0U) {
    return hipErrorInvalidValue;
  }
  ++state.stream_destroy_calls;
  delete stream;
  return hipSuccess;
}

hipError_t hipStreamSynchronize(const hipStream_t stream) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.streams.count(stream) == 0U ? hipErrorInvalidValue : hipSuccess;
}

hipError_t hipMalloc(void **const pointer, const std::size_t size) noexcept {
  if (pointer == nullptr || size == 0U) {
    return hipErrorInvalidValue;
  }
  /* Large public-buffer sizes are metadata-only in this fake runtime.  Keep
   * the allocation bounded so row-count overflow tests can exercise ABI
   * validation without touching or emulating tensor data. */
  const std::size_t allocation_size =
      size > (std::size_t{1} << 32U) ? std::size_t{1} : size;
  *pointer = std::malloc(allocation_size);
  if (*pointer == nullptr) {
    return hipErrorUnknown;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.allocations.insert(*pointer);
  return hipSuccess;
}

hipError_t hipFree(void *const pointer) noexcept {
  if (pointer == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.allocations.erase(pointer) == 0U) {
    return hipErrorInvalidValue;
  }
  ++state.allocation_free_calls;
  std::free(pointer);
  return hipSuccess;
}

hipError_t hipEventCreateWithFlags(hipEvent_t *const event,
                                   const unsigned int /*flags*/) noexcept {
  if (event == nullptr) {
    return hipErrorInvalidValue;
  }
  *event = new (std::nothrow) FakeHipEvent;
  if (*event == nullptr) {
    return hipErrorUnknown;
  }
  std::unique_lock<std::mutex> lock(state.mutex);
  state.events.insert(*event);
  if (state.event_create_gate) {
    state.event_create_entered = true;
    state.condition.notify_all();
    state.condition.wait(lock, [] { return !state.event_create_gate; });
  }
  return hipSuccess;
}

hipError_t hipEventDestroy(const hipEvent_t event) noexcept {
  if (event == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.events.erase(event) == 0U) {
    return hipErrorInvalidValue;
  }
  ++state.event_destroy_calls;
  delete event;
  return hipSuccess;
}

hipError_t hipEventRecord(const hipEvent_t event,
                          const hipStream_t stream) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.event_record_status != hipSuccess) {
    return state.event_record_status;
  }
  if (state.events.count(event) == 0U || state.streams.count(stream) == 0U) {
    return hipErrorInvalidValue;
  }
  event->recorded = true;
  return hipSuccess;
}

hipError_t hipEventQuery(const hipEvent_t event) noexcept {
  std::unique_lock<std::mutex> lock(state.mutex);
  if (state.events.count(event) == 0U) {
    return hipErrorInvalidValue;
  }
  if (state.event_query_gate) {
    state.event_query_entered = true;
    state.condition.notify_all();
    state.condition.wait(lock, [] { return !state.event_query_gate; });
  }
  return event->recorded ? hipSuccess : hipErrorNotReady;
}

hipError_t hipMemcpyAsync(void *const destination, const void *const source,
                          const std::size_t size, const hipMemcpyKind kind,
                          const hipStream_t stream) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.streams.count(stream) == 0U || destination == nullptr ||
      source == nullptr ||
      (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost)) {
    return hipErrorInvalidValue;
  }
  std::memcpy(destination, source, size);
  return hipSuccess;
}
