#ifndef SLLM_TEST_FAKE_HIP_RUNTIME_H
#define SLLM_TEST_FAKE_HIP_RUNTIME_H

#include <cstddef>
#include <cstdint>

enum hipError_t : int {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorNotReady = 600,
  hipErrorUnknown = 999,
};

struct FakeHipStream;
struct FakeHipEvent {
  bool recorded = false;
};

using hipStream_t = FakeHipStream *;
using hipEvent_t = FakeHipEvent *;

struct hipDeviceProp_t {
  char name[256];
  char gcnArchName[256];
  std::size_t totalGlobalMem;
  int warpSize;
};

enum : unsigned int {
  hipStreamNonBlocking = 1U,
  hipEventDisableTiming = 2U,
};

enum hipMemcpyKind : int {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
};

const char *hipGetErrorString(hipError_t error) noexcept;

hipError_t hipGetDeviceCount(int *count) noexcept;
hipError_t hipGetDeviceProperties(hipDeviceProp_t *properties,
                                  unsigned int device) noexcept;
hipError_t hipSetDevice(int device) noexcept;
hipError_t hipStreamCreateWithFlags(hipStream_t *stream,
                                    unsigned int flags) noexcept;
hipError_t hipStreamDestroy(hipStream_t stream) noexcept;
hipError_t hipStreamSynchronize(hipStream_t stream) noexcept;
hipError_t hipMalloc(void **pointer, std::size_t size) noexcept;
hipError_t hipFree(void *pointer) noexcept;
hipError_t hipEventCreateWithFlags(hipEvent_t *event,
                                   unsigned int flags) noexcept;
hipError_t hipEventDestroy(hipEvent_t event) noexcept;
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) noexcept;
hipError_t hipEventElapsedTime(float *milliseconds, hipEvent_t start,
                               hipEvent_t end) noexcept;
hipError_t hipEventQuery(hipEvent_t event) noexcept;
hipError_t hipMemcpyAsync(void *destination, const void *source,
                          std::size_t size, hipMemcpyKind kind,
                          hipStream_t stream) noexcept;

namespace fake_hip {

void reset() noexcept;
hipError_t rmsnorm_launch(const uint16_t *activation, const uint16_t *raw_scale,
                          uint16_t *output, uint32_t normalized_size,
                          uint32_t row_count, float epsilon,
                          hipStream_t stream) noexcept;
void set_rmsnorm_launch_status(hipError_t status) noexcept;
std::size_t rmsnorm_launch_calls() noexcept;
uint32_t rmsnorm_last_normalized_size() noexcept;
uint32_t rmsnorm_last_row_count() noexcept;
void set_event_record_status(hipError_t status) noexcept;
void set_event_create_gate(bool enabled) noexcept;
void wait_event_create_entered();
void release_event_create_gate() noexcept;
void set_event_query_gate(bool enabled) noexcept;
void wait_event_query_entered();
void release_event_query_gate() noexcept;
std::size_t event_destroy_calls() noexcept;
std::size_t stream_destroy_calls() noexcept;
std::size_t allocation_free_calls() noexcept;
std::size_t live_events() noexcept;
std::size_t live_streams() noexcept;
std::size_t live_allocations() noexcept;

} // namespace fake_hip

#endif // SLLM_TEST_FAKE_HIP_RUNTIME_H
