#ifndef SLLM_TEST_FAKE_HIP_RUNTIME_H
#define SLLM_TEST_FAKE_HIP_RUNTIME_H

#include <cstddef>
#include <cstdint>

enum hipError_t : int {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorNotSupported = 801,
  hipErrorNotReady = 600,
  hipErrorUnknown = 999,
};

struct FakeHipStream {};
struct FakeHipMemHandle;
struct FakeHipEvent {
  bool recorded = false;
};
using hipStream_t = FakeHipStream *;
using hipEvent_t = FakeHipEvent *;
using hipMemGenericAllocationHandle_t = FakeHipMemHandle *;

enum hipMemAllocationType : int { hipMemAllocationTypePinned = 1 };
enum hipMemLocationType : int { hipMemLocationTypeDevice = 1 };
enum hipMemAllocationGranularity_flags : int {
  hipMemAllocationGranularityMinimum = 0,
  hipMemAllocationGranularityRecommended = 1,
};
enum hipMemAccessFlags : int {
  hipMemAccessFlagsProtRead = 1,
  hipMemAccessFlagsProtReadWrite = 3,
};
struct hipMemLocation {
  hipMemLocationType type;
  int id;
};
struct hipMemAllocationProp {
  hipMemAllocationType type;
  hipMemLocation location;
};
struct hipMemAccessDesc {
  hipMemLocation location;
  hipMemAccessFlags flags;
};

struct hipDeviceProp_t {
  char name[256];
  char gcnArchName[256];
  std::size_t totalGlobalMem;
  int warpSize;
};

enum hipDeviceAttribute_t : int {
  hipDeviceAttributeVirtualMemoryManagementSupported = 1,
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
hipError_t hipDeviceGetAttribute(int *value, hipDeviceAttribute_t attribute,
                                 int device) noexcept;
hipError_t hipMemGetInfo(std::size_t *available, std::size_t *total) noexcept;
hipError_t hipMemGetAllocationGranularity(
    std::size_t *granularity, const hipMemAllocationProp *properties,
    hipMemAllocationGranularity_flags option) noexcept;
hipError_t hipMemAddressReserve(void **pointer, std::size_t size,
                                std::size_t alignment, void *requested,
                                unsigned long long flags) noexcept;
hipError_t hipMemAddressFree(void *pointer, std::size_t size) noexcept;
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t *handle,
                        std::size_t size,
                        const hipMemAllocationProp *properties,
                        unsigned long long flags) noexcept;
hipError_t hipMemMap(void *pointer, std::size_t size, std::size_t offset,
                     hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags) noexcept;
hipError_t hipMemSetAccess(void *pointer, std::size_t size,
                           const hipMemAccessDesc *descriptors,
                           std::size_t count) noexcept;
hipError_t hipMemUnmap(void *pointer, std::size_t size) noexcept;
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) noexcept;
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

enum class VmmOperation : uint8_t {
  AddressReserve,
  AddressFree,
  Create,
  Map,
  SetAccess,
  Unmap,
  Release,
};

uint32_t f16_to_f32_bits_for_test(uint16_t raw) noexcept;
void reset() noexcept;
void set_vmm_supported(bool supported) noexcept;
/* The next `successful_calls` matching operations pass, then one operation
 * fails with hipErrorOutOfMemory.  A zero value fails the next operation. */
void set_vmm_failure_after(VmmOperation operation,
                           uint64_t successful_calls) noexcept;
void clear_vmm_failures() noexcept;
std::size_t vmm_operation_calls(VmmOperation operation) noexcept;
std::size_t device_property_calls() noexcept;
hipError_t rmsnorm_launch(const uint16_t *activation, const uint16_t *raw_scale,
                          uint16_t *output, uint32_t normalized_size,
                          uint32_t row_count, float epsilon,
                          uint32_t scale_mode, hipStream_t stream) noexcept;
hipError_t residual_rmsnorm_launch(const uint16_t *residual,
                                   const uint16_t *addend,
                                   const uint16_t *raw_scale,
                                   uint16_t *residual_output, uint16_t *output,
                                   uint32_t normalized_size, uint32_t row_count,
                                   float epsilon, uint32_t scale_mode,
                                   hipStream_t stream) noexcept;
hipError_t elementwise_copy_launch(const uint16_t *input, uint16_t *output,
                                   uint64_t element_count,
                                   hipStream_t stream) noexcept;
hipError_t elementwise_add_launch(const uint16_t *input0,
                                  const uint16_t *input1, uint16_t *output,
                                  uint64_t element_count,
                                  hipStream_t stream) noexcept;
hipError_t
elementwise_broadcast_add_launch(const uint16_t *input, const uint16_t *vector,
                                 uint16_t *output, uint64_t element_count,
                                 uint64_t width, hipStream_t stream) noexcept;
hipError_t elementwise_silu_mul_launch(const uint16_t *gate, const uint16_t *up,
                                       uint16_t *output, uint64_t element_count,
                                       hipStream_t stream) noexcept;
hipError_t elementwise_sigmoid_mul_launch(const uint16_t *gate,
                                          const uint16_t *attention_value,
                                          uint16_t *output,
                                          uint64_t element_count,
                                          hipStream_t stream) noexcept;
hipError_t elementwise_scalar_mul_launch(const uint16_t *input,
                                         const uint16_t *scalar,
                                         uint16_t *output,
                                         uint64_t element_count,
                                         hipStream_t stream) noexcept;
hipError_t elementwise_gelu_tanh_mul_launch(const uint16_t *gate,
                                            const uint16_t *up,
                                            uint16_t *output,
                                            uint64_t element_count,
                                            hipStream_t stream) noexcept;
hipError_t elementwise_tanh_softcap_launch(const uint16_t *input,
                                           const uint16_t *cap,
                                           uint16_t *output,
                                           uint64_t element_count,
                                           hipStream_t stream) noexcept;
hipError_t embedding_gather_launch(const uint16_t *weight,
                                   const int32_t *token_ids, uint16_t *output,
                                   uint64_t token_count, uint64_t hidden_size,
                                   hipStream_t stream) noexcept;
hipError_t matmul_launch(const uint16_t *activation, const uint16_t *weight,
                         uint16_t *output, uint64_t m, uint64_t k, uint64_t n,
                         hipStream_t stream) noexcept;
hipError_t argmax_launch(const uint16_t *logits, int32_t *output, uint64_t m,
                         uint64_t v, hipStream_t stream) noexcept;
hipError_t attention_preprocess_launch(
    const uint16_t *packed_q_gate, const uint16_t *k,
    const uint16_t *q_raw_scale, const uint16_t *k_raw_scale,
    const int32_t *positions, uint16_t *q_output, uint16_t *gate_output,
    uint16_t *k_output, uint32_t m, hipStream_t stream) noexcept;
hipError_t rotary_launch(const uint16_t *query, const uint16_t *key,
                         const int32_t *positions, uint16_t *query_output,
                         uint16_t *key_output, uint32_t token_count,
                         uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                         uint32_t rotary_dim, float theta,
                         hipStream_t stream) noexcept;
hipError_t windowed_attention_launch(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint64_t sliding_window, hipStream_t stream) noexcept;
hipError_t
kv_state_append_launch(const uint16_t *key_input, const uint16_t *value_input,
                       uint16_t *key_output, uint16_t *value_output,
                       uint32_t token_count, uint64_t capacity_tokens,
                       uint64_t start_position, hipStream_t stream) noexcept;
hipError_t causal_attention_launch(const uint16_t *query, const uint16_t *key,
                                   const uint16_t *value, uint16_t *output,
                                   uint32_t query_count,
                                   uint64_t capacity_tokens,
                                   uint64_t start_position,
                                   uint64_t committed_kv_length,
                                   hipStream_t stream) noexcept;
std::size_t embedding_gather_launch_calls() noexcept;
void set_gcn_arch_name(const char *name) noexcept;
std::size_t matmul_launch_calls() noexcept;
std::size_t attention_preprocess_launch_calls() noexcept;
std::size_t rotary_launch_calls() noexcept;
std::size_t windowed_attention_launch_calls() noexcept;
uint32_t rotary_last_token_count() noexcept;
uint32_t attention_preprocess_last_m() noexcept;
std::size_t kv_state_append_launch_calls() noexcept;
std::size_t causal_attention_launch_calls() noexcept;
void set_causal_attention_launch_status(hipError_t status) noexcept;
uint32_t kv_state_last_token_count() noexcept;
uint64_t kv_state_last_capacity_tokens() noexcept;
uint64_t kv_state_last_start_position() noexcept;
void set_kv_state_append_launch_status(hipError_t status) noexcept;
bool copy_kv_key_output(uint16_t *destination, uint64_t element_count) noexcept;
bool copy_kv_value_output(uint16_t *destination,
                          uint64_t element_count) noexcept;
uint64_t matmul_last_m() noexcept;
uint64_t matmul_last_k() noexcept;
uint64_t matmul_last_n() noexcept;
uint64_t matmul_last_output_elements() noexcept;
void set_elementwise_launch_status(hipError_t status) noexcept;
std::size_t elementwise_copy_launch_calls() noexcept;
std::size_t elementwise_add_launch_calls() noexcept;
std::size_t elementwise_broadcast_add_launch_calls() noexcept;
std::size_t elementwise_silu_mul_launch_calls() noexcept;
std::size_t elementwise_sigmoid_mul_launch_calls() noexcept;
std::size_t elementwise_scalar_mul_launch_calls() noexcept;
std::size_t elementwise_gelu_tanh_mul_launch_calls() noexcept;
std::size_t elementwise_tanh_softcap_launch_calls() noexcept;
uint64_t elementwise_last_element_count() noexcept;
void set_rmsnorm_launch_status(hipError_t status) noexcept;
void set_rmsnorm_numerical_execution(bool enabled) noexcept;
std::size_t rmsnorm_launch_calls() noexcept;
std::size_t residual_rmsnorm_launch_calls() noexcept;
uint32_t rmsnorm_last_normalized_size() noexcept;
uint32_t rmsnorm_last_row_count() noexcept;
uint32_t rmsnorm_last_scale_mode() noexcept;
void set_matmul_launch_status(hipError_t status) noexcept;
std::size_t argmax_launch_calls() noexcept;
uint64_t argmax_last_m() noexcept;
uint64_t argmax_last_v() noexcept;
void set_argmax_launch_status(hipError_t status) noexcept;
void set_event_record_status(hipError_t status) noexcept;
void set_event_create_gate(bool enabled) noexcept;
void wait_event_create_entered();
void release_event_create_gate() noexcept;
void set_event_query_gate(bool enabled) noexcept;
void set_completion_pending(bool enabled) noexcept;
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
