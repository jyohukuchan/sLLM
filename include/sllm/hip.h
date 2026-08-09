#ifndef SLLM_HIP_H
#define SLLM_HIP_H

#include <stdint.h>

#if defined(_WIN32) && defined(SLLM_HIP_SHARED)
#if defined(SLLM_HIP_BUILD)
#define SLLM_HIP_API __declspec(dllexport)
#else
#define SLLM_HIP_API __declspec(dllimport)
#endif
#else
#define SLLM_HIP_API
#endif

#ifdef __cplusplus
#define SLLM_HIP_NOEXCEPT noexcept
#else
#define SLLM_HIP_NOEXCEPT
#endif

#define SLLM_HIP_ABI_VERSION UINT32_C(1)
#define SLLM_HIP_LIBRARY_VERSION_MAJOR UINT32_C(0)
#define SLLM_HIP_LIBRARY_VERSION_MINOR UINT32_C(1)
#define SLLM_HIP_LIBRARY_VERSION_PATCH UINT32_C(0)

typedef uint32_t sllm_status_t;

#define SLLM_STATUS_OK UINT32_C(0)
#define SLLM_STATUS_INVALID_ARGUMENT UINT32_C(1)
#define SLLM_STATUS_BUFFER_TOO_SMALL UINT32_C(2)
#define SLLM_STATUS_UNSUPPORTED UINT32_C(3)
#define SLLM_STATUS_HIP_UNAVAILABLE UINT32_C(4)
#define SLLM_STATUS_INVALID_ABI_VERSION UINT32_C(5)
#define SLLM_STATUS_RESERVED_NONZERO UINT32_C(6)
#define SLLM_STATUS_INTERNAL_ERROR UINT32_C(7)

/* Public runtime statuses use a separate numeric range from private evidence
 * statuses.  They are additive to the Phase 1 status set. */
#define SLLM_STATUS_PUBLIC_PENDING UINT32_C(0x100)
#define SLLM_STATUS_PUBLIC_TIMEOUT UINT32_C(0x101)
#define SLLM_STATUS_PUBLIC_INVALID_HANDLE UINT32_C(0x102)
#define SLLM_STATUS_PUBLIC_DEVICE_MISMATCH UINT32_C(0x103)
#define SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR UINT32_C(0x104)
#define SLLM_STATUS_PUBLIC_BUSY UINT32_C(0x105)
#define SLLM_STATUS_PUBLIC_NOT_READY UINT32_C(0x106)
/* RMSNorm execution is additive to public ABI v1. */
#define SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR UINT32_C(0x107)
#define SLLM_STATUS_INVALID_TENSOR_BINDING UINT32_C(0x108)
#define SLLM_STATUS_ZERO_EXTENT UINT32_C(0x109)
#define SLLM_STATUS_SHAPE_MISMATCH UINT32_C(0x10a)
#define SLLM_STATUS_STRIDE_MISMATCH UINT32_C(0x10b)
#define SLLM_STATUS_METADATA_OVERFLOW UINT32_C(0x10c)
#define SLLM_STATUS_BUFFER_OUT_OF_BOUNDS UINT32_C(0x10d)
#define SLLM_STATUS_MISALIGNED_OFFSET UINT32_C(0x10e)
#define SLLM_STATUS_UNSUPPORTED_DTYPE UINT32_C(0x10f)
#define SLLM_STATUS_UNSUPPORTED_ENCODING UINT32_C(0x110)
#define SLLM_STATUS_INVALID_EPSILON UINT32_C(0x111)
#define SLLM_STATUS_UNSUPPORTED_SCALE_MODE UINT32_C(0x112)
#define SLLM_STATUS_ALIAS_OVERLAP UINT32_C(0x113)
#define SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH UINT32_C(0x114)

#define SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_RMSNORM_KERNEL_ID_BASELINE_WAVE32_V1 UINT32_C(1)
#define SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_RMSNORM_MAX_N UINT64_C(4096)
#define SLLM_HIP_RMSNORM_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_RMSNORM_MAX_ROWS UINT64_C(4294967295)

#define SLLM_BACKEND_HIP UINT32_C(1)

typedef uint32_t sllm_access_mode_t;

#define SLLM_ACCESS_READ UINT32_C(1)
#define SLLM_ACCESS_WRITE UINT32_C(2)
#define SLLM_ACCESS_READ_WRITE UINT32_C(3)

#define SLLM_HIP_MAX_DEVICE_NAME UINT32_C(128)
#define SLLM_HIP_MAX_GCN_ARCH_NAME UINT32_C(64)
#define SLLM_HIP_MAX_TRANSFER_BYTES UINT64_C(1073741824)

#define SLLM_HIP_RMSNORM_VERSION UINT32_C(1)
#define SLLM_HIP_TENSOR_MAX_RANK UINT32_C(8)

typedef uint32_t sllm_tensor_dtype_t;
#define SLLM_TENSOR_DTYPE_BF16 UINT32_C(0)
#define SLLM_TENSOR_DTYPE_F32 UINT32_C(2)

typedef uint32_t sllm_tensor_encoding_t;
#define SLLM_TENSOR_ENCODING_UNQUANTIZED UINT32_C(0)

typedef uint32_t sllm_rmsnorm_accumulation_dtype_t;
#define SLLM_RMSNORM_ACCUMULATION_F32 UINT32_C(2)

typedef uint32_t sllm_rmsnorm_scale_mode_t;
#define SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE UINT32_C(1)

typedef uint32_t sllm_rmsnorm_alias_policy_t;
#define SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP UINT32_C(1)

#define SLLM_COMPLETION_STATE_PENDING UINT32_C(0)
#define SLLM_COMPLETION_STATE_SUCCESS UINT32_C(1)
#define SLLM_COMPLETION_STATE_FAILURE UINT32_C(2)

/* These handles have no public layout and must not be dereferenced by callers.
 */
typedef struct sllm_context_t sllm_context_t;
typedef struct sllm_queue_t sllm_queue_t;
typedef struct sllm_buffer_t sllm_buffer_t;
typedef struct sllm_event_t sllm_event_t;
typedef struct sllm_completion_t sllm_completion_t;
typedef struct sllm_rmsnorm_plan_t sllm_rmsnorm_plan_t;

typedef struct sllm_completion_timing_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid;
  uint32_t reserved0;
  uint64_t elapsed_ns;
  uint32_t reserved[4];
} sllm_completion_timing_t;

typedef struct sllm_error_sink_t {
  uint32_t struct_size;
  uint32_t abi_version;
  char *message;
  uint64_t message_capacity;
  uint64_t message_length;
  uint64_t reserved[2];
} sllm_error_sink_t;

/* message_capacity includes space for the terminating NUL.  On a diagnostic
 * error, message_length is the required message length excluding that NUL,
 * even when the message is truncated.  A valid sink copies at most
 * message_capacity - 1 bytes and always NUL-terminates a non-zero-capacity
 * buffer.  If the complete message does not fit, the API returns
 * SLLM_STATUS_BUFFER_TOO_SMALL.  A null sink leaves the primary operation
 * status unchanged. */

/* The argument must be a character array, not a pointer. */
// clang-format off
#define SLLM_ERROR_SINK_INIT(buffer)                                           \
  {sizeof(sllm_error_sink_t),                                                  \
   SLLM_HIP_ABI_VERSION,                                                       \
   (buffer),                                                                   \
   sizeof(buffer),                                                             \
   0,                                                                          \
   {0, 0}}
// clang-format on

typedef struct sllm_version_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t reserved[3];
} sllm_version_info_t;

typedef struct sllm_backend_probe_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t backend;
  uint32_t available;
  uint32_t hip_runtime_present;
  uint32_t reserved[3];
} sllm_backend_probe_result_t;

typedef struct sllm_context_probe_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t context_present;
  uint32_t hip_available;
  uint32_t reserved[4];
} sllm_context_probe_result_t;

typedef struct sllm_device_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t device_index;
  uint32_t visible_device_count;
  uint64_t total_memory_bytes;
  uint32_t wavefront_size;
  uint32_t reserved0;
  char name[SLLM_HIP_MAX_DEVICE_NAME];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[4];
} sllm_device_info_t;

typedef struct sllm_context_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t device_index;
  uint32_t flags;
  char expected_gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[4];
} sllm_context_create_info_t;

typedef struct sllm_queue_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t flags;
  uint32_t reserved[5];
} sllm_queue_create_info_t;

typedef struct sllm_buffer_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint32_t flags;
  uint32_t reserved[5];
} sllm_buffer_create_info_t;

typedef struct sllm_transfer_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  /* H2D source is copied before the call returns.  D2H does not retain this
   * pointer; callers read staged output with sllm_completion_read(). */
  void *host_pointer;
  uint64_t buffer_offset_bytes;
  uint64_t size_bytes;
  uint32_t reserved[4];
} sllm_transfer_desc_t;

typedef struct sllm_completion_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t state;
  uint32_t reserved0;
  uint64_t transfer_size_bytes;
  uint64_t available_bytes;
  uint32_t reserved[4];
} sllm_completion_result_t;

/* Tensor bindings are descriptors, not ownership transfers.  prepare copies
 * all metadata immediately and never retains this struct or its address. */
typedef struct sllm_tensor_binding_t {
  uint32_t struct_size;
  uint32_t abi_version;
  const sllm_buffer_t *buffer;
  uint64_t byte_offset;
  sllm_tensor_dtype_t dtype;
  sllm_tensor_encoding_t encoding;
  uint32_t rank;
  uint32_t reserved0;
  uint64_t shape[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t stride_elements[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t reserved[2];
} sllm_tensor_binding_t;

typedef struct sllm_rmsnorm_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  sllm_rmsnorm_accumulation_dtype_t accumulation_dtype;
  sllm_rmsnorm_scale_mode_t scale_mode;
  sllm_rmsnorm_alias_policy_t alias_policy;
  uint32_t epsilon_bits;
  uint32_t reserved[3];
  sllm_tensor_binding_t activation;
  sllm_tensor_binding_t raw_scale;
  sllm_tensor_binding_t output;
} sllm_rmsnorm_desc_t;

typedef struct sllm_rmsnorm_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t row_count;
  uint64_t normalized_size;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_rmsnorm_dispatch_info_t;

#ifdef __cplusplus
extern "C" {
#endif

SLLM_HIP_API sllm_status_t sllm_get_abi_version(
    uint32_t *abi_version, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_query_version(sllm_version_info_t *version,
                                              sllm_error_sink_t *error_sink)
    SLLM_HIP_NOEXCEPT;

/* The host build returns SLLM_STATUS_HIP_UNAVAILABLE.  The HIP build reports
 * the visible device set without CPU fallback. */
SLLM_HIP_API sllm_status_t
sllm_backend_probe(uint32_t backend, sllm_backend_probe_result_t *result,
                   sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* A null context reports runtime availability.  An opaque non-null context is
 * validated by the HIP build and is never dereferenced by callers. */
SLLM_HIP_API sllm_status_t sllm_context_probe(
    const sllm_context_t *context, sllm_context_probe_result_t *result,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_device_count(
    uint32_t *count, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_device_query(uint32_t device_index, sllm_device_info_t *info,
                  sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_context_create(
    const sllm_context_create_info_t *info, sllm_context_t **context,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_context_release(
    sllm_context_t **context, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_queue_create(
    const sllm_context_t *context, const sllm_queue_create_info_t *info,
    sllm_queue_t **queue, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_queue_release(
    sllm_queue_t **queue, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_create(
    const sllm_context_t *context, const sllm_buffer_create_info_t *info,
    sllm_buffer_t **buffer, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_release(
    sllm_buffer_t **buffer, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_buffer_size(const sllm_buffer_t *buffer, uint64_t *size_bytes,
                 sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_event_create(const sllm_context_t *context, sllm_event_t **event,
                  sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_event_release(
    sllm_event_t **event, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_copy_h2d(
    const sllm_queue_t *queue, const sllm_buffer_t *buffer,
    const sllm_transfer_desc_t *transfer, sllm_completion_t **completion,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_copy_d2h(
    const sllm_queue_t *queue, const sllm_buffer_t *buffer,
    const sllm_transfer_desc_t *transfer, sllm_completion_t **completion,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_completion_query(
    sllm_completion_t *completion, sllm_completion_result_t *result,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_completion_wait(sllm_completion_t *completion, uint32_t timeout_ms,
                     sllm_completion_result_t *result,
                     sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_completion_read(sllm_completion_t *completion, void *destination,
                     uint64_t destination_capacity, uint64_t *bytes_written,
                     sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* RMSNorm completions retain a pair of timing-enabled HIP events.  The
 * elapsed value is available only after successful completion and is never a
 * host-clock or CPU-fallback estimate.  Other completion kinds return
 * SLLM_STATUS_UNSUPPORTED. */
SLLM_HIP_API sllm_status_t sllm_completion_timing(
    sllm_completion_t *completion, sllm_completion_timing_t *timing,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_completion_release(sllm_completion_t **completion,
                        sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* RMSNorm preparation captures immutable metadata; execution is a separate
 * asynchronous baseline dispatch operation on the reusable plan. */
SLLM_HIP_API sllm_status_t sllm_rmsnorm_prepare(
    const sllm_context_t *context, const sllm_rmsnorm_desc_t *descriptor,
    sllm_rmsnorm_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_rmsnorm_plan_release(sllm_rmsnorm_plan_t **plan,
                          sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_rmsnorm_execute(
    const sllm_rmsnorm_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion, sllm_rmsnorm_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLLM_HIP_H */

#undef SLLM_HIP_NOEXCEPT
