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

#define SLLM_BACKEND_HIP UINT32_C(1)

typedef uint32_t sllm_access_mode_t;

#define SLLM_ACCESS_READ UINT32_C(1)
#define SLLM_ACCESS_WRITE UINT32_C(2)
#define SLLM_ACCESS_READ_WRITE UINT32_C(3)

/* These handles have no public layout and must not be dereferenced by callers.
 */
typedef struct sllm_context_t sllm_context_t;
typedef struct sllm_queue_t sllm_queue_t;
typedef struct sllm_buffer_t sllm_buffer_t;
typedef struct sllm_event_t sllm_event_t;
typedef struct sllm_completion_t sllm_completion_t;

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

#ifdef __cplusplus
extern "C" {
#endif

SLLM_HIP_API sllm_status_t sllm_get_abi_version(uint32_t *abi_version,
                                                sllm_error_sink_t *error_sink);

SLLM_HIP_API sllm_status_t sllm_query_version(sllm_version_info_t *version,
                                              sllm_error_sink_t *error_sink);

/* Phase 1 always returns SLLM_STATUS_HIP_UNAVAILABLE for SLLM_BACKEND_HIP. */
SLLM_HIP_API sllm_status_t
sllm_backend_probe(uint32_t backend, sllm_backend_probe_result_t *result,
                   sllm_error_sink_t *error_sink);

/* No context can be created in Phase 1; a null or opaque handle is never
 * dereferenced. */
SLLM_HIP_API sllm_status_t sllm_context_probe(
    const sllm_context_t *context, sllm_context_probe_result_t *result,
    sllm_error_sink_t *error_sink);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLLM_HIP_H */
