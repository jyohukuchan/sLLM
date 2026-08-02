#ifndef ULLM_HIP_H
#define ULLM_HIP_H

#include <stdint.h>

#if defined(_WIN32) && defined(ULLM_HIP_SHARED)
#if defined(ULLM_HIP_BUILD)
#define ULLM_HIP_API __declspec(dllexport)
#else
#define ULLM_HIP_API __declspec(dllimport)
#endif
#else
#define ULLM_HIP_API
#endif

#define ULLM_HIP_ABI_VERSION UINT32_C(1)
#define ULLM_HIP_LIBRARY_VERSION_MAJOR UINT32_C(0)
#define ULLM_HIP_LIBRARY_VERSION_MINOR UINT32_C(1)
#define ULLM_HIP_LIBRARY_VERSION_PATCH UINT32_C(0)

typedef uint32_t ullm_status_t;

#define ULLM_STATUS_OK UINT32_C(0)
#define ULLM_STATUS_INVALID_ARGUMENT UINT32_C(1)
#define ULLM_STATUS_BUFFER_TOO_SMALL UINT32_C(2)
#define ULLM_STATUS_UNSUPPORTED UINT32_C(3)
#define ULLM_STATUS_HIP_UNAVAILABLE UINT32_C(4)
#define ULLM_STATUS_INVALID_ABI_VERSION UINT32_C(5)
#define ULLM_STATUS_RESERVED_NONZERO UINT32_C(6)
#define ULLM_STATUS_INTERNAL_ERROR UINT32_C(7)

#define ULLM_BACKEND_HIP UINT32_C(1)

typedef uint32_t ullm_access_mode_t;

#define ULLM_ACCESS_READ UINT32_C(1)
#define ULLM_ACCESS_WRITE UINT32_C(2)
#define ULLM_ACCESS_READ_WRITE UINT32_C(3)

/* These handles have no public layout and must not be dereferenced by callers.
 */
typedef struct ullm_context_t ullm_context_t;
typedef struct ullm_queue_t ullm_queue_t;
typedef struct ullm_buffer_t ullm_buffer_t;
typedef struct ullm_event_t ullm_event_t;
typedef struct ullm_completion_t ullm_completion_t;

typedef struct ullm_error_sink_t {
  uint32_t struct_size;
  uint32_t abi_version;
  char *message;
  uint64_t message_capacity;
  uint64_t message_length;
  uint64_t reserved[2];
} ullm_error_sink_t;

/* message_capacity includes space for the terminating NUL.  On a diagnostic
 * error, message_length is the required message length excluding that NUL,
 * even when the message is truncated.  A valid sink copies at most
 * message_capacity - 1 bytes and always NUL-terminates a non-zero-capacity
 * buffer.  If the complete message does not fit, the API returns
 * ULLM_STATUS_BUFFER_TOO_SMALL.  A null sink leaves the primary operation
 * status unchanged. */

/* The argument must be a character array, not a pointer. */
// clang-format off
#define ULLM_ERROR_SINK_INIT(buffer)                                           \
  {sizeof(ullm_error_sink_t),                                                  \
   ULLM_HIP_ABI_VERSION,                                                       \
   (buffer),                                                                   \
   sizeof(buffer),                                                             \
   0,                                                                          \
   {0, 0}}
// clang-format on

typedef struct ullm_version_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t reserved[3];
} ullm_version_info_t;

typedef struct ullm_backend_probe_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t backend;
  uint32_t available;
  uint32_t hip_runtime_present;
  uint32_t reserved[3];
} ullm_backend_probe_result_t;

typedef struct ullm_context_probe_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t context_present;
  uint32_t hip_available;
  uint32_t reserved[4];
} ullm_context_probe_result_t;

#ifdef __cplusplus
extern "C" {
#endif

ULLM_HIP_API ullm_status_t ullm_get_abi_version(uint32_t *abi_version,
                                                ullm_error_sink_t *error_sink);

ULLM_HIP_API ullm_status_t ullm_query_version(ullm_version_info_t *version,
                                              ullm_error_sink_t *error_sink);

/* Phase 1 always returns ULLM_STATUS_HIP_UNAVAILABLE for ULLM_BACKEND_HIP. */
ULLM_HIP_API ullm_status_t
ullm_backend_probe(uint32_t backend, ullm_backend_probe_result_t *result,
                   ullm_error_sink_t *error_sink);

/* No context can be created in Phase 1; a null or opaque handle is never
 * dereferenced. */
ULLM_HIP_API ullm_status_t ullm_context_probe(
    const ullm_context_t *context, ullm_context_probe_result_t *result,
    ullm_error_sink_t *error_sink);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ULLM_HIP_H */
