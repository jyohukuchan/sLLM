#ifndef ULLM_HIP_EVIDENCE_ABI_H
#define ULLM_HIP_EVIDENCE_ABI_H

#include "ullm/hip.h"

#include <stdint.h>

/* Private model-free evidence ABI. This header is deliberately not installed.
 */
#define ULLM_HIP_EVIDENCE_ABI_VERSION UINT32_C(1)
#define ULLM_HIP_EVIDENCE_TRANSFORM_XOR UINT8_C(0x5a)

#define ULLM_STATUS_HIP_TIMEOUT UINT32_C(8)
#define ULLM_STATUS_HIP_INVALID_HANDLE UINT32_C(9)
#define ULLM_STATUS_HIP_ZERO_DISPATCH UINT32_C(10)
#define ULLM_STATUS_HIP_RUNTIME_ERROR UINT32_C(11)
#define ULLM_STATUS_HIP_DISPATCH_CONTRACT UINT32_C(12)

typedef struct ullm_hip_evidence_request_t {
  uint32_t struct_size;
  uint32_t abi_version;
  const uint8_t *input;
  uint64_t input_size;
  uint32_t reserved[4];
} ullm_hip_evidence_request_t;

typedef struct ullm_hip_evidence_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t output_size;
  /* Successful device hipMalloc allocations; excludes hipHostMalloc. */
  uint64_t allocation_count;
  /* Successful hipMemcpyAsync transfers; excludes CPU memcpy operations. */
  uint64_t copy_count;
  uint64_t dispatch_count;
  uint32_t selected_backend;
  uint32_t fallback_used;
  uint32_t terminal;
  uint32_t reserved[4];
} ullm_hip_evidence_result_t;

typedef struct ullm_hip_evidence_completion_t ullm_hip_evidence_completion_t;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t ullm_hip_evidence_submit(const ullm_hip_evidence_request_t *request,
                                  ullm_hip_evidence_completion_t **completion,
                                  struct ullm_error_sink_t *error_sink);

uint32_t ullm_hip_evidence_wait(ullm_hip_evidence_completion_t *completion,
                                uint32_t timeout_ms, uint8_t *output,
                                uint64_t output_capacity,
                                ullm_hip_evidence_result_t *result,
                                struct ullm_error_sink_t *error_sink);

uint32_t ullm_hip_evidence_destroy(ullm_hip_evidence_completion_t **completion,
                                   struct ullm_error_sink_t *error_sink);

#ifdef __cplusplus
} /* extern "C" */

#include <cstddef>

static_assert(sizeof(ullm_hip_evidence_request_t) == 40U,
              "private evidence request ABI layout changed");
static_assert(alignof(ullm_hip_evidence_request_t) == 8U,
              "private evidence request ABI alignment changed");
static_assert(offsetof(ullm_hip_evidence_request_t, struct_size) == 0U,
              "private evidence request struct_size offset changed");
static_assert(offsetof(ullm_hip_evidence_request_t, abi_version) == 4U,
              "private evidence request abi_version offset changed");
static_assert(offsetof(ullm_hip_evidence_request_t, input) == 8U,
              "private evidence request input offset changed");
static_assert(offsetof(ullm_hip_evidence_request_t, input_size) == 16U,
              "private evidence request input_size offset changed");
static_assert(offsetof(ullm_hip_evidence_request_t, reserved) == 24U,
              "private evidence request reserved offset changed");
static_assert(sizeof(ullm_hip_evidence_result_t) == 72U,
              "private evidence result ABI layout changed");
static_assert(alignof(ullm_hip_evidence_result_t) == 8U,
              "private evidence result ABI alignment changed");
static_assert(offsetof(ullm_hip_evidence_result_t, struct_size) == 0U,
              "private evidence result struct_size offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, abi_version) == 4U,
              "private evidence result abi_version offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, output_size) == 8U,
              "private evidence result output_size offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, allocation_count) == 16U,
              "private evidence result allocation_count offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, copy_count) == 24U,
              "private evidence result copy_count offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, dispatch_count) == 32U,
              "private evidence result dispatch_count offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, selected_backend) == 40U,
              "private evidence result selected_backend offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, fallback_used) == 44U,
              "private evidence result fallback_used offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, terminal) == 48U,
              "private evidence result terminal offset changed");
static_assert(offsetof(ullm_hip_evidence_result_t, reserved) == 52U,
              "private evidence result reserved offset changed");
static_assert(sizeof(ullm_error_sink_t) == 48U,
              "evidence error sink ABI layout changed");
static_assert(alignof(ullm_error_sink_t) == 8U,
              "evidence error sink ABI alignment changed");
static_assert(offsetof(ullm_error_sink_t, struct_size) == 0U,
              "evidence error sink struct_size offset changed");
static_assert(offsetof(ullm_error_sink_t, abi_version) == 4U,
              "evidence error sink abi_version offset changed");
static_assert(offsetof(ullm_error_sink_t, message) == 8U,
              "evidence error sink message offset changed");
static_assert(offsetof(ullm_error_sink_t, message_capacity) == 16U,
              "evidence error sink message_capacity offset changed");
static_assert(offsetof(ullm_error_sink_t, message_length) == 24U,
              "evidence error sink message_length offset changed");
static_assert(offsetof(ullm_error_sink_t, reserved) == 32U,
              "evidence error sink reserved offset changed");
#endif

#endif /* ULLM_HIP_EVIDENCE_ABI_H */
