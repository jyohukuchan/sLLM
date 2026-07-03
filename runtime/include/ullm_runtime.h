// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_RUNTIME_H
#define ULLM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ULLM_RUNTIME_ABI_VERSION 1u

typedef enum ullm_status {
    ULLM_STATUS_OK = 0,
    ULLM_STATUS_INVALID_ARGUMENT = 1,
    ULLM_STATUS_BUFFER_TOO_SMALL = 2,
    ULLM_STATUS_RUNTIME_ERROR = 3,
} ullm_status;

typedef struct ullm_device_info {
    int32_t device_id;
    char backend[16];
    char name[128];
    uint64_t total_global_mem;
    int32_t compute_major;
    int32_t compute_minor;
    char gcn_arch_name[64];
    uint32_t flags;
} ullm_device_info;

typedef struct ullm_runtime_context ullm_runtime_context;

typedef struct ullm_runtime_buffer ullm_runtime_buffer;

typedef struct ullm_runtime_stream ullm_runtime_stream;

uint32_t ullm_runtime_abi_version(void);

ullm_status ullm_runtime_get_last_error(char *buffer, size_t *buffer_len);

ullm_status ullm_runtime_get_device_count(uint32_t *count);

ullm_status ullm_runtime_get_device_info(uint32_t index, ullm_device_info *info);

ullm_status ullm_runtime_context_create(uint32_t device_index, ullm_runtime_context **context);

ullm_status ullm_runtime_context_destroy(ullm_runtime_context *context);

ullm_status ullm_runtime_context_device_info(
    const ullm_runtime_context *context,
    ullm_device_info *info);

ullm_status ullm_runtime_buffer_alloc(
    ullm_runtime_context *context,
    size_t bytes,
    ullm_runtime_buffer **buffer);

ullm_status ullm_runtime_buffer_destroy(ullm_runtime_buffer *buffer);

ullm_status ullm_runtime_buffer_size(
    const ullm_runtime_buffer *buffer,
    size_t *bytes);

ullm_status ullm_runtime_stream_create(
    ullm_runtime_context *context,
    ullm_runtime_stream **stream);

ullm_status ullm_runtime_stream_destroy(ullm_runtime_stream *stream);

ullm_status ullm_runtime_stream_synchronize(ullm_runtime_stream *stream);

ullm_status ullm_runtime_smoke_add_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    size_t count);

#ifdef __cplusplus
}
#endif

#endif
