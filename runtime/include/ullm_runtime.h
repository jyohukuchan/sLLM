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

ullm_status ullm_runtime_buffer_copy_from_host(
    ullm_runtime_buffer *buffer,
    size_t offset,
    const void *src,
    size_t bytes,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_buffer_copy_to_host(
    const ullm_runtime_buffer *buffer,
    size_t offset,
    void *dst,
    size_t bytes,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_stream_create(
    ullm_runtime_context *context,
    ullm_runtime_stream **stream);

ullm_status ullm_runtime_stream_destroy(ullm_runtime_stream *stream);

ullm_status ullm_runtime_stream_synchronize(ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_dequant_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const float *scale_values,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_matvec_f32(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_rmsnorm_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_silu_mul_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_add_f32(
    const ullm_runtime_buffer *lhs_buffer,
    const ullm_runtime_buffer *rhs_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_depthwise_conv1d_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t channels,
    size_t sequence_len,
    size_t kernel_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_linear_attn_gate_beta_f32(
    const ullm_runtime_buffer *a_buffer,
    const ullm_runtime_buffer *b_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t sequence_len,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_linear_attn_recurrent_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *beta_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t sequence_len,
    size_t key_dim,
    size_t value_dim,
    ullm_runtime_buffer *state_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_smoke_add_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    size_t count);

#ifdef __cplusplus
}
#endif

#endif
