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

ullm_status ullm_runtime_aq4_row_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_batch_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_top1_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_add_f32(
    const ullm_runtime_buffer *index_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *codebook_buffer,
    const ullm_runtime_buffer *scale_values_buffer,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *residual_buffer,
    const ullm_runtime_buffer *row_scale_buffer,
    size_t scale_count,
    size_t group_size,
    float tensor_scale,
    size_t row_scale_count,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_pair_f32(
    const ullm_runtime_buffer *left_index_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    const ullm_runtime_buffer *left_codebook_buffer,
    const ullm_runtime_buffer *left_scale_values_buffer,
    const ullm_runtime_buffer *left_row_scale_buffer,
    size_t left_scale_count,
    size_t left_group_size,
    float left_tensor_scale,
    size_t left_row_scale_count,
    const ullm_runtime_buffer *right_index_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    const ullm_runtime_buffer *right_codebook_buffer,
    const ullm_runtime_buffer *right_scale_values_buffer,
    const ullm_runtime_buffer *right_row_scale_buffer,
    size_t right_scale_count,
    size_t right_group_size,
    float right_tensor_scale,
    size_t right_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_triple_f32(
    const ullm_runtime_buffer *first_index_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    const ullm_runtime_buffer *first_codebook_buffer,
    const ullm_runtime_buffer *first_scale_values_buffer,
    const ullm_runtime_buffer *first_row_scale_buffer,
    size_t first_scale_count,
    size_t first_group_size,
    float first_tensor_scale,
    size_t first_row_scale_count,
    const ullm_runtime_buffer *second_index_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    const ullm_runtime_buffer *second_codebook_buffer,
    const ullm_runtime_buffer *second_scale_values_buffer,
    const ullm_runtime_buffer *second_row_scale_buffer,
    size_t second_scale_count,
    size_t second_group_size,
    float second_tensor_scale,
    size_t second_row_scale_count,
    const ullm_runtime_buffer *third_index_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    const ullm_runtime_buffer *third_codebook_buffer,
    const ullm_runtime_buffer *third_scale_values_buffer,
    const ullm_runtime_buffer *third_row_scale_buffer,
    size_t third_scale_count,
    size_t third_group_size,
    float third_tensor_scale,
    size_t third_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_qkv_z_gate_beta_f32(
    const ullm_runtime_buffer *qkv_index_buffer,
    const ullm_runtime_buffer *qkv_scale_buffer,
    const ullm_runtime_buffer *qkv_codebook_buffer,
    const ullm_runtime_buffer *qkv_scale_values_buffer,
    const ullm_runtime_buffer *qkv_row_scale_buffer,
    size_t qkv_scale_count,
    size_t qkv_group_size,
    float qkv_tensor_scale,
    size_t qkv_row_scale_count,
    const ullm_runtime_buffer *z_index_buffer,
    const ullm_runtime_buffer *z_scale_buffer,
    const ullm_runtime_buffer *z_codebook_buffer,
    const ullm_runtime_buffer *z_scale_values_buffer,
    const ullm_runtime_buffer *z_row_scale_buffer,
    size_t z_scale_count,
    size_t z_group_size,
    float z_tensor_scale,
    size_t z_row_scale_count,
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t qkv_rows,
    size_t z_rows,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *qkv_output_buffer,
    ullm_runtime_buffer *z_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_silu_mul_f32(
    const ullm_runtime_buffer *gate_index_buffer,
    const ullm_runtime_buffer *gate_scale_buffer,
    const ullm_runtime_buffer *gate_codebook_buffer,
    const ullm_runtime_buffer *gate_scale_values_buffer,
    const ullm_runtime_buffer *gate_row_scale_buffer,
    size_t gate_scale_count,
    size_t gate_group_size,
    float gate_tensor_scale,
    size_t gate_row_scale_count,
    const ullm_runtime_buffer *up_index_buffer,
    const ullm_runtime_buffer *up_scale_buffer,
    const ullm_runtime_buffer *up_codebook_buffer,
    const ullm_runtime_buffer *up_scale_values_buffer,
    const ullm_runtime_buffer *up_row_scale_buffer,
    size_t up_scale_count,
    size_t up_group_size,
    float up_tensor_scale,
    size_t up_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_aq4_matvec_gate_beta_f32(
    const ullm_runtime_buffer *a_index_buffer,
    const ullm_runtime_buffer *a_scale_buffer,
    const ullm_runtime_buffer *a_codebook_buffer,
    const ullm_runtime_buffer *a_scale_values_buffer,
    const ullm_runtime_buffer *a_row_scale_buffer,
    size_t a_scale_count,
    size_t a_group_size,
    float a_tensor_scale,
    size_t a_row_scale_count,
    const ullm_runtime_buffer *b_index_buffer,
    const ullm_runtime_buffer *b_scale_buffer,
    const ullm_runtime_buffer *b_codebook_buffer,
    const ullm_runtime_buffer *b_scale_values_buffer,
    const ullm_runtime_buffer *b_row_scale_buffer,
    size_t b_scale_count,
    size_t b_group_size,
    float b_tensor_scale,
    size_t b_row_scale_count,
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *a_log_buffer,
    const ullm_runtime_buffer *dt_bias_buffer,
    size_t heads,
    size_t cols,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_buffer *beta_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_matvec_f32(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_matvec_bf16_f32(
    const ullm_runtime_buffer *matrix_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_bf16_row_f32(
    const ullm_runtime_buffer *matrix_buffer,
    size_t rows,
    size_t cols,
    size_t row_index,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_top1_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_top1_pairs_f32(
    const ullm_runtime_buffer *values_buffer,
    const ullm_runtime_buffer *indices_buffer,
    size_t elements,
    ullm_runtime_buffer *partial_values_buffer,
    ullm_runtime_buffer *partial_indices_buffer,
    size_t partial_count,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_rmsnorm_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t elements,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_segmented_rmsnorm_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_segmented_rmsnorm_silu_mul_f32(
    const ullm_runtime_buffer *input_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *gate_buffer,
    size_t segments,
    size_t segment_size,
    float epsilon,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_silu_mul_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *up_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_sigmoid_mul_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t elements,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_qwen35_split_q_gate_f32(
    const ullm_runtime_buffer *projected_buffer,
    size_t q_heads,
    size_t head_dim,
    ullm_runtime_buffer *query_output_buffer,
    ullm_runtime_buffer *gate_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_qwen35_qk_norm_rope_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_rope_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_qwen35_qk_norm_rope_paged_kv_write_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *v_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    float epsilon,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    ullm_runtime_buffer *q_gate_output_buffer,
    ullm_runtime_buffer *q_rope_output_buffer,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
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

ullm_status ullm_runtime_rope_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t sequence_len,
    size_t heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t position_offset,
    float rope_base,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_causal_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_decode_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cache_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_paged_decode_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_paged_decode_attn_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_len,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_paged_kv_write_f32(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_position,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_linear_attn_qkv_prepare_f32(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    float q_scale,
    int qk_l2_norm,
    ullm_runtime_buffer *conv_output_buffer,
    ullm_runtime_buffer *q_output_buffer,
    ullm_runtime_buffer *k_output_buffer,
    ullm_runtime_buffer *v_output_buffer,
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
