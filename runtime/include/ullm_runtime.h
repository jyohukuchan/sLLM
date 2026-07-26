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

typedef enum ullm_sq_fp8_execution_path {
    ULLM_SQ_FP8_EXECUTION_PATH_CPU_REFERENCE = 0,
    ULLM_SQ_FP8_EXECUTION_PATH_HIP_KERNEL = 1,
} ullm_sq_fp8_execution_path;

typedef enum ullm_sq8_ck_implementation {
    ULLM_SQ8_CK_IMPLEMENTATION_UNAVAILABLE = 0,
    ULLM_SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X128 = 1,
    ULLM_SQ8_CK_MEM_V1_KPADDING_TILE_16X128X256 = 2,
    ULLM_SQ8_CK_MEM_V1_DEFAULT_TILE_16X256X128 = 3,
    ULLM_SQ8_CK_MEM_V1_DEFAULT_TILE_16X128X256 = 4,
} ullm_sq8_ck_implementation;

/*
 * Storage type for MoE matrix tensors. MoE activations, routing scores, and
 * outputs are always F32 in the correctness-first ABI; only the checkpoint
 * weights use this selector. The BF16 value is raw IEEE BF16, little-endian,
 * exactly as stored in safetensors.
 */
typedef enum ullm_moe_weight_dtype {
    ULLM_MOE_WEIGHT_DTYPE_F32 = 0,
    ULLM_MOE_WEIGHT_DTYPE_BF16 = 1,
} ullm_moe_weight_dtype;

/*
 * Stable, test-visible AQ4 batch dispatch classification. New values are additive;
 * the existing ABI remains version 1 because no function signature or prior value changed.
 */
typedef enum ullm_aq4_matvec_batch_dispatch_kind {
    ULLM_AQ4_MATVEC_BATCH_DISPATCH_LEGACY = 0,
    ULLM_AQ4_MATVEC_BATCH_DISPATCH_TILED_LDS_BM8 = 1,
    ULLM_AQ4_MATVEC_BATCH_DISPATCH_TILED = ULLM_AQ4_MATVEC_BATCH_DISPATCH_TILED_LDS_BM8,
    ULLM_AQ4_MATVEC_BATCH_DISPATCH_REGISTER_BM4 = 2,
    ULLM_AQ4_MATVEC_BATCH_DISPATCH_REGISTER_BM8 = 3,
    ULLM_AQ4_MATVEC_BATCH_DISPATCH_REGISTER_BM8_GROUP8 = 4,
} ullm_aq4_matvec_batch_dispatch_kind;

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

ullm_status ullm_runtime_buffer_zero(
    ullm_runtime_buffer *buffer,
    size_t offset,
    size_t bytes,
    ullm_runtime_stream *stream);

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

ullm_status ullm_runtime_buffer_copy(
    ullm_runtime_buffer *dst_buffer,
    size_t dst_offset,
    const ullm_runtime_buffer *src_buffer,
    size_t src_offset,
    size_t bytes,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_stream_create(
    ullm_runtime_context *context,
    ullm_runtime_stream **stream);

ullm_status ullm_runtime_stream_destroy(ullm_runtime_stream *stream);

ullm_status ullm_runtime_stream_synchronize(ullm_runtime_stream *stream);

/*
 * Enqueues exact OCP E4M3 RNE row-by-K128 activation quantization from F32
 * input[M,K] to byte output[M,K] and F32 scales[M,K/128]. The outputs may be
 * reused by multiple projection calls on the same stream.
 */
ullm_status ullm_runtime_sq8_ck_quantize_activation_f32(
    const ullm_runtime_buffer *input_buffer,
    size_t m,
    size_t k,
    ullm_runtime_buffer *quantized_buffer,
    ullm_runtime_buffer *scale_buffer,
    ullm_runtime_stream *stream);

/*
 * Enqueues a measured gfx1201 CK ABScale projection from quantized A[M,K],
 * F32 A scales[M,K/128], canonical FP8 weight[N,K], and F32 weight scales
 * [N/128,K/128]. It then converts workspace_buffer[M,N] from BF16 into F32
 * output_buffer[M,N] on the same stream.
 */
ullm_status ullm_runtime_sq8_ck_projection_f32(
    const ullm_runtime_buffer *quantized_activation_buffer,
    const ullm_runtime_buffer *activation_scale_buffer,
    const ullm_runtime_buffer *weight_buffer,
    const ullm_runtime_buffer *weight_scale_buffer,
    size_t m,
    size_t n,
    size_t k,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    ullm_sq8_ck_implementation *implementation);

ullm_status ullm_runtime_wmma_fp8_probe(
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_wmma_fp8_qk_probe(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_rocwmma_fp8_qk_probe(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_rocwmma_fp8_attn_probe(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

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

/*
 * Direct-only gfx1201 M=1 AQ4 experiment. It uses sixteen-byte uint4 packed-index loads for
 * complete 32-element chunks and has no CPU path or production-dispatch admission.
 */
ullm_status ullm_runtime_aq4_matvec_wide_load_prototype_f32(
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

/*
 * Direct-only gfx1201 M=1 AQ4 experiment. It preserves the promoted uint4 packed-index loads,
 * but fixes RPB=32 and reduces each row with a width-8 sub-wave shuffle. It has no CPU path or
 * production-dispatch admission.
 */
ullm_status ullm_runtime_aq4_matvec_shuffle_prototype_f32(
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

/*
 * Directly launches the cached gfx1201/group16 register BM8 kernel. This additive ABI has no
 * environment dependency and never falls back to another implementation. Unsupported backend
 * or geometry is rejected before launch, so ULLM_RUNTIME_ABI_VERSION remains 1.
 */
ullm_status ullm_runtime_aq4_matvec_batch_register_bm8_f32(
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

/*
 * Directly launches the cached gfx1201/group8 register BM8 kernel. This additive ABI has no
 * environment dependency and never falls back to another implementation. Unsupported backend
 * or geometry is rejected before launch, so ULLM_RUNTIME_ABI_VERSION remains 1.
 */
ullm_status ullm_runtime_aq4_matvec_batch_register_bm8_group8_f32(
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

/*
 * Direct production gfx1201 rocWMMA AQ4 group16 GEMM for M=128. This stable ABI invokes the
 * double-buffered HIPRTC kernel without a fallback and accepts only nonzero shapes with rows
 * divisible by 16 and cols divisible by 32; those constraints preserve complete 16-row output
 * tiles and the Wide-K 16-byte source loads.
 */
ullm_status ullm_runtime_aq4_matvec_batch_wmma_prototype_f32(
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

/*
 * Direct-only gfx1201/group16 ragged-M WMMA experiment. The kernel always computes one physical
 * M=128 CTA, while m_actual (1..128) controls its guarded activation loads and output stores.
 * It is isolated from production dispatch and accepts the same complete 16-row/Wide-K geometry.
 */
ullm_status ullm_runtime_aq4_matvec_batch_wmma_ragged_m_prototype_f32(
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
    size_t m_actual,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/*
 * Isolated v3 experiment for the gfx1201/group16/M=128 WMMA GEMM. It uses two 32-K Wide-K
 * microtiles per LDS stage and therefore additionally requires cols divisible by 64. It has its
 * own HIPRTC module/cache, has no fallback, and does not affect production dispatch.
 */
ullm_status ullm_runtime_aq4_matvec_batch_wmma_prototype_v3_f32(
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

/*
 * Direct production gfx1201 rocWMMA AQ4 group8 GEMM for M=128. This stable ABI uses a separate
 * HIPRTC module and accepts only nonzero shapes with rows divisible by 16 and cols divisible by
 * 32; those constraints preserve complete 16-row output tiles and the Wide-K 16-byte source
 * loads. It has no fallback.
 */
ullm_status ullm_runtime_aq4_matvec_batch_wmma_group8_prototype_f32(
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

/*
 * Direct-only gfx1201/group8 ragged-M WMMA experiment. The kernel computes one physical M=128
 * CTA while m_actual (1..128) guards activation loads and final output stores, allowing exact
 * logical-M input/output buffers. It is independent of the M=128 group8 production module.
 */
ullm_status ullm_runtime_aq4_matvec_batch_wmma_group8_ragged_m_prototype_f32(
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
    size_t m_actual,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/*
 * Classifies the active AQ4 batch path, including explicit experiment environment gates,
 * without launching or timing a kernel.
 * device_index follows ullm_runtime_context_create: 0 is CPU and HIP devices start at 1.
 */
ullm_aq4_matvec_batch_dispatch_kind ullm_runtime_aq4_matvec_batch_dispatch_kind_for_shape(
    uint32_t device_index,
    size_t group_size,
    size_t rows,
    size_t cols,
    size_t batch_count);

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

/*
 * Direct-only gfx1201 AQ4 matvec-add experiment. It uses aligned sixteen-byte uint4 loads for
 * complete group8/group16 32-element chunks and is never selected by production dispatch.
 */
ullm_status ullm_runtime_aq4_matvec_add_wide_load_prototype_f32(
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

/*
 * Direct-only gfx1201 AQ4 matvec-add experiment. It retains the promoted uint4 weight loads and
 * residual-add epilogue, while replacing only the exact-RPB=8 per-row LDS reduction with shuffles.
 */
ullm_status ullm_runtime_aq4_matvec_add_shuffle_prototype_f32(
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

/* Direct-only gfx1201 experiment: scalar packed loads and wave32 reductions for three streams. */
ullm_status ullm_runtime_aq4_matvec_triple_shuffle_prototype_f32(
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, size_t, size_t, size_t, size_t,
    ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_stream *);

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

/* Direct-only gfx1201 M=1 experiment: all four packed streams use aligned uint4 loads. */
ullm_status ullm_runtime_aq4_matvec_qkv_z_gate_beta_wide_load_prototype_f32(
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, size_t, size_t,
    ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_stream *);

/* Direct-only gfx1201 M=1 experiment: scalar packed loads, wave32 reductions, and a two-wave LDS combine. */
ullm_status ullm_runtime_aq4_matvec_qkv_z_gate_beta_shuffle_prototype_f32(
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, size_t, size_t,
    ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_buffer *, ullm_runtime_stream *);

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

/* Direct-only gfx1201 M=1 experiment: gate and up packed streams use aligned uint4 loads. */
ullm_status ullm_runtime_aq4_matvec_silu_mul_wide_load_prototype_f32(
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, size_t, size_t, ullm_runtime_buffer *, ullm_runtime_stream *);

/* Direct-only gfx1201 M=1 experiment: scalar packed loads and wave32 reductions for gate/up. */
ullm_status ullm_runtime_aq4_matvec_silu_mul_shuffle_prototype_f32(
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, const ullm_runtime_buffer *, size_t, size_t, float, size_t,
    const ullm_runtime_buffer *, size_t, size_t, ullm_runtime_buffer *, ullm_runtime_stream *);

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

ullm_status ullm_runtime_sq_fp8_matvec_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_sq_fp8_matvec_batch_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    uint32_t scale_kind,
    size_t scale_block_cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_sq_fp8_matvec_block2d_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t scale_block_rows,
    size_t scale_block_cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    ullm_sq_fp8_execution_path *execution_path);

ullm_status ullm_runtime_sq_fp8_matvec_block2d_batch_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t scale_block_rows,
    size_t scale_block_cols,
    size_t batch_count,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream,
    ullm_sq_fp8_execution_path *execution_path);

ullm_status ullm_runtime_sq_fp8_matvec_pair_f32(
    const ullm_runtime_buffer *left_payload_buffer,
    const ullm_runtime_buffer *left_scale_buffer,
    uint32_t left_scale_kind,
    size_t left_scale_block_cols,
    const ullm_runtime_buffer *right_payload_buffer,
    const ullm_runtime_buffer *right_scale_buffer,
    uint32_t right_scale_kind,
    size_t right_scale_block_cols,
    const ullm_runtime_buffer *input_buffer,
    size_t left_rows,
    size_t right_rows,
    size_t cols,
    ullm_runtime_buffer *left_output_buffer,
    ullm_runtime_buffer *right_output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_sq_fp8_matvec_triple_f32(
    const ullm_runtime_buffer *first_payload_buffer,
    const ullm_runtime_buffer *first_scale_buffer,
    uint32_t first_scale_kind,
    size_t first_scale_block_cols,
    const ullm_runtime_buffer *second_payload_buffer,
    const ullm_runtime_buffer *second_scale_buffer,
    uint32_t second_scale_kind,
    size_t second_scale_block_cols,
    const ullm_runtime_buffer *third_payload_buffer,
    const ullm_runtime_buffer *third_scale_buffer,
    uint32_t third_scale_kind,
    size_t third_scale_block_cols,
    const ullm_runtime_buffer *input_buffer,
    size_t first_rows,
    size_t second_rows,
    size_t third_rows,
    size_t cols,
    ullm_runtime_buffer *first_output_buffer,
    ullm_runtime_buffer *second_output_buffer,
    ullm_runtime_buffer *third_output_buffer,
    ullm_runtime_stream *stream);

/*
 * SQ8_1 payload is signed I8 row-major with payload_row_stride=round_up(cols,16).
 * Scale is a separate little-endian FP16 [rows,ceil(cols/32)] plane.  This is the
 * default W8A16 path; it does not quantize activations.
 */
ullm_status ullm_runtime_sq8_1_matvec_w8a16_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/*
 * Explicit-only SQ8_1 W8A8 reference path.  It quantizes input in K=32 blocks
 * and applies signed-I32 dot * s_w * s_a.  It is intentionally separate from
 * the default W8A16 entry point while full-model W8A8 quality remains gated.
 */
ullm_status ullm_runtime_sq8_1_matvec_w8a8_explicit_f32(
    const ullm_runtime_buffer *payload_buffer,
    const ullm_runtime_buffer *scale_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t rows,
    size_t cols,
    size_t payload_row_stride,
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

/*
 * Qwen3.5 MoE correctness-first primitives.
 *
 * The router weight layout is [num_experts, hidden_size]. `routing_scores`
 * and `selected_expert_ids` are [tokens, top_k], row-major. IDs are int32.
 * `boundary_tie_flags` is [tokens] uint32: zero means the kth boundary was
 * unique; one means that the kth probability tied with an unselected expert.
 * HF/PyTorch does not specify a stable tie order, so full forward execution
 * must fail closed when this flag is nonzero instead of assigning a private
 * semantic tie-break.
 */
ullm_status ullm_runtime_moe_route_f32(
    const ullm_runtime_buffer *hidden_buffer,
    const ullm_runtime_buffer *router_weight_buffer,
    ullm_moe_weight_dtype router_weight_dtype,
    size_t tokens,
    size_t hidden_size,
    size_t num_experts,
    size_t top_k,
    ullm_runtime_buffer *routing_scores_buffer,
    ullm_runtime_buffer *selected_expert_ids_buffer,
    ullm_runtime_buffer *boundary_tie_flags_buffer,
    ullm_runtime_stream *stream);

/* Repeats each token row `top_k` times, producing assignment-major [tokens*top_k, hidden_size]. */
ullm_status ullm_runtime_moe_gather_f32(
    const ullm_runtime_buffer *hidden_buffer,
    size_t tokens,
    size_t hidden_size,
    size_t top_k,
    ullm_runtime_buffer *gathered_hidden_buffer,
    ullm_runtime_stream *stream);

/*
 * Generic variable-group GEMM. `weight_buffer` is row-major
 * [num_experts, rows_per_expert, cols]. `expert_ids_buffer` is int32
 * [assignments]. Input is [assignments, cols], output is
 * [assignments, rows_per_expert]. Empty expert groups are valid because no
 * assignment names them.
 */
ullm_status ullm_runtime_moe_grouped_gemm_f32(
    const ullm_runtime_buffer *weight_buffer,
    ullm_moe_weight_dtype weight_dtype,
    const ullm_runtime_buffer *expert_ids_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t assignments,
    size_t num_experts,
    size_t rows_per_expert,
    size_t cols,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/* Splits [assignments, 2*intermediate_size] into gate/up halves and writes SiLU(gate)*up. */
ullm_status ullm_runtime_moe_gated_silu_f32(
    const ullm_runtime_buffer *gate_up_buffer,
    size_t assignments,
    size_t intermediate_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/* Reduces assignment-major [tokens*top_k, hidden_size] using [tokens, top_k] routing scores. */
ullm_status ullm_runtime_moe_scatter_weighted_f32(
    const ullm_runtime_buffer *expert_output_buffer,
    const ullm_runtime_buffer *routing_scores_buffer,
    size_t tokens,
    size_t top_k,
    size_t hidden_size,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/* Broadcasts one shared-expert gate per token: sigmoid(gate[t]) * input[t,:]. */
ullm_status ullm_runtime_moe_sigmoid_gate_f32(
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *input_buffer,
    size_t tokens,
    size_t hidden_size,
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

/*
 * Direct-only gfx1201 experiment for Qwen3.5's M=1 hidden=4096 RMSNorm. This never changes
 * production RMSNorm dispatch and never falls back; it is for differential and timing checks.
 */
ullm_status ullm_runtime_rmsnorm_shuffle_prototype_f32(
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

/* Direct-only gfx1201 experiment for Qwen3.5's M=1 linear-attention post shape
 * (32 value-head segments, each 128 values). This never changes production dispatch and never
 * falls back; it is solely for isolated differential and timing comparisons. */
ullm_status ullm_runtime_segmented_rmsnorm_silu_mul_shuffle_prototype_f32(
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

ullm_status ullm_runtime_qwen35_qk_norm_rope_batch_f32(
    const ullm_runtime_buffer *q_projected_buffer,
    const ullm_runtime_buffer *k_projected_buffer,
    const ullm_runtime_buffer *q_weight_buffer,
    const ullm_runtime_buffer *k_weight_buffer,
    size_t q_heads,
    size_t kv_heads,
    size_t sequence_len,
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

ullm_status ullm_runtime_causal_attn_f32_flash2(
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

ullm_status ullm_runtime_causal_attn_batch_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t batch_count,
    size_t sequence_len,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_causal_attn_batch_f32_flash2(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    size_t batch_count,
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

ullm_status ullm_runtime_cached_prefix_attn_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_cached_prefix_attn_f32_flash2(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2_fp8q(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_cached_prefix_attn_fp8_e4m3_rocwmma(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    size_t cached_prefix_len,
    size_t new_tokens,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    float q_scale,
    float k_scale,
    float v_scale,
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

ullm_status ullm_runtime_paged_decode_attn_split_f32(
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
    size_t source_tile,
    ullm_runtime_buffer *workspace_buffer,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_paged_decode_attn_split_sigmoid_gate_f32(
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
    size_t source_tile,
    ullm_runtime_buffer *workspace_buffer,
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

/*
 * Writes a contiguous M-token K/V chunk into the logical paged cache. K and V
 * use [M,kv_heads,dim] row-major layout. cache_start is the logical position
 * of row zero; block_table maps logical blocks to physical cache blocks.
 */
ullm_status ullm_runtime_paged_kv_write_chunk_f32(
    const ullm_runtime_buffer *k_buffer,
    const ullm_runtime_buffer *v_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cache_start,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    ullm_runtime_buffer *k_cache_buffer,
    ullm_runtime_buffer *v_cache_buffer,
    ullm_runtime_stream *stream);

/*
 * Computes causal GQA attention for an M-token query chunk against a paged
 * cache. Query row i attends to logical source positions
 * [0,cached_prefix_len+i+1). Q is [M,q_heads,head_dim], output is
 * [M,q_heads,value_dim], and no M-by-context workspace is required.
 */
ullm_status ullm_runtime_paged_causal_gqa_chunk_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/* As above, then multiplies each output element by sigmoid(gate). gate follows
 * the existing M=1 contract [M,q_heads,head_dim]; the gated ABI requires
 * head_dim == value_dim so the gate and output element indexing agree. */
ullm_status ullm_runtime_paged_causal_gqa_chunk_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/*
 * Production, direct-only gfx1201 rocWMMA Qwen3.5 cold-prefill implementation. It accepts the
 * same paged F32 buffer layouts as the scalar reader, with validated M=1..=128 chunks and
 * Q=16, KV=4, head_dim=value_dim=256, and a 256-token page. It uses FP16 Q/K staging and FP32
 * online softmax/output accumulation; AV deliberately remains scalar.
 */
ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/* As above, then applies the Qwen3.5 self-attention output gate, sigmoid(gate), to the final
 * normalized F32 attention output. The gate ABI is [M,q_heads,head_dim]. */
ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

/*
 * Compatibility aliases for the validate-first direct ABI names exported before this kernel was
 * promoted. They retain the production M=1..=128 contract and forward to the entries above, keeping
 * this additive runtime ABI at version 1 for existing callers.
 */
ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_prototype_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
    ullm_runtime_stream *stream);

ullm_status ullm_runtime_paged_causal_gqa_chunk_wmma_prototype_sigmoid_gate_f32(
    const ullm_runtime_buffer *q_buffer,
    const ullm_runtime_buffer *gate_buffer,
    const ullm_runtime_buffer *k_cache_buffer,
    const ullm_runtime_buffer *v_cache_buffer,
    const ullm_runtime_buffer *block_table_buffer,
    size_t cached_prefix_len,
    size_t m,
    size_t block_size,
    size_t cache_blocks,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t value_dim,
    float softmax_scale,
    ullm_runtime_buffer *output_buffer,
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

/* Direct-only gfx1201 Qwen3.5 M=1 experiment. This never changes production prepare dispatch
 * or stages through CPU; it is solely for isolated differential and timing comparisons. */
ullm_status ullm_runtime_linear_attn_qkv_prepare_shuffle_prototype_f32(
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

ullm_status ullm_runtime_linear_attn_qkv_prepare_batch_f32(
    const ullm_runtime_buffer *qkv_buffer,
    const ullm_runtime_buffer *conv_weight_buffer,
    ullm_runtime_buffer *conv_history_buffer,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    size_t kernel_size,
    size_t sequence_len,
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
