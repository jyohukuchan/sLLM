// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_SQ8_FP32_GPU_REFERENCE_GFX1201_H
#define ULLM_SQ8_FP32_GPU_REFERENCE_GFX1201_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Deliberately independent, non-production F32 control for the canonical
 * Qwen3-14B SQ8_0 artifact.  It owns HIP allocations directly and must not
 * be routed through the optimized SQ8 runtime/CK/WMMA dispatchers.
 */
struct ullm_sq8_fp32_gpu_reference_gfx1201_session;

struct ullm_sq8_fp32_gpu_reference_gfx1201_device_info {
    uint64_t total_global_mem_bytes;
    uint64_t free_global_mem_bytes;
    char name[128];
    char gcn_arch_name[64];
    char pci_bdf[32];
};

int ullm_sq8_fp32_gpu_reference_gfx1201_create(
    size_t max_context,
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session** session,
    char* error,
    size_t error_capacity);

void ullm_sq8_fp32_gpu_reference_gfx1201_destroy(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session);

int ullm_sq8_fp32_gpu_reference_gfx1201_device_info(
    const struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    struct ullm_sq8_fp32_gpu_reference_gfx1201_device_info* info,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_reserve_sq8_weight(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* tensor_name,
    size_t rows,
    size_t cols,
    const void* scales_bf16,
    size_t scale_bytes,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_upload_sq8_weight_chunk(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* tensor_name,
    size_t offset_bytes,
    const void* source,
    size_t bytes,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_reserve_bf16_tensor(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* slot,
    size_t elements,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_upload_bf16_tensor_chunk(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* slot,
    size_t offset_bytes,
    const void* source,
    size_t bytes,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_upload_layer_norms(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    size_t layer_index,
    const void* input_norm_bf16,
    const void* post_attention_norm_bf16,
    const void* q_norm_bf16,
    const void* k_norm_bf16,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_upload_final_norm(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const void* final_norm_bf16,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_finalize_model(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_forward(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    uint32_t token_id,
    float* logits_f32,
    size_t logits_elements,
    float* final_hidden_f32,
    size_t final_hidden_elements,
    float* layer_hidden_f32,
    size_t layer_hidden_elements,
    char* error,
    size_t error_capacity);

int ullm_sq8_fp32_gpu_reference_gfx1201_reset(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
