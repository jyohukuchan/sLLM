// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_SQ8_HANDWRITTEN_GFX1201_H
#define ULLM_SQ8_HANDWRITTEN_GFX1201_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Internal SQ8_0 feasibility prototype only.  The payload inputs are the
 * canonical OCP E4M3 raw bytes. Activation scales are F32 values emitted by
 * the resident [1,128] CK activation quantizer; weight scales are F32 values
 * decoded from canonical BF16 [128,128] blocks. This header is intentionally
 * not installed and is absent from runtime/include/ullm_runtime.h.
 */
int ullm_sq8_handwritten_gfx1201_m1_wmma_projection(
    const void* quantized_activation_ocp_e4m3,
    const void* activation_scale_f32,
    const void* weight_ocp_e4m3,
    const void* weight_scale_f32,
    size_t n,
    size_t k,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

/* HIP-reported post-compilation resource allocation for the private kernel. */
int ullm_sq8_handwritten_gfx1201_m1_wmma_resources(
    int device_id,
    uint32_t* vgpr_per_thread,
    size_t* static_lds_bytes,
    size_t* local_bytes_per_thread,
    int* threads_per_block,
    int* active_blocks_per_cu,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
