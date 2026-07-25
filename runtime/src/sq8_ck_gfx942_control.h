// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_SQ8_CK_GFX942_CONTROL_H
#define ULLM_SQ8_CK_GFX942_CONTROL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * B control path: canonical SQ8_0 OCP bytes are decoded directly to BF16 and
 * multiplied by hipBLAS.  This function never converts to FNUZ and never
 * calls the CK FP8 MFMA operation used by A′.
 */
int ullm_sq8_ck_gfx942_control_dequant_ocp_bf16_projection(
    const void* activation_ocp_bytes,
    const void* activation_scale_f32,
    const void* weight_ocp_bytes,
    const void* weight_scale_f32,
    size_t m,
    size_t n,
    size_t k,
    void* activation_bf16,
    void* weight_bf16,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
