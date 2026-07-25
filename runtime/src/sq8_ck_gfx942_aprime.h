// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_SQ8_CK_GFX942_APRIME_H
#define ULLM_SQ8_CK_GFX942_APRIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A′ is an opaque f8_ocp_t ABI reuse, not an OCP-FP8 semantic path.  Both
 * byte inputs must have been derived by the SQ8_0 OCP-to-FNUZ oracle before
 * they reach this function.  Every scale is independently precompensated x2
 * for its converted operand; the mathematical A×B scale product is x4.
 *
 * No raw canonical OCP payload is accepted by an A′ entry point.
 */
int ullm_sq8_ck_gfx942_aprime_projection_fnuz_prepacked(
    const void* activation_fnuz_prepacked_bytes,
    const void* activation_scale_f32_x2,
    const void* weight_fnuz_prepacked_bytes,
    const void* weight_scale_f32_x2,
    size_t m,
    size_t n,
    size_t k,
    void* workspace_bf16,
    void* output_f32,
    void* stream,
    int device_id,
    uint32_t implementation,
    char* error,
    size_t error_capacity);

/*
 * Physical-gfx942-only fragment diagnostic.  Inputs are raw E4M3FNUZ bytes in
 * A[16,32] row-major and B[32,16] column-major order.  It emits both the
 * logical 16x16 matrix and the four accumulator registers held by each of the
 * 64 lanes.  It deliberately has no OCP input parameter.
 */
int ullm_sq8_ck_gfx942_aprime_fragment_probe_fnuz(
    const void* a_fnuz_16x32_row_major,
    const void* b_fnuz_32x16_column_major,
    void* matrix_f32_16x16,
    void* fragment_f32_lane64x4,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
