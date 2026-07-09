// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_SQ8_CK_GFX1201_H
#define ULLM_SQ8_CK_GFX1201_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int ullm_sq8_ck_gfx1201_quantize_activation(
    const void *input,
    void *quantized,
    void *scales,
    size_t m,
    size_t k,
    void *stream,
    int device_id,
    char *error,
    size_t error_capacity);

int ullm_sq8_ck_gfx1201_projection(
    const void *quantized_activation,
    const void *activation_scales,
    const void *weight,
    const void *weight_scales,
    size_t m,
    size_t n,
    size_t k,
    void *workspace,
    void *output,
    void *stream,
    int device_id,
    uint32_t implementation,
    char *error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
