// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ULLM_MOE_GFX1201_H
#define ULLM_MOE_GFX1201_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Internal static-HIP entry points used by ullm_runtime_api_moe.inc.
 * They accept device pointers and return zero on a launch/validation failure.
 * Public buffer bounds and backend validation stays in the C ABI layer. */
int ullm_moe_gfx1201_route_f32(
    const void* hidden_f32,
    const void* router_weights,
    uint32_t weight_dtype,
    size_t tokens,
    size_t hidden_size,
    size_t num_experts,
    size_t top_k,
    void* routing_scores_f32,
    void* selected_expert_ids_i32,
    void* boundary_tie_flags_u32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

int ullm_moe_gfx1201_gather_f32(
    const void* hidden_f32,
    size_t tokens,
    size_t hidden_size,
    size_t top_k,
    void* gathered_hidden_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

int ullm_moe_gfx1201_grouped_gemm_f32(
    const void* weights,
    uint32_t weight_dtype,
    const void* expert_ids_i32,
    const void* input_f32,
    size_t assignments,
    size_t num_experts,
    size_t rows_per_expert,
    size_t cols,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

int ullm_moe_gfx1201_gated_silu_f32(
    const void* gate_up_f32,
    size_t assignments,
    size_t intermediate_size,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

int ullm_moe_gfx1201_scatter_weighted_f32(
    const void* expert_output_f32,
    const void* routing_scores_f32,
    size_t tokens,
    size_t top_k,
    size_t hidden_size,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

int ullm_moe_gfx1201_sigmoid_gate_f32(
    const void* gate_f32,
    const void* input_f32,
    size_t tokens,
    size_t hidden_size,
    void* output_f32,
    void* stream,
    int device_id,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
