// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::NonNull;

const STATUS_OK: c_int = 0;
const STATUS_INVALID_ARGUMENT: c_int = 1;
const STATUS_BUFFER_TOO_SMALL: c_int = 2;

enum RawRuntimeContext {}

enum RawRuntimeBuffer {}

enum RawRuntimeStream {}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawDeviceInfo {
    device_id: i32,
    backend: [c_char; 16],
    name: [c_char; 128],
    total_global_mem: u64,
    compute_major: i32,
    compute_minor: i32,
    gcn_arch_name: [c_char; 64],
    flags: u32,
}

unsafe extern "C" {
    fn ullm_runtime_abi_version() -> u32;
    fn ullm_runtime_get_last_error(buffer: *mut c_char, buffer_len: *mut usize) -> c_int;
    fn ullm_runtime_get_device_count(count: *mut u32) -> c_int;
    fn ullm_runtime_get_device_info(index: u32, info: *mut RawDeviceInfo) -> c_int;
    fn ullm_runtime_context_create(index: u32, context: *mut *mut RawRuntimeContext) -> c_int;
    fn ullm_runtime_context_destroy(context: *mut RawRuntimeContext) -> c_int;
    fn ullm_runtime_context_device_info(
        context: *const RawRuntimeContext,
        info: *mut RawDeviceInfo,
    ) -> c_int;
    fn ullm_runtime_buffer_alloc(
        context: *mut RawRuntimeContext,
        bytes: usize,
        buffer: *mut *mut RawRuntimeBuffer,
    ) -> c_int;
    fn ullm_runtime_buffer_destroy(buffer: *mut RawRuntimeBuffer) -> c_int;
    fn ullm_runtime_buffer_size(buffer: *const RawRuntimeBuffer, bytes: *mut usize) -> c_int;
    fn ullm_runtime_buffer_zero(
        buffer: *mut RawRuntimeBuffer,
        offset: usize,
        bytes: usize,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_buffer_copy_from_host(
        buffer: *mut RawRuntimeBuffer,
        offset: usize,
        src: *const c_void,
        bytes: usize,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_buffer_copy_to_host(
        buffer: *const RawRuntimeBuffer,
        offset: usize,
        dst: *mut c_void,
        bytes: usize,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_buffer_copy(
        dst_buffer: *mut RawRuntimeBuffer,
        dst_offset: usize,
        src_buffer: *const RawRuntimeBuffer,
        src_offset: usize,
        bytes: usize,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_stream_create(
        context: *mut RawRuntimeContext,
        stream: *mut *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_stream_destroy(stream: *mut RawRuntimeStream) -> c_int;
    fn ullm_runtime_stream_synchronize(stream: *mut RawRuntimeStream) -> c_int;
    fn ullm_runtime_wmma_fp8_probe(
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_wmma_fp8_qk_probe(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_rocwmma_fp8_qk_probe(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_rocwmma_fp8_attn_probe(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_dequant_f32(
        index_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        codebook_buffer: *const RawRuntimeBuffer,
        scale_values: *const f32,
        scale_count: usize,
        group_size: usize,
        tensor_scale: f32,
        elements: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_row_f32(
        index_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        codebook_buffer: *const RawRuntimeBuffer,
        scale_values_buffer: *const RawRuntimeBuffer,
        row_scale_buffer: *const RawRuntimeBuffer,
        scale_count: usize,
        group_size: usize,
        tensor_scale: f32,
        row_scale_count: usize,
        rows: usize,
        cols: usize,
        row_index: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_f32(
        index_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        codebook_buffer: *const RawRuntimeBuffer,
        scale_values_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        row_scale_buffer: *const RawRuntimeBuffer,
        scale_count: usize,
        group_size: usize,
        tensor_scale: f32,
        row_scale_count: usize,
        rows: usize,
        cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_batch_f32(
        index_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        codebook_buffer: *const RawRuntimeBuffer,
        scale_values_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        row_scale_buffer: *const RawRuntimeBuffer,
        scale_count: usize,
        group_size: usize,
        tensor_scale: f32,
        row_scale_count: usize,
        rows: usize,
        cols: usize,
        batch_count: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_batch_dispatch_kind_for_shape(
        device_index: u32,
        group_size: usize,
        rows: usize,
        cols: usize,
        batch_count: usize,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_top1_f32(
        index_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        codebook_buffer: *const RawRuntimeBuffer,
        scale_values_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        row_scale_buffer: *const RawRuntimeBuffer,
        scale_count: usize,
        group_size: usize,
        tensor_scale: f32,
        row_scale_count: usize,
        rows: usize,
        cols: usize,
        partial_values_buffer: *mut RawRuntimeBuffer,
        partial_indices_buffer: *mut RawRuntimeBuffer,
        partial_count: usize,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_add_f32(
        index_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        codebook_buffer: *const RawRuntimeBuffer,
        scale_values_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        residual_buffer: *const RawRuntimeBuffer,
        row_scale_buffer: *const RawRuntimeBuffer,
        scale_count: usize,
        group_size: usize,
        tensor_scale: f32,
        row_scale_count: usize,
        rows: usize,
        cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_pair_f32(
        left_index_buffer: *const RawRuntimeBuffer,
        left_scale_buffer: *const RawRuntimeBuffer,
        left_codebook_buffer: *const RawRuntimeBuffer,
        left_scale_values_buffer: *const RawRuntimeBuffer,
        left_row_scale_buffer: *const RawRuntimeBuffer,
        left_scale_count: usize,
        left_group_size: usize,
        left_tensor_scale: f32,
        left_row_scale_count: usize,
        right_index_buffer: *const RawRuntimeBuffer,
        right_scale_buffer: *const RawRuntimeBuffer,
        right_codebook_buffer: *const RawRuntimeBuffer,
        right_scale_values_buffer: *const RawRuntimeBuffer,
        right_row_scale_buffer: *const RawRuntimeBuffer,
        right_scale_count: usize,
        right_group_size: usize,
        right_tensor_scale: f32,
        right_row_scale_count: usize,
        input_buffer: *const RawRuntimeBuffer,
        left_rows: usize,
        right_rows: usize,
        cols: usize,
        left_output_buffer: *mut RawRuntimeBuffer,
        right_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_triple_f32(
        first_index_buffer: *const RawRuntimeBuffer,
        first_scale_buffer: *const RawRuntimeBuffer,
        first_codebook_buffer: *const RawRuntimeBuffer,
        first_scale_values_buffer: *const RawRuntimeBuffer,
        first_row_scale_buffer: *const RawRuntimeBuffer,
        first_scale_count: usize,
        first_group_size: usize,
        first_tensor_scale: f32,
        first_row_scale_count: usize,
        second_index_buffer: *const RawRuntimeBuffer,
        second_scale_buffer: *const RawRuntimeBuffer,
        second_codebook_buffer: *const RawRuntimeBuffer,
        second_scale_values_buffer: *const RawRuntimeBuffer,
        second_row_scale_buffer: *const RawRuntimeBuffer,
        second_scale_count: usize,
        second_group_size: usize,
        second_tensor_scale: f32,
        second_row_scale_count: usize,
        third_index_buffer: *const RawRuntimeBuffer,
        third_scale_buffer: *const RawRuntimeBuffer,
        third_codebook_buffer: *const RawRuntimeBuffer,
        third_scale_values_buffer: *const RawRuntimeBuffer,
        third_row_scale_buffer: *const RawRuntimeBuffer,
        third_scale_count: usize,
        third_group_size: usize,
        third_tensor_scale: f32,
        third_row_scale_count: usize,
        input_buffer: *const RawRuntimeBuffer,
        first_rows: usize,
        second_rows: usize,
        third_rows: usize,
        cols: usize,
        first_output_buffer: *mut RawRuntimeBuffer,
        second_output_buffer: *mut RawRuntimeBuffer,
        third_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_qkv_z_gate_beta_f32(
        qkv_index_buffer: *const RawRuntimeBuffer,
        qkv_scale_buffer: *const RawRuntimeBuffer,
        qkv_codebook_buffer: *const RawRuntimeBuffer,
        qkv_scale_values_buffer: *const RawRuntimeBuffer,
        qkv_row_scale_buffer: *const RawRuntimeBuffer,
        qkv_scale_count: usize,
        qkv_group_size: usize,
        qkv_tensor_scale: f32,
        qkv_row_scale_count: usize,
        z_index_buffer: *const RawRuntimeBuffer,
        z_scale_buffer: *const RawRuntimeBuffer,
        z_codebook_buffer: *const RawRuntimeBuffer,
        z_scale_values_buffer: *const RawRuntimeBuffer,
        z_row_scale_buffer: *const RawRuntimeBuffer,
        z_scale_count: usize,
        z_group_size: usize,
        z_tensor_scale: f32,
        z_row_scale_count: usize,
        a_index_buffer: *const RawRuntimeBuffer,
        a_scale_buffer: *const RawRuntimeBuffer,
        a_codebook_buffer: *const RawRuntimeBuffer,
        a_scale_values_buffer: *const RawRuntimeBuffer,
        a_row_scale_buffer: *const RawRuntimeBuffer,
        a_scale_count: usize,
        a_group_size: usize,
        a_tensor_scale: f32,
        a_row_scale_count: usize,
        b_index_buffer: *const RawRuntimeBuffer,
        b_scale_buffer: *const RawRuntimeBuffer,
        b_codebook_buffer: *const RawRuntimeBuffer,
        b_scale_values_buffer: *const RawRuntimeBuffer,
        b_row_scale_buffer: *const RawRuntimeBuffer,
        b_scale_count: usize,
        b_group_size: usize,
        b_tensor_scale: f32,
        b_row_scale_count: usize,
        input_buffer: *const RawRuntimeBuffer,
        a_log_buffer: *const RawRuntimeBuffer,
        dt_bias_buffer: *const RawRuntimeBuffer,
        qkv_rows: usize,
        z_rows: usize,
        heads: usize,
        cols: usize,
        qkv_output_buffer: *mut RawRuntimeBuffer,
        z_output_buffer: *mut RawRuntimeBuffer,
        gate_output_buffer: *mut RawRuntimeBuffer,
        beta_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_silu_mul_f32(
        gate_index_buffer: *const RawRuntimeBuffer,
        gate_scale_buffer: *const RawRuntimeBuffer,
        gate_codebook_buffer: *const RawRuntimeBuffer,
        gate_scale_values_buffer: *const RawRuntimeBuffer,
        gate_row_scale_buffer: *const RawRuntimeBuffer,
        gate_scale_count: usize,
        gate_group_size: usize,
        gate_tensor_scale: f32,
        gate_row_scale_count: usize,
        up_index_buffer: *const RawRuntimeBuffer,
        up_scale_buffer: *const RawRuntimeBuffer,
        up_codebook_buffer: *const RawRuntimeBuffer,
        up_scale_values_buffer: *const RawRuntimeBuffer,
        up_row_scale_buffer: *const RawRuntimeBuffer,
        up_scale_count: usize,
        up_group_size: usize,
        up_tensor_scale: f32,
        up_row_scale_count: usize,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_aq4_matvec_gate_beta_f32(
        a_index_buffer: *const RawRuntimeBuffer,
        a_scale_buffer: *const RawRuntimeBuffer,
        a_codebook_buffer: *const RawRuntimeBuffer,
        a_scale_values_buffer: *const RawRuntimeBuffer,
        a_row_scale_buffer: *const RawRuntimeBuffer,
        a_scale_count: usize,
        a_group_size: usize,
        a_tensor_scale: f32,
        a_row_scale_count: usize,
        b_index_buffer: *const RawRuntimeBuffer,
        b_scale_buffer: *const RawRuntimeBuffer,
        b_codebook_buffer: *const RawRuntimeBuffer,
        b_scale_values_buffer: *const RawRuntimeBuffer,
        b_row_scale_buffer: *const RawRuntimeBuffer,
        b_scale_count: usize,
        b_group_size: usize,
        b_tensor_scale: f32,
        b_row_scale_count: usize,
        input_buffer: *const RawRuntimeBuffer,
        a_log_buffer: *const RawRuntimeBuffer,
        dt_bias_buffer: *const RawRuntimeBuffer,
        heads: usize,
        cols: usize,
        gate_output_buffer: *mut RawRuntimeBuffer,
        beta_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_matvec_f32(
        matrix_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sq_fp8_matvec_f32(
        payload_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        scale_kind: u32,
        scale_block_cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sq_fp8_matvec_batch_f32(
        payload_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        scale_kind: u32,
        scale_block_cols: usize,
        batch_count: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sq_fp8_matvec_block2d_f32(
        payload_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        scale_block_rows: usize,
        scale_block_cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
        execution_path: *mut c_int,
    ) -> c_int;
    fn ullm_runtime_sq_fp8_matvec_block2d_batch_f32(
        payload_buffer: *const RawRuntimeBuffer,
        scale_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        scale_block_rows: usize,
        scale_block_cols: usize,
        batch_count: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
        execution_path: *mut c_int,
    ) -> c_int;
    fn ullm_runtime_sq_fp8_matvec_pair_f32(
        left_payload_buffer: *const RawRuntimeBuffer,
        left_scale_buffer: *const RawRuntimeBuffer,
        left_scale_kind: u32,
        left_scale_block_cols: usize,
        right_payload_buffer: *const RawRuntimeBuffer,
        right_scale_buffer: *const RawRuntimeBuffer,
        right_scale_kind: u32,
        right_scale_block_cols: usize,
        input_buffer: *const RawRuntimeBuffer,
        left_rows: usize,
        right_rows: usize,
        cols: usize,
        left_output_buffer: *mut RawRuntimeBuffer,
        right_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sq_fp8_matvec_triple_f32(
        first_payload_buffer: *const RawRuntimeBuffer,
        first_scale_buffer: *const RawRuntimeBuffer,
        first_scale_kind: u32,
        first_scale_block_cols: usize,
        second_payload_buffer: *const RawRuntimeBuffer,
        second_scale_buffer: *const RawRuntimeBuffer,
        second_scale_kind: u32,
        second_scale_block_cols: usize,
        third_payload_buffer: *const RawRuntimeBuffer,
        third_scale_buffer: *const RawRuntimeBuffer,
        third_scale_kind: u32,
        third_scale_block_cols: usize,
        input_buffer: *const RawRuntimeBuffer,
        first_rows: usize,
        second_rows: usize,
        third_rows: usize,
        cols: usize,
        first_output_buffer: *mut RawRuntimeBuffer,
        second_output_buffer: *mut RawRuntimeBuffer,
        third_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_matvec_bf16_f32(
        matrix_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_bf16_row_f32(
        matrix_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        row_index: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_top1_f32(
        input_buffer: *const RawRuntimeBuffer,
        elements: usize,
        partial_values_buffer: *mut RawRuntimeBuffer,
        partial_indices_buffer: *mut RawRuntimeBuffer,
        partial_count: usize,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_top1_pairs_f32(
        values_buffer: *const RawRuntimeBuffer,
        indices_buffer: *const RawRuntimeBuffer,
        elements: usize,
        partial_values_buffer: *mut RawRuntimeBuffer,
        partial_indices_buffer: *mut RawRuntimeBuffer,
        partial_count: usize,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_rmsnorm_f32(
        input_buffer: *const RawRuntimeBuffer,
        weight_buffer: *const RawRuntimeBuffer,
        elements: usize,
        epsilon: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_segmented_rmsnorm_f32(
        input_buffer: *const RawRuntimeBuffer,
        weight_buffer: *const RawRuntimeBuffer,
        segments: usize,
        segment_size: usize,
        epsilon: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_segmented_rmsnorm_silu_mul_f32(
        input_buffer: *const RawRuntimeBuffer,
        weight_buffer: *const RawRuntimeBuffer,
        gate_buffer: *const RawRuntimeBuffer,
        segments: usize,
        segment_size: usize,
        epsilon: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_silu_mul_f32(
        gate_buffer: *const RawRuntimeBuffer,
        up_buffer: *const RawRuntimeBuffer,
        elements: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sigmoid_mul_f32(
        gate_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        elements: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_qwen35_split_q_gate_f32(
        projected_buffer: *const RawRuntimeBuffer,
        q_heads: usize,
        head_dim: usize,
        query_output_buffer: *mut RawRuntimeBuffer,
        gate_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_qwen35_qk_norm_rope_f32(
        q_projected_buffer: *const RawRuntimeBuffer,
        k_projected_buffer: *const RawRuntimeBuffer,
        q_weight_buffer: *const RawRuntimeBuffer,
        k_weight_buffer: *const RawRuntimeBuffer,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position_offset: usize,
        rope_base: f32,
        epsilon: f32,
        q_gate_output_buffer: *mut RawRuntimeBuffer,
        q_rope_output_buffer: *mut RawRuntimeBuffer,
        k_rope_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_qwen35_qk_norm_rope_batch_f32(
        q_projected_buffer: *const RawRuntimeBuffer,
        k_projected_buffer: *const RawRuntimeBuffer,
        q_weight_buffer: *const RawRuntimeBuffer,
        k_weight_buffer: *const RawRuntimeBuffer,
        q_heads: usize,
        kv_heads: usize,
        sequence_len: usize,
        head_dim: usize,
        rotary_dim: usize,
        position_offset: usize,
        rope_base: f32,
        epsilon: f32,
        q_gate_output_buffer: *mut RawRuntimeBuffer,
        q_rope_output_buffer: *mut RawRuntimeBuffer,
        k_rope_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_qwen35_qk_norm_rope_paged_kv_write_f32(
        q_projected_buffer: *const RawRuntimeBuffer,
        k_projected_buffer: *const RawRuntimeBuffer,
        v_projected_buffer: *const RawRuntimeBuffer,
        q_weight_buffer: *const RawRuntimeBuffer,
        k_weight_buffer: *const RawRuntimeBuffer,
        block_table_buffer: *const RawRuntimeBuffer,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        rotary_dim: usize,
        position_offset: usize,
        rope_base: f32,
        epsilon: f32,
        cache_position: usize,
        block_size: usize,
        cache_blocks: usize,
        q_gate_output_buffer: *mut RawRuntimeBuffer,
        q_rope_output_buffer: *mut RawRuntimeBuffer,
        k_cache_buffer: *mut RawRuntimeBuffer,
        v_cache_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_add_f32(
        lhs_buffer: *const RawRuntimeBuffer,
        rhs_buffer: *const RawRuntimeBuffer,
        elements: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_rope_f32(
        input_buffer: *const RawRuntimeBuffer,
        sequence_len: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position_offset: usize,
        rope_base: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_causal_attn_f32(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        sequence_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_causal_attn_batch_f32(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        batch_count: usize,
        sequence_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_causal_attn_f32_flash2(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        sequence_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_causal_attn_batch_f32_flash2(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        batch_count: usize,
        sequence_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_decode_attn_f32(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        cache_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_cached_prefix_attn_f32(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_cached_prefix_attn_f32_flash2(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_cached_prefix_attn_fp8_e4m3(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        k_scale: f32,
        v_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        k_scale: f32,
        v_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2_fp8q(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_cached_prefix_attn_fp8_e4m3_rocwmma(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        new_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_paged_decode_attn_f32(
        q: *const RawRuntimeBuffer,
        k_cache: *const RawRuntimeBuffer,
        v_cache: *const RawRuntimeBuffer,
        block_table: *const RawRuntimeBuffer,
        cache_len: usize,
        block_size: usize,
        cache_blocks: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_paged_decode_attn_sigmoid_gate_f32(
        q: *const RawRuntimeBuffer,
        gate: *const RawRuntimeBuffer,
        k_cache: *const RawRuntimeBuffer,
        v_cache: *const RawRuntimeBuffer,
        block_table: *const RawRuntimeBuffer,
        cache_len: usize,
        block_size: usize,
        cache_blocks: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_paged_kv_write_f32(
        k: *const RawRuntimeBuffer,
        v: *const RawRuntimeBuffer,
        block_table: *const RawRuntimeBuffer,
        cache_position: usize,
        block_size: usize,
        cache_blocks: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        k_cache: *mut RawRuntimeBuffer,
        v_cache: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_paged_kv_write_chunk_f32(
        k: *const RawRuntimeBuffer,
        v: *const RawRuntimeBuffer,
        block_table: *const RawRuntimeBuffer,
        cache_start: usize,
        m: usize,
        block_size: usize,
        cache_blocks: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        k_cache: *mut RawRuntimeBuffer,
        v_cache: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_paged_causal_gqa_chunk_f32(
        q: *const RawRuntimeBuffer,
        k_cache: *const RawRuntimeBuffer,
        v_cache: *const RawRuntimeBuffer,
        block_table: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        m: usize,
        block_size: usize,
        cache_blocks: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_paged_causal_gqa_chunk_sigmoid_gate_f32(
        q: *const RawRuntimeBuffer,
        gate: *const RawRuntimeBuffer,
        k_cache: *const RawRuntimeBuffer,
        v_cache: *const RawRuntimeBuffer,
        block_table: *const RawRuntimeBuffer,
        cached_prefix_len: usize,
        m: usize,
        block_size: usize,
        cache_blocks: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
        output: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_linear_attn_qkv_prepare_f32(
        qkv_buffer: *const RawRuntimeBuffer,
        conv_weight_buffer: *const RawRuntimeBuffer,
        conv_history_buffer: *mut RawRuntimeBuffer,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        q_scale: f32,
        qk_l2_norm: c_int,
        conv_output_buffer: *mut RawRuntimeBuffer,
        q_output_buffer: *mut RawRuntimeBuffer,
        k_output_buffer: *mut RawRuntimeBuffer,
        v_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_linear_attn_qkv_prepare_batch_f32(
        qkv_buffer: *const RawRuntimeBuffer,
        conv_weight_buffer: *const RawRuntimeBuffer,
        conv_history_buffer: *mut RawRuntimeBuffer,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        sequence_len: usize,
        q_scale: f32,
        qk_l2_norm: c_int,
        conv_output_buffer: *mut RawRuntimeBuffer,
        q_output_buffer: *mut RawRuntimeBuffer,
        k_output_buffer: *mut RawRuntimeBuffer,
        v_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_linear_attn_gate_beta_f32(
        a_buffer: *const RawRuntimeBuffer,
        b_buffer: *const RawRuntimeBuffer,
        a_log_buffer: *const RawRuntimeBuffer,
        dt_bias_buffer: *const RawRuntimeBuffer,
        heads: usize,
        sequence_len: usize,
        gate_output_buffer: *mut RawRuntimeBuffer,
        beta_output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_linear_attn_recurrent_f32(
        q_buffer: *const RawRuntimeBuffer,
        k_buffer: *const RawRuntimeBuffer,
        v_buffer: *const RawRuntimeBuffer,
        gate_buffer: *const RawRuntimeBuffer,
        beta_buffer: *const RawRuntimeBuffer,
        key_heads: usize,
        value_heads: usize,
        sequence_len: usize,
        key_dim: usize,
        value_dim: usize,
        state_buffer: *mut RawRuntimeBuffer,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_depthwise_conv1d_f32(
        input_buffer: *const RawRuntimeBuffer,
        weight_buffer: *const RawRuntimeBuffer,
        channels: usize,
        sequence_len: usize,
        kernel_size: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_smoke_add_f32(
        lhs: *const f32,
        rhs: *const f32,
        out: *mut f32,
        count: usize,
    ) -> c_int;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub device_id: i32,
    pub backend: String,
    pub name: String,
    pub total_global_mem: u64,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub gcn_arch_name: String,
    pub flags: u32,
}

#[derive(Debug)]
pub struct RuntimeContext {
    raw: NonNull<RawRuntimeContext>,
}

#[derive(Debug)]
pub struct RuntimeBuffer {
    raw: NonNull<RawRuntimeBuffer>,
}

#[derive(Debug)]
pub struct RuntimeStream {
    raw: NonNull<RawRuntimeStream>,
}

pub fn abi_version() -> u32 {
    unsafe { ullm_runtime_abi_version() }
}

pub fn device_count() -> Result<u32, String> {
    let mut count = 0_u32;
    status_to_result(unsafe { ullm_runtime_get_device_count(&mut count) })?;
    Ok(count)
}

pub fn device_info(index: u32) -> Result<DeviceInfo, String> {
    let mut raw = RawDeviceInfo {
        device_id: 0,
        backend: [0; 16],
        name: [0; 128],
        total_global_mem: 0,
        compute_major: 0,
        compute_minor: 0,
        gcn_arch_name: [0; 64],
        flags: 0,
    };
    status_to_result(unsafe { ullm_runtime_get_device_info(index, &mut raw) })?;
    Ok(DeviceInfo {
        device_id: raw.device_id,
        backend: c_array_to_string(&raw.backend),
        name: c_array_to_string(&raw.name),
        total_global_mem: raw.total_global_mem,
        compute_major: raw.compute_major,
        compute_minor: raw.compute_minor,
        gcn_arch_name: c_array_to_string(&raw.gcn_arch_name),
        flags: raw.flags,
    })
}

impl RuntimeContext {
    pub fn create(device_index: u32) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        status_to_result(unsafe { ullm_runtime_context_create(device_index, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| "runtime returned a null context".to_string())?;
        Ok(Self { raw })
    }

    pub fn device_info(&self) -> Result<DeviceInfo, String> {
        let mut raw = RawDeviceInfo {
            device_id: 0,
            backend: [0; 16],
            name: [0; 128],
            total_global_mem: 0,
            compute_major: 0,
            compute_minor: 0,
            gcn_arch_name: [0; 64],
            flags: 0,
        };
        status_to_result(unsafe { ullm_runtime_context_device_info(self.raw.as_ptr(), &mut raw) })?;
        Ok(DeviceInfo {
            device_id: raw.device_id,
            backend: c_array_to_string(&raw.backend),
            name: c_array_to_string(&raw.name),
            total_global_mem: raw.total_global_mem,
            compute_major: raw.compute_major,
            compute_minor: raw.compute_minor,
            gcn_arch_name: c_array_to_string(&raw.gcn_arch_name),
            flags: raw.flags,
        })
    }

    pub fn alloc_buffer(&mut self, bytes: usize) -> Result<RuntimeBuffer, String> {
        let mut raw = std::ptr::null_mut();
        status_to_result(unsafe { ullm_runtime_buffer_alloc(self.raw.as_ptr(), bytes, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| "runtime returned a null buffer".to_string())?;
        Ok(RuntimeBuffer { raw })
    }

    pub fn create_stream(&mut self) -> Result<RuntimeStream, String> {
        let mut raw = std::ptr::null_mut();
        status_to_result(unsafe { ullm_runtime_stream_create(self.raw.as_ptr(), &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| "runtime returned a null stream".to_string())?;
        Ok(RuntimeStream { raw })
    }
}

impl Drop for RuntimeContext {
    fn drop(&mut self) {
        let _ = unsafe { ullm_runtime_context_destroy(self.raw.as_ptr()) };
    }
}

impl RuntimeBuffer {
    pub fn size(&self) -> Result<usize, String> {
        let mut bytes = 0_usize;
        status_to_result(unsafe { ullm_runtime_buffer_size(self.raw.as_ptr(), &mut bytes) })?;
        Ok(bytes)
    }

    pub fn zero(
        &mut self,
        offset: usize,
        bytes: usize,
        stream: Option<&mut RuntimeStream>,
    ) -> Result<(), String> {
        check_copy_range(offset, bytes, self.size()?)?;
        let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
        status_to_result(unsafe {
            ullm_runtime_buffer_zero(self.raw.as_ptr(), offset, bytes, stream)
        })
    }

    pub fn copy_from_host(
        &mut self,
        offset: usize,
        src: &[u8],
        stream: Option<&mut RuntimeStream>,
    ) -> Result<(), String> {
        check_copy_range(offset, src.len(), self.size()?)?;
        let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
        status_to_result(unsafe {
            ullm_runtime_buffer_copy_from_host(
                self.raw.as_ptr(),
                offset,
                src.as_ptr().cast::<c_void>(),
                src.len(),
                stream,
            )
        })
    }

    pub fn copy_to_host(
        &self,
        offset: usize,
        dst: &mut [u8],
        stream: Option<&mut RuntimeStream>,
    ) -> Result<(), String> {
        check_copy_range(offset, dst.len(), self.size()?)?;
        let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
        status_to_result(unsafe {
            ullm_runtime_buffer_copy_to_host(
                self.raw.as_ptr(),
                offset,
                dst.as_mut_ptr().cast::<c_void>(),
                dst.len(),
                stream,
            )
        })
    }

    pub fn copy_from_buffer(
        &mut self,
        dst_offset: usize,
        src: &RuntimeBuffer,
        src_offset: usize,
        bytes: usize,
        stream: Option<&mut RuntimeStream>,
    ) -> Result<(), String> {
        check_copy_range(dst_offset, bytes, self.size()?)?;
        check_copy_range(src_offset, bytes, src.size()?)?;
        let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
        status_to_result(unsafe {
            ullm_runtime_buffer_copy(
                self.raw.as_ptr(),
                dst_offset,
                src.raw.as_ptr(),
                src_offset,
                bytes,
                stream,
            )
        })
    }
}

impl Drop for RuntimeBuffer {
    fn drop(&mut self) {
        let _ = unsafe { ullm_runtime_buffer_destroy(self.raw.as_ptr()) };
    }
}

impl RuntimeStream {
    pub fn synchronize(&mut self) -> Result<(), String> {
        status_to_result(unsafe { ullm_runtime_stream_synchronize(self.raw.as_ptr()) })
    }
}

impl Drop for RuntimeStream {
    fn drop(&mut self) {
        let _ = unsafe { ullm_runtime_stream_destroy(self.raw.as_ptr()) };
    }
}

pub fn smoke_add_f32(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, String> {
    if lhs.len() != rhs.len() {
        return Err("smoke_add_f32 input lengths differ".to_string());
    }
    let mut out = vec![0.0_f32; lhs.len()];
    status_to_result(unsafe {
        ullm_runtime_smoke_add_f32(lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr(), out.len())
    })?;
    Ok(out)
}

pub fn wmma_fp8_probe(
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    check_copy_range(0, std::mem::size_of::<u32>(), output.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe { ullm_runtime_wmma_fp8_probe(output.raw.as_ptr(), stream) })
}

pub fn wmma_fp8_qk_probe(
    q_buffer: &RuntimeBuffer,
    k_buffer: &RuntimeBuffer,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let tile_elements = 16_usize * 16_usize;
    let output_bytes = tile_elements * std::mem::size_of::<f32>();
    check_copy_range(0, tile_elements, q_buffer.size()?)?;
    check_copy_range(0, tile_elements, k_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_wmma_fp8_qk_probe(
            q_buffer.raw.as_ptr(),
            k_buffer.raw.as_ptr(),
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn rocwmma_fp8_qk_probe(
    q_buffer: &RuntimeBuffer,
    k_buffer: &RuntimeBuffer,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let tile_elements = 16_usize * 16_usize;
    let output_bytes = tile_elements * std::mem::size_of::<f32>();
    check_copy_range(0, tile_elements, q_buffer.size()?)?;
    check_copy_range(0, tile_elements, k_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_rocwmma_fp8_qk_probe(
            q_buffer.raw.as_ptr(),
            k_buffer.raw.as_ptr(),
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn rocwmma_fp8_attn_probe(
    q_buffer: &RuntimeBuffer,
    k_buffer: &RuntimeBuffer,
    v_buffer: &RuntimeBuffer,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let q_bytes = 16_usize * 16_usize;
    let k_bytes = 32_usize * 16_usize;
    let v_bytes = 32_usize * 16_usize * std::mem::size_of::<f32>();
    let output_bytes = 16_usize * 16_usize * std::mem::size_of::<f32>();
    check_copy_range(0, q_bytes, q_buffer.size()?)?;
    check_copy_range(0, k_bytes, k_buffer.size()?)?;
    check_copy_range(0, v_bytes, v_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_rocwmma_fp8_attn_probe(
            q_buffer.raw.as_ptr(),
            k_buffer.raw.as_ptr(),
            v_buffer.raw.as_ptr(),
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn aq4_dequant_f32(
    index_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    codebook_buffer: &RuntimeBuffer,
    scale_values: &[f32],
    group_size: usize,
    tensor_scale: f32,
    elements: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if scale_values.is_empty() {
        return Err("AQ4 dequant scale table is empty".to_string());
    }
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 dequant output byte size overflows".to_string())?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_aq4_dequant_f32(
            index_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            codebook_buffer.raw.as_ptr(),
            scale_values.as_ptr(),
            scale_values.len(),
            group_size,
            tensor_scale,
            elements,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_row_f32(
    index_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    codebook_buffer: &RuntimeBuffer,
    scale_values_buffer: &RuntimeBuffer,
    row_scale_buffer: Option<&RuntimeBuffer>,
    scale_count: usize,
    group_size: usize,
    tensor_scale: f32,
    row_scale_count: usize,
    rows: usize,
    cols: usize,
    row_index: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if scale_count == 0 {
        return Err("AQ4 row scale table is empty".to_string());
    }
    if group_size == 0 {
        return Err("AQ4 row group size must be greater than zero".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("AQ4 row rows and cols must be greater than zero".to_string());
    }
    if row_index >= rows {
        return Err("AQ4 row index is out of range".to_string());
    }
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err("AQ4 row tensor scale must be finite and greater than zero".to_string());
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 row matrix element count overflows".to_string())?;
    let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
    let groups = elements / group_size + usize::from(!elements.is_multiple_of(group_size));
    let scale_value_bytes = scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 row scale value byte size overflows".to_string())?;
    let row_scale_bytes = row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 row scale override byte size overflows".to_string())?;
    let output_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 row output byte size overflows".to_string())?;
    check_copy_range(0, index_bytes, index_buffer.size()?)?;
    check_copy_range(0, groups, scale_buffer.size()?)?;
    check_copy_range(0, 16 * std::mem::size_of::<f32>(), codebook_buffer.size()?)?;
    check_copy_range(0, scale_value_bytes, scale_values_buffer.size()?)?;
    if let Some(row_scale_buffer) = row_scale_buffer {
        check_copy_range(0, row_scale_bytes, row_scale_buffer.size()?)?;
    }
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let row_scale_raw = row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_aq4_row_f32(
            index_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            codebook_buffer.raw.as_ptr(),
            scale_values_buffer.raw.as_ptr(),
            row_scale_raw,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            row_index,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_f32(
    index_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    codebook_buffer: &RuntimeBuffer,
    scale_values_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    row_scale_buffer: Option<&RuntimeBuffer>,
    scale_count: usize,
    group_size: usize,
    tensor_scale: f32,
    row_scale_count: usize,
    rows: usize,
    cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if scale_count == 0 {
        return Err("AQ4 matvec scale table is empty".to_string());
    }
    if group_size == 0 {
        return Err("AQ4 matvec group size must be greater than zero".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("AQ4 matvec rows and cols must be greater than zero".to_string());
    }
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err("AQ4 matvec tensor scale must be finite and greater than zero".to_string());
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec matrix element count overflows".to_string())?;
    let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
    let groups = elements / group_size + usize::from(!elements.is_multiple_of(group_size));
    let scale_value_bytes = scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec scale value byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec input byte size overflows".to_string())?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec output byte size overflows".to_string())?;
    let row_scale_bytes = row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec row scale byte size overflows".to_string())?;
    check_copy_range(0, index_bytes, index_buffer.size()?)?;
    check_copy_range(0, groups, scale_buffer.size()?)?;
    check_copy_range(0, 16 * std::mem::size_of::<f32>(), codebook_buffer.size()?)?;
    check_copy_range(0, scale_value_bytes, scale_values_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    if let Some(row_scale_buffer) = row_scale_buffer {
        check_copy_range(0, row_scale_bytes, row_scale_buffer.size()?)?;
    }
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let row_scale_raw = row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_f32(
            index_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            codebook_buffer.raw.as_ptr(),
            scale_values_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            row_scale_raw,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_batch_f32(
    index_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    codebook_buffer: &RuntimeBuffer,
    scale_values_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    row_scale_buffer: Option<&RuntimeBuffer>,
    scale_count: usize,
    group_size: usize,
    tensor_scale: f32,
    row_scale_count: usize,
    rows: usize,
    cols: usize,
    batch_count: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if scale_count == 0 {
        return Err("AQ4 matvec batch scale table is empty".to_string());
    }
    if group_size == 0 {
        return Err("AQ4 matvec batch group size must be greater than zero".to_string());
    }
    if rows == 0 || cols == 0 || batch_count == 0 {
        return Err(
            "AQ4 matvec batch rows, cols, and batch count must be greater than zero".to_string(),
        );
    }
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err(
            "AQ4 matvec batch tensor scale must be finite and greater than zero".to_string(),
        );
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec batch matrix element count overflows".to_string())?;
    let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
    let groups = elements / group_size + usize::from(!elements.is_multiple_of(group_size));
    let scale_value_bytes = scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec batch scale value byte size overflows".to_string())?;
    let input_elements = batch_count
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec batch input element count overflows".to_string())?;
    let output_elements = batch_count
        .checked_mul(rows)
        .ok_or_else(|| "AQ4 matvec batch output element count overflows".to_string())?;
    let input_bytes = input_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec batch input byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec batch output byte size overflows".to_string())?;
    let row_scale_bytes = row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec batch row scale byte size overflows".to_string())?;
    check_copy_range(0, index_bytes, index_buffer.size()?)?;
    check_copy_range(0, groups, scale_buffer.size()?)?;
    check_copy_range(0, 16 * std::mem::size_of::<f32>(), codebook_buffer.size()?)?;
    check_copy_range(0, scale_value_bytes, scale_values_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    if let Some(row_scale_buffer) = row_scale_buffer {
        check_copy_range(0, row_scale_bytes, row_scale_buffer.size()?)?;
    }
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let row_scale_raw = row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_batch_f32(
            index_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            codebook_buffer.raw.as_ptr(),
            scale_values_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            row_scale_raw,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            batch_count,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

/// Returns the runtime's shape-only AQ4 batch dispatch decision.
///
/// `device_index` uses the public runtime indexing convention (CPU is 0, HIP starts at 1).
/// A value of `true` means the gfx1201 tiled GEMM candidate is selected; `false` is the exact
/// legacy/fallback path.
pub fn aq4_matvec_batch_dispatch_tiled_for_shape(
    device_index: u32,
    group_size: usize,
    rows: usize,
    cols: usize,
    batch_count: usize,
) -> bool {
    unsafe {
        ullm_runtime_aq4_matvec_batch_dispatch_kind_for_shape(
            device_index,
            group_size,
            rows,
            cols,
            batch_count,
        ) == 1
    }
}

pub fn aq4_matvec_top1_partial_count(rows: usize) -> Result<usize, String> {
    if rows == 0 {
        return Err("AQ4 matvec top1 rows must be greater than zero".to_string());
    }
    let rows_per_block = aq4_matvec_top1_rows_per_block();
    rows.checked_add(rows_per_block - 1)
        .map(|value| value / rows_per_block)
        .ok_or_else(|| "AQ4 matvec top1 partial count overflows".to_string())
}

fn aq4_matvec_top1_rows_per_block() -> usize {
    const DEFAULT_ROWS_PER_BLOCK: usize = 8;
    std::env::var("ULLM_AQ4_MATVEC_TOP1_RPB")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value >= 1 && *value <= 32 && 256 % *value == 0)
        .unwrap_or(DEFAULT_ROWS_PER_BLOCK)
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_top1_f32(
    index_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    codebook_buffer: &RuntimeBuffer,
    scale_values_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    row_scale_buffer: Option<&RuntimeBuffer>,
    scale_count: usize,
    group_size: usize,
    tensor_scale: f32,
    row_scale_count: usize,
    rows: usize,
    cols: usize,
    partial_values_buffer: &mut RuntimeBuffer,
    partial_indices_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<usize, String> {
    if scale_count == 0 {
        return Err("AQ4 matvec top1 scale table is empty".to_string());
    }
    if group_size == 0 {
        return Err("AQ4 matvec top1 group size must be greater than zero".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("AQ4 matvec top1 rows and cols must be greater than zero".to_string());
    }
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err(
            "AQ4 matvec top1 tensor scale must be finite and greater than zero".to_string(),
        );
    }
    let partial_count = aq4_matvec_top1_partial_count(rows)?;
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec top1 matrix element count overflows".to_string())?;
    let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
    let groups = elements / group_size + usize::from(!elements.is_multiple_of(group_size));
    let scale_value_bytes = scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec top1 scale value byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec top1 input byte size overflows".to_string())?;
    let row_scale_bytes = row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec top1 row scale byte size overflows".to_string())?;
    let partial_values_bytes = partial_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec top1 partial value byte size overflows".to_string())?;
    let partial_indices_bytes = partial_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "AQ4 matvec top1 partial index byte size overflows".to_string())?;
    check_copy_range(0, index_bytes, index_buffer.size()?)?;
    check_copy_range(0, groups, scale_buffer.size()?)?;
    check_copy_range(0, 16 * std::mem::size_of::<f32>(), codebook_buffer.size()?)?;
    check_copy_range(0, scale_value_bytes, scale_values_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    if let Some(row_scale_buffer) = row_scale_buffer {
        check_copy_range(0, row_scale_bytes, row_scale_buffer.size()?)?;
    }
    check_copy_range(0, partial_values_bytes, partial_values_buffer.size()?)?;
    check_copy_range(0, partial_indices_bytes, partial_indices_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let row_scale_raw = row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_top1_f32(
            index_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            codebook_buffer.raw.as_ptr(),
            scale_values_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            row_scale_raw,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            partial_values_buffer.raw.as_ptr(),
            partial_indices_buffer.raw.as_ptr(),
            partial_count,
            stream,
        )
    })?;
    Ok(partial_count)
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_add_f32(
    index_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    codebook_buffer: &RuntimeBuffer,
    scale_values_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    residual_buffer: &RuntimeBuffer,
    row_scale_buffer: Option<&RuntimeBuffer>,
    scale_count: usize,
    group_size: usize,
    tensor_scale: f32,
    row_scale_count: usize,
    rows: usize,
    cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if scale_count == 0 {
        return Err("AQ4 matvec add scale table is empty".to_string());
    }
    if group_size == 0 {
        return Err("AQ4 matvec add group size must be greater than zero".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("AQ4 matvec add rows and cols must be greater than zero".to_string());
    }
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err("AQ4 matvec add tensor scale must be finite and greater than zero".to_string());
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec add matrix element count overflows".to_string())?;
    let index_bytes = elements / 2 + usize::from(elements % 2 != 0);
    let groups = elements / group_size + usize::from(!elements.is_multiple_of(group_size));
    let scale_value_bytes = scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec add scale value byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec add input byte size overflows".to_string())?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec add output byte size overflows".to_string())?;
    let row_scale_bytes = row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec add row scale byte size overflows".to_string())?;
    check_copy_range(0, index_bytes, index_buffer.size()?)?;
    check_copy_range(0, groups, scale_buffer.size()?)?;
    check_copy_range(0, scale_value_bytes, scale_values_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, output_bytes, residual_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    if let Some(row_scale_buffer) = row_scale_buffer {
        check_copy_range(0, row_scale_bytes, row_scale_buffer.size()?)?;
    }
    let row_scale_ptr = row_scale_buffer.map_or(std::ptr::null(), |buffer| buffer.raw.as_ptr());
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_add_f32(
            index_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            codebook_buffer.raw.as_ptr(),
            scale_values_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            residual_buffer.raw.as_ptr(),
            row_scale_ptr,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_pair_f32(
    left_index_buffer: &RuntimeBuffer,
    left_scale_buffer: &RuntimeBuffer,
    left_codebook_buffer: &RuntimeBuffer,
    left_scale_values_buffer: &RuntimeBuffer,
    left_row_scale_buffer: Option<&RuntimeBuffer>,
    left_scale_count: usize,
    left_group_size: usize,
    left_tensor_scale: f32,
    left_row_scale_count: usize,
    right_index_buffer: &RuntimeBuffer,
    right_scale_buffer: &RuntimeBuffer,
    right_codebook_buffer: &RuntimeBuffer,
    right_scale_values_buffer: &RuntimeBuffer,
    right_row_scale_buffer: Option<&RuntimeBuffer>,
    right_scale_count: usize,
    right_group_size: usize,
    right_tensor_scale: f32,
    right_row_scale_count: usize,
    input_buffer: &RuntimeBuffer,
    left_rows: usize,
    right_rows: usize,
    cols: usize,
    left_output_buffer: &mut RuntimeBuffer,
    right_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if left_scale_count == 0 || right_scale_count == 0 {
        return Err("AQ4 matvec pair scale table is empty".to_string());
    }
    if left_group_size == 0 || right_group_size == 0 {
        return Err("AQ4 matvec pair group size must be greater than zero".to_string());
    }
    if left_rows == 0 || right_rows == 0 || cols == 0 {
        return Err("AQ4 matvec pair rows and cols must be greater than zero".to_string());
    }
    if !left_tensor_scale.is_finite()
        || left_tensor_scale <= 0.0
        || !right_tensor_scale.is_finite()
        || right_tensor_scale <= 0.0
    {
        return Err(
            "AQ4 matvec pair tensor scale must be finite and greater than zero".to_string(),
        );
    }
    let left_elements = left_rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec pair left matrix element count overflows".to_string())?;
    let right_elements = right_rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec pair right matrix element count overflows".to_string())?;
    let left_index_bytes = left_elements / 2 + usize::from(!left_elements.is_multiple_of(2));
    let right_index_bytes = right_elements / 2 + usize::from(!right_elements.is_multiple_of(2));
    let left_groups = left_elements / left_group_size
        + usize::from(!left_elements.is_multiple_of(left_group_size));
    let right_groups = right_elements / right_group_size
        + usize::from(!right_elements.is_multiple_of(right_group_size));
    let left_scale_value_bytes = left_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec pair left scale value byte size overflows".to_string())?;
    let right_scale_value_bytes = right_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec pair right scale value byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec pair input byte size overflows".to_string())?;
    let left_output_bytes = left_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec pair left output byte size overflows".to_string())?;
    let right_output_bytes = right_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec pair right output byte size overflows".to_string())?;
    let left_row_scale_bytes = left_row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec pair left row scale byte size overflows".to_string())?;
    let right_row_scale_bytes = right_row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec pair right row scale byte size overflows".to_string())?;
    check_copy_range(0, left_index_bytes, left_index_buffer.size()?)?;
    check_copy_range(0, right_index_bytes, right_index_buffer.size()?)?;
    check_copy_range(0, left_groups, left_scale_buffer.size()?)?;
    check_copy_range(0, right_groups, right_scale_buffer.size()?)?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        left_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        right_codebook_buffer.size()?,
    )?;
    check_copy_range(0, left_scale_value_bytes, left_scale_values_buffer.size()?)?;
    check_copy_range(
        0,
        right_scale_value_bytes,
        right_scale_values_buffer.size()?,
    )?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    if let Some(left_row_scale_buffer) = left_row_scale_buffer {
        check_copy_range(0, left_row_scale_bytes, left_row_scale_buffer.size()?)?;
    }
    if let Some(right_row_scale_buffer) = right_row_scale_buffer {
        check_copy_range(0, right_row_scale_bytes, right_row_scale_buffer.size()?)?;
    }
    check_copy_range(0, left_output_bytes, left_output_buffer.size()?)?;
    check_copy_range(0, right_output_bytes, right_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let left_row_scale_raw = left_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let right_row_scale_raw = right_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_pair_f32(
            left_index_buffer.raw.as_ptr(),
            left_scale_buffer.raw.as_ptr(),
            left_codebook_buffer.raw.as_ptr(),
            left_scale_values_buffer.raw.as_ptr(),
            left_row_scale_raw,
            left_scale_count,
            left_group_size,
            left_tensor_scale,
            left_row_scale_count,
            right_index_buffer.raw.as_ptr(),
            right_scale_buffer.raw.as_ptr(),
            right_codebook_buffer.raw.as_ptr(),
            right_scale_values_buffer.raw.as_ptr(),
            right_row_scale_raw,
            right_scale_count,
            right_group_size,
            right_tensor_scale,
            right_row_scale_count,
            input_buffer.raw.as_ptr(),
            left_rows,
            right_rows,
            cols,
            left_output_buffer.raw.as_ptr(),
            right_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_triple_f32(
    first_index_buffer: &RuntimeBuffer,
    first_scale_buffer: &RuntimeBuffer,
    first_codebook_buffer: &RuntimeBuffer,
    first_scale_values_buffer: &RuntimeBuffer,
    first_row_scale_buffer: Option<&RuntimeBuffer>,
    first_scale_count: usize,
    first_group_size: usize,
    first_tensor_scale: f32,
    first_row_scale_count: usize,
    second_index_buffer: &RuntimeBuffer,
    second_scale_buffer: &RuntimeBuffer,
    second_codebook_buffer: &RuntimeBuffer,
    second_scale_values_buffer: &RuntimeBuffer,
    second_row_scale_buffer: Option<&RuntimeBuffer>,
    second_scale_count: usize,
    second_group_size: usize,
    second_tensor_scale: f32,
    second_row_scale_count: usize,
    third_index_buffer: &RuntimeBuffer,
    third_scale_buffer: &RuntimeBuffer,
    third_codebook_buffer: &RuntimeBuffer,
    third_scale_values_buffer: &RuntimeBuffer,
    third_row_scale_buffer: Option<&RuntimeBuffer>,
    third_scale_count: usize,
    third_group_size: usize,
    third_tensor_scale: f32,
    third_row_scale_count: usize,
    input_buffer: &RuntimeBuffer,
    first_rows: usize,
    second_rows: usize,
    third_rows: usize,
    cols: usize,
    first_output_buffer: &mut RuntimeBuffer,
    second_output_buffer: &mut RuntimeBuffer,
    third_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if first_scale_count == 0 || second_scale_count == 0 || third_scale_count == 0 {
        return Err("AQ4 matvec triple scale table is empty".to_string());
    }
    if first_group_size == 0 || second_group_size == 0 || third_group_size == 0 {
        return Err("AQ4 matvec triple group size must be greater than zero".to_string());
    }
    if first_rows == 0 || second_rows == 0 || third_rows == 0 || cols == 0 {
        return Err("AQ4 matvec triple rows and cols must be greater than zero".to_string());
    }
    if !first_tensor_scale.is_finite()
        || first_tensor_scale <= 0.0
        || !second_tensor_scale.is_finite()
        || second_tensor_scale <= 0.0
        || !third_tensor_scale.is_finite()
        || third_tensor_scale <= 0.0
    {
        return Err(
            "AQ4 matvec triple tensor scale must be finite and greater than zero".to_string(),
        );
    }
    let layout = |label: &str,
                  rows: usize,
                  group_size: usize,
                  scale_count: usize,
                  row_scale_count: usize| {
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| format!("AQ4 matvec triple {label} element count overflows"))?;
        let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
        let groups = elements / group_size + usize::from(!elements.is_multiple_of(group_size));
        let scale_value_bytes = scale_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("AQ4 matvec triple {label} scale value byte size overflows"))?;
        let row_scale_bytes = row_scale_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("AQ4 matvec triple {label} row scale byte size overflows"))?;
        let output_bytes = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("AQ4 matvec triple {label} output byte size overflows"))?;
        Ok::<_, String>((
            index_bytes,
            groups,
            scale_value_bytes,
            row_scale_bytes,
            output_bytes,
        ))
    };
    let (
        first_index_bytes,
        first_groups,
        first_scale_value_bytes,
        first_row_scale_bytes,
        first_output_bytes,
    ) = layout(
        "first",
        first_rows,
        first_group_size,
        first_scale_count,
        first_row_scale_count,
    )?;
    let (
        second_index_bytes,
        second_groups,
        second_scale_value_bytes,
        second_row_scale_bytes,
        second_output_bytes,
    ) = layout(
        "second",
        second_rows,
        second_group_size,
        second_scale_count,
        second_row_scale_count,
    )?;
    let (
        third_index_bytes,
        third_groups,
        third_scale_value_bytes,
        third_row_scale_bytes,
        third_output_bytes,
    ) = layout(
        "third",
        third_rows,
        third_group_size,
        third_scale_count,
        third_row_scale_count,
    )?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec triple input byte size overflows".to_string())?;
    check_copy_range(0, first_index_bytes, first_index_buffer.size()?)?;
    check_copy_range(0, second_index_bytes, second_index_buffer.size()?)?;
    check_copy_range(0, third_index_bytes, third_index_buffer.size()?)?;
    check_copy_range(0, first_groups, first_scale_buffer.size()?)?;
    check_copy_range(0, second_groups, second_scale_buffer.size()?)?;
    check_copy_range(0, third_groups, third_scale_buffer.size()?)?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        first_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        second_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        third_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        first_scale_value_bytes,
        first_scale_values_buffer.size()?,
    )?;
    check_copy_range(
        0,
        second_scale_value_bytes,
        second_scale_values_buffer.size()?,
    )?;
    check_copy_range(
        0,
        third_scale_value_bytes,
        third_scale_values_buffer.size()?,
    )?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    if let Some(first_row_scale_buffer) = first_row_scale_buffer {
        check_copy_range(0, first_row_scale_bytes, first_row_scale_buffer.size()?)?;
    }
    if let Some(second_row_scale_buffer) = second_row_scale_buffer {
        check_copy_range(0, second_row_scale_bytes, second_row_scale_buffer.size()?)?;
    }
    if let Some(third_row_scale_buffer) = third_row_scale_buffer {
        check_copy_range(0, third_row_scale_bytes, third_row_scale_buffer.size()?)?;
    }
    check_copy_range(0, first_output_bytes, first_output_buffer.size()?)?;
    check_copy_range(0, second_output_bytes, second_output_buffer.size()?)?;
    check_copy_range(0, third_output_bytes, third_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let first_row_scale_raw = first_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let second_row_scale_raw = second_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let third_row_scale_raw = third_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_triple_f32(
            first_index_buffer.raw.as_ptr(),
            first_scale_buffer.raw.as_ptr(),
            first_codebook_buffer.raw.as_ptr(),
            first_scale_values_buffer.raw.as_ptr(),
            first_row_scale_raw,
            first_scale_count,
            first_group_size,
            first_tensor_scale,
            first_row_scale_count,
            second_index_buffer.raw.as_ptr(),
            second_scale_buffer.raw.as_ptr(),
            second_codebook_buffer.raw.as_ptr(),
            second_scale_values_buffer.raw.as_ptr(),
            second_row_scale_raw,
            second_scale_count,
            second_group_size,
            second_tensor_scale,
            second_row_scale_count,
            third_index_buffer.raw.as_ptr(),
            third_scale_buffer.raw.as_ptr(),
            third_codebook_buffer.raw.as_ptr(),
            third_scale_values_buffer.raw.as_ptr(),
            third_row_scale_raw,
            third_scale_count,
            third_group_size,
            third_tensor_scale,
            third_row_scale_count,
            input_buffer.raw.as_ptr(),
            first_rows,
            second_rows,
            third_rows,
            cols,
            first_output_buffer.raw.as_ptr(),
            second_output_buffer.raw.as_ptr(),
            third_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_qkv_z_gate_beta_f32(
    qkv_index_buffer: &RuntimeBuffer,
    qkv_scale_buffer: &RuntimeBuffer,
    qkv_codebook_buffer: &RuntimeBuffer,
    qkv_scale_values_buffer: &RuntimeBuffer,
    qkv_row_scale_buffer: Option<&RuntimeBuffer>,
    qkv_scale_count: usize,
    qkv_group_size: usize,
    qkv_tensor_scale: f32,
    qkv_row_scale_count: usize,
    z_index_buffer: &RuntimeBuffer,
    z_scale_buffer: &RuntimeBuffer,
    z_codebook_buffer: &RuntimeBuffer,
    z_scale_values_buffer: &RuntimeBuffer,
    z_row_scale_buffer: Option<&RuntimeBuffer>,
    z_scale_count: usize,
    z_group_size: usize,
    z_tensor_scale: f32,
    z_row_scale_count: usize,
    a_index_buffer: &RuntimeBuffer,
    a_scale_buffer: &RuntimeBuffer,
    a_codebook_buffer: &RuntimeBuffer,
    a_scale_values_buffer: &RuntimeBuffer,
    a_row_scale_buffer: Option<&RuntimeBuffer>,
    a_scale_count: usize,
    a_group_size: usize,
    a_tensor_scale: f32,
    a_row_scale_count: usize,
    b_index_buffer: &RuntimeBuffer,
    b_scale_buffer: &RuntimeBuffer,
    b_codebook_buffer: &RuntimeBuffer,
    b_scale_values_buffer: &RuntimeBuffer,
    b_row_scale_buffer: Option<&RuntimeBuffer>,
    b_scale_count: usize,
    b_group_size: usize,
    b_tensor_scale: f32,
    b_row_scale_count: usize,
    input_buffer: &RuntimeBuffer,
    a_log_buffer: &RuntimeBuffer,
    dt_bias_buffer: &RuntimeBuffer,
    qkv_rows: usize,
    z_rows: usize,
    heads: usize,
    cols: usize,
    qkv_output_buffer: &mut RuntimeBuffer,
    z_output_buffer: &mut RuntimeBuffer,
    gate_output_buffer: &mut RuntimeBuffer,
    beta_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if qkv_scale_count == 0 || z_scale_count == 0 || a_scale_count == 0 || b_scale_count == 0 {
        return Err("AQ4 qkv/z gate/beta scale table is empty".to_string());
    }
    if qkv_group_size == 0 || z_group_size == 0 || a_group_size == 0 || b_group_size == 0 {
        return Err("AQ4 qkv/z gate/beta group size must be greater than zero".to_string());
    }
    if qkv_rows == 0 || z_rows == 0 || heads == 0 || cols == 0 {
        return Err("AQ4 qkv/z gate/beta rows and cols must be greater than zero".to_string());
    }
    if !qkv_tensor_scale.is_finite()
        || qkv_tensor_scale <= 0.0
        || !z_tensor_scale.is_finite()
        || z_tensor_scale <= 0.0
        || !a_tensor_scale.is_finite()
        || a_tensor_scale <= 0.0
        || !b_tensor_scale.is_finite()
        || b_tensor_scale <= 0.0
    {
        return Err(
            "AQ4 qkv/z gate/beta tensor scale must be finite and greater than zero".to_string(),
        );
    }
    let layout = |label: &str,
                  rows: usize,
                  group_size: usize,
                  scale_count: usize,
                  row_scale_count: usize|
     -> Result<(usize, usize, usize, usize, usize), String> {
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| format!("AQ4 qkv/z gate/beta {label} element count overflows"))?;
        let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
        let groups = elements / group_size + usize::from(!elements.is_multiple_of(group_size));
        let scale_value_bytes = scale_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("AQ4 qkv/z gate/beta {label} scale value byte size overflows")
            })?;
        let row_scale_bytes = row_scale_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("AQ4 qkv/z gate/beta {label} row scale byte size overflows"))?;
        let output_bytes = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("AQ4 qkv/z gate/beta {label} output byte size overflows"))?;
        Ok((
            index_bytes,
            groups,
            scale_value_bytes,
            row_scale_bytes,
            output_bytes,
        ))
    };
    let (qkv_index_bytes, qkv_groups, qkv_scale_value_bytes, qkv_row_scale_bytes, qkv_output_bytes) =
        layout(
            "qkv",
            qkv_rows,
            qkv_group_size,
            qkv_scale_count,
            qkv_row_scale_count,
        )?;
    let (z_index_bytes, z_groups, z_scale_value_bytes, z_row_scale_bytes, z_output_bytes) =
        layout("z", z_rows, z_group_size, z_scale_count, z_row_scale_count)?;
    let (a_index_bytes, a_groups, a_scale_value_bytes, a_row_scale_bytes, gate_output_bytes) =
        layout("a", heads, a_group_size, a_scale_count, a_row_scale_count)?;
    let (b_index_bytes, b_groups, b_scale_value_bytes, b_row_scale_bytes, beta_output_bytes) =
        layout("b", heads, b_group_size, b_scale_count, b_row_scale_count)?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 qkv/z gate/beta input byte size overflows".to_string())?;
    check_copy_range(0, qkv_index_bytes, qkv_index_buffer.size()?)?;
    check_copy_range(0, z_index_bytes, z_index_buffer.size()?)?;
    check_copy_range(0, a_index_bytes, a_index_buffer.size()?)?;
    check_copy_range(0, b_index_bytes, b_index_buffer.size()?)?;
    check_copy_range(0, qkv_groups, qkv_scale_buffer.size()?)?;
    check_copy_range(0, z_groups, z_scale_buffer.size()?)?;
    check_copy_range(0, a_groups, a_scale_buffer.size()?)?;
    check_copy_range(0, b_groups, b_scale_buffer.size()?)?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        qkv_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        z_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        a_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        b_codebook_buffer.size()?,
    )?;
    check_copy_range(0, qkv_scale_value_bytes, qkv_scale_values_buffer.size()?)?;
    check_copy_range(0, z_scale_value_bytes, z_scale_values_buffer.size()?)?;
    check_copy_range(0, a_scale_value_bytes, a_scale_values_buffer.size()?)?;
    check_copy_range(0, b_scale_value_bytes, b_scale_values_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, gate_output_bytes, a_log_buffer.size()?)?;
    check_copy_range(0, gate_output_bytes, dt_bias_buffer.size()?)?;
    if let Some(qkv_row_scale_buffer) = qkv_row_scale_buffer {
        check_copy_range(0, qkv_row_scale_bytes, qkv_row_scale_buffer.size()?)?;
    }
    if let Some(z_row_scale_buffer) = z_row_scale_buffer {
        check_copy_range(0, z_row_scale_bytes, z_row_scale_buffer.size()?)?;
    }
    if let Some(a_row_scale_buffer) = a_row_scale_buffer {
        check_copy_range(0, a_row_scale_bytes, a_row_scale_buffer.size()?)?;
    }
    if let Some(b_row_scale_buffer) = b_row_scale_buffer {
        check_copy_range(0, b_row_scale_bytes, b_row_scale_buffer.size()?)?;
    }
    check_copy_range(0, qkv_output_bytes, qkv_output_buffer.size()?)?;
    check_copy_range(0, z_output_bytes, z_output_buffer.size()?)?;
    check_copy_range(0, gate_output_bytes, gate_output_buffer.size()?)?;
    check_copy_range(0, beta_output_bytes, beta_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let qkv_row_scale_raw = qkv_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let z_row_scale_raw = z_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let a_row_scale_raw = a_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let b_row_scale_raw = b_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_qkv_z_gate_beta_f32(
            qkv_index_buffer.raw.as_ptr(),
            qkv_scale_buffer.raw.as_ptr(),
            qkv_codebook_buffer.raw.as_ptr(),
            qkv_scale_values_buffer.raw.as_ptr(),
            qkv_row_scale_raw,
            qkv_scale_count,
            qkv_group_size,
            qkv_tensor_scale,
            qkv_row_scale_count,
            z_index_buffer.raw.as_ptr(),
            z_scale_buffer.raw.as_ptr(),
            z_codebook_buffer.raw.as_ptr(),
            z_scale_values_buffer.raw.as_ptr(),
            z_row_scale_raw,
            z_scale_count,
            z_group_size,
            z_tensor_scale,
            z_row_scale_count,
            a_index_buffer.raw.as_ptr(),
            a_scale_buffer.raw.as_ptr(),
            a_codebook_buffer.raw.as_ptr(),
            a_scale_values_buffer.raw.as_ptr(),
            a_row_scale_raw,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_count,
            b_index_buffer.raw.as_ptr(),
            b_scale_buffer.raw.as_ptr(),
            b_codebook_buffer.raw.as_ptr(),
            b_scale_values_buffer.raw.as_ptr(),
            b_row_scale_raw,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_count,
            input_buffer.raw.as_ptr(),
            a_log_buffer.raw.as_ptr(),
            dt_bias_buffer.raw.as_ptr(),
            qkv_rows,
            z_rows,
            heads,
            cols,
            qkv_output_buffer.raw.as_ptr(),
            z_output_buffer.raw.as_ptr(),
            gate_output_buffer.raw.as_ptr(),
            beta_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_silu_mul_f32(
    gate_index_buffer: &RuntimeBuffer,
    gate_scale_buffer: &RuntimeBuffer,
    gate_codebook_buffer: &RuntimeBuffer,
    gate_scale_values_buffer: &RuntimeBuffer,
    gate_row_scale_buffer: Option<&RuntimeBuffer>,
    gate_scale_count: usize,
    gate_group_size: usize,
    gate_tensor_scale: f32,
    gate_row_scale_count: usize,
    up_index_buffer: &RuntimeBuffer,
    up_scale_buffer: &RuntimeBuffer,
    up_codebook_buffer: &RuntimeBuffer,
    up_scale_values_buffer: &RuntimeBuffer,
    up_row_scale_buffer: Option<&RuntimeBuffer>,
    up_scale_count: usize,
    up_group_size: usize,
    up_tensor_scale: f32,
    up_row_scale_count: usize,
    input_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if gate_scale_count == 0 || up_scale_count == 0 {
        return Err("AQ4 matvec SiLU-mul scale table is empty".to_string());
    }
    if gate_group_size == 0 || up_group_size == 0 {
        return Err("AQ4 matvec SiLU-mul group size must be greater than zero".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("AQ4 matvec SiLU-mul rows and cols must be greater than zero".to_string());
    }
    if !gate_tensor_scale.is_finite()
        || gate_tensor_scale <= 0.0
        || !up_tensor_scale.is_finite()
        || up_tensor_scale <= 0.0
    {
        return Err(
            "AQ4 matvec SiLU-mul tensor scale must be finite and greater than zero".to_string(),
        );
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec SiLU-mul matrix element count overflows".to_string())?;
    let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
    let gate_groups =
        elements / gate_group_size + usize::from(!elements.is_multiple_of(gate_group_size));
    let up_groups = elements / up_group_size + usize::from(!elements.is_multiple_of(up_group_size));
    let gate_scale_value_bytes = gate_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec SiLU-mul gate scale value byte size overflows".to_string())?;
    let up_scale_value_bytes = up_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec SiLU-mul up scale value byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec SiLU-mul input byte size overflows".to_string())?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec SiLU-mul output byte size overflows".to_string())?;
    let gate_row_scale_bytes = gate_row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec SiLU-mul gate row scale byte size overflows".to_string())?;
    let up_row_scale_bytes = up_row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec SiLU-mul up row scale byte size overflows".to_string())?;
    check_copy_range(0, index_bytes, gate_index_buffer.size()?)?;
    check_copy_range(0, index_bytes, up_index_buffer.size()?)?;
    check_copy_range(0, gate_groups, gate_scale_buffer.size()?)?;
    check_copy_range(0, up_groups, up_scale_buffer.size()?)?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        gate_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        up_codebook_buffer.size()?,
    )?;
    check_copy_range(0, gate_scale_value_bytes, gate_scale_values_buffer.size()?)?;
    check_copy_range(0, up_scale_value_bytes, up_scale_values_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    if let Some(gate_row_scale_buffer) = gate_row_scale_buffer {
        check_copy_range(0, gate_row_scale_bytes, gate_row_scale_buffer.size()?)?;
    }
    if let Some(up_row_scale_buffer) = up_row_scale_buffer {
        check_copy_range(0, up_row_scale_bytes, up_row_scale_buffer.size()?)?;
    }
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let gate_row_scale_raw = gate_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let up_row_scale_raw = up_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_silu_mul_f32(
            gate_index_buffer.raw.as_ptr(),
            gate_scale_buffer.raw.as_ptr(),
            gate_codebook_buffer.raw.as_ptr(),
            gate_scale_values_buffer.raw.as_ptr(),
            gate_row_scale_raw,
            gate_scale_count,
            gate_group_size,
            gate_tensor_scale,
            gate_row_scale_count,
            up_index_buffer.raw.as_ptr(),
            up_scale_buffer.raw.as_ptr(),
            up_codebook_buffer.raw.as_ptr(),
            up_scale_values_buffer.raw.as_ptr(),
            up_row_scale_raw,
            up_scale_count,
            up_group_size,
            up_tensor_scale,
            up_row_scale_count,
            input_buffer.raw.as_ptr(),
            rows,
            cols,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn aq4_matvec_gate_beta_f32(
    a_index_buffer: &RuntimeBuffer,
    a_scale_buffer: &RuntimeBuffer,
    a_codebook_buffer: &RuntimeBuffer,
    a_scale_values_buffer: &RuntimeBuffer,
    a_row_scale_buffer: Option<&RuntimeBuffer>,
    a_scale_count: usize,
    a_group_size: usize,
    a_tensor_scale: f32,
    a_row_scale_count: usize,
    b_index_buffer: &RuntimeBuffer,
    b_scale_buffer: &RuntimeBuffer,
    b_codebook_buffer: &RuntimeBuffer,
    b_scale_values_buffer: &RuntimeBuffer,
    b_row_scale_buffer: Option<&RuntimeBuffer>,
    b_scale_count: usize,
    b_group_size: usize,
    b_tensor_scale: f32,
    b_row_scale_count: usize,
    input_buffer: &RuntimeBuffer,
    a_log_buffer: &RuntimeBuffer,
    dt_bias_buffer: &RuntimeBuffer,
    heads: usize,
    cols: usize,
    gate_output_buffer: &mut RuntimeBuffer,
    beta_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if a_scale_count == 0 || b_scale_count == 0 {
        return Err("AQ4 matvec gate/beta scale table is empty".to_string());
    }
    if a_group_size == 0 || b_group_size == 0 {
        return Err("AQ4 matvec gate/beta group size must be greater than zero".to_string());
    }
    if heads == 0 || cols == 0 {
        return Err("AQ4 matvec gate/beta heads and cols must be greater than zero".to_string());
    }
    if !a_tensor_scale.is_finite()
        || a_tensor_scale <= 0.0
        || !b_tensor_scale.is_finite()
        || b_tensor_scale <= 0.0
    {
        return Err(
            "AQ4 matvec gate/beta tensor scale must be finite and greater than zero".to_string(),
        );
    }
    let elements = heads
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec gate/beta matrix element count overflows".to_string())?;
    let index_bytes = elements / 2 + usize::from(!elements.is_multiple_of(2));
    let a_groups = elements / a_group_size + usize::from(!elements.is_multiple_of(a_group_size));
    let b_groups = elements / b_group_size + usize::from(!elements.is_multiple_of(b_group_size));
    let a_scale_value_bytes = a_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec gate/beta a scale value byte size overflows".to_string())?;
    let b_scale_value_bytes = b_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec gate/beta b scale value byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec gate/beta input byte size overflows".to_string())?;
    let output_bytes = heads
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec gate/beta output byte size overflows".to_string())?;
    let a_row_scale_bytes = a_row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec gate/beta a row scale byte size overflows".to_string())?;
    let b_row_scale_bytes = b_row_scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "AQ4 matvec gate/beta b row scale byte size overflows".to_string())?;
    check_copy_range(0, index_bytes, a_index_buffer.size()?)?;
    check_copy_range(0, index_bytes, b_index_buffer.size()?)?;
    check_copy_range(0, a_groups, a_scale_buffer.size()?)?;
    check_copy_range(0, b_groups, b_scale_buffer.size()?)?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        a_codebook_buffer.size()?,
    )?;
    check_copy_range(
        0,
        16 * std::mem::size_of::<f32>(),
        b_codebook_buffer.size()?,
    )?;
    check_copy_range(0, a_scale_value_bytes, a_scale_values_buffer.size()?)?;
    check_copy_range(0, b_scale_value_bytes, b_scale_values_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, output_bytes, a_log_buffer.size()?)?;
    check_copy_range(0, output_bytes, dt_bias_buffer.size()?)?;
    if let Some(a_row_scale_buffer) = a_row_scale_buffer {
        check_copy_range(0, a_row_scale_bytes, a_row_scale_buffer.size()?)?;
    }
    if let Some(b_row_scale_buffer) = b_row_scale_buffer {
        check_copy_range(0, b_row_scale_bytes, b_row_scale_buffer.size()?)?;
    }
    check_copy_range(0, output_bytes, gate_output_buffer.size()?)?;
    check_copy_range(0, output_bytes, beta_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let a_row_scale_raw = a_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let b_row_scale_raw = b_row_scale_buffer
        .map(|buffer| buffer.raw.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    status_to_result(unsafe {
        ullm_runtime_aq4_matvec_gate_beta_f32(
            a_index_buffer.raw.as_ptr(),
            a_scale_buffer.raw.as_ptr(),
            a_codebook_buffer.raw.as_ptr(),
            a_scale_values_buffer.raw.as_ptr(),
            a_row_scale_raw,
            a_scale_count,
            a_group_size,
            a_tensor_scale,
            a_row_scale_count,
            b_index_buffer.raw.as_ptr(),
            b_scale_buffer.raw.as_ptr(),
            b_codebook_buffer.raw.as_ptr(),
            b_scale_values_buffer.raw.as_ptr(),
            b_row_scale_raw,
            b_scale_count,
            b_group_size,
            b_tensor_scale,
            b_row_scale_count,
            input_buffer.raw.as_ptr(),
            a_log_buffer.raw.as_ptr(),
            dt_bias_buffer.raw.as_ptr(),
            heads,
            cols,
            gate_output_buffer.raw.as_ptr(),
            beta_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn matvec_f32(
    matrix_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err("f32 matvec rows and cols must be greater than zero".to_string());
    }
    let matrix_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "f32 matvec matrix element count overflows".to_string())?;
    let matrix_bytes = matrix_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 matvec matrix byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 matvec input byte size overflows".to_string())?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 matvec output byte size overflows".to_string())?;
    check_copy_range(0, matrix_bytes, matrix_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_matvec_f32(
            matrix_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            rows,
            cols,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub const SQ_FP8_SCALE_TENSOR: u32 = 0;
pub const SQ_FP8_SCALE_ROW: u32 = 1;
pub const SQ_FP8_SCALE_ROW_BLOCK: u32 = 2;
pub const SQ_FP8_EXECUTION_PATH_CPU_REFERENCE: i32 = 0;
pub const SQ_FP8_EXECUTION_PATH_HIP_KERNEL: i32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqFp8ExecutionPath {
    CpuReference,
    HipKernel,
}

fn sq_fp8_execution_path(raw: c_int) -> Result<SqFp8ExecutionPath, String> {
    match raw {
        SQ_FP8_EXECUTION_PATH_CPU_REFERENCE => Ok(SqFp8ExecutionPath::CpuReference),
        SQ_FP8_EXECUTION_PATH_HIP_KERNEL => Ok(SqFp8ExecutionPath::HipKernel),
        _ => Err(format!("SQ FP8 runtime returned unknown execution path {raw}")),
    }
}

pub fn sq_fp8_matvec_f32(
    payload_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    scale_kind: u32,
    scale_block_cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err("SQ FP8 matvec rows and cols must be greater than zero".to_string());
    }
    if scale_kind > SQ_FP8_SCALE_ROW_BLOCK {
        return Err(
            "SQ FP8 matvec scale kind must be tensor(0), row(1), or row_block(2)".to_string(),
        );
    }
    if scale_kind == SQ_FP8_SCALE_ROW_BLOCK && scale_block_cols == 0 {
        return Err(
            "SQ FP8 matvec row_block scale_block_cols must be greater than zero".to_string(),
        );
    }
    let matrix_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec matrix element count overflows".to_string())?;
    let scale_values = if scale_kind == SQ_FP8_SCALE_TENSOR {
        1
    } else if scale_kind == SQ_FP8_SCALE_ROW {
        rows
    } else {
        let blocks_per_row = cols
            .checked_add(scale_block_cols - 1)
            .and_then(|value| value.checked_div(scale_block_cols))
            .ok_or_else(|| "SQ FP8 matvec row_block count overflows".to_string())?;
        rows.checked_mul(blocks_per_row)
            .ok_or_else(|| "SQ FP8 matvec row_block scale count overflows".to_string())?
    };
    let scale_bytes = scale_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec scale byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec input byte size overflows".to_string())?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec output byte size overflows".to_string())?;
    check_copy_range(0, matrix_elements, payload_buffer.size()?)?;
    check_copy_range(0, scale_bytes, scale_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq_fp8_matvec_f32(
            payload_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            rows,
            cols,
            scale_kind,
            scale_block_cols,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn sq_fp8_matvec_batch_f32(
    payload_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    scale_kind: u32,
    scale_block_cols: usize,
    batch_count: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if rows == 0 || cols == 0 || batch_count == 0 {
        return Err(
            "SQ FP8 matvec batch rows, cols, and batch count must be greater than zero".to_string(),
        );
    }
    if scale_kind > SQ_FP8_SCALE_ROW_BLOCK {
        return Err(
            "SQ FP8 matvec batch scale kind must be tensor(0), row(1), or row_block(2)".to_string(),
        );
    }
    if scale_kind == SQ_FP8_SCALE_ROW_BLOCK && scale_block_cols == 0 {
        return Err(
            "SQ FP8 matvec batch row_block scale_block_cols must be greater than zero".to_string(),
        );
    }
    let matrix_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec batch matrix element count overflows".to_string())?;
    let scale_values = if scale_kind == SQ_FP8_SCALE_TENSOR {
        1
    } else if scale_kind == SQ_FP8_SCALE_ROW {
        rows
    } else {
        let blocks_per_row = cols
            .checked_add(scale_block_cols - 1)
            .and_then(|value| value.checked_div(scale_block_cols))
            .ok_or_else(|| "SQ FP8 matvec batch row_block count overflows".to_string())?;
        rows.checked_mul(blocks_per_row)
            .ok_or_else(|| "SQ FP8 matvec batch row_block scale count overflows".to_string())?
    };
    let scale_bytes = scale_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec batch scale byte size overflows".to_string())?;
    let input_elements = batch_count
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec batch input element count overflows".to_string())?;
    let output_elements = batch_count
        .checked_mul(rows)
        .ok_or_else(|| "SQ FP8 matvec batch output element count overflows".to_string())?;
    let input_bytes = input_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec batch input byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec batch output byte size overflows".to_string())?;
    check_copy_range(0, matrix_elements, payload_buffer.size()?)?;
    check_copy_range(0, scale_bytes, scale_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq_fp8_matvec_batch_f32(
            payload_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            rows,
            cols,
            scale_kind,
            scale_block_cols,
            batch_count,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

struct SqFp8Block2dBufferLengths {
    payload: usize,
    scales: usize,
    input: usize,
    output: usize,
}

fn sq_fp8_block2d_buffer_lengths(
    rows: usize,
    cols: usize,
    scale_block_rows: usize,
    scale_block_cols: usize,
    batch_count: usize,
) -> Result<SqFp8Block2dBufferLengths, String> {
    if rows == 0 || cols == 0 || batch_count == 0 {
        return Err(
            "SQ FP8 block2d matvec rows, cols, and batch count must be greater than zero"
                .to_string(),
        );
    }
    if scale_block_rows == 0 || scale_block_cols == 0 {
        return Err(
            "SQ FP8 block2d matvec scale block rows and cols must be greater than zero"
                .to_string(),
        );
    }
    let payload = rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 block2d matvec matrix element count overflows".to_string())?;
    let scale_block_row_count = 1 + (rows - 1) / scale_block_rows;
    let scale_block_col_count = 1 + (cols - 1) / scale_block_cols;
    let scale_values = scale_block_row_count
        .checked_mul(scale_block_col_count)
        .ok_or_else(|| "SQ FP8 block2d matvec scale count overflows".to_string())?;
    let scales = scale_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 block2d matvec scale byte size overflows".to_string())?;
    let input_elements = batch_count
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 block2d matvec input element count overflows".to_string())?;
    let input = input_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 block2d matvec input byte size overflows".to_string())?;
    let output_elements = batch_count
        .checked_mul(rows)
        .ok_or_else(|| "SQ FP8 block2d matvec output element count overflows".to_string())?;
    let output = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 block2d matvec output byte size overflows".to_string())?;
    Ok(SqFp8Block2dBufferLengths {
        payload,
        scales,
        input,
        output,
    })
}

fn validate_sq_fp8_block2d_buffers(
    payload_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    output_buffer: &RuntimeBuffer,
    lengths: &SqFp8Block2dBufferLengths,
) -> Result<(), String> {
    check_copy_range(0, lengths.payload, payload_buffer.size()?)?;
    check_copy_range(0, lengths.scales, scale_buffer.size()?)?;
    check_copy_range(0, lengths.input, input_buffer.size()?)?;
    check_copy_range(0, lengths.output, output_buffer.size()?)?;
    Ok(())
}

pub fn sq_fp8_matvec_block2d_f32(
    payload_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    scale_block_rows: usize,
    scale_block_cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<SqFp8ExecutionPath, String> {
    let lengths =
        sq_fp8_block2d_buffer_lengths(rows, cols, scale_block_rows, scale_block_cols, 1)?;
    validate_sq_fp8_block2d_buffers(
        payload_buffer,
        scale_buffer,
        input_buffer,
        output_buffer,
        &lengths,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let mut execution_path = -1;
    status_to_result(unsafe {
        ullm_runtime_sq_fp8_matvec_block2d_f32(
            payload_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            rows,
            cols,
            scale_block_rows,
            scale_block_cols,
            output_buffer.raw.as_ptr(),
            stream,
            &mut execution_path,
        )
    })?;
    sq_fp8_execution_path(execution_path)
}

pub fn sq_fp8_matvec_block2d_batch_f32(
    payload_buffer: &RuntimeBuffer,
    scale_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    scale_block_rows: usize,
    scale_block_cols: usize,
    batch_count: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<SqFp8ExecutionPath, String> {
    let lengths = sq_fp8_block2d_buffer_lengths(
        rows,
        cols,
        scale_block_rows,
        scale_block_cols,
        batch_count,
    )?;
    validate_sq_fp8_block2d_buffers(
        payload_buffer,
        scale_buffer,
        input_buffer,
        output_buffer,
        &lengths,
    )?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let mut execution_path = -1;
    status_to_result(unsafe {
        ullm_runtime_sq_fp8_matvec_block2d_batch_f32(
            payload_buffer.raw.as_ptr(),
            scale_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            rows,
            cols,
            scale_block_rows,
            scale_block_cols,
            batch_count,
            output_buffer.raw.as_ptr(),
            stream,
            &mut execution_path,
        )
    })?;
    sq_fp8_execution_path(execution_path)
}

fn sq_fp8_scale_byte_len(
    rows: usize,
    cols: usize,
    scale_kind: u32,
    scale_block_cols: usize,
    label: &str,
) -> Result<usize, String> {
    if scale_kind > SQ_FP8_SCALE_ROW_BLOCK {
        return Err(format!(
            "{label} scale kind must be tensor(0), row(1), or row_block(2)"
        ));
    }
    if scale_kind == SQ_FP8_SCALE_ROW_BLOCK && scale_block_cols == 0 {
        return Err(format!(
            "{label} row_block scale_block_cols must be greater than zero"
        ));
    }
    let scale_values = if scale_kind == SQ_FP8_SCALE_TENSOR {
        1
    } else if scale_kind == SQ_FP8_SCALE_ROW {
        rows
    } else {
        let blocks_per_row = cols
            .checked_add(scale_block_cols - 1)
            .and_then(|value| value.checked_div(scale_block_cols))
            .ok_or_else(|| format!("{label} row_block count overflows"))?;
        rows.checked_mul(blocks_per_row)
            .ok_or_else(|| format!("{label} row_block scale count overflows"))?
    };
    scale_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} scale byte size overflows"))
}

#[allow(clippy::too_many_arguments)]
pub fn sq_fp8_matvec_pair_f32(
    left_payload_buffer: &RuntimeBuffer,
    left_scale_buffer: &RuntimeBuffer,
    left_scale_kind: u32,
    left_scale_block_cols: usize,
    right_payload_buffer: &RuntimeBuffer,
    right_scale_buffer: &RuntimeBuffer,
    right_scale_kind: u32,
    right_scale_block_cols: usize,
    input_buffer: &RuntimeBuffer,
    left_rows: usize,
    right_rows: usize,
    cols: usize,
    left_output_buffer: &mut RuntimeBuffer,
    right_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if left_rows == 0 || right_rows == 0 || cols == 0 {
        return Err("SQ FP8 matvec pair rows and cols must be greater than zero".to_string());
    }
    let left_payload_bytes = left_rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec pair left payload byte size overflows".to_string())?;
    let right_payload_bytes = right_rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec pair right payload byte size overflows".to_string())?;
    let left_scale_bytes = sq_fp8_scale_byte_len(
        left_rows,
        cols,
        left_scale_kind,
        left_scale_block_cols,
        "SQ FP8 matvec pair left",
    )?;
    let right_scale_bytes = sq_fp8_scale_byte_len(
        right_rows,
        cols,
        right_scale_kind,
        right_scale_block_cols,
        "SQ FP8 matvec pair right",
    )?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec pair input byte size overflows".to_string())?;
    let left_output_bytes = left_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec pair left output byte size overflows".to_string())?;
    let right_output_bytes = right_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec pair right output byte size overflows".to_string())?;
    check_copy_range(0, left_payload_bytes, left_payload_buffer.size()?)?;
    check_copy_range(0, right_payload_bytes, right_payload_buffer.size()?)?;
    check_copy_range(0, left_scale_bytes, left_scale_buffer.size()?)?;
    check_copy_range(0, right_scale_bytes, right_scale_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, left_output_bytes, left_output_buffer.size()?)?;
    check_copy_range(0, right_output_bytes, right_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq_fp8_matvec_pair_f32(
            left_payload_buffer.raw.as_ptr(),
            left_scale_buffer.raw.as_ptr(),
            left_scale_kind,
            left_scale_block_cols,
            right_payload_buffer.raw.as_ptr(),
            right_scale_buffer.raw.as_ptr(),
            right_scale_kind,
            right_scale_block_cols,
            input_buffer.raw.as_ptr(),
            left_rows,
            right_rows,
            cols,
            left_output_buffer.raw.as_ptr(),
            right_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn sq_fp8_matvec_triple_f32(
    first_payload_buffer: &RuntimeBuffer,
    first_scale_buffer: &RuntimeBuffer,
    first_scale_kind: u32,
    first_scale_block_cols: usize,
    second_payload_buffer: &RuntimeBuffer,
    second_scale_buffer: &RuntimeBuffer,
    second_scale_kind: u32,
    second_scale_block_cols: usize,
    third_payload_buffer: &RuntimeBuffer,
    third_scale_buffer: &RuntimeBuffer,
    third_scale_kind: u32,
    third_scale_block_cols: usize,
    input_buffer: &RuntimeBuffer,
    first_rows: usize,
    second_rows: usize,
    third_rows: usize,
    cols: usize,
    first_output_buffer: &mut RuntimeBuffer,
    second_output_buffer: &mut RuntimeBuffer,
    third_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if first_rows == 0 || second_rows == 0 || third_rows == 0 || cols == 0 {
        return Err("SQ FP8 matvec triple rows and cols must be greater than zero".to_string());
    }
    let first_payload_bytes = first_rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec triple first payload byte size overflows".to_string())?;
    let second_payload_bytes = second_rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec triple second payload byte size overflows".to_string())?;
    let third_payload_bytes = third_rows
        .checked_mul(cols)
        .ok_or_else(|| "SQ FP8 matvec triple third payload byte size overflows".to_string())?;
    let first_scale_bytes = sq_fp8_scale_byte_len(
        first_rows,
        cols,
        first_scale_kind,
        first_scale_block_cols,
        "SQ FP8 matvec triple first",
    )?;
    let second_scale_bytes = sq_fp8_scale_byte_len(
        second_rows,
        cols,
        second_scale_kind,
        second_scale_block_cols,
        "SQ FP8 matvec triple second",
    )?;
    let third_scale_bytes = sq_fp8_scale_byte_len(
        third_rows,
        cols,
        third_scale_kind,
        third_scale_block_cols,
        "SQ FP8 matvec triple third",
    )?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec triple input byte size overflows".to_string())?;
    let first_output_bytes = first_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec triple first output byte size overflows".to_string())?;
    let second_output_bytes = second_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec triple second output byte size overflows".to_string())?;
    let third_output_bytes = third_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 matvec triple third output byte size overflows".to_string())?;
    check_copy_range(0, first_payload_bytes, first_payload_buffer.size()?)?;
    check_copy_range(0, second_payload_bytes, second_payload_buffer.size()?)?;
    check_copy_range(0, third_payload_bytes, third_payload_buffer.size()?)?;
    check_copy_range(0, first_scale_bytes, first_scale_buffer.size()?)?;
    check_copy_range(0, second_scale_bytes, second_scale_buffer.size()?)?;
    check_copy_range(0, third_scale_bytes, third_scale_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, first_output_bytes, first_output_buffer.size()?)?;
    check_copy_range(0, second_output_bytes, second_output_buffer.size()?)?;
    check_copy_range(0, third_output_bytes, third_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq_fp8_matvec_triple_f32(
            first_payload_buffer.raw.as_ptr(),
            first_scale_buffer.raw.as_ptr(),
            first_scale_kind,
            first_scale_block_cols,
            second_payload_buffer.raw.as_ptr(),
            second_scale_buffer.raw.as_ptr(),
            second_scale_kind,
            second_scale_block_cols,
            third_payload_buffer.raw.as_ptr(),
            third_scale_buffer.raw.as_ptr(),
            third_scale_kind,
            third_scale_block_cols,
            input_buffer.raw.as_ptr(),
            first_rows,
            second_rows,
            third_rows,
            cols,
            first_output_buffer.raw.as_ptr(),
            second_output_buffer.raw.as_ptr(),
            third_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn matvec_bf16_f32(
    matrix_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err("BF16 matvec rows and cols must be greater than zero".to_string());
    }
    let matrix_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "BF16 matvec matrix element count overflows".to_string())?;
    let matrix_bytes = matrix_elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "BF16 matvec matrix byte size overflows".to_string())?;
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "BF16 matvec input byte size overflows".to_string())?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "BF16 matvec output byte size overflows".to_string())?;
    check_copy_range(0, matrix_bytes, matrix_buffer.size()?)?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_matvec_bf16_f32(
            matrix_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            rows,
            cols,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn bf16_row_f32(
    matrix_buffer: &RuntimeBuffer,
    rows: usize,
    cols: usize,
    row_index: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err("BF16 row rows and cols must be greater than zero".to_string());
    }
    if row_index >= rows {
        return Err("BF16 row index is out of range".to_string());
    }
    let matrix_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "BF16 row matrix element count overflows".to_string())?;
    let matrix_bytes = matrix_elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "BF16 row matrix byte size overflows".to_string())?;
    let output_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "BF16 row output byte size overflows".to_string())?;
    check_copy_range(0, matrix_bytes, matrix_buffer.size()?)?;
    check_copy_range(0, output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_bf16_row_f32(
            matrix_buffer.raw.as_ptr(),
            rows,
            cols,
            row_index,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn top1_partial_count(elements: usize) -> Result<usize, String> {
    if elements == 0 {
        return Err("f32 top1 elements must be greater than zero".to_string());
    }
    const BLOCK_SIZE: usize = 256;
    elements
        .checked_add(BLOCK_SIZE - 1)
        .map(|value| value / BLOCK_SIZE)
        .ok_or_else(|| "f32 top1 partial count overflows".to_string())
}

pub fn top1_f32(
    input_buffer: &RuntimeBuffer,
    elements: usize,
    partial_values_buffer: &mut RuntimeBuffer,
    partial_indices_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<usize, String> {
    let partial_count = top1_partial_count(elements)?;
    let input_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 top1 input byte size overflows".to_string())?;
    let partial_values_bytes = partial_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 top1 partial value byte size overflows".to_string())?;
    let partial_indices_bytes = partial_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "f32 top1 partial index byte size overflows".to_string())?;
    check_copy_range(0, input_bytes, input_buffer.size()?)?;
    check_copy_range(0, partial_values_bytes, partial_values_buffer.size()?)?;
    check_copy_range(0, partial_indices_bytes, partial_indices_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_top1_f32(
            input_buffer.raw.as_ptr(),
            elements,
            partial_values_buffer.raw.as_ptr(),
            partial_indices_buffer.raw.as_ptr(),
            partial_count,
            stream,
        )
    })?;
    Ok(partial_count)
}

pub fn top1_pairs_f32_in_place(
    values_buffer: &mut RuntimeBuffer,
    indices_buffer: &mut RuntimeBuffer,
    elements: usize,
    stream: Option<&mut RuntimeStream>,
) -> Result<usize, String> {
    let partial_count = top1_partial_count(elements)?;
    let values_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 top1 pairs value byte size overflows".to_string())?;
    let indices_bytes = elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "f32 top1 pairs index byte size overflows".to_string())?;
    let partial_values_bytes = partial_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 top1 pairs partial value byte size overflows".to_string())?;
    let partial_indices_bytes = partial_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "f32 top1 pairs partial index byte size overflows".to_string())?;
    check_copy_range(0, values_bytes, values_buffer.size()?)?;
    check_copy_range(0, indices_bytes, indices_buffer.size()?)?;
    check_copy_range(0, partial_values_bytes, values_buffer.size()?)?;
    check_copy_range(0, partial_indices_bytes, indices_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_top1_pairs_f32(
            values_buffer.raw.as_ptr(),
            indices_buffer.raw.as_ptr(),
            elements,
            values_buffer.raw.as_ptr(),
            indices_buffer.raw.as_ptr(),
            partial_count,
            stream,
        )
    })?;
    Ok(partial_count)
}

pub fn rmsnorm_f32(
    input_buffer: &RuntimeBuffer,
    weight_buffer: &RuntimeBuffer,
    elements: usize,
    epsilon: f32,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if elements == 0 {
        return Err("f32 RMSNorm elements must be greater than zero".to_string());
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("f32 RMSNorm epsilon must be finite and greater than zero".to_string());
    }
    let required_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 RMSNorm byte size overflows".to_string())?;
    check_copy_range(0, required_bytes, input_buffer.size()?)?;
    check_copy_range(0, required_bytes, weight_buffer.size()?)?;
    check_copy_range(0, required_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_rmsnorm_f32(
            input_buffer.raw.as_ptr(),
            weight_buffer.raw.as_ptr(),
            elements,
            epsilon,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn segmented_rmsnorm_f32(
    input_buffer: &RuntimeBuffer,
    weight_buffer: &RuntimeBuffer,
    segments: usize,
    segment_size: usize,
    epsilon: f32,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if segments == 0 || segment_size == 0 {
        return Err(
            "f32 segmented RMSNorm segments and segment size must be greater than zero".to_string(),
        );
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(
            "f32 segmented RMSNorm epsilon must be finite and greater than zero".to_string(),
        );
    }
    let elements = segments
        .checked_mul(segment_size)
        .ok_or_else(|| "f32 segmented RMSNorm element count overflows".to_string())?;
    let input_output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 segmented RMSNorm input/output byte size overflows".to_string())?;
    let weight_bytes = segment_size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 segmented RMSNorm weight byte size overflows".to_string())?;
    check_copy_range(0, input_output_bytes, input_buffer.size()?)?;
    check_copy_range(0, weight_bytes, weight_buffer.size()?)?;
    check_copy_range(0, input_output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_segmented_rmsnorm_f32(
            input_buffer.raw.as_ptr(),
            weight_buffer.raw.as_ptr(),
            segments,
            segment_size,
            epsilon,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn segmented_rmsnorm_silu_mul_f32(
    input_buffer: &RuntimeBuffer,
    weight_buffer: &RuntimeBuffer,
    gate_buffer: &RuntimeBuffer,
    segments: usize,
    segment_size: usize,
    epsilon: f32,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if segments == 0 || segment_size == 0 {
        return Err(
            "f32 segmented RMSNorm SiLU-mul segments and segment size must be greater than zero"
                .to_string(),
        );
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(
            "f32 segmented RMSNorm SiLU-mul epsilon must be finite and greater than zero"
                .to_string(),
        );
    }
    let elements = segments
        .checked_mul(segment_size)
        .ok_or_else(|| "f32 segmented RMSNorm SiLU-mul element count overflows".to_string())?;
    let input_output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "f32 segmented RMSNorm SiLU-mul input/output byte size overflows".to_string()
        })?;
    let weight_bytes = segment_size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 segmented RMSNorm SiLU-mul weight byte size overflows".to_string())?;
    check_copy_range(0, input_output_bytes, input_buffer.size()?)?;
    check_copy_range(0, weight_bytes, weight_buffer.size()?)?;
    check_copy_range(0, input_output_bytes, gate_buffer.size()?)?;
    check_copy_range(0, input_output_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_segmented_rmsnorm_silu_mul_f32(
            input_buffer.raw.as_ptr(),
            weight_buffer.raw.as_ptr(),
            gate_buffer.raw.as_ptr(),
            segments,
            segment_size,
            epsilon,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn silu_mul_f32(
    gate_buffer: &RuntimeBuffer,
    up_buffer: &RuntimeBuffer,
    elements: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if elements == 0 {
        return Err("f32 SiLU-mul elements must be greater than zero".to_string());
    }
    let required_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 SiLU-mul byte size overflows".to_string())?;
    check_copy_range(0, required_bytes, gate_buffer.size()?)?;
    check_copy_range(0, required_bytes, up_buffer.size()?)?;
    check_copy_range(0, required_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_silu_mul_f32(
            gate_buffer.raw.as_ptr(),
            up_buffer.raw.as_ptr(),
            elements,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn sigmoid_mul_f32(
    gate_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    elements: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if elements == 0 {
        return Err("f32 Sigmoid-mul elements must be greater than zero".to_string());
    }
    let required_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 Sigmoid-mul byte size overflows".to_string())?;
    check_copy_range(0, required_bytes, gate_buffer.size()?)?;
    check_copy_range(0, required_bytes, input_buffer.size()?)?;
    check_copy_range(0, required_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sigmoid_mul_f32(
            gate_buffer.raw.as_ptr(),
            input_buffer.raw.as_ptr(),
            elements,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn sigmoid_mul_f32_in_place(
    gate_buffer: &RuntimeBuffer,
    input_output_buffer: &mut RuntimeBuffer,
    elements: usize,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if elements == 0 {
        return Err("f32 Sigmoid-mul elements must be greater than zero".to_string());
    }
    let required_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 Sigmoid-mul byte size overflows".to_string())?;
    check_copy_range(0, required_bytes, gate_buffer.size()?)?;
    check_copy_range(0, required_bytes, input_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sigmoid_mul_f32(
            gate_buffer.raw.as_ptr(),
            input_output_buffer.raw.as_ptr(),
            elements,
            input_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn qwen35_split_q_gate_f32(
    projected_buffer: &RuntimeBuffer,
    q_heads: usize,
    head_dim: usize,
    query_output_buffer: &mut RuntimeBuffer,
    gate_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if q_heads == 0 || head_dim == 0 {
        return Err("Qwen3.5 q/gate split q_heads and head_dim must be greater than zero".into());
    }
    let output_elements = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Qwen3.5 q/gate split element count overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3.5 q/gate split output byte size overflows".to_string())?;
    let projected_bytes = output_bytes
        .checked_mul(2)
        .ok_or_else(|| "Qwen3.5 q/gate split projected byte size overflows".to_string())?;
    check_copy_range(0, projected_bytes, projected_buffer.size()?)?;
    check_copy_range(0, output_bytes, query_output_buffer.size()?)?;
    check_copy_range(0, output_bytes, gate_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_qwen35_split_q_gate_f32(
            projected_buffer.raw.as_ptr(),
            q_heads,
            head_dim,
            query_output_buffer.raw.as_ptr(),
            gate_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_qk_norm_rope_f32(
    q_projected_buffer: &RuntimeBuffer,
    k_projected_buffer: &RuntimeBuffer,
    q_weight_buffer: &RuntimeBuffer,
    k_weight_buffer: &RuntimeBuffer,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
    epsilon: f32,
    q_gate_output_buffer: &mut RuntimeBuffer,
    q_rope_output_buffer: &mut RuntimeBuffer,
    k_rope_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err("Qwen3.5 q/k norm RoPE heads and head_dim must be greater than zero".into());
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(
            "Qwen3.5 q/k norm RoPE rotary_dim must be even and no greater than head_dim".into(),
        );
    }
    if !rope_base.is_finite() || rope_base <= 1.0 {
        return Err("Qwen3.5 q/k norm RoPE base must be finite and greater than one".into());
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("Qwen3.5 q/k norm RoPE epsilon must be finite and greater than zero".into());
    }
    let q_output_elements = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE q element count overflows".to_string())?;
    let k_output_elements = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE k element count overflows".to_string())?;
    let q_output_bytes = q_output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE q output byte size overflows".to_string())?;
    let k_output_bytes = k_output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE k output byte size overflows".to_string())?;
    let q_projected_bytes = q_output_bytes
        .checked_mul(2)
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE q projected byte size overflows".to_string())?;
    let weight_bytes = head_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE weight byte size overflows".to_string())?;
    check_copy_range(0, q_projected_bytes, q_projected_buffer.size()?)?;
    check_copy_range(0, k_output_bytes, k_projected_buffer.size()?)?;
    check_copy_range(0, weight_bytes, q_weight_buffer.size()?)?;
    check_copy_range(0, weight_bytes, k_weight_buffer.size()?)?;
    check_copy_range(0, q_output_bytes, q_gate_output_buffer.size()?)?;
    check_copy_range(0, q_output_bytes, q_rope_output_buffer.size()?)?;
    check_copy_range(0, k_output_bytes, k_rope_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_qwen35_qk_norm_rope_f32(
            q_projected_buffer.raw.as_ptr(),
            k_projected_buffer.raw.as_ptr(),
            q_weight_buffer.raw.as_ptr(),
            k_weight_buffer.raw.as_ptr(),
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            q_gate_output_buffer.raw.as_ptr(),
            q_rope_output_buffer.raw.as_ptr(),
            k_rope_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_qk_norm_rope_batch_f32(
    q_projected_buffer: &RuntimeBuffer,
    k_projected_buffer: &RuntimeBuffer,
    q_weight_buffer: &RuntimeBuffer,
    k_weight_buffer: &RuntimeBuffer,
    q_heads: usize,
    kv_heads: usize,
    sequence_len: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
    epsilon: f32,
    q_gate_output_buffer: &mut RuntimeBuffer,
    q_rope_output_buffer: &mut RuntimeBuffer,
    k_rope_output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if q_heads == 0 || kv_heads == 0 || sequence_len == 0 || head_dim == 0 {
        return Err(
            "Qwen3.5 q/k norm RoPE batch heads, sequence_len, and head_dim must be greater than zero"
                .into(),
        );
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(
            "Qwen3.5 q/k norm RoPE batch rotary_dim must be even and no greater than head_dim"
                .into(),
        );
    }
    if !rope_base.is_finite() || rope_base <= 1.0 {
        return Err("Qwen3.5 q/k norm RoPE batch base must be finite and greater than one".into());
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(
            "Qwen3.5 q/k norm RoPE batch epsilon must be finite and greater than zero".into(),
        );
    }
    let q_output_elements = sequence_len
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE batch q element count overflows".to_string())?;
    let k_output_elements = sequence_len
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE batch k element count overflows".to_string())?;
    let q_output_bytes = q_output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE batch q output byte size overflows".to_string())?;
    let k_output_bytes = k_output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE batch k output byte size overflows".to_string())?;
    let q_projected_bytes = q_output_bytes
        .checked_mul(2)
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE batch q projected byte size overflows".to_string())?;
    let weight_bytes = head_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE batch weight byte size overflows".to_string())?;
    check_copy_range(0, q_projected_bytes, q_projected_buffer.size()?)?;
    check_copy_range(0, k_output_bytes, k_projected_buffer.size()?)?;
    check_copy_range(0, weight_bytes, q_weight_buffer.size()?)?;
    check_copy_range(0, weight_bytes, k_weight_buffer.size()?)?;
    check_copy_range(0, q_output_bytes, q_gate_output_buffer.size()?)?;
    check_copy_range(0, q_output_bytes, q_rope_output_buffer.size()?)?;
    check_copy_range(0, k_output_bytes, k_rope_output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_qwen35_qk_norm_rope_batch_f32(
            q_projected_buffer.raw.as_ptr(),
            k_projected_buffer.raw.as_ptr(),
            q_weight_buffer.raw.as_ptr(),
            k_weight_buffer.raw.as_ptr(),
            q_heads,
            kv_heads,
            sequence_len,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            q_gate_output_buffer.raw.as_ptr(),
            q_rope_output_buffer.raw.as_ptr(),
            k_rope_output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_qk_norm_rope_paged_kv_write_f32(
    q_projected_buffer: &RuntimeBuffer,
    k_projected_buffer: &RuntimeBuffer,
    v_projected_buffer: &RuntimeBuffer,
    q_weight_buffer: &RuntimeBuffer,
    k_weight_buffer: &RuntimeBuffer,
    block_table_buffer: &RuntimeBuffer,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
    epsilon: f32,
    cache_position: usize,
    block_size: usize,
    cache_blocks: usize,
    q_gate_output_buffer: &mut RuntimeBuffer,
    q_rope_output_buffer: &mut RuntimeBuffer,
    k_cache_buffer: &mut RuntimeBuffer,
    v_cache_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 {
        return Err(
            "Qwen3.5 q/k norm RoPE paged KV write heads and dims must be greater than zero".into(),
        );
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(
            "Qwen3.5 q/k norm RoPE paged KV write rotary_dim must be even and no greater than head_dim"
                .into(),
        );
    }
    if block_size == 0 || cache_blocks == 0 {
        return Err(
            "Qwen3.5 q/k norm RoPE paged KV write block_size and cache_blocks must be greater than zero"
                .into(),
        );
    }
    if !rope_base.is_finite() || rope_base <= 1.0 {
        return Err(
            "Qwen3.5 q/k norm RoPE paged KV write base must be finite and greater than one".into(),
        );
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(
            "Qwen3.5 q/k norm RoPE paged KV write epsilon must be finite and greater than zero"
                .into(),
        );
    }
    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "Qwen3.5 q/k norm RoPE paged KV write cache size overflows".to_string())?;
    if cache_position >= physical_tokens {
        return Err(
            "Qwen3.5 q/k norm RoPE paged KV write cache_position exceeds cache capacity".into(),
        );
    }
    let block_table_entries = cache_position / block_size + 1;
    let q_output_elements = q_heads.checked_mul(head_dim).ok_or_else(|| {
        "Qwen3.5 q/k norm RoPE paged KV write q element count overflows".to_string()
    })?;
    let k_elements = kv_heads.checked_mul(head_dim).ok_or_else(|| {
        "Qwen3.5 q/k norm RoPE paged KV write k element count overflows".to_string()
    })?;
    let v_elements = kv_heads.checked_mul(value_dim).ok_or_else(|| {
        "Qwen3.5 q/k norm RoPE paged KV write v element count overflows".to_string()
    })?;
    let kv_head_cache = physical_tokens.checked_mul(kv_heads).ok_or_else(|| {
        "Qwen3.5 q/k norm RoPE paged KV write cache head count overflows".to_string()
    })?;
    let k_cache_elements = kv_head_cache.checked_mul(head_dim).ok_or_else(|| {
        "Qwen3.5 q/k norm RoPE paged KV write k cache element count overflows".to_string()
    })?;
    let v_cache_elements = kv_head_cache.checked_mul(value_dim).ok_or_else(|| {
        "Qwen3.5 q/k norm RoPE paged KV write v cache element count overflows".to_string()
    })?;
    let q_output_bytes = q_output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Qwen3.5 q/k norm RoPE paged KV write q output byte size overflows".to_string()
        })?;
    let q_projected_bytes = q_output_bytes.checked_mul(2).ok_or_else(|| {
        "Qwen3.5 q/k norm RoPE paged KV write q projected byte size overflows".to_string()
    })?;
    let k_projected_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Qwen3.5 q/k norm RoPE paged KV write k projected byte size overflows".to_string()
        })?;
    let v_projected_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Qwen3.5 q/k norm RoPE paged KV write v projected byte size overflows".to_string()
        })?;
    let weight_bytes = head_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Qwen3.5 q/k norm RoPE paged KV write weight byte size overflows".to_string()
        })?;
    let block_table_bytes = block_table_entries
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            "Qwen3.5 q/k norm RoPE paged KV write block table byte size overflows".to_string()
        })?;
    let k_cache_bytes = k_cache_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Qwen3.5 q/k norm RoPE paged KV write k cache byte size overflows".to_string()
        })?;
    let v_cache_bytes = v_cache_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Qwen3.5 q/k norm RoPE paged KV write v cache byte size overflows".to_string()
        })?;
    check_copy_range(0, q_projected_bytes, q_projected_buffer.size()?)?;
    check_copy_range(0, k_projected_bytes, k_projected_buffer.size()?)?;
    check_copy_range(0, v_projected_bytes, v_projected_buffer.size()?)?;
    check_copy_range(0, weight_bytes, q_weight_buffer.size()?)?;
    check_copy_range(0, weight_bytes, k_weight_buffer.size()?)?;
    check_copy_range(0, block_table_bytes, block_table_buffer.size()?)?;
    check_copy_range(0, q_output_bytes, q_gate_output_buffer.size()?)?;
    check_copy_range(0, q_output_bytes, q_rope_output_buffer.size()?)?;
    check_copy_range(0, k_cache_bytes, k_cache_buffer.size()?)?;
    check_copy_range(0, v_cache_bytes, v_cache_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_qwen35_qk_norm_rope_paged_kv_write_f32(
            q_projected_buffer.raw.as_ptr(),
            k_projected_buffer.raw.as_ptr(),
            v_projected_buffer.raw.as_ptr(),
            q_weight_buffer.raw.as_ptr(),
            k_weight_buffer.raw.as_ptr(),
            block_table_buffer.raw.as_ptr(),
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            cache_position,
            block_size,
            cache_blocks,
            q_gate_output_buffer.raw.as_ptr(),
            q_rope_output_buffer.raw.as_ptr(),
            k_cache_buffer.raw.as_ptr(),
            v_cache_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn add_f32(
    lhs_buffer: &RuntimeBuffer,
    rhs_buffer: &RuntimeBuffer,
    elements: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if elements == 0 {
        return Err("f32 add elements must be greater than zero".to_string());
    }
    let required_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 add byte size overflows".to_string())?;
    check_copy_range(0, required_bytes, lhs_buffer.size()?)?;
    check_copy_range(0, required_bytes, rhs_buffer.size()?)?;
    check_copy_range(0, required_bytes, output_buffer.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_add_f32(
            lhs_buffer.raw.as_ptr(),
            rhs_buffer.raw.as_ptr(),
            elements,
            output_buffer.raw.as_ptr(),
            stream,
        )
    })
}

pub fn rope_f32(
    input: &RuntimeBuffer,
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if sequence_len == 0 {
        return Err("f32 RoPE sequence_len must be greater than zero".to_string());
    }
    if heads == 0 {
        return Err("f32 RoPE heads must be greater than zero".to_string());
    }
    if head_dim == 0 {
        return Err("f32 RoPE head_dim must be greater than zero".to_string());
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err("f32 RoPE rotary_dim must be even and no greater than head_dim".to_string());
    }
    if !rope_base.is_finite() || rope_base <= 1.0 {
        return Err("f32 RoPE base must be finite and greater than one".to_string());
    }
    let head_sequence = sequence_len
        .checked_mul(heads)
        .ok_or_else(|| "f32 RoPE head-sequence element count overflows".to_string())?;
    let elements = head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 RoPE element count overflows".to_string())?;
    let required_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 RoPE byte size overflows".to_string())?;
    check_copy_range(0, required_bytes, input.size()?)?;
    check_copy_range(0, required_bytes, output.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_rope_f32(
            input.raw.as_ptr(),
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn causal_attn_f32(
    q: &RuntimeBuffer,
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if sequence_len == 0 {
        return Err("f32 causal attention sequence_len must be greater than zero".to_string());
    }
    if q_heads == 0 {
        return Err("f32 causal attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 causal attention kv_heads must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err("f32 causal attention q_heads must be a multiple of kv_heads".to_string());
    }
    if head_dim == 0 {
        return Err("f32 causal attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 causal attention value_dim must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 causal attention softmax scale must be finite and greater than zero".to_string(),
        );
    }

    let q_head_sequence = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| "f32 causal attention q head-sequence count overflows".to_string())?;
    let kv_head_sequence = sequence_len
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 causal attention kv head-sequence count overflows".to_string())?;
    let q_elements = q_head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 causal attention q element count overflows".to_string())?;
    let k_elements = kv_head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 causal attention k element count overflows".to_string())?;
    let v_elements = kv_head_sequence
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 causal attention v element count overflows".to_string())?;
    let output_elements = q_head_sequence
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 causal attention output element count overflows".to_string())?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k.size()?)?;
    check_copy_range(0, v_bytes, v.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_causal_attn_f32(
            q.raw.as_ptr(),
            k.raw.as_ptr(),
            v.raw.as_ptr(),
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn causal_attn_f32_flash2(
    q: &RuntimeBuffer,
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if sequence_len == 0 {
        return Err("f32 causal attention sequence_len must be greater than zero".to_string());
    }
    if q_heads == 0 {
        return Err("f32 causal attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 causal attention kv_heads must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err("f32 causal attention q_heads must be a multiple of kv_heads".to_string());
    }
    if head_dim == 0 {
        return Err("f32 causal attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 causal attention value_dim must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 causal attention softmax scale must be finite and greater than zero".to_string(),
        );
    }

    let q_head_sequence = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| "f32 causal attention q head-sequence count overflows".to_string())?;
    let kv_head_sequence = sequence_len
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 causal attention kv head-sequence count overflows".to_string())?;
    let q_elements = q_head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 causal attention q element count overflows".to_string())?;
    let k_elements = kv_head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 causal attention k element count overflows".to_string())?;
    let v_elements = kv_head_sequence
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 causal attention v element count overflows".to_string())?;
    let output_elements = q_head_sequence
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 causal attention output element count overflows".to_string())?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 causal attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k.size()?)?;
    check_copy_range(0, v_bytes, v.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_causal_attn_f32_flash2(
            q.raw.as_ptr(),
            k.raw.as_ptr(),
            v.raw.as_ptr(),
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn causal_attn_batch_f32(
    q: &RuntimeBuffer,
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    batch_count: usize,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if batch_count == 0 {
        return Err(
            "f32 batched causal attention batch_count must be greater than zero".to_string(),
        );
    }
    if sequence_len == 0 {
        return Err(
            "f32 batched causal attention sequence_len must be greater than zero".to_string(),
        );
    }
    if q_heads == 0 {
        return Err("f32 batched causal attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 batched causal attention kv_heads must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "f32 batched causal attention q_heads must be a multiple of kv_heads".to_string(),
        );
    }
    if head_dim == 0 {
        return Err("f32 batched causal attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 batched causal attention value_dim must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 batched causal attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }

    let q_head_sequence_per_batch = sequence_len.checked_mul(q_heads).ok_or_else(|| {
        "f32 batched causal attention q head-sequence count overflows".to_string()
    })?;
    let kv_head_sequence_per_batch = sequence_len.checked_mul(kv_heads).ok_or_else(|| {
        "f32 batched causal attention kv head-sequence count overflows".to_string()
    })?;
    let q_elements_per_batch = q_head_sequence_per_batch
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 batched causal attention q element count overflows".to_string())?;
    let k_elements_per_batch = kv_head_sequence_per_batch
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 batched causal attention k element count overflows".to_string())?;
    let v_elements_per_batch = kv_head_sequence_per_batch
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 batched causal attention v element count overflows".to_string())?;
    let output_elements_per_batch = q_head_sequence_per_batch
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 batched causal attention output element count overflows".to_string())?;

    let q_elements = q_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention q batch element count overflows".to_string()
        })?;
    let k_elements = k_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention k batch element count overflows".to_string()
        })?;
    let v_elements = v_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention v batch element count overflows".to_string()
        })?;
    let output_elements = output_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention output batch element count overflows".to_string()
        })?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k.size()?)?;
    check_copy_range(0, v_bytes, v.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_causal_attn_batch_f32(
            q.raw.as_ptr(),
            k.raw.as_ptr(),
            v.raw.as_ptr(),
            batch_count,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn causal_attn_batch_f32_flash2(
    q: &RuntimeBuffer,
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    batch_count: usize,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if batch_count == 0 {
        return Err(
            "f32 batched causal attention batch_count must be greater than zero".to_string(),
        );
    }
    if sequence_len == 0 {
        return Err(
            "f32 batched causal attention sequence_len must be greater than zero".to_string(),
        );
    }
    if q_heads == 0 {
        return Err("f32 batched causal attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 batched causal attention kv_heads must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "f32 batched causal attention q_heads must be a multiple of kv_heads".to_string(),
        );
    }
    if head_dim == 0 {
        return Err("f32 batched causal attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 batched causal attention value_dim must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 batched causal attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }

    let q_head_sequence_per_batch = sequence_len.checked_mul(q_heads).ok_or_else(|| {
        "f32 batched causal attention q head-sequence count overflows".to_string()
    })?;
    let kv_head_sequence_per_batch = sequence_len.checked_mul(kv_heads).ok_or_else(|| {
        "f32 batched causal attention kv head-sequence count overflows".to_string()
    })?;
    let q_elements_per_batch = q_head_sequence_per_batch
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 batched causal attention q element count overflows".to_string())?;
    let k_elements_per_batch = kv_head_sequence_per_batch
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 batched causal attention k element count overflows".to_string())?;
    let v_elements_per_batch = kv_head_sequence_per_batch
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 batched causal attention v element count overflows".to_string())?;
    let output_elements_per_batch = q_head_sequence_per_batch
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 batched causal attention output element count overflows".to_string())?;

    let q_elements = q_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention q batch element count overflows".to_string()
        })?;
    let k_elements = k_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention k batch element count overflows".to_string()
        })?;
    let v_elements = v_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention v batch element count overflows".to_string()
        })?;
    let output_elements = output_elements_per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| {
            "f32 batched causal attention output batch element count overflows".to_string()
        })?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 batched causal attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k.size()?)?;
    check_copy_range(0, v_bytes, v.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_causal_attn_batch_f32_flash2(
            q.raw.as_ptr(),
            k.raw.as_ptr(),
            v.raw.as_ptr(),
            batch_count,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn decode_attn_f32(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if cache_len == 0 {
        return Err("f32 decode attention cache_len must be greater than zero".to_string());
    }
    if q_heads == 0 {
        return Err("f32 decode attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 decode attention kv_heads must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err("f32 decode attention q_heads must be a multiple of kv_heads".to_string());
    }
    if head_dim == 0 {
        return Err("f32 decode attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 decode attention value_dim must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 decode attention softmax scale must be finite and greater than zero".to_string(),
        );
    }

    let q_elements = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 decode attention q element count overflows".to_string())?;
    let kv_head_sequence = cache_len
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 decode attention kv head-cache count overflows".to_string())?;
    let k_elements = kv_head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 decode attention k element count overflows".to_string())?;
    let v_elements = kv_head_sequence
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 decode attention v element count overflows".to_string())?;
    let output_elements = q_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 decode attention output element count overflows".to_string())?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 decode attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 decode attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 decode attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 decode attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k_cache.size()?)?;
    check_copy_range(0, v_bytes, v_cache.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_decode_attn_f32(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn cached_prefix_attn_f32(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    cached_prefix_len: usize,
    new_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if new_tokens == 0 {
        return Err("f32 cached prefix attention new_tokens must be greater than zero".to_string());
    }
    if q_heads == 0 {
        return Err("f32 cached prefix attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 cached prefix attention kv_heads must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "f32 cached prefix attention q_heads must be a multiple of kv_heads".to_string(),
        );
    }
    if head_dim == 0 {
        return Err("f32 cached prefix attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 cached prefix attention value_dim must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 cached prefix attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }

    let total_context = cached_prefix_len
        .checked_add(new_tokens)
        .ok_or_else(|| "f32 cached prefix attention total context overflows".to_string())?;
    let q_head_sequence = new_tokens
        .checked_mul(q_heads)
        .ok_or_else(|| "f32 cached prefix attention q head-sequence count overflows".to_string())?;
    let q_elements = q_head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 cached prefix attention q element count overflows".to_string())?;
    let kv_head_context = total_context
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 cached prefix attention kv head-context count overflows".to_string())?;
    let k_elements = kv_head_context
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 cached prefix attention k element count overflows".to_string())?;
    let v_elements = kv_head_context
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 cached prefix attention v element count overflows".to_string())?;
    let output_elements = q_head_sequence
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 cached prefix attention output element count overflows".to_string())?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 cached prefix attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 cached prefix attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 cached prefix attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 cached prefix attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k_cache.size()?)?;
    check_copy_range(0, v_bytes, v_cache.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_cached_prefix_attn_f32(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn cached_prefix_attn_f32_flash2(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    cached_prefix_len: usize,
    new_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if new_tokens == 0 {
        return Err(
            "f32 cached prefix flash2 attention new_tokens must be greater than zero".to_string(),
        );
    }
    if q_heads == 0 {
        return Err(
            "f32 cached prefix flash2 attention q_heads must be greater than zero".to_string(),
        );
    }
    if kv_heads == 0 {
        return Err(
            "f32 cached prefix flash2 attention kv_heads must be greater than zero".to_string(),
        );
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "f32 cached prefix flash2 attention q_heads must be a multiple of kv_heads".to_string(),
        );
    }
    if head_dim == 0 {
        return Err(
            "f32 cached prefix flash2 attention head_dim must be greater than zero".to_string(),
        );
    }
    if value_dim == 0 {
        return Err(
            "f32 cached prefix flash2 attention value_dim must be greater than zero".to_string(),
        );
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 cached prefix flash2 attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }

    let total_context = cached_prefix_len
        .checked_add(new_tokens)
        .ok_or_else(|| "f32 cached prefix flash2 attention total context overflows".to_string())?;
    let q_head_sequence = new_tokens.checked_mul(q_heads).ok_or_else(|| {
        "f32 cached prefix flash2 attention q head-sequence count overflows".to_string()
    })?;
    let q_elements = q_head_sequence.checked_mul(head_dim).ok_or_else(|| {
        "f32 cached prefix flash2 attention q element count overflows".to_string()
    })?;
    let kv_head_context = total_context.checked_mul(kv_heads).ok_or_else(|| {
        "f32 cached prefix flash2 attention kv head-context count overflows".to_string()
    })?;
    let k_elements = kv_head_context.checked_mul(head_dim).ok_or_else(|| {
        "f32 cached prefix flash2 attention k element count overflows".to_string()
    })?;
    let v_elements = kv_head_context.checked_mul(value_dim).ok_or_else(|| {
        "f32 cached prefix flash2 attention v element count overflows".to_string()
    })?;
    let output_elements = q_head_sequence.checked_mul(value_dim).ok_or_else(|| {
        "f32 cached prefix flash2 attention output element count overflows".to_string()
    })?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 cached prefix flash2 attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 cached prefix flash2 attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 cached prefix flash2 attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "f32 cached prefix flash2 attention output byte size overflows".to_string()
        })?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k_cache.size()?)?;
    check_copy_range(0, v_bytes, v_cache.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_cached_prefix_attn_f32_flash2(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn cached_prefix_attn_fp8_e4m3(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    cached_prefix_len: usize,
    new_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    k_scale: f32,
    v_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if new_tokens == 0 {
        return Err(
            "fp8 e4m3 cached prefix attention new_tokens must be greater than zero".to_string(),
        );
    }
    if q_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix attention q_heads must be greater than zero".to_string(),
        );
    }
    if kv_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix attention kv_heads must be greater than zero".to_string(),
        );
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "fp8 e4m3 cached prefix attention q_heads must be a multiple of kv_heads".to_string(),
        );
    }
    if head_dim == 0 {
        return Err(
            "fp8 e4m3 cached prefix attention head_dim must be greater than zero".to_string(),
        );
    }
    if value_dim == 0 {
        return Err(
            "fp8 e4m3 cached prefix attention value_dim must be greater than zero".to_string(),
        );
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "fp8 e4m3 cached prefix attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }
    if !k_scale.is_finite() || k_scale <= 0.0 || !v_scale.is_finite() || v_scale <= 0.0 {
        return Err(
            "fp8 e4m3 cached prefix attention scales must be finite and greater than zero"
                .to_string(),
        );
    }

    let total_context = cached_prefix_len
        .checked_add(new_tokens)
        .ok_or_else(|| "fp8 e4m3 cached prefix attention total context overflows".to_string())?;
    let q_head_sequence = new_tokens.checked_mul(q_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix attention q head-sequence count overflows".to_string()
    })?;
    let q_elements = q_head_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| "fp8 e4m3 cached prefix attention q element count overflows".to_string())?;
    let kv_head_context = total_context.checked_mul(kv_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix attention kv head-context count overflows".to_string()
    })?;
    let k_elements = kv_head_context
        .checked_mul(head_dim)
        .ok_or_else(|| "fp8 e4m3 cached prefix attention k element count overflows".to_string())?;
    let v_elements = kv_head_context
        .checked_mul(value_dim)
        .ok_or_else(|| "fp8 e4m3 cached prefix attention v element count overflows".to_string())?;
    let output_elements = q_head_sequence.checked_mul(value_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix attention output element count overflows".to_string()
    })?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "fp8 e4m3 cached prefix attention q byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "fp8 e4m3 cached prefix attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_elements, k_cache.size()?)?;
    check_copy_range(0, v_elements, v_cache.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_cached_prefix_attn_fp8_e4m3(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn cached_prefix_attn_fp8_e4m3_flash2(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    cached_prefix_len: usize,
    new_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    k_scale: f32,
    v_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if new_tokens == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention new_tokens must be greater than zero"
                .to_string(),
        );
    }
    if q_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention q_heads must be greater than zero".to_string(),
        );
    }
    if kv_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention kv_heads must be greater than zero"
                .to_string(),
        );
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention q_heads must be a multiple of kv_heads"
                .to_string(),
        );
    }
    if head_dim == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention head_dim must be greater than zero"
                .to_string(),
        );
    }
    if value_dim == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention value_dim must be greater than zero"
                .to_string(),
        );
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }
    if !k_scale.is_finite() || k_scale <= 0.0 || !v_scale.is_finite() || v_scale <= 0.0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 attention scales must be finite and greater than zero"
                .to_string(),
        );
    }

    let total_context = cached_prefix_len.checked_add(new_tokens).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 attention total context overflows".to_string()
    })?;
    let q_head_sequence = new_tokens.checked_mul(q_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 attention q head-sequence count overflows".to_string()
    })?;
    let q_elements = q_head_sequence.checked_mul(head_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 attention q element count overflows".to_string()
    })?;
    let kv_head_context = total_context.checked_mul(kv_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 attention kv head-context count overflows".to_string()
    })?;
    let k_elements = kv_head_context.checked_mul(head_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 attention k element count overflows".to_string()
    })?;
    let v_elements = kv_head_context.checked_mul(value_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 attention v element count overflows".to_string()
    })?;
    let output_elements = q_head_sequence.checked_mul(value_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 attention output element count overflows".to_string()
    })?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "fp8 e4m3 cached prefix flash2 attention q byte size overflows".to_string()
        })?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "fp8 e4m3 cached prefix flash2 attention output byte size overflows".to_string()
        })?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_elements, k_cache.size()?)?;
    check_copy_range(0, v_elements, v_cache.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            k_scale,
            v_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn cached_prefix_attn_fp8_e4m3_flash2_fp8q(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    cached_prefix_len: usize,
    new_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if new_tokens == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention new_tokens must be greater than zero"
                .to_string(),
        );
    }
    if q_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention q_heads must be greater than zero"
                .to_string(),
        );
    }
    if kv_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention kv_heads must be greater than zero"
                .to_string(),
        );
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention q_heads must be a multiple of kv_heads"
                .to_string(),
        );
    }
    if head_dim == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention head_dim must be greater than zero"
                .to_string(),
        );
    }
    if value_dim == 0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention value_dim must be greater than zero"
                .to_string(),
        );
    }
    if value_dim > 256 {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention value_dim exceeds 256".to_string(),
        );
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }
    if !q_scale.is_finite()
        || q_scale <= 0.0
        || !k_scale.is_finite()
        || k_scale <= 0.0
        || !v_scale.is_finite()
        || v_scale <= 0.0
    {
        return Err(
            "fp8 e4m3 cached prefix flash2 fp8q attention scales must be finite and greater than zero"
                .to_string(),
        );
    }

    let total_context = cached_prefix_len.checked_add(new_tokens).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 fp8q attention total context overflows".to_string()
    })?;
    let q_head_sequence = new_tokens.checked_mul(q_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 fp8q attention q head-sequence count overflows".to_string()
    })?;
    let q_elements = q_head_sequence.checked_mul(head_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 fp8q attention q element count overflows".to_string()
    })?;
    let kv_head_context = total_context.checked_mul(kv_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 fp8q attention kv head-context count overflows".to_string()
    })?;
    let k_elements = kv_head_context.checked_mul(head_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 fp8q attention k element count overflows".to_string()
    })?;
    let v_elements = kv_head_context.checked_mul(value_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 fp8q attention v element count overflows".to_string()
    })?;
    let output_elements = q_head_sequence.checked_mul(value_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix flash2 fp8q attention output element count overflows".to_string()
    })?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "fp8 e4m3 cached prefix flash2 fp8q attention output byte size overflows".to_string()
        })?;

    check_copy_range(0, q_elements, q.size()?)?;
    check_copy_range(0, k_elements, k_cache.size()?)?;
    check_copy_range(0, v_elements, v_cache.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2_fp8q(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn cached_prefix_attn_fp8_e4m3_rocwmma(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    cached_prefix_len: usize,
    new_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if new_tokens == 0 {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention new_tokens must be greater than zero"
                .to_string(),
        );
    }
    if q_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention q_heads must be greater than zero"
                .to_string(),
        );
    }
    if kv_heads == 0 {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention kv_heads must be greater than zero"
                .to_string(),
        );
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention q_heads must be a multiple of kv_heads"
                .to_string(),
        );
    }
    if !(q_heads / kv_heads).is_multiple_of(16) {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention requires q_heads/kv_heads to be a multiple of 16"
                .to_string(),
        );
    }
    if !head_dim.is_multiple_of(16) || !value_dim.is_multiple_of(16) {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention currently requires head_dim and value_dim to be multiples of 16"
                .to_string(),
        );
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }
    if !q_scale.is_finite()
        || q_scale <= 0.0
        || !k_scale.is_finite()
        || k_scale <= 0.0
        || !v_scale.is_finite()
        || v_scale <= 0.0
    {
        return Err(
            "fp8 e4m3 cached prefix rocWMMA attention scales must be finite and greater than zero"
                .to_string(),
        );
    }

    let total_context = cached_prefix_len.checked_add(new_tokens).ok_or_else(|| {
        "fp8 e4m3 cached prefix rocWMMA attention total context overflows".to_string()
    })?;
    let q_head_sequence = new_tokens.checked_mul(q_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix rocWMMA attention q head-sequence count overflows".to_string()
    })?;
    let q_elements = q_head_sequence.checked_mul(head_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix rocWMMA attention q element count overflows".to_string()
    })?;
    let kv_head_context = total_context.checked_mul(kv_heads).ok_or_else(|| {
        "fp8 e4m3 cached prefix rocWMMA attention kv head-context count overflows".to_string()
    })?;
    let k_elements = kv_head_context.checked_mul(head_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix rocWMMA attention k element count overflows".to_string()
    })?;
    let v_elements = kv_head_context.checked_mul(value_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix rocWMMA attention v element count overflows".to_string()
    })?;
    let output_elements = q_head_sequence.checked_mul(value_dim).ok_or_else(|| {
        "fp8 e4m3 cached prefix rocWMMA attention output element count overflows".to_string()
    })?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            "fp8 e4m3 cached prefix rocWMMA attention output byte size overflows".to_string()
        })?;

    check_copy_range(0, q_elements, q.size()?)?;
    check_copy_range(0, k_elements, k_cache.size()?)?;
    check_copy_range(0, v_elements, v_cache.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_cached_prefix_attn_fp8_e4m3_rocwmma(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            cached_prefix_len,
            new_tokens,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn paged_decode_attn_f32(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    block_table: &RuntimeBuffer,
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if cache_len == 0 {
        return Err("f32 paged decode attention cache_len must be greater than zero".to_string());
    }
    if block_size == 0 {
        return Err("f32 paged decode attention block_size must be greater than zero".to_string());
    }
    if cache_blocks == 0 {
        return Err(
            "f32 paged decode attention cache_blocks must be greater than zero".to_string(),
        );
    }
    if q_heads == 0 {
        return Err("f32 paged decode attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 paged decode attention kv_heads must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(
            "f32 paged decode attention q_heads must be a multiple of kv_heads".to_string(),
        );
    }
    if head_dim == 0 {
        return Err("f32 paged decode attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 paged decode attention value_dim must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "f32 paged decode attention softmax scale must be finite and greater than zero"
                .to_string(),
        );
    }

    let block_table_len = cache_len
        .checked_sub(1)
        .and_then(|value| value.checked_div(block_size))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "f32 paged decode attention block table length overflows".to_string())?;
    let q_elements = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 paged decode attention q element count overflows".to_string())?;
    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "f32 paged decode attention physical cache size overflows".to_string())?;
    let kv_head_cache = physical_tokens
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 paged decode attention kv head-cache count overflows".to_string())?;
    let k_elements = kv_head_cache
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 paged decode attention k element count overflows".to_string())?;
    let v_elements = kv_head_cache
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 paged decode attention v element count overflows".to_string())?;
    let output_elements = q_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 paged decode attention output element count overflows".to_string())?;
    let block_table_elements = block_table_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "f32 paged decode attention block_table byte size overflows".to_string())?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged decode attention q byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged decode attention k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged decode attention v byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged decode attention output byte size overflows".to_string())?;

    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k_cache.size()?)?;
    check_copy_range(0, v_bytes, v_cache.size()?)?;
    check_copy_range(0, block_table_elements, block_table.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_paged_decode_attn_f32(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            block_table.raw.as_ptr(),
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_sigmoid_gate_f32(
    q: &RuntimeBuffer,
    gate: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    block_table: &RuntimeBuffer,
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if q_heads == 0 || value_dim == 0 {
        return Err(
            "f32 paged decode attention sigmoid gate q_heads/value_dim must be greater than zero"
                .to_string(),
        );
    }
    let gate_bytes = q_heads
        .checked_mul(value_dim)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "f32 paged decode attention sigmoid gate byte size overflows".to_string())?;
    check_copy_range(0, gate_bytes, gate.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_paged_decode_attn_sigmoid_gate_f32(
            q.raw.as_ptr(),
            gate.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            block_table.raw.as_ptr(),
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn paged_kv_write_f32(
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    block_table: &RuntimeBuffer,
    cache_position: usize,
    block_size: usize,
    cache_blocks: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    k_cache: &mut RuntimeBuffer,
    v_cache: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if block_size == 0 {
        return Err("f32 paged KV write block_size must be greater than zero".to_string());
    }
    if cache_blocks == 0 {
        return Err("f32 paged KV write cache_blocks must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("f32 paged KV write kv_heads must be greater than zero".to_string());
    }
    if head_dim == 0 {
        return Err("f32 paged KV write head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("f32 paged KV write value_dim must be greater than zero".to_string());
    }

    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "f32 paged KV write physical cache size overflows".to_string())?;
    if cache_position >= physical_tokens {
        return Err(
            "f32 paged KV write cache_position exceeds physical cache capacity".to_string(),
        );
    }
    let block_table_len = cache_position
        .checked_div(block_size)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "f32 paged KV write block table length overflows".to_string())?;
    let k_elements = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 paged KV write k element count overflows".to_string())?;
    let v_elements = kv_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 paged KV write v element count overflows".to_string())?;
    let kv_head_cache = physical_tokens
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 paged KV write kv head-cache count overflows".to_string())?;
    let k_cache_elements = kv_head_cache
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 paged KV write k cache element count overflows".to_string())?;
    let v_cache_elements = kv_head_cache
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 paged KV write v cache element count overflows".to_string())?;

    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged KV write k byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged KV write v byte size overflows".to_string())?;
    let k_cache_bytes = k_cache_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged KV write k cache byte size overflows".to_string())?;
    let v_cache_bytes = v_cache_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged KV write v cache byte size overflows".to_string())?;
    let block_table_bytes = block_table_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "f32 paged KV write block_table byte size overflows".to_string())?;

    check_copy_range(0, k_bytes, k.size()?)?;
    check_copy_range(0, v_bytes, v.size()?)?;
    check_copy_range(0, block_table_bytes, block_table.size()?)?;
    check_copy_range(0, k_cache_bytes, k_cache.size()?)?;
    check_copy_range(0, v_cache_bytes, v_cache.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_paged_kv_write_f32(
            k.raw.as_ptr(),
            v.raw.as_ptr(),
            block_table.raw.as_ptr(),
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn paged_kv_write_chunk_f32(
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    block_table: &RuntimeBuffer,
    cache_start: usize,
    m: usize,
    block_size: usize,
    cache_blocks: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    k_cache: &mut RuntimeBuffer,
    v_cache: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if m == 0 || m > 128 {
        return Err("f32 paged KV write chunk m must be in 1..=128".to_string());
    }
    if block_size == 0 || cache_blocks == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 {
        return Err("f32 paged KV write chunk dimensions must be greater than zero".to_string());
    }
    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "f32 paged KV write chunk physical cache size overflows".to_string())?;
    let end = cache_start
        .checked_add(m)
        .ok_or_else(|| "f32 paged KV write chunk logical range overflows".to_string())?;
    if end > physical_tokens {
        return Err("f32 paged KV write chunk logical range exceeds physical cache capacity".to_string());
    }
    let table_entries = (end - 1) / block_size + 1;
    let k_row = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 paged KV write chunk k row elements overflow".to_string())?;
    let v_row = kv_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 paged KV write chunk v row elements overflow".to_string())?;
    let k_bytes = m
        .checked_mul(k_row)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "f32 paged KV write chunk k byte size overflows".to_string())?;
    let v_bytes = m
        .checked_mul(v_row)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "f32 paged KV write chunk v byte size overflows".to_string())?;
    let cache_heads = physical_tokens
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 paged KV write chunk cache elements overflow".to_string())?;
    let k_cache_bytes = cache_heads
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "f32 paged KV write chunk k cache byte size overflows".to_string())?;
    let v_cache_bytes = cache_heads
        .checked_mul(value_dim)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "f32 paged KV write chunk v cache byte size overflows".to_string())?;
    let table_bytes = table_entries
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "f32 paged KV write chunk table byte size overflows".to_string())?;
    check_copy_range(0, k_bytes, k.size()?)?;
    check_copy_range(0, v_bytes, v.size()?)?;
    check_copy_range(0, table_bytes, block_table.size()?)?;
    check_copy_range(0, k_cache_bytes, k_cache.size()?)?;
    check_copy_range(0, v_cache_bytes, v_cache.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_paged_kv_write_chunk_f32(
            k.raw.as_ptr(),
            v.raw.as_ptr(),
            block_table.raw.as_ptr(),
            cache_start,
            m,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn paged_causal_gqa_chunk_sizes(
    cached_prefix_len: usize,
    m: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
) -> Result<(usize, usize, usize, usize, usize, usize), String> {
    if m == 0 || m > 128 {
        return Err("f32 paged causal GQA chunk m must be in 1..=128".to_string());
    }
    if block_size == 0 || cache_blocks == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 {
        return Err("f32 paged causal GQA chunk dimensions must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err("f32 paged causal GQA chunk q_heads must be a multiple of kv_heads".to_string());
    }
    if head_dim > 256 || value_dim > 256 {
        return Err("f32 paged causal GQA chunk head_dim/value_dim must be at most 256".to_string());
    }
    let total_context = cached_prefix_len
        .checked_add(m)
        .ok_or_else(|| "f32 paged causal GQA chunk total context overflows".to_string())?;
    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "f32 paged causal GQA chunk physical cache size overflows".to_string())?;
    if total_context > physical_tokens {
        return Err("f32 paged causal GQA chunk context exceeds physical cache capacity".to_string());
    }
    let table_entries = (total_context - 1) / block_size + 1;
    let q_elements = m
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "f32 paged causal GQA chunk q element count overflows".to_string())?;
    let output_elements = m
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| "f32 paged causal GQA chunk output element count overflows".to_string())?;
    let cache_heads = physical_tokens
        .checked_mul(kv_heads)
        .ok_or_else(|| "f32 paged causal GQA chunk cache element count overflows".to_string())?;
    let k_elements = cache_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "f32 paged causal GQA chunk k cache element count overflows".to_string())?;
    let v_elements = cache_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "f32 paged causal GQA chunk v cache element count overflows".to_string())?;
    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged causal GQA chunk q byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged causal GQA chunk output byte size overflows".to_string())?;
    let k_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged causal GQA chunk k cache byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "f32 paged causal GQA chunk v cache byte size overflows".to_string())?;
    let table_bytes = table_entries
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "f32 paged causal GQA chunk table byte size overflows".to_string())?;
    Ok((q_bytes, output_bytes, q_bytes, k_bytes, v_bytes, table_bytes))
}

#[allow(clippy::too_many_arguments)]
pub fn paged_causal_gqa_chunk_f32(
    q: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    block_table: &RuntimeBuffer,
    cached_prefix_len: usize,
    m: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err("f32 paged causal GQA chunk softmax scale must be finite and greater than zero".to_string());
    }
    let (q_bytes, output_bytes, _, k_bytes, v_bytes, table_bytes) = paged_causal_gqa_chunk_sizes(
        cached_prefix_len, m, block_size, cache_blocks, q_heads, kv_heads, head_dim, value_dim,
    )?;
    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, k_bytes, k_cache.size()?)?;
    check_copy_range(0, v_bytes, v_cache.size()?)?;
    check_copy_range(0, table_bytes, block_table.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_paged_causal_gqa_chunk_f32(
            q.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            block_table.raw.as_ptr(),
            cached_prefix_len,
            m,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn paged_causal_gqa_chunk_sigmoid_gate_f32(
    q: &RuntimeBuffer,
    gate: &RuntimeBuffer,
    k_cache: &RuntimeBuffer,
    v_cache: &RuntimeBuffer,
    block_table: &RuntimeBuffer,
    cached_prefix_len: usize,
    m: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if head_dim != value_dim {
        return Err("f32 paged causal GQA chunk gated ABI requires head_dim == value_dim".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err("f32 paged causal GQA chunk softmax scale must be finite and greater than zero".to_string());
    }
    let (q_bytes, output_bytes, gate_bytes, k_bytes, v_bytes, table_bytes) = paged_causal_gqa_chunk_sizes(
        cached_prefix_len, m, block_size, cache_blocks, q_heads, kv_heads, head_dim, value_dim,
    )?;
    check_copy_range(0, q_bytes, q.size()?)?;
    check_copy_range(0, gate_bytes, gate.size()?)?;
    check_copy_range(0, k_bytes, k_cache.size()?)?;
    check_copy_range(0, v_bytes, v_cache.size()?)?;
    check_copy_range(0, table_bytes, block_table.size()?)?;
    check_copy_range(0, output_bytes, output.size()?)?;
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_paged_causal_gqa_chunk_sigmoid_gate_f32(
            q.raw.as_ptr(),
            gate.raw.as_ptr(),
            k_cache.raw.as_ptr(),
            v_cache.raw.as_ptr(),
            block_table.raw.as_ptr(),
            cached_prefix_len,
            m,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attn_qkv_prepare_f32(
    qkv: &RuntimeBuffer,
    conv_weight: &RuntimeBuffer,
    conv_history: &mut RuntimeBuffer,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel_size: usize,
    q_scale: f32,
    qk_l2_norm: bool,
    conv_output: &mut RuntimeBuffer,
    q_output: &mut RuntimeBuffer,
    k_output: &mut RuntimeBuffer,
    v_output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 || kernel_size == 0 {
        return Err(
            "linear attention qkv prepare dimensions must be greater than zero".to_string(),
        );
    }
    if !q_scale.is_finite() {
        return Err("linear attention qkv prepare q_scale must be finite".to_string());
    }
    let q_elements = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "linear attention qkv prepare q element count overflows".to_string())?;
    let v_elements = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "linear attention qkv prepare v element count overflows".to_string())?;
    let channels = q_elements
        .checked_add(q_elements)
        .and_then(|value| value.checked_add(v_elements))
        .ok_or_else(|| "linear attention qkv prepare channel count overflows".to_string())?;
    let qkv_bytes = channels
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention qkv prepare qkv byte size overflows".to_string())?;
    let history_bytes = channels
        .checked_mul(kernel_size)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "linear attention qkv prepare history byte size overflows".to_string())?;
    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention qkv prepare q byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention qkv prepare v byte size overflows".to_string())?;

    check_copy_range(0, qkv_bytes, qkv.size()?)?;
    check_copy_range(0, history_bytes, conv_weight.size()?)?;
    check_copy_range(0, history_bytes, conv_history.size()?)?;
    check_copy_range(0, qkv_bytes, conv_output.size()?)?;
    check_copy_range(0, q_bytes, q_output.size()?)?;
    check_copy_range(0, q_bytes, k_output.size()?)?;
    check_copy_range(0, v_bytes, v_output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_linear_attn_qkv_prepare_f32(
            qkv.raw.as_ptr(),
            conv_weight.raw.as_ptr(),
            conv_history.raw.as_ptr(),
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            if qk_l2_norm { 1 } else { 0 },
            conv_output.raw.as_ptr(),
            q_output.raw.as_ptr(),
            k_output.raw.as_ptr(),
            v_output.raw.as_ptr(),
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attn_qkv_prepare_batch_f32(
    qkv: &RuntimeBuffer,
    conv_weight: &RuntimeBuffer,
    conv_history: &mut RuntimeBuffer,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel_size: usize,
    sequence_len: usize,
    q_scale: f32,
    qk_l2_norm: bool,
    conv_output: &mut RuntimeBuffer,
    q_output: &mut RuntimeBuffer,
    k_output: &mut RuntimeBuffer,
    v_output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || kernel_size == 0
        || sequence_len == 0
    {
        return Err(
            "linear attention qkv prepare batch dimensions must be greater than zero".to_string(),
        );
    }
    if !q_scale.is_finite() {
        return Err("linear attention qkv prepare batch q_scale must be finite".to_string());
    }
    let q_elements = key_heads.checked_mul(key_dim).ok_or_else(|| {
        "linear attention qkv prepare batch q element count overflows".to_string()
    })?;
    let v_elements = value_heads.checked_mul(value_dim).ok_or_else(|| {
        "linear attention qkv prepare batch v element count overflows".to_string()
    })?;
    let channels = q_elements
        .checked_add(q_elements)
        .and_then(|value| value.checked_add(v_elements))
        .ok_or_else(|| "linear attention qkv prepare batch channel count overflows".to_string())?;
    let qkv_bytes = channels
        .checked_mul(sequence_len)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "linear attention qkv prepare batch qkv byte size overflows".to_string())?;
    let history_bytes = channels
        .checked_mul(kernel_size)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            "linear attention qkv prepare batch history byte size overflows".to_string()
        })?;
    let q_bytes = q_elements
        .checked_mul(sequence_len)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "linear attention qkv prepare batch q byte size overflows".to_string())?;
    let v_bytes = v_elements
        .checked_mul(sequence_len)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "linear attention qkv prepare batch v byte size overflows".to_string())?;

    check_copy_range(0, qkv_bytes, qkv.size()?)?;
    check_copy_range(0, history_bytes, conv_weight.size()?)?;
    check_copy_range(0, history_bytes, conv_history.size()?)?;
    check_copy_range(0, qkv_bytes, conv_output.size()?)?;
    check_copy_range(0, q_bytes, q_output.size()?)?;
    check_copy_range(0, q_bytes, k_output.size()?)?;
    check_copy_range(0, v_bytes, v_output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_linear_attn_qkv_prepare_batch_f32(
            qkv.raw.as_ptr(),
            conv_weight.raw.as_ptr(),
            conv_history.raw.as_ptr(),
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            sequence_len,
            q_scale,
            if qk_l2_norm { 1 } else { 0 },
            conv_output.raw.as_ptr(),
            q_output.raw.as_ptr(),
            k_output.raw.as_ptr(),
            v_output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn linear_attn_gate_beta_f32(
    a: &RuntimeBuffer,
    b: &RuntimeBuffer,
    a_log: &RuntimeBuffer,
    dt_bias: &RuntimeBuffer,
    heads: usize,
    sequence_len: usize,
    gate_output: &mut RuntimeBuffer,
    beta_output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if heads == 0 {
        return Err("linear attention gate beta heads must be greater than zero".to_string());
    }
    if sequence_len == 0 {
        return Err(
            "linear attention gate beta sequence_len must be greater than zero".to_string(),
        );
    }

    let output_elements = heads
        .checked_mul(sequence_len)
        .ok_or_else(|| "linear attention gate beta output element count overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention gate beta output byte size overflows".to_string())?;
    let param_bytes = heads
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention gate beta parameter byte size overflows".to_string())?;

    check_linear_attention_gate_beta_copy_range("a", 0, output_bytes, a.size()?)?;
    check_linear_attention_gate_beta_copy_range("b", 0, output_bytes, b.size()?)?;
    check_linear_attention_gate_beta_copy_range("a_log", 0, param_bytes, a_log.size()?)?;
    check_linear_attention_gate_beta_copy_range("dt_bias", 0, param_bytes, dt_bias.size()?)?;
    check_linear_attention_gate_beta_copy_range(
        "gate_output",
        0,
        output_bytes,
        gate_output.size()?,
    )?;
    check_linear_attention_gate_beta_copy_range(
        "beta_output",
        0,
        output_bytes,
        beta_output.size()?,
    )?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_linear_attn_gate_beta_f32(
            a.raw.as_ptr(),
            b.raw.as_ptr(),
            a_log.raw.as_ptr(),
            dt_bias.raw.as_ptr(),
            heads,
            sequence_len,
            gate_output.raw.as_ptr(),
            beta_output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn linear_attn_recurrent_f32(
    q: &RuntimeBuffer,
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    gate: &RuntimeBuffer,
    beta: &RuntimeBuffer,
    key_heads: usize,
    value_heads: usize,
    sequence_len: usize,
    key_dim: usize,
    value_dim: usize,
    state: &mut RuntimeBuffer,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if key_heads == 0 {
        return Err("linear attention recurrent key_heads must be greater than zero".to_string());
    }
    if value_heads == 0 {
        return Err("linear attention recurrent value_heads must be greater than zero".to_string());
    }
    if !value_heads.is_multiple_of(key_heads) {
        return Err(
            "linear attention recurrent value_heads must be a multiple of key_heads".to_string(),
        );
    }
    if sequence_len == 0 {
        return Err(
            "linear attention recurrent sequence_len must be greater than zero".to_string(),
        );
    }
    if key_dim == 0 {
        return Err("linear attention recurrent key_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("linear attention recurrent value_dim must be greater than zero".to_string());
    }

    let key_head_sequence_elements = key_heads.checked_mul(sequence_len).ok_or_else(|| {
        "linear attention recurrent key head-sequence element count overflows".to_string()
    })?;
    let value_head_sequence_elements = value_heads.checked_mul(sequence_len).ok_or_else(|| {
        "linear attention recurrent value head-sequence element count overflows".to_string()
    })?;
    let q_elements = key_head_sequence_elements
        .checked_mul(key_dim)
        .ok_or_else(|| "linear attention recurrent q element count overflows".to_string())?;
    let v_elements = value_head_sequence_elements
        .checked_mul(value_dim)
        .ok_or_else(|| "linear attention recurrent v/output element count overflows".to_string())?;
    let state_elements = value_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "linear attention recurrent state element count overflows".to_string())?
        .checked_mul(value_dim)
        .ok_or_else(|| "linear attention recurrent state element count overflows".to_string())?;

    let q_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention recurrent q byte size overflows".to_string())?;
    let k_bytes = q_bytes;
    let v_output_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention recurrent v/output byte size overflows".to_string())?;
    let gate_beta_bytes = value_head_sequence_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention recurrent gate/beta byte size overflows".to_string())?;
    let state_bytes = state_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "linear attention recurrent state byte size overflows".to_string())?;

    check_linear_attention_recurrent_copy_range("q", 0, q_bytes, q.size()?)?;
    check_linear_attention_recurrent_copy_range("k", 0, k_bytes, k.size()?)?;
    check_linear_attention_recurrent_copy_range("v", 0, v_output_bytes, v.size()?)?;
    check_linear_attention_recurrent_copy_range("gate", 0, gate_beta_bytes, gate.size()?)?;
    check_linear_attention_recurrent_copy_range("beta", 0, gate_beta_bytes, beta.size()?)?;
    check_linear_attention_recurrent_copy_range("state", 0, state_bytes, state.size()?)?;
    check_linear_attention_recurrent_copy_range("output", 0, v_output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_linear_attn_recurrent_f32(
            q.raw.as_ptr(),
            k.raw.as_ptr(),
            v.raw.as_ptr(),
            gate.raw.as_ptr(),
            beta.raw.as_ptr(),
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            state.raw.as_ptr(),
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn depthwise_conv1d_f32(
    input: &RuntimeBuffer,
    weight: &RuntimeBuffer,
    channels: usize,
    sequence_len: usize,
    kernel_size: usize,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    if channels == 0 {
        return Err("depthwise conv1d channels must be greater than zero".to_string());
    }
    if sequence_len == 0 {
        return Err("depthwise conv1d sequence_len must be greater than zero".to_string());
    }
    if kernel_size == 0 {
        return Err("depthwise conv1d kernel_size must be greater than zero".to_string());
    }

    let input_elements = channels
        .checked_mul(sequence_len)
        .ok_or_else(|| "depthwise conv1d input element count overflows".to_string())?;
    let weight_elements = channels
        .checked_mul(kernel_size)
        .ok_or_else(|| "depthwise conv1d weight element count overflows".to_string())?;
    let output_elements = channels
        .checked_mul(sequence_len)
        .ok_or_else(|| "depthwise conv1d output element count overflows".to_string())?;

    let input_bytes = input_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "depthwise conv1d input byte size overflows".to_string())?;
    let weight_bytes = weight_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "depthwise conv1d weight byte size overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "depthwise conv1d output byte size overflows".to_string())?;

    check_depthwise_copy_range("input", 0, input_bytes, input.size()?)?;
    check_depthwise_copy_range("weight", 0, weight_bytes, weight.size()?)?;
    check_depthwise_copy_range("output", 0, output_bytes, output.size()?)?;

    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_depthwise_conv1d_f32(
            input.raw.as_ptr(),
            weight.raw.as_ptr(),
            channels,
            sequence_len,
            kernel_size,
            output.raw.as_ptr(),
            stream,
        )
    })
}

fn status_to_result(status: c_int) -> Result<(), String> {
    match status {
        STATUS_OK => Ok(()),
        STATUS_INVALID_ARGUMENT => Err(last_error()),
        STATUS_BUFFER_TOO_SMALL => Err(last_error()),
        _ => Err(last_error()),
    }
}

fn last_error() -> String {
    let mut len = 0_usize;
    let status = unsafe { ullm_runtime_get_last_error(std::ptr::null_mut(), &mut len) };
    if status != STATUS_BUFFER_TOO_SMALL || len == 0 {
        return "unknown runtime error".to_string();
    }
    let mut buffer = vec![0_i8; len];
    let status = unsafe { ullm_runtime_get_last_error(buffer.as_mut_ptr(), &mut len) };
    if status != STATUS_OK {
        return "unknown runtime error".to_string();
    }
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn check_copy_range(offset: usize, bytes: usize, total: usize) -> Result<(), String> {
    if offset <= total && bytes <= total - offset {
        Ok(())
    } else {
        Err(format!(
            "runtime buffer copy range is out of bounds: offset={offset} bytes={bytes} total={total}"
        ))
    }
}

fn c_array_to_string<const N: usize>(value: &[c_char; N]) -> String {
    let nul = value.iter().position(|&ch| ch == 0).unwrap_or(N);
    let bytes: Vec<u8> = value[..nul].iter().map(|&ch| ch as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn check_depthwise_copy_range(
    kind: &str,
    offset: usize,
    bytes: usize,
    total: usize,
) -> Result<(), String> {
    if offset <= total && bytes <= total - offset {
        Ok(())
    } else {
        Err(format!(
            "depthwise conv1d {kind} buffer is too small: offset={offset} bytes={bytes} total={total}"
        ))
    }
}

fn check_linear_attention_gate_beta_copy_range(
    kind: &str,
    offset: usize,
    bytes: usize,
    total: usize,
) -> Result<(), String> {
    if offset <= total && bytes <= total - offset {
        Ok(())
    } else {
        Err(format!(
            "linear attention gate beta {kind} buffer is too small: offset={offset} bytes={bytes} total={total}"
        ))
    }
}

fn check_linear_attention_recurrent_copy_range(
    kind: &str,
    offset: usize,
    bytes: usize,
    total: usize,
) -> Result<(), String> {
    if offset <= total && bytes <= total - offset {
        Ok(())
    } else {
        Err(format!(
            "linear attention recurrent {kind} buffer is too small: offset={offset} bytes={bytes} total={total}"
        ))
    }
}
