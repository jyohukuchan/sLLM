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
    fn ullm_runtime_stream_create(
        context: *mut RawRuntimeContext,
        stream: *mut *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_stream_destroy(stream: *mut RawRuntimeStream) -> c_int;
    fn ullm_runtime_stream_synchronize(stream: *mut RawRuntimeStream) -> c_int;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_reports_abi_version() {
        assert_eq!(abi_version(), 1);
    }

    #[test]
    fn runtime_has_at_least_cpu_device() {
        let count = device_count().unwrap();
        assert!(count >= 1);
        let info = device_info(0).unwrap();
        assert_eq!(info.backend, "cpu");
    }

    #[test]
    fn smoke_adds_f32_values() {
        let out = smoke_add_f32(&[1.0, 2.5, -3.0], &[4.0, -1.5, 3.5]).unwrap();
        assert_eq!(out, vec![5.0, 1.0, 0.5]);
    }

    #[test]
    fn cpu_context_allocates_runtime_buffer() {
        let mut context = RuntimeContext::create(0).unwrap();
        let info = context.device_info().unwrap();
        assert_eq!(info.backend, "cpu");
        let buffer = context.alloc_buffer(4096).unwrap();
        assert_eq!(buffer.size().unwrap(), 4096);
    }

    #[test]
    fn cpu_context_creates_and_synchronizes_stream() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn cpu_buffer_roundtrips_host_data() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut buffer = context.alloc_buffer(64).unwrap();
        let input: Vec<u8> = (0..48).map(|value| (value * 17 + 3) as u8).collect();
        buffer.copy_from_host(8, &input, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; input.len()];
        buffer
            .copy_to_host(8, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn zero_byte_buffer_copy_accepts_end_offset() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut buffer = context.alloc_buffer(8).unwrap();
        buffer.copy_from_host(8, &[], None).unwrap();
        let mut output = [];
        buffer.copy_to_host(8, &mut output, None).unwrap();
    }

    #[test]
    fn buffer_copy_rejects_out_of_bounds_range() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut buffer = context.alloc_buffer(4).unwrap();
        let err = buffer.copy_from_host(3, &[1_u8, 2], None).unwrap_err();
        assert!(err.contains("out of bounds"));
    }

    #[test]
    fn cpu_matvec_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        matvec_f32(&matrix, &input, 2, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.5, 9.0]);
    }

    #[test]
    fn cpu_matvec_bf16_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<u16>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_bf16_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        matvec_bf16_f32(&matrix, &input, 2, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.5, 9.0]);
    }

    #[test]
    fn cpu_bf16_row_f32_reads_selected_row() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<u16>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_bf16_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        stream.synchronize().unwrap();

        bf16_row_f32(&matrix, 2, 3, 1, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 3 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn cpu_matvec_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let matrix = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        let err = matvec_f32(&matrix, &input, 2, 3, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_top1_f32_writes_partial_maxima() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut values = vec![-1.0_f32; 300];
        values[123] = 8.0;
        values[259] = 9.0;
        values[260] = 9.0;
        let mut input = context
            .alloc_buffer(values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let partial_count = top1_partial_count(values.len()).unwrap();
        assert_eq!(partial_count, 2);
        let mut partial_values = context
            .alloc_buffer(partial_count * std::mem::size_of::<f32>())
            .unwrap();
        let mut partial_indices = context
            .alloc_buffer(partial_count * std::mem::size_of::<u32>())
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let written = top1_f32(
            &input,
            values.len(),
            &mut partial_values,
            &mut partial_indices,
            Some(&mut stream),
        )
        .unwrap();
        assert_eq!(written, partial_count);
        stream.synchronize().unwrap();

        let mut value_bytes = vec![0_u8; partial_count * std::mem::size_of::<f32>()];
        let mut index_bytes = vec![0_u8; partial_count * std::mem::size_of::<u32>()];
        partial_values
            .copy_to_host(0, &mut value_bytes, Some(&mut stream))
            .unwrap();
        partial_indices
            .copy_to_host(0, &mut index_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&value_bytes), vec![8.0, 9.0]);
        assert_eq!(le_bytes_to_u32s(&index_bytes), vec![123, 259]);
    }

    #[test]
    fn cpu_rmsnorm_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0];
        let weight_values = [0.5_f32, 1.0, 1.5, -2.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rmsnorm_f32(
            &input,
            &weight,
            input_values.len(),
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_rmsnorm(&input_values, &weight_values, epsilon);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_segmented_rmsnorm_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_f32(
            &input,
            &weight,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_segmented_rmsnorm_silu_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let gate_values = [-1.0_f32, 0.25, 1.0, -2.0, 0.5, 3.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_silu_mul_f32(
            &input,
            &weight,
            &gate,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let normed = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected = expected_silu_mul(&gate_values, &normed);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_rmsnorm_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let input = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let weight = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = rmsnorm_f32(&input, &weight, 4, 1e-5, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_silu_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let up_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut up = context
            .alloc_buffer(up_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        up.copy_from_host(0, &f32s_to_le_bytes(&up_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        silu_mul_f32(
            &gate,
            &up,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&gate_values, &up_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_silu_mul_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let gate = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let up = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = silu_mul_f32(&gate, &up, 4, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_sigmoid_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let input_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sigmoid_mul_f32(
            &gate,
            &input,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_sigmoid_mul(&gate_values, &input_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_sigmoid_mul_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let gate = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let input = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = sigmoid_mul_f32(&gate, &input, 4, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_qwen35_split_q_gate_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let projected_values = [1.0_f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let mut projected = context
            .alloc_buffer(projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut query = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        projected
            .copy_from_host(0, &f32s_to_le_bytes(&projected_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_split_q_gate_f32(&projected, 2, 2, &mut query, &mut gate, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let mut query_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        let mut gate_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        query
            .copy_to_host(0, &mut query_bytes, Some(&mut stream))
            .unwrap();
        gate.copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&query_bytes), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(le_bytes_to_f32s(&gate_bytes), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn cpu_qwen35_qk_norm_rope_f32_matches_split_norm_rope() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let q_projected_values = (0..q_heads * head_dim * 2)
            .map(|index| (index as f32 - 7.0) / 9.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..kv_heads * head_dim)
            .map(|index| (index as f32 + 3.0) / -8.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let q_output_elements = q_heads * head_dim;
        let k_output_elements = kv_heads * head_dim;
        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_rope = context
            .alloc_buffer(k_output_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_f32(
            &q_projected,
            &k_projected,
            &q_weight,
            &k_weight,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            &mut q_gate,
            &mut q_rope,
            &mut k_rope,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_rope_bytes = vec![0_u8; k_output_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_rope
            .copy_to_host(0, &mut k_rope_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let expected_q_gate = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base + head_dim..source_base + 2 * head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_query = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base..source_base + head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            1,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            1,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&k_rope_bytes), &expected_k_rope, 1e-5);
    }

    #[test]
    fn cpu_add_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let lhs_values = [-1.0_f32, 0.0, 1.0, 2.0, 8.5];
        let rhs_values = [3.0_f32, -4.0, 5.0, 6.0, -0.25];
        let mut lhs = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut rhs = context
            .alloc_buffer(rhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        lhs.copy_from_host(0, &f32s_to_le_bytes(&lhs_values), Some(&mut stream))
            .unwrap();
        rhs.copy_from_host(0, &f32s_to_le_bytes(&rhs_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        add_f32(&lhs, &rhs, lhs_values.len(), &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; lhs_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = lhs_values
            .iter()
            .zip(rhs_values.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn cpu_add_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let lhs = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let rhs = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();

        let err = add_f32(&lhs, &rhs, 4, &mut output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_rope_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 2_usize;
        let heads = 2_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 3_usize;
        let rope_base = 10000.0_f32;
        let elements = sequence_len * heads * head_dim;
        let input_values = (0..elements)
            .map(|index| (index as f32 - 11.0) / 7.0)
            .collect::<Vec<_>>();
        let expected = expected_rope(
            &input_values,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rope_f32(
            &input,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_rope_f32_rejects_invalid_rotary_dim_or_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let input = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();

        let err = rope_f32(&input, 1, 1, 6, 5, 0, 10000.0, &mut output, None).unwrap_err();
        assert!(err.contains("rotary_dim"));

        let mut short_output = context
            .alloc_buffer(5 * std::mem::size_of::<f32>())
            .unwrap();
        let err = rope_f32(&input, 1, 1, 6, 4, 0, 10000.0, &mut short_output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_causal_attn_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..sequence_len * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..sequence_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_causal_attn(
            &q_values,
            &k_values,
            &v_values,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_f32(
            &q,
            &k,
            &v,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; expected.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_causal_attn_f32_rejects_invalid_heads_or_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let q = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let k = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let v = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        let err = causal_attn_f32(&q, &k, &v, 1, 3, 2, 1, 1, 1.0, &mut output, None).unwrap_err();
        assert!(err.contains("q_heads"));

        let mut short_output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let err =
            causal_attn_f32(&q, &k, &v, 1, 4, 2, 1, 1, 1.0, &mut short_output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_decode_attn_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..cache_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..cache_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_decode_attn(
            &q_values,
            &k_values,
            &v_values,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        decode_attn_f32(
            &q,
            &k,
            &v,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; expected.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_decode_attn_f32_rejects_invalid_shape_or_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let q = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let k = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let v = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(8 * std::mem::size_of::<f32>())
            .unwrap();

        let err = decode_attn_f32(&q, &k, &v, 3, 3, 2, 1, 2, 1.0, &mut output, None).unwrap_err();
        assert!(err.contains("q_heads"));

        let mut short_output = context
            .alloc_buffer(7 * std::mem::size_of::<f32>())
            .unwrap();
        let err =
            decode_attn_f32(&q, &k, &v, 3, 4, 2, 1, 2, 1.0, &mut short_output, None).unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_paged_decode_attn_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 5_usize;
        let block_size = 2_usize;
        let cache_blocks = 4_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_cache_values = (0..cache_blocks * block_size * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_cache_values = (0..cache_blocks * block_size * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let block_table_values = vec![2_u32, 0_u32, 3_u32];
        let expected = expected_paged_decode_attn(
            &q_values,
            &k_cache_values,
            &v_cache_values,
            &block_table_values,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(k_cache_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(v_cache_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k_cache
            .copy_from_host(0, &f32s_to_le_bytes(&k_cache_values), Some(&mut stream))
            .unwrap();
        v_cache
            .copy_from_host(0, &f32s_to_le_bytes(&v_cache_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; expected.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_paged_decode_attn_f32_rejects_invalid_shape_short_output_or_short_block_table() {
        let mut context = RuntimeContext::create(0).unwrap();
        let q = context
            .alloc_buffer(12 * std::mem::size_of::<f32>())
            .unwrap();
        let k_cache = context
            .alloc_buffer(4 * 2 * 2 * 3 * std::mem::size_of::<f32>())
            .unwrap();
        let v_cache = context
            .alloc_buffer(4 * 2 * 2 * 2 * std::mem::size_of::<f32>())
            .unwrap();
        let block_table = context
            .alloc_buffer(3 * std::mem::size_of::<u32>())
            .unwrap();

        let mut short_output = context
            .alloc_buffer(7 * std::mem::size_of::<f32>())
            .unwrap();
        let err = paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            5,
            2,
            4,
            3,
            2,
            3,
            2,
            1.0,
            &mut short_output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("q_heads"));

        let mut output = context
            .alloc_buffer(7 * std::mem::size_of::<f32>())
            .unwrap();
        let err = paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            5,
            2,
            4,
            4,
            2,
            3,
            2,
            1.0,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));

        let short_block_table = context
            .alloc_buffer(2 * std::mem::size_of::<u32>())
            .unwrap();
        let err = paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &short_block_table,
            5,
            2,
            4,
            4,
            2,
            3,
            2,
            1.0,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("block"));
    }

    #[test]
    fn cpu_paged_kv_write_f32_writes_expected_physical_slot() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_position = 3_usize;
        let block_size = 2_usize;
        let cache_blocks = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let k_values = vec![0.25_f32, -0.5, 1.25, 2.0, -3.0, 4.0];
        let v_values = vec![-0.75_f32, 0.5, 1.5, -2.5];
        let block_table_values = vec![2_u32, 0_u32];
        let physical_tokens = cache_blocks * block_size;
        let mut expected_k_cache = vec![0.0_f32; physical_tokens * kv_heads * head_dim];
        let mut expected_v_cache = vec![0.0_f32; physical_tokens * kv_heads * value_dim];
        let physical_timestep = block_table_values[cache_position / block_size] as usize
            * block_size
            + (cache_position % block_size);
        for kv_head in 0..kv_heads {
            let k_src = kv_head * head_dim;
            let k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
            expected_k_cache[k_dst..k_dst + head_dim]
                .copy_from_slice(&k_values[k_src..k_src + head_dim]);

            let v_src = kv_head * value_dim;
            let v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
            expected_v_cache[v_dst..v_dst + value_dim]
                .copy_from_slice(&v_values[v_src..v_src + value_dim]);
        }

        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(expected_k_cache.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(expected_v_cache.len() * std::mem::size_of::<f32>())
            .unwrap();

        k_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.0_f32; expected_k_cache.len()]),
                Some(&mut stream),
            )
            .unwrap();
        v_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.0_f32; expected_v_cache.len()]),
                Some(&mut stream),
            )
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        paged_kv_write_f32(
            &k,
            &v,
            &block_table,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            &mut k_cache,
            &mut v_cache,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut k_cache_bytes = vec![0_u8; expected_k_cache.len() * std::mem::size_of::<f32>()];
        let mut v_cache_bytes = vec![0_u8; expected_v_cache.len() * std::mem::size_of::<f32>()];
        k_cache
            .copy_to_host(0, &mut k_cache_bytes, Some(&mut stream))
            .unwrap();
        v_cache
            .copy_to_host(0, &mut v_cache_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&k_cache_bytes), &expected_k_cache, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&v_cache_bytes), &expected_v_cache, 1e-6);
    }

    #[test]
    fn cpu_paged_kv_write_f32_rejects_short_cache_or_block_table() {
        let mut context = RuntimeContext::create(0).unwrap();
        let k = context
            .alloc_buffer(2 * 3 * std::mem::size_of::<f32>())
            .unwrap();
        let v = context
            .alloc_buffer(2 * 2 * std::mem::size_of::<f32>())
            .unwrap();
        let short_block_table = context.alloc_buffer(std::mem::size_of::<u32>()).unwrap();
        let block_table = context
            .alloc_buffer(2 * std::mem::size_of::<u32>())
            .unwrap();
        let mut short_k_cache = context
            .alloc_buffer((4 * 2 * 2 * 3 - 1) * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(4 * 2 * 2 * 2 * std::mem::size_of::<f32>())
            .unwrap();

        let err = paged_kv_write_f32(
            &k,
            &v,
            &short_block_table,
            3,
            2,
            4,
            2,
            3,
            2,
            &mut short_k_cache,
            &mut v_cache,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("block_table"));

        let err = paged_kv_write_f32(
            &k,
            &v,
            &block_table,
            3,
            2,
            4,
            2,
            3,
            2,
            &mut short_k_cache,
            &mut v_cache,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("cache"));
    }

    #[test]
    fn cpu_linear_attn_gate_beta_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let heads = 3_usize;
        let sequence_len = 4_usize;
        let a = [
            0.1_f32, -0.2, 1.2, 0.9, 0.8, -1.1, -0.7, 0.5, 1.4, -0.3, 0.2, -0.6,
        ];
        let b = [
            1.0_f32, -1.2, 0.3, -0.8, 0.6, 1.1, -0.5, 0.9, 0.0, -0.4, 1.3, -0.7,
        ];
        let a_log = [-1.0_f32, 0.25, -0.5];
        let dt_bias = [0.3_f32, -0.2, 0.4];
        let (expected_gate, expected_beta) =
            expected_linear_attn_gate_beta(&a, &b, &a_log, &dt_bias, heads, sequence_len);

        let mut a_buffer = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_buffer = context
            .alloc_buffer(b.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log_buffer = context
            .alloc_buffer(a_log.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut dt_bias_buffer = context
            .alloc_buffer(dt_bias.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();

        a_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&a), Some(&mut stream))
            .unwrap();
        b_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&b), Some(&mut stream))
            .unwrap();
        a_log_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&a_log), Some(&mut stream))
            .unwrap();
        dt_bias_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&dt_bias), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_gate_beta_f32(
            &a_buffer,
            &b_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            heads,
            sequence_len,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut gate_output_bytes = vec![0_u8; gate_output.size().unwrap()];
        gate_output
            .copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let mut beta_output_bytes = vec![0_u8; beta_output.size().unwrap()];
        beta_output
            .copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let gate_output_values = le_bytes_to_f32s(&gate_output_bytes);
        let beta_output_values = le_bytes_to_f32s(&beta_output_bytes);
        assert_f32s_close(&gate_output_values, &expected_gate, 1e-5);
        assert_f32s_close(&beta_output_values, &expected_beta, 1e-5);
    }

    #[test]
    fn cpu_linear_attn_gate_beta_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let a = [
            0.1_f32, -0.2, 1.2, 0.9, 0.8, -1.1, -0.7, 0.5, 1.4, -0.3, 0.2, -0.6,
        ];
        let b = [
            1.0_f32, -1.2, 0.3, -0.8, 0.6, 1.1, -0.5, 0.9, 0.0, -0.4, 1.3, -0.7,
        ];
        let a_log = [-1.0_f32, 0.25, -0.5];
        let dt_bias = [0.3_f32, -0.2, 0.4];
        let a_buffer = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let b_buffer = context
            .alloc_buffer(b.len() * std::mem::size_of::<f32>())
            .unwrap();
        let a_log_buffer = context
            .alloc_buffer(a_log.len() * std::mem::size_of::<f32>())
            .unwrap();
        let dt_bias_buffer = context
            .alloc_buffer(dt_bias.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer((a.len() - 1) * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();

        let err = linear_attn_gate_beta_f32(
            &a_buffer,
            &b_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            3,
            4,
            &mut gate_output,
            &mut beta_output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("linear attention gate beta"));
        assert!(err.contains("gate_output"));
    }

    #[test]
    fn cpu_linear_attn_recurrent_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 1_usize;
        let value_heads = 2_usize;
        let sequence_len = 3_usize;
        let key_dim = 2_usize;
        let value_dim = 2_usize;
        let q = [0.2_f32, 0.1, -0.4, 0.7, 0.8, 0.2];
        let k = [0.3_f32, 0.6, 0.7, -0.5, 0.4, 0.9];
        let v = [
            0.4_f32, -0.1, 0.6, 0.3, -0.2, 0.4, 0.1, -0.3, 0.5, 0.2, -0.4, 0.6,
        ];
        let gate = [0.05_f32, -0.1, 0.2, 0.15, -0.25, 0.3];
        let beta = [0.9_f32, 1.1, 0.7, 0.8, 0.6, 0.5];
        let initial_state = [0.1_f32, 0.2, 0.3, 0.4, -0.1, 0.0, 0.05, -0.05];
        let (expected_output, expected_state) = expected_linear_attn_recurrent_f32(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &initial_state,
        );

        let mut q_buffer = context
            .alloc_buffer(q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_buffer = context
            .alloc_buffer(k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_buffer = context
            .alloc_buffer(gate.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_buffer = context
            .alloc_buffer(beta.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut state_buffer = context
            .alloc_buffer(initial_state.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();

        q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&q), Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&k), Some(&mut stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&v), Some(&mut stream))
            .unwrap();
        gate_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&gate), Some(&mut stream))
            .unwrap();
        beta_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&beta), Some(&mut stream))
            .unwrap();
        state_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&initial_state), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut output_buffer,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; output_buffer.size().unwrap()];
        output_buffer
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        let mut state_bytes = vec![0_u8; state_buffer.size().unwrap()];
        state_buffer
            .copy_to_host(0, &mut state_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected_output, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&state_bytes), &expected_state, 1e-5);
    }

    #[test]
    fn cpu_linear_attn_recurrent_f32_rejects_short_output_or_state() {
        let mut context = RuntimeContext::create(0).unwrap();
        let key_heads = 1_usize;
        let value_heads = 2_usize;
        let sequence_len = 2_usize;
        let key_dim = 2_usize;
        let value_dim = 2_usize;
        let q_buffer = context
            .alloc_buffer(key_heads * sequence_len * key_dim * std::mem::size_of::<f32>())
            .unwrap();
        let k_buffer = context
            .alloc_buffer(key_heads * sequence_len * key_dim * std::mem::size_of::<f32>())
            .unwrap();
        let v_buffer = context
            .alloc_buffer(value_heads * sequence_len * value_dim * std::mem::size_of::<f32>())
            .unwrap();
        let gate_buffer = context
            .alloc_buffer(value_heads * sequence_len * std::mem::size_of::<f32>())
            .unwrap();
        let beta_buffer = context
            .alloc_buffer(value_heads * sequence_len * std::mem::size_of::<f32>())
            .unwrap();
        let mut state_buffer = context
            .alloc_buffer(value_heads * key_dim * value_dim * std::mem::size_of::<f32>())
            .unwrap();
        let mut short_output = context
            .alloc_buffer((value_heads * sequence_len * value_dim - 1) * std::mem::size_of::<f32>())
            .unwrap();
        let mut short_state = context
            .alloc_buffer((value_heads * key_dim * value_dim - 1) * std::mem::size_of::<f32>())
            .unwrap();

        let err = linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut short_output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("linear attention recurrent"));
        assert!(err.contains("output"));

        let mut full_output = context
            .alloc_buffer(value_heads * sequence_len * value_dim * std::mem::size_of::<f32>())
            .unwrap();
        let state_error = linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut short_state,
            &mut full_output,
            None,
        )
        .unwrap_err();
        assert!(state_error.contains("linear attention recurrent"));
        assert!(state_error.contains("state"));
    }

    #[test]
    fn cpu_depthwise_conv1d_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let channels = 3_usize;
        let sequence_len = 5_usize;
        let kernel_size = 3_usize;
        let input_values = [
            1.0_f32, 0.5, -1.0, 2.0, 1.0, 0.5, 3.0, -0.5, 0.5, 4.0, -1.0, 1.5, 5.0, 0.0, -2.0,
        ];
        let weight_values = [1.0_f32, -1.0, 2.0, 0.5_f32, 1.0, -0.5, -1.0, 1.0, 1.5];
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(sequence_len * channels * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        depthwise_conv1d_f32(
            &input,
            &weight,
            channels,
            sequence_len,
            kernel_size,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; sequence_len * channels * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_depthwise_conv1d(
            &input_values,
            &weight_values,
            channels,
            sequence_len,
            kernel_size,
        );
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_depthwise_conv1d_f32_uses_causal_conv1d_weight_order() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, 3.0];
        let weight_values = [10.0_f32, 100.0, 1000.0];
        let expected = [1000.0_f32, 2100.0, 3210.0];
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        depthwise_conv1d_f32(&input, &weight, 1, 3, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn cpu_depthwise_conv1d_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let input = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let weight = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(5 * std::mem::size_of::<f32>())
            .unwrap();

        let err = depthwise_conv1d_f32(&input, &weight, 2, 3, 1, &mut output, None).unwrap_err();
        assert!(err.contains("depthwise conv1d"));
    }

    #[test]
    fn cpu_linear_attn_qkv_prepare_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 2;
        let value_heads = 2;
        let key_dim = 2;
        let value_dim = 2;
        let kernel_size = 3;
        let q_scale = 0.5;
        let channels = key_heads * key_dim * 2 + value_heads * value_dim;
        let qkv_values: Vec<f32> = (0..channels)
            .map(|index| index as f32 * 0.1 + 0.1)
            .collect();
        let conv_weight_values: Vec<f32> =
            (0..channels).flat_map(|_| [0.25_f32, 0.5, 1.0]).collect();
        let history_values = vec![0.0_f32; channels * kernel_size];
        let mut expected_history = history_values.clone();
        let (expected_conv, expected_q, expected_k, expected_v) = expected_linear_attn_qkv_prepare(
            &qkv_values,
            &conv_weight_values,
            &mut expected_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            true,
        );

        let mut qkv = context
            .alloc_buffer(channels * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_weight = context
            .alloc_buffer(conv_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_history = context
            .alloc_buffer(history_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_output = context
            .alloc_buffer(channels * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_output = context
            .alloc_buffer(expected_q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_output = context
            .alloc_buffer(expected_k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_output = context
            .alloc_buffer(expected_v.len() * std::mem::size_of::<f32>())
            .unwrap();

        qkv.copy_from_host(0, &f32s_to_le_bytes(&qkv_values), Some(&mut stream))
            .unwrap();
        conv_weight
            .copy_from_host(0, &f32s_to_le_bytes(&conv_weight_values), Some(&mut stream))
            .unwrap();
        conv_history
            .copy_from_host(0, &f32s_to_le_bytes(&history_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_qkv_prepare_f32(
            &qkv,
            &conv_weight,
            &mut conv_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            true,
            &mut conv_output,
            &mut q_output,
            &mut k_output,
            &mut v_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut conv_bytes = vec![0_u8; expected_conv.len() * std::mem::size_of::<f32>()];
        let mut q_bytes = vec![0_u8; expected_q.len() * std::mem::size_of::<f32>()];
        let mut k_bytes = vec![0_u8; expected_k.len() * std::mem::size_of::<f32>()];
        let mut v_bytes = vec![0_u8; expected_v.len() * std::mem::size_of::<f32>()];
        let mut history_bytes = vec![0_u8; expected_history.len() * std::mem::size_of::<f32>()];
        conv_output
            .copy_to_host(0, &mut conv_bytes, Some(&mut stream))
            .unwrap();
        q_output
            .copy_to_host(0, &mut q_bytes, Some(&mut stream))
            .unwrap();
        k_output
            .copy_to_host(0, &mut k_bytes, Some(&mut stream))
            .unwrap();
        v_output
            .copy_to_host(0, &mut v_bytes, Some(&mut stream))
            .unwrap();
        conv_history
            .copy_to_host(0, &mut history_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        assert_f32s_close(&le_bytes_to_f32s(&conv_bytes), &expected_conv, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_bytes), &expected_q, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&k_bytes), &expected_k, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&v_bytes), &expected_v, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&history_bytes), &expected_history, 1e-6);
    }

    #[test]
    fn cpu_aq4_dequant_f32_materializes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(2).unwrap();
        let mut scale = context.alloc_buffer(2).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x30], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_dequant_f32(
            &index,
            &scale,
            &codebook,
            &[0.5, 2.0],
            2,
            10.0,
            4,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![5.0, 10.0, 0.0, 60.0]);
    }

    #[test]
    fn cpu_aq4_dequant_f32_rejects_short_output() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut index = context.alloc_buffer(2).unwrap();
        let mut scale = context.alloc_buffer(2).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        index.copy_from_host(0, &[0x21_u8, 0x30], None).unwrap();
        scale.copy_from_host(0, &[0_u8, 1], None).unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), None)
            .unwrap();

        let err = aq4_dequant_f32(
            &index,
            &scale,
            &codebook,
            &[0.5, 2.0],
            2,
            10.0,
            4,
            &mut output,
            None,
        )
        .unwrap_err();
        assert!(err.contains("out of bounds") || err.contains("output"));
    }

    #[test]
    fn cpu_aq4_matvec_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![112.5, 30.0]);
    }

    #[test]
    fn cpu_aq4_matvec_pair_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut left_index = context.alloc_buffer(2).unwrap();
        let mut left_scale = context.alloc_buffer(2).unwrap();
        let mut left_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut right_index = context.alloc_buffer(1).unwrap();
        let mut right_scale = context.alloc_buffer(1).unwrap();
        let mut right_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        left_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        left_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        right_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        right_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        left_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        right_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        left_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        right_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_pair_f32(
            &left_index,
            &left_scale,
            &left_codebook,
            &left_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &right_index,
            &right_scale,
            &right_codebook,
            &right_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            2,
            &mut left_output,
            &mut right_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut left_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut right_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        left_output
            .copy_to_host(0, &mut left_output_bytes, Some(&mut stream))
            .unwrap();
        right_output
            .copy_to_host(0, &mut right_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&left_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&right_output_bytes), vec![28.0]);
    }

    #[test]
    fn cpu_aq4_matvec_triple_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut first_index = context.alloc_buffer(2).unwrap();
        let mut first_scale = context.alloc_buffer(2).unwrap();
        let mut first_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut second_index = context.alloc_buffer(1).unwrap();
        let mut second_scale = context.alloc_buffer(1).unwrap();
        let mut second_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_index = context.alloc_buffer(1).unwrap();
        let mut third_scale = context.alloc_buffer(1).unwrap();
        let mut third_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut third_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        first_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        first_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        second_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        second_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        third_index
            .copy_from_host(0, &[0x87_u8], Some(&mut stream))
            .unwrap();
        third_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        for codebook in [
            &mut first_codebook,
            &mut second_codebook,
            &mut third_codebook,
        ] {
            codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
                .unwrap();
        }
        for scale_values in [
            &mut first_scale_values,
            &mut second_scale_values,
            &mut third_scale_values,
        ] {
            scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
                .unwrap();
        }
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_triple_f32(
            &first_index,
            &first_scale,
            &first_codebook,
            &first_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &second_index,
            &second_scale,
            &second_codebook,
            &second_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &third_index,
            &third_scale,
            &third_codebook,
            &third_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            1,
            2,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut first_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut second_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut third_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        first_output
            .copy_to_host(0, &mut first_output_bytes, Some(&mut stream))
            .unwrap();
        second_output
            .copy_to_host(0, &mut second_output_bytes, Some(&mut stream))
            .unwrap();
        third_output
            .copy_to_host(0, &mut third_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&first_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&second_output_bytes), vec![28.0]);
        assert_eq!(le_bytes_to_f32s(&third_output_bytes), vec![38.0]);
    }

    #[test]
    fn cpu_aq4_matvec_qkv_z_gate_beta_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut qkv_index = context.alloc_buffer(2).unwrap();
        let mut qkv_scale = context.alloc_buffer(2).unwrap();
        let mut qkv_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut qkv_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut z_index = context.alloc_buffer(1).unwrap();
        let mut z_scale = context.alloc_buffer(1).unwrap();
        let mut z_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut z_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut a_index = context.alloc_buffer(1).unwrap();
        let mut a_scale = context.alloc_buffer(1).unwrap();
        let mut a_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut b_index = context.alloc_buffer(1).unwrap();
        let mut b_scale = context.alloc_buffer(1).unwrap();
        let mut b_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut dt_bias = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut qkv_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut z_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut gate_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut beta_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        qkv_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        qkv_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        z_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        z_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        a_index
            .copy_from_host(0, &[0x87_u8], Some(&mut stream))
            .unwrap();
        a_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        b_index
            .copy_from_host(0, &[0xa9_u8], Some(&mut stream))
            .unwrap();
        b_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        for codebook in [
            &mut qkv_codebook,
            &mut z_codebook,
            &mut a_codebook,
            &mut b_codebook,
        ] {
            codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
                .unwrap();
        }
        for scale_values in [
            &mut qkv_scale_values,
            &mut z_scale_values,
            &mut a_scale_values,
            &mut b_scale_values,
        ] {
            scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
                .unwrap();
        }
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        a_log
            .copy_from_host(0, &f32s_to_le_bytes(&[0.0]), Some(&mut stream))
            .unwrap();
        dt_bias
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_qkv_z_gate_beta_f32(
            &qkv_index,
            &qkv_scale,
            &qkv_codebook,
            &qkv_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &z_index,
            &z_scale,
            &z_codebook,
            &z_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &a_index,
            &a_scale,
            &a_codebook,
            &a_scale_values,
            None,
            1,
            2,
            0.1,
            0,
            &b_index,
            &b_scale,
            &b_codebook,
            &b_scale_values,
            None,
            1,
            2,
            0.1,
            0,
            &input,
            &a_log,
            &dt_bias,
            2,
            1,
            1,
            2,
            &mut qkv_output,
            &mut z_output,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut qkv_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut z_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut gate_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut beta_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        qkv_output
            .copy_to_host(0, &mut qkv_output_bytes, Some(&mut stream))
            .unwrap();
        z_output
            .copy_to_host(0, &mut z_output_bytes, Some(&mut stream))
            .unwrap();
        gate_output
            .copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
            .unwrap();
        beta_output
            .copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let x = 3.8_f32 + 0.5;
        let expected_gate = -x.exp().ln_1p();
        let expected_beta = 1.0_f32 / (1.0 + (-4.8_f32).exp());
        assert_eq!(le_bytes_to_f32s(&qkv_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&z_output_bytes), vec![28.0]);
        assert_f32s_close(
            &le_bytes_to_f32s(&gate_output_bytes),
            &[expected_gate],
            1e-5,
        );
        assert_f32s_close(
            &le_bytes_to_f32s(&beta_output_bytes),
            &[expected_beta],
            1e-6,
        );
    }

    #[test]
    fn cpu_aq4_matvec_add_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut residual = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        residual
            .copy_from_host(0, &f32s_to_le_bytes(&[1.25, -2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_add_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            &residual,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![113.75, 28.0]);
    }

    #[test]
    fn cpu_aq4_matvec_silu_mul_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut gate_index = context.alloc_buffer(3).unwrap();
        let mut gate_scale = context.alloc_buffer(3).unwrap();
        let mut gate_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_index = context.alloc_buffer(3).unwrap();
        let mut up_scale = context.alloc_buffer(3).unwrap();
        let mut up_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        gate_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        up_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        gate_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        up_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        gate_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        up_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        gate_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        up_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_silu_mul_f32(
            &gate_index,
            &gate_scale,
            &gate_codebook,
            &gate_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &up_index,
            &up_scale,
            &up_codebook,
            &up_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&[1.125, 0.3], &[2.25, 0.6]);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn cpu_aq4_matvec_gate_beta_f32_computes_expected_values() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut a_index = context.alloc_buffer(3).unwrap();
        let mut a_scale = context.alloc_buffer(3).unwrap();
        let mut a_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_index = context.alloc_buffer(3).unwrap();
        let mut b_scale = context.alloc_buffer(3).unwrap();
        let mut b_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut dt_bias = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        a_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        b_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        a_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        b_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        a_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        b_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        a_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        b_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        a_log
            .copy_from_host(0, &f32s_to_le_bytes(&[0.0, 0.5]), Some(&mut stream))
            .unwrap();
        dt_bias
            .copy_from_host(0, &f32s_to_le_bytes(&[0.1, -0.2]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_gate_beta_f32(
            &a_index,
            &a_scale,
            &a_codebook,
            &a_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &b_index,
            &b_scale,
            &b_codebook,
            &b_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            &a_log,
            &dt_bias,
            2,
            3,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut gate_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut beta_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        gate_output
            .copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .unwrap();
        beta_output
            .copy_to_host(0, &mut beta_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let (expected_gate, expected_beta) = expected_linear_attn_gate_beta(
            &[1.125, 0.3],
            &[2.25, 0.6],
            &[0.0, 0.5],
            &[0.1, -0.2],
            2,
            1,
        );
        assert_f32s_close(&le_bytes_to_f32s(&gate_bytes), &expected_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&beta_bytes), &expected_beta, 1e-6);
    }

    #[test]
    fn first_hip_aq4_dequant_f32_materializes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(2).unwrap();
        let mut scale = context.alloc_buffer(2).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x30], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_dequant_f32(
            &index,
            &scale,
            &codebook,
            &[0.5, 2.0],
            2,
            10.0,
            4,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![5.0, 10.0, 0.0, 60.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![112.5, 30.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_pair_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut left_index = context.alloc_buffer(2).unwrap();
        let mut left_scale = context.alloc_buffer(2).unwrap();
        let mut left_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut right_index = context.alloc_buffer(1).unwrap();
        let mut right_scale = context.alloc_buffer(1).unwrap();
        let mut right_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut left_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut right_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        left_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        left_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        right_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        right_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        left_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        right_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        left_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        right_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_pair_f32(
            &left_index,
            &left_scale,
            &left_codebook,
            &left_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &right_index,
            &right_scale,
            &right_codebook,
            &right_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            2,
            &mut left_output,
            &mut right_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut left_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut right_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        left_output
            .copy_to_host(0, &mut left_output_bytes, Some(&mut stream))
            .unwrap();
        right_output
            .copy_to_host(0, &mut right_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&left_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&right_output_bytes), vec![28.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_triple_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut first_index = context.alloc_buffer(2).unwrap();
        let mut first_scale = context.alloc_buffer(2).unwrap();
        let mut first_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut second_index = context.alloc_buffer(1).unwrap();
        let mut second_scale = context.alloc_buffer(1).unwrap();
        let mut second_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_index = context.alloc_buffer(1).unwrap();
        let mut third_scale = context.alloc_buffer(1).unwrap();
        let mut third_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut third_scale_values = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut input = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut first_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut second_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let mut third_output = context.alloc_buffer(std::mem::size_of::<f32>()).unwrap();

        first_index
            .copy_from_host(0, &[0x21_u8, 0x43], Some(&mut stream))
            .unwrap();
        first_scale
            .copy_from_host(0, &[0_u8, 0], Some(&mut stream))
            .unwrap();
        second_index
            .copy_from_host(0, &[0x65_u8], Some(&mut stream))
            .unwrap();
        second_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        third_index
            .copy_from_host(0, &[0x87_u8], Some(&mut stream))
            .unwrap();
        third_scale
            .copy_from_host(0, &[0_u8], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        for codebook in [
            &mut first_codebook,
            &mut second_codebook,
            &mut third_codebook,
        ] {
            codebook
                .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
                .unwrap();
        }
        for scale_values in [
            &mut first_scale_values,
            &mut second_scale_values,
            &mut third_scale_values,
        ] {
            scale_values
                .copy_from_host(0, &f32s_to_le_bytes(&[1.0]), Some(&mut stream))
                .unwrap();
        }
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[2.0, 3.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_triple_f32(
            &first_index,
            &first_scale,
            &first_codebook,
            &first_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &second_index,
            &second_scale,
            &second_codebook,
            &second_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &third_index,
            &third_scale,
            &third_codebook,
            &third_scale_values,
            None,
            1,
            2,
            1.0,
            0,
            &input,
            2,
            1,
            1,
            2,
            &mut first_output,
            &mut second_output,
            &mut third_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut first_output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut second_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        let mut third_output_bytes = vec![0_u8; std::mem::size_of::<f32>()];
        first_output
            .copy_to_host(0, &mut first_output_bytes, Some(&mut stream))
            .unwrap();
        second_output
            .copy_to_host(0, &mut second_output_bytes, Some(&mut stream))
            .unwrap();
        third_output
            .copy_to_host(0, &mut third_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&first_output_bytes), vec![8.0, 18.0]);
        assert_eq!(le_bytes_to_f32s(&second_output_bytes), vec![28.0]);
        assert_eq!(le_bytes_to_f32s(&third_output_bytes), vec![38.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_add_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut residual = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        residual
            .copy_from_host(0, &f32s_to_le_bytes(&[1.25, -2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_add_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            &residual,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![113.75, 28.0]);
    }

    #[test]
    fn first_hip_aq4_matvec_silu_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut gate_index = context.alloc_buffer(3).unwrap();
        let mut gate_scale = context.alloc_buffer(3).unwrap();
        let mut gate_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_index = context.alloc_buffer(3).unwrap();
        let mut up_scale = context.alloc_buffer(3).unwrap();
        let mut up_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut up_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        gate_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        up_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        gate_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        up_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        gate_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        up_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        gate_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        up_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_silu_mul_f32(
            &gate_index,
            &gate_scale,
            &gate_codebook,
            &gate_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &up_index,
            &up_scale,
            &up_codebook,
            &up_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            2,
            3,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&[1.125, 0.3], &[2.25, 0.6]);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn first_hip_aq4_matvec_gate_beta_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut a_index = context.alloc_buffer(3).unwrap();
        let mut a_scale = context.alloc_buffer(3).unwrap();
        let mut a_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_index = context.alloc_buffer(3).unwrap();
        let mut b_scale = context.alloc_buffer(3).unwrap();
        let mut b_codebook = context
            .alloc_buffer(16 * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_scale_values = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut dt_bias = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        a_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        b_index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        a_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        b_scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        a_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        b_codebook
            .copy_from_host(0, &f32s_to_le_bytes(&codebook_values), Some(&mut stream))
            .unwrap();
        a_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        b_scale_values
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, 2.0]), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        a_log
            .copy_from_host(0, &f32s_to_le_bytes(&[0.0, 0.5]), Some(&mut stream))
            .unwrap();
        dt_bias
            .copy_from_host(0, &f32s_to_le_bytes(&[0.1, -0.2]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        aq4_matvec_gate_beta_f32(
            &a_index,
            &a_scale,
            &a_codebook,
            &a_scale_values,
            None,
            2,
            2,
            0.1,
            0,
            &b_index,
            &b_scale,
            &b_codebook,
            &b_scale_values,
            None,
            2,
            2,
            0.2,
            0,
            &input,
            &a_log,
            &dt_bias,
            2,
            3,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut gate_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        let mut beta_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        gate_output
            .copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .unwrap();
        beta_output
            .copy_to_host(0, &mut beta_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let (expected_gate, expected_beta) = expected_linear_attn_gate_beta(
            &[1.125, 0.3],
            &[2.25, 0.6],
            &[0.0, 0.5],
            &[0.1, -0.2],
            2,
            1,
        );
        assert_f32s_close(&le_bytes_to_f32s(&gate_bytes), &expected_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&beta_bytes), &expected_beta, 1e-6);
    }

    #[test]
    fn first_hip_matvec_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut matrix = context
            .alloc_buffer(6 * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(3 * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(2 * std::mem::size_of::<f32>())
            .unwrap();

        matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&[0.5, -1.0, 2.0]), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        matvec_f32(&matrix, &input, 2, 3, &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; 2 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(le_bytes_to_f32s(&output_bytes), vec![4.5, 9.0]);
    }

    #[test]
    fn first_hip_rmsnorm_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0];
        let weight_values = [0.5_f32, 1.0, 1.5, -2.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rmsnorm_f32(
            &input,
            &weight,
            input_values.len(),
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_rmsnorm(&input_values, &weight_values, epsilon);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_segmented_rmsnorm_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_f32(
            &input,
            &weight,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_segmented_rmsnorm_silu_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let input_values = [1.0_f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let weight_values = [0.5_f32, 1.0, -1.5];
        let gate_values = [-1.0_f32, 0.25, 1.0, -2.0, 0.5, 3.0];
        let epsilon = 1e-5_f32;
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        segmented_rmsnorm_silu_mul_f32(
            &input,
            &weight,
            &gate,
            2,
            3,
            epsilon,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let normed = input_values
            .chunks_exact(3)
            .flat_map(|segment| expected_rmsnorm(segment, &weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected = expected_silu_mul(&gate_values, &normed);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_silu_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let up_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut up = context
            .alloc_buffer(up_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        up.copy_from_host(0, &f32s_to_le_bytes(&up_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        silu_mul_f32(
            &gate,
            &up,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_silu_mul(&gate_values, &up_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_sigmoid_mul_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let gate_values = [-1.0_f32, 0.0, 1.0, 2.0];
        let input_values = [3.0_f32, -4.0, 5.0, 6.0];
        let mut gate = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(gate_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        gate.copy_from_host(0, &f32s_to_le_bytes(&gate_values), Some(&mut stream))
            .unwrap();
        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        sigmoid_mul_f32(
            &gate,
            &input,
            gate_values.len(),
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; gate_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = expected_sigmoid_mul(&gate_values, &input_values);
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_add_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let lhs_values = [-1.0_f32, 0.0, 1.0, 2.0, 8.5];
        let rhs_values = [3.0_f32, -4.0, 5.0, 6.0, -0.25];
        let mut lhs = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut rhs = context
            .alloc_buffer(rhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(lhs_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        lhs.copy_from_host(0, &f32s_to_le_bytes(&lhs_values), Some(&mut stream))
            .unwrap();
        rhs.copy_from_host(0, &f32s_to_le_bytes(&rhs_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        add_f32(&lhs, &rhs, lhs_values.len(), &mut output, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; lhs_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let expected = lhs_values
            .iter()
            .zip(rhs_values.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect::<Vec<_>>();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-6);
    }

    #[test]
    fn first_hip_rope_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 2_usize;
        let heads = 2_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 3_usize;
        let rope_base = 10000.0_f32;
        let elements = sequence_len * heads * head_dim;
        let input_values = (0..elements)
            .map(|index| (index as f32 - 11.0) / 7.0)
            .collect::<Vec<_>>();
        let expected = expected_rope(
            &input_values,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        rope_f32(
            &input,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; input_values.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-4);
    }

    #[test]
    fn first_hip_qwen35_qk_norm_rope_f32_matches_split_norm_rope_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_heads = 2_usize;
        let kv_heads = 1_usize;
        let head_dim = 6_usize;
        let rotary_dim = 4_usize;
        let position_offset = 5_usize;
        let rope_base = 10000.0_f32;
        let epsilon = 1e-5_f32;
        let q_projected_values = (0..q_heads * head_dim * 2)
            .map(|index| (index as f32 - 7.0) / 9.0)
            .collect::<Vec<_>>();
        let k_projected_values = (0..kv_heads * head_dim)
            .map(|index| (index as f32 + 3.0) / -8.0)
            .collect::<Vec<_>>();
        let q_weight_values = [0.5_f32, -1.0, 1.25, 0.75, -0.5, 1.5];
        let k_weight_values = [-0.25_f32, 0.5, 1.0, -1.5, 0.75, 1.25];
        let q_output_elements = q_heads * head_dim;
        let k_output_elements = kv_heads * head_dim;
        let mut q_projected = context
            .alloc_buffer(q_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_projected = context
            .alloc_buffer(k_projected_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_weight = context
            .alloc_buffer(q_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_weight = context
            .alloc_buffer(k_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_gate = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_rope = context
            .alloc_buffer(q_output_elements * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_rope = context
            .alloc_buffer(k_output_elements * std::mem::size_of::<f32>())
            .unwrap();

        q_projected
            .copy_from_host(0, &f32s_to_le_bytes(&q_projected_values), Some(&mut stream))
            .unwrap();
        k_projected
            .copy_from_host(0, &f32s_to_le_bytes(&k_projected_values), Some(&mut stream))
            .unwrap();
        q_weight
            .copy_from_host(0, &f32s_to_le_bytes(&q_weight_values), Some(&mut stream))
            .unwrap();
        k_weight
            .copy_from_host(0, &f32s_to_le_bytes(&k_weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        qwen35_qk_norm_rope_f32(
            &q_projected,
            &k_projected,
            &q_weight,
            &k_weight,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
            epsilon,
            &mut q_gate,
            &mut q_rope,
            &mut k_rope,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut q_gate_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut q_rope_bytes = vec![0_u8; q_output_elements * std::mem::size_of::<f32>()];
        let mut k_rope_bytes = vec![0_u8; k_output_elements * std::mem::size_of::<f32>()];
        q_gate
            .copy_to_host(0, &mut q_gate_bytes, Some(&mut stream))
            .unwrap();
        q_rope
            .copy_to_host(0, &mut q_rope_bytes, Some(&mut stream))
            .unwrap();
        k_rope
            .copy_to_host(0, &mut k_rope_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let expected_q_gate = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base + head_dim..source_base + 2 * head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_query = (0..q_heads)
            .flat_map(|head| {
                let source_base = head * 2 * head_dim;
                q_projected_values[source_base..source_base + head_dim].to_vec()
            })
            .collect::<Vec<_>>();
        let expected_q_normed = expected_q_query
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &q_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_k_normed = k_projected_values
            .chunks_exact(head_dim)
            .flat_map(|segment| expected_rmsnorm(segment, &k_weight_values, epsilon))
            .collect::<Vec<_>>();
        let expected_q_rope = expected_rope(
            &expected_q_normed,
            1,
            q_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope(
            &expected_k_normed,
            1,
            kv_heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        assert_f32s_close(&le_bytes_to_f32s(&q_gate_bytes), &expected_q_gate, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_rope_bytes), &expected_q_rope, 1e-4);
        assert_f32s_close(&le_bytes_to_f32s(&k_rope_bytes), &expected_k_rope, 1e-4);
    }

    #[test]
    fn first_hip_causal_attn_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..sequence_len * q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..sequence_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..sequence_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_causal_attn(
            &q_values,
            &k_values,
            &v_values,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        causal_attn_f32(
            &q,
            &k,
            &v,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; expected.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-4);
    }

    #[test]
    fn first_hip_decode_attn_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_values = (0..cache_len * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_values = (0..cache_len * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let expected = expected_decode_attn(
            &q_values,
            &k_values,
            &v_values,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        decode_attn_f32(
            &q,
            &k,
            &v,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; expected.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-4);
    }

    #[test]
    fn first_hip_paged_decode_attn_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_len = 5_usize;
        let block_size = 2_usize;
        let cache_blocks = 4_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let q_values = (0..q_heads * head_dim)
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let k_cache_values = (0..cache_blocks * block_size * kv_heads * head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_cache_values = (0..cache_blocks * block_size * kv_heads * value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let block_table_values = vec![2_u32, 0_u32, 3_u32];
        let expected = expected_paged_decode_attn(
            &q_values,
            &k_cache_values,
            &v_cache_values,
            &block_table_values,
            cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );

        let mut q = context
            .alloc_buffer(q_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(k_cache_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(v_cache_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(expected.len() * std::mem::size_of::<f32>())
            .unwrap();

        q.copy_from_host(0, &f32s_to_le_bytes(&q_values), Some(&mut stream))
            .unwrap();
        k_cache
            .copy_from_host(0, &f32s_to_le_bytes(&k_cache_values), Some(&mut stream))
            .unwrap();
        v_cache
            .copy_from_host(0, &f32s_to_le_bytes(&v_cache_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; expected.len() * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-4);
    }

    #[test]
    fn first_hip_paged_kv_write_f32_writes_expected_physical_slot_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let cache_position = 3_usize;
        let block_size = 2_usize;
        let cache_blocks = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 3_usize;
        let value_dim = 2_usize;
        let k_values = vec![0.25_f32, -0.5, 1.25, 2.0, -3.0, 4.0];
        let v_values = vec![-0.75_f32, 0.5, 1.5, -2.5];
        let block_table_values = vec![2_u32, 0_u32];
        let physical_tokens = cache_blocks * block_size;
        let mut expected_k_cache = vec![0.0_f32; physical_tokens * kv_heads * head_dim];
        let mut expected_v_cache = vec![0.0_f32; physical_tokens * kv_heads * value_dim];
        let physical_timestep = block_table_values[cache_position / block_size] as usize
            * block_size
            + (cache_position % block_size);
        for kv_head in 0..kv_heads {
            let k_src = kv_head * head_dim;
            let k_dst = (physical_timestep * kv_heads + kv_head) * head_dim;
            expected_k_cache[k_dst..k_dst + head_dim]
                .copy_from_slice(&k_values[k_src..k_src + head_dim]);

            let v_src = kv_head * value_dim;
            let v_dst = (physical_timestep * kv_heads + kv_head) * value_dim;
            expected_v_cache[v_dst..v_dst + value_dim]
                .copy_from_slice(&v_values[v_src..v_src + value_dim]);
        }

        let mut k = context
            .alloc_buffer(k_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v = context
            .alloc_buffer(v_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut block_table = context
            .alloc_buffer(block_table_values.len() * std::mem::size_of::<u32>())
            .unwrap();
        let mut k_cache = context
            .alloc_buffer(expected_k_cache.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_cache = context
            .alloc_buffer(expected_v_cache.len() * std::mem::size_of::<f32>())
            .unwrap();

        k_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.0_f32; expected_k_cache.len()]),
                Some(&mut stream),
            )
            .unwrap();
        v_cache
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&vec![0.0_f32; expected_v_cache.len()]),
                Some(&mut stream),
            )
            .unwrap();
        k.copy_from_host(0, &f32s_to_le_bytes(&k_values), Some(&mut stream))
            .unwrap();
        v.copy_from_host(0, &f32s_to_le_bytes(&v_values), Some(&mut stream))
            .unwrap();
        let block_table_bytes: Vec<u8> = block_table_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        block_table
            .copy_from_host(0, &block_table_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        paged_kv_write_f32(
            &k,
            &v,
            &block_table,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            &mut k_cache,
            &mut v_cache,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut k_cache_bytes = vec![0_u8; expected_k_cache.len() * std::mem::size_of::<f32>()];
        let mut v_cache_bytes = vec![0_u8; expected_v_cache.len() * std::mem::size_of::<f32>()];
        k_cache
            .copy_to_host(0, &mut k_cache_bytes, Some(&mut stream))
            .unwrap();
        v_cache
            .copy_to_host(0, &mut v_cache_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&k_cache_bytes), &expected_k_cache, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&v_cache_bytes), &expected_v_cache, 1e-5);
    }

    #[test]
    fn first_hip_linear_attn_gate_beta_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let heads = 3_usize;
        let sequence_len = 4_usize;
        let a = [
            0.1_f32, -0.2, 1.2, 0.9, 0.8, -1.1, -0.7, 0.5, 1.4, -0.3, 0.2, -0.6,
        ];
        let b = [
            1.0_f32, -1.2, 0.3, -0.8, 0.6, 1.1, -0.5, 0.9, 0.0, -0.4, 1.3, -0.7,
        ];
        let a_log = [-1.0_f32, 0.25, -0.5];
        let dt_bias = [0.3_f32, -0.2, 0.4];
        let (expected_gate, expected_beta) =
            expected_linear_attn_gate_beta(&a, &b, &a_log, &dt_bias, heads, sequence_len);

        let mut a_buffer = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut b_buffer = context
            .alloc_buffer(b.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut a_log_buffer = context
            .alloc_buffer(a_log.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut dt_bias_buffer = context
            .alloc_buffer(dt_bias.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_output = context
            .alloc_buffer(a.len() * std::mem::size_of::<f32>())
            .unwrap();

        a_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&a), Some(&mut stream))
            .unwrap();
        b_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&b), Some(&mut stream))
            .unwrap();
        a_log_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&a_log), Some(&mut stream))
            .unwrap();
        dt_bias_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&dt_bias), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_gate_beta_f32(
            &a_buffer,
            &b_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            heads,
            sequence_len,
            &mut gate_output,
            &mut beta_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut gate_output_bytes = vec![0_u8; gate_output.size().unwrap()];
        gate_output
            .copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        let mut beta_output_bytes = vec![0_u8; beta_output.size().unwrap()];
        beta_output
            .copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let gate_output_values = le_bytes_to_f32s(&gate_output_bytes);
        let beta_output_values = le_bytes_to_f32s(&beta_output_bytes);
        assert_f32s_close(&gate_output_values, &expected_gate, 1e-5);
        assert_f32s_close(&beta_output_values, &expected_beta, 1e-5);
    }

    #[test]
    fn first_hip_linear_attn_recurrent_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 1_usize;
        let value_heads = 2_usize;
        let sequence_len = 3_usize;
        let key_dim = 2_usize;
        let value_dim = 2_usize;
        let q = [0.2_f32, 0.1, -0.4, 0.7, 0.8, 0.2];
        let k = [0.3_f32, 0.6, 0.7, -0.5, 0.4, 0.9];
        let v = [
            0.4_f32, -0.1, 0.6, 0.3, -0.2, 0.4, 0.1, -0.3, 0.5, 0.2, -0.4, 0.6,
        ];
        let gate = [0.05_f32, -0.1, 0.2, 0.15, -0.25, 0.3];
        let beta = [0.9_f32, 1.1, 0.7, 0.8, 0.6, 0.5];
        let initial_state = [0.1_f32, 0.2, 0.3, 0.4, -0.1, 0.0, 0.05, -0.05];
        let (expected_output, expected_state) = expected_linear_attn_recurrent_f32(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &initial_state,
        );

        let mut q_buffer = context
            .alloc_buffer(q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_buffer = context
            .alloc_buffer(k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_buffer = context
            .alloc_buffer(gate.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_buffer = context
            .alloc_buffer(beta.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut state_buffer = context
            .alloc_buffer(initial_state.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();

        q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&q), Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&k), Some(&mut stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&v), Some(&mut stream))
            .unwrap();
        gate_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&gate), Some(&mut stream))
            .unwrap();
        beta_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&beta), Some(&mut stream))
            .unwrap();
        state_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&initial_state), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut output_buffer,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; output_buffer.size().unwrap()];
        output_buffer
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        let mut state_bytes = vec![0_u8; state_buffer.size().unwrap()];
        state_buffer
            .copy_to_host(0, &mut state_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected_output, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&state_bytes), &expected_state, 1e-5);
    }

    #[test]
    fn first_hip_linear_attn_recurrent_f32_decode_step_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 1_usize;
        let value_heads = 2_usize;
        let sequence_len = 1_usize;
        let key_dim = 4_usize;
        let value_dim = 3_usize;
        let q = [0.2_f32, -0.1, 0.4, 0.7];
        let k = [-0.3_f32, 0.6, 0.2, -0.5];
        let v = [0.4_f32, -0.1, 0.6, 0.3, -0.2, 0.4];
        let gate = [0.05_f32, -0.1];
        let beta = [0.9_f32, 1.1];
        let initial_state = [
            0.1_f32, 0.2, 0.3, 0.4, -0.1, 0.0, 0.05, -0.05, 0.2, 0.1, -0.2, 0.3, -0.3, 0.25, 0.15,
            -0.1, 0.05, 0.35, -0.15, 0.45, 0.2, -0.25, 0.1, -0.05,
        ];
        let (expected_output, expected_state) = expected_linear_attn_recurrent_f32(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &initial_state,
        );

        let mut q_buffer = context
            .alloc_buffer(q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_buffer = context
            .alloc_buffer(k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut gate_buffer = context
            .alloc_buffer(gate.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut beta_buffer = context
            .alloc_buffer(beta.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut state_buffer = context
            .alloc_buffer(initial_state.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output_buffer = context
            .alloc_buffer(v.len() * std::mem::size_of::<f32>())
            .unwrap();

        q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&q), Some(&mut stream))
            .unwrap();
        k_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&k), Some(&mut stream))
            .unwrap();
        v_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&v), Some(&mut stream))
            .unwrap();
        gate_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&gate), Some(&mut stream))
            .unwrap();
        beta_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&beta), Some(&mut stream))
            .unwrap();
        state_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&initial_state), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut output_buffer,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; output_buffer.size().unwrap()];
        output_buffer
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        let mut state_bytes = vec![0_u8; state_buffer.size().unwrap()];
        state_buffer
            .copy_to_host(0, &mut state_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected_output, 1e-5);
        assert_f32s_close(&le_bytes_to_f32s(&state_bytes), &expected_state, 1e-5);
    }

    #[test]
    fn first_hip_depthwise_conv1d_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let channels = 2_usize;
        let sequence_len = 6_usize;
        let kernel_size = 4_usize;
        let input_values = [
            0.5_f32, -1.0, 1.0, 2.0, -1.5, 0.75, -0.25, 3.5, 4.0, -2.0, 1.25, -0.5, 2.0, -3.0, 1.5,
            -0.75, 0.0, 0.5, 0.25, -1.25, 3.0, 1.0, -0.5, 2.5,
        ];
        let weight_values = [1.0_f32, 0.5, -1.0, 0.25, -0.5_f32, 1.0, -0.25, 2.0];

        let mut input = context
            .alloc_buffer(input_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut weight = context
            .alloc_buffer(weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut output = context
            .alloc_buffer(sequence_len * channels * std::mem::size_of::<f32>())
            .unwrap();

        input
            .copy_from_host(0, &f32s_to_le_bytes(&input_values), Some(&mut stream))
            .unwrap();
        weight
            .copy_from_host(0, &f32s_to_le_bytes(&weight_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        depthwise_conv1d_f32(
            &input,
            &weight,
            channels,
            sequence_len,
            kernel_size,
            &mut output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut output_bytes = vec![0_u8; sequence_len * channels * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let expected = expected_depthwise_conv1d(
            &input_values,
            &weight_values,
            channels,
            sequence_len,
            kernel_size,
        );
        assert_f32s_close(&le_bytes_to_f32s(&output_bytes), &expected, 1e-5);
    }

    #[test]
    fn first_hip_linear_attn_qkv_prepare_f32_computes_expected_values_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let key_heads = 2;
        let value_heads = 2;
        let key_dim = 2;
        let value_dim = 2;
        let kernel_size = 3;
        let q_scale = 0.5;
        let channels = key_heads * key_dim * 2 + value_heads * value_dim;
        let qkv_values: Vec<f32> = (0..channels)
            .map(|index| index as f32 * 0.1 + 0.1)
            .collect();
        let conv_weight_values: Vec<f32> =
            (0..channels).flat_map(|_| [0.25_f32, 0.5, 1.0]).collect();
        let history_values = vec![0.0_f32; channels * kernel_size];
        let mut expected_history = history_values.clone();
        let (expected_conv, expected_q, expected_k, expected_v) = expected_linear_attn_qkv_prepare(
            &qkv_values,
            &conv_weight_values,
            &mut expected_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            true,
        );

        let mut qkv = context
            .alloc_buffer(channels * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_weight = context
            .alloc_buffer(conv_weight_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_history = context
            .alloc_buffer(history_values.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut conv_output = context
            .alloc_buffer(channels * std::mem::size_of::<f32>())
            .unwrap();
        let mut q_output = context
            .alloc_buffer(expected_q.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_output = context
            .alloc_buffer(expected_k.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_output = context
            .alloc_buffer(expected_v.len() * std::mem::size_of::<f32>())
            .unwrap();

        qkv.copy_from_host(0, &f32s_to_le_bytes(&qkv_values), Some(&mut stream))
            .unwrap();
        conv_weight
            .copy_from_host(0, &f32s_to_le_bytes(&conv_weight_values), Some(&mut stream))
            .unwrap();
        conv_history
            .copy_from_host(0, &f32s_to_le_bytes(&history_values), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        linear_attn_qkv_prepare_f32(
            &qkv,
            &conv_weight,
            &mut conv_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            true,
            &mut conv_output,
            &mut q_output,
            &mut k_output,
            &mut v_output,
            Some(&mut stream),
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut conv_bytes = vec![0_u8; expected_conv.len() * std::mem::size_of::<f32>()];
        let mut q_bytes = vec![0_u8; expected_q.len() * std::mem::size_of::<f32>()];
        let mut k_bytes = vec![0_u8; expected_k.len() * std::mem::size_of::<f32>()];
        let mut v_bytes = vec![0_u8; expected_v.len() * std::mem::size_of::<f32>()];
        let mut history_bytes = vec![0_u8; expected_history.len() * std::mem::size_of::<f32>()];
        conv_output
            .copy_to_host(0, &mut conv_bytes, Some(&mut stream))
            .unwrap();
        q_output
            .copy_to_host(0, &mut q_bytes, Some(&mut stream))
            .unwrap();
        k_output
            .copy_to_host(0, &mut k_bytes, Some(&mut stream))
            .unwrap();
        v_output
            .copy_to_host(0, &mut v_bytes, Some(&mut stream))
            .unwrap();
        conv_history
            .copy_to_host(0, &mut history_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        assert_f32s_close(&le_bytes_to_f32s(&conv_bytes), &expected_conv, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&q_bytes), &expected_q, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&k_bytes), &expected_k, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&v_bytes), &expected_v, 1e-6);
        assert_f32s_close(&le_bytes_to_f32s(&history_bytes), &expected_history, 1e-6);
    }

    #[test]
    fn first_hip_context_allocates_runtime_buffer_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let info = context.device_info().unwrap();
        assert_eq!(info.backend, "hip");
        let buffer = context.alloc_buffer(4096).unwrap();
        assert_eq!(buffer.size().unwrap(), 4096);
    }

    #[test]
    fn first_hip_context_creates_stream_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn first_hip_buffer_roundtrips_host_data_when_available() {
        if device_count().unwrap() < 2 {
            return;
        }
        let mut context = RuntimeContext::create(1).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut buffer = context.alloc_buffer(4096).unwrap();
        let input: Vec<u8> = (0..4096).map(|value| (value * 31 + 7) as u8).collect();
        buffer.copy_from_host(0, &input, Some(&mut stream)).unwrap();
        stream.synchronize().unwrap();

        let mut output = vec![0_u8; input.len()];
        buffer
            .copy_to_host(0, &mut output, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(output, input);
    }

    fn f32s_to_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn f32s_to_bf16_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values) / 2);
        for value in values {
            let bits = value.to_bits();
            let bf16 = (bits >> 16) as u16;
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        bytes
    }

    fn le_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn le_bytes_to_u32s(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn expected_rmsnorm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
        let sum_squares = input.iter().map(|value| value * value).sum::<f32>();
        let inv_rms = 1.0 / (sum_squares / input.len() as f32 + epsilon).sqrt();
        input
            .iter()
            .zip(weight)
            .map(|(input, weight)| input * inv_rms * weight)
            .collect()
    }

    fn expected_silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(gate, up)| {
                let gate = *gate;
                let sigmoid = 1.0 / (1.0 + (-gate).exp());
                gate * sigmoid * *up
            })
            .collect()
    }

    fn expected_sigmoid_mul(gate: &[f32], input: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(input)
            .map(|(gate, input)| {
                let sigmoid = 1.0 / (1.0 + (-*gate).exp());
                sigmoid * *input
            })
            .collect()
    }

    fn expected_rope(
        input: &[f32],
        sequence_len: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position_offset: usize,
        rope_base: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; input.len()];
        let half = rotary_dim / 2;
        for timestep in 0..sequence_len {
            let position = (position_offset + timestep) as f32;
            for head in 0..heads {
                let base = (timestep * heads + head) * head_dim;
                for pair_dim in 0..half {
                    let exponent = (2.0 * pair_dim as f32) / rotary_dim as f32;
                    let theta = position / rope_base.powf(exponent);
                    let c = theta.cos();
                    let s = theta.sin();
                    let first = input[base + pair_dim];
                    let second = input[base + half + pair_dim];
                    output[base + pair_dim] = first * c - second * s;
                    output[base + half + pair_dim] = second * c + first * s;
                }
                output[base + rotary_dim..base + head_dim]
                    .copy_from_slice(&input[base + rotary_dim..base + head_dim]);
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn expected_causal_attn(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sequence_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; sequence_len * q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for timestep in 0..sequence_len {
            for q_head in 0..q_heads {
                let kv_head = q_head / q_per_kv;
                let q_base = (timestep * q_heads + q_head) * head_dim;
                let mut scores = Vec::with_capacity(timestep + 1);
                for source_timestep in 0..=timestep {
                    let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    let score = (0..head_dim)
                        .map(|dim| q[q_base + dim] * k[k_base + dim])
                        .sum::<f32>()
                        * softmax_scale;
                    scores.push(score);
                }
                let max_score = scores
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, |max, score| max.max(score));
                let weights = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                let output_base = (timestep * q_heads + q_head) * value_dim;
                for value in 0..value_dim {
                    let mut weighted = 0.0_f32;
                    for (source_timestep, weight) in weights.iter().enumerate() {
                        let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value;
                        weighted += *weight * v[v_index];
                    }
                    output[output_base + value] = weighted / denominator;
                }
            }
        }
        output
    }

    fn expected_decode_attn(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        cache_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for q_head in 0..q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = q_head * head_dim;
            let mut scores = Vec::with_capacity(cache_len);
            for cache_t in 0..cache_len {
                let k_base = (cache_t * kv_heads + kv_head) * head_dim;
                let score = (0..head_dim)
                    .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                    .sum::<f32>()
                    * softmax_scale;
                scores.push(score);
            }
            let max_score = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |max, score| max.max(score));
            let weights = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            let output_base = q_head * value_dim;
            for value in 0..value_dim {
                let mut weighted = 0.0_f32;
                for (cache_t, weight) in weights.iter().enumerate() {
                    let v_index = (cache_t * kv_heads + kv_head) * value_dim + value;
                    weighted += *weight * v_cache[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
        output
    }

    fn expected_paged_decode_attn(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        block_table: &[u32],
        cache_len: usize,
        block_size: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for q_head in 0..q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = q_head * head_dim;
            let mut scores = Vec::with_capacity(cache_len);
            for cache_t in 0..cache_len {
                let block = block_table[cache_t / block_size] as usize;
                let offset = cache_t % block_size;
                let k_base = ((block * block_size + offset) * kv_heads + kv_head) * head_dim;
                let score = (0..head_dim)
                    .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                    .sum::<f32>()
                    * softmax_scale;
                scores.push(score);
            }
            let max_score = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |max, score| max.max(score));
            let weights = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            let output_base = q_head * value_dim;
            for value in 0..value_dim {
                let mut weighted = 0.0_f32;
                for (cache_t, weight) in weights.iter().enumerate() {
                    let block = block_table[cache_t / block_size] as usize;
                    let offset = cache_t % block_size;
                    let v_base = ((block * block_size + offset) * kv_heads + kv_head) * value_dim;
                    weighted += *weight * v_cache[v_base + value];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
        output
    }

    fn expected_depthwise_conv1d(
        input: &[f32],
        weight: &[f32],
        channels: usize,
        sequence_len: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; channels * sequence_len];
        for t in 0..sequence_len {
            for c in 0..channels {
                let mut value = 0.0_f32;
                for k in 0..kernel_size {
                    let left_padding = kernel_size - 1 - k;
                    if t >= left_padding {
                        value +=
                            input[(t - left_padding) * channels + c] * weight[c * kernel_size + k];
                    }
                }
                output[t * channels + c] = value;
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn expected_linear_attn_qkv_prepare(
        qkv: &[f32],
        conv_weight: &[f32],
        conv_history: &mut [f32],
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        q_scale: f32,
        qk_l2_norm: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_elements = key_heads * key_dim;
        let v_elements = value_heads * value_dim;
        let channels = q_elements * 2 + v_elements;
        assert_eq!(qkv.len(), channels);
        assert_eq!(conv_weight.len(), channels * kernel_size);
        assert_eq!(conv_history.len(), channels * kernel_size);

        let mut conv_output = vec![0.0_f32; channels];
        for channel in 0..channels {
            for kernel in 0..kernel_size - 1 {
                conv_history[kernel * channels + channel] =
                    conv_history[(kernel + 1) * channels + channel];
            }
            conv_history[(kernel_size - 1) * channels + channel] = qkv[channel];
            let mut sum = 0.0_f32;
            for kernel in 0..kernel_size {
                sum += conv_history[kernel * channels + channel]
                    * conv_weight[channel * kernel_size + kernel];
            }
            let sigmoid = 1.0 / (1.0 + (-sum).exp());
            conv_output[channel] = sum * sigmoid;
        }

        let mut q = vec![0.0_f32; q_elements];
        let mut k = vec![0.0_f32; q_elements];
        let mut v = vec![0.0_f32; v_elements];
        for head in 0..key_heads {
            let q_base = head * key_dim;
            let k_base = q_elements + head * key_dim;
            let target = head * key_dim;
            let q_norm = (conv_output[q_base..q_base + key_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt();
            let k_norm = (conv_output[k_base..k_base + key_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt();
            for dim in 0..key_dim {
                let q_value = conv_output[q_base + dim];
                let k_value = conv_output[k_base + dim];
                q[target + dim] = if qk_l2_norm {
                    q_value / q_norm * q_scale
                } else {
                    q_value * q_scale
                };
                k[target + dim] = if qk_l2_norm {
                    k_value / k_norm
                } else {
                    k_value
                };
            }
        }
        let v_base = q_elements * 2;
        v.copy_from_slice(&conv_output[v_base..v_base + v_elements]);
        (conv_output, q, k, v)
    }

    fn expected_linear_attn_gate_beta(
        a: &[f32],
        b: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        heads: usize,
        sequence_len: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut gate = Vec::with_capacity(heads * sequence_len);
        let mut beta = Vec::with_capacity(heads * sequence_len);
        for t in 0..sequence_len {
            for h in 0..heads {
                let index = t * heads + h;
                let x = a[index] + dt_bias[h];
                let softplus = if x <= 20.0 { (1.0 + x.exp()).ln() } else { x };
                gate.push(-a_log[h].exp() * softplus);
                beta.push(1.0 / (1.0 + (-b[index]).exp()));
            }
        }
        (gate, beta)
    }

    fn expected_linear_attn_recurrent_f32(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        key_heads: usize,
        value_heads: usize,
        sequence_len: usize,
        key_dim: usize,
        value_dim: usize,
        initial_state: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = initial_state.to_vec();
        let mut output = vec![0.0_f32; sequence_len * value_heads * value_dim];
        let state_row_size = key_dim * value_dim;
        let key_head_group = value_heads / key_heads;
        for t in 0..sequence_len {
            for value_head in 0..value_heads {
                let key_head = value_head / key_head_group;
                let gate_index = t * value_heads + value_head;
                let factor = gate[gate_index].exp();
                for key in 0..key_dim {
                    for value in 0..value_dim {
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        state[state_index] *= factor;
                    }
                }
                for value in 0..value_dim {
                    let mut current = 0.0_f32;
                    for key in 0..key_dim {
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        let k_index = (t * key_heads + key_head) * key_dim + key;
                        current += state[state_index] * k[k_index];
                    }
                    let v_index = (t * value_heads + value_head) * value_dim + value;
                    let v_prime = (v[v_index] - current) * beta[gate_index];
                    for key in 0..key_dim {
                        let k_index = (t * key_heads + key_head) * key_dim + key;
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        state[state_index] += k[k_index] * v_prime;
                    }
                }
                for value in 0..value_dim {
                    let mut value_output = 0.0_f32;
                    for key in 0..key_dim {
                        let state_index = value_head * state_row_size + key * value_dim + value;
                        let q_index = (t * key_heads + key_head) * key_dim + key;
                        value_output += state[state_index] * q[q_index];
                    }
                    let output_index = (t * value_heads + value_head) * value_dim + value;
                    output[output_index] = value_output;
                }
            }
        }
        (output, state)
    }

    fn assert_f32s_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual} expected={expected}"
            );
        }
    }
}
