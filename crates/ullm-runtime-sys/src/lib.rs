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
    fn ullm_runtime_matvec_f32(
        matrix_buffer: *const RawRuntimeBuffer,
        input_buffer: *const RawRuntimeBuffer,
        rows: usize,
        cols: usize,
        output_buffer: *mut RawRuntimeBuffer,
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
    fn ullm_runtime_silu_mul_f32(
        gate_buffer: *const RawRuntimeBuffer,
        up_buffer: *const RawRuntimeBuffer,
        elements: usize,
        output_buffer: *mut RawRuntimeBuffer,
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

    fn le_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
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
                    if t >= k {
                        value += input[(t - k) * channels + c] * weight[c * kernel_size + k];
                    }
                }
                output[t * channels + c] = value;
            }
        }
        output
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
