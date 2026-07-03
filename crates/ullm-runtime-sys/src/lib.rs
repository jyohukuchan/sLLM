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
