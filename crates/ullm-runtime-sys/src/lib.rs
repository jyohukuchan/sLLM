// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;

const STATUS_OK: c_int = 0;
const STATUS_INVALID_ARGUMENT: c_int = 1;
const STATUS_BUFFER_TOO_SMALL: c_int = 2;

enum RawRuntimeContext {}

enum RawRuntimeBuffer {}

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
}

impl Drop for RuntimeBuffer {
    fn drop(&mut self) {
        let _ = unsafe { ullm_runtime_buffer_destroy(self.raw.as_ptr()) };
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
}
