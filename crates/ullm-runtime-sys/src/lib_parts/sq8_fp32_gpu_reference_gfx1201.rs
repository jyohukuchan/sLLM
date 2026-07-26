// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Opaque bindings for the standalone gfx1201 F32 reference control.  This is
// purposefully not expressed in terms of RuntimeContext/RuntimeBuffer: the
// C++ side owns a separate HIP stream, allocations, and hipBLAS handle so it
// cannot accidentally enter the optimized SQ8 dispatch path.

enum RawSq8Fp32GpuReferenceGfx1201Session {}

#[repr(C)]
struct RawSq8Fp32GpuReferenceGfx1201DeviceInfo {
    total_global_mem_bytes: u64,
    free_global_mem_bytes: u64,
    name: [c_char; 128],
    gcn_arch_name: [c_char; 64],
    pci_bdf: [c_char; 32],
}

unsafe extern "C" {
    fn ullm_sq8_fp32_gpu_reference_gfx1201_create(
        max_context: usize,
        session: *mut *mut RawSq8Fp32GpuReferenceGfx1201Session,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_destroy(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
    );
    fn ullm_sq8_fp32_gpu_reference_gfx1201_device_info(
        session: *const RawSq8Fp32GpuReferenceGfx1201Session,
        info: *mut RawSq8Fp32GpuReferenceGfx1201DeviceInfo,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_reserve_sq8_weight(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        tensor_name: *const c_char,
        rows: usize,
        cols: usize,
        scales_bf16: *const c_void,
        scale_bytes: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_upload_sq8_weight_chunk(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        tensor_name: *const c_char,
        offset_bytes: usize,
        source: *const c_void,
        bytes: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_reserve_bf16_tensor(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        slot: *const c_char,
        elements: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_upload_bf16_tensor_chunk(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        slot: *const c_char,
        offset_bytes: usize,
        source: *const c_void,
        bytes: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_upload_layer_norms(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        layer_index: usize,
        input_norm_bf16: *const c_void,
        post_attention_norm_bf16: *const c_void,
        q_norm_bf16: *const c_void,
        k_norm_bf16: *const c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_upload_final_norm(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        final_norm_bf16: *const c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_finalize_model(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_forward(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        token_id: u32,
        logits_f32: *mut f32,
        logits_elements: usize,
        final_hidden_f32: *mut f32,
        final_hidden_elements: usize,
        layer_hidden_f32: *mut f32,
        layer_hidden_elements: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn ullm_sq8_fp32_gpu_reference_gfx1201_reset(
        session: *mut RawSq8Fp32GpuReferenceGfx1201Session,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

const SQ8_FP32_GPU_REFERENCE_SUCCESS: c_int = 1;
const SQ8_FP32_GPU_REFERENCE_ERROR_CAPACITY: usize = 4096;

fn sq8_fp32_gpu_reference_call(
    operation: &str,
    call: impl FnOnce(*mut c_char, usize) -> c_int,
) -> Result<(), String> {
    let mut error = [0_i8; SQ8_FP32_GPU_REFERENCE_ERROR_CAPACITY];
    let result = call(error.as_mut_ptr(), error.len());
    if result == SQ8_FP32_GPU_REFERENCE_SUCCESS {
        return Ok(());
    }
    let detail = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    if detail.is_empty() {
        Err(format!(
            "GPU F32 reference {operation} failed without an error message"
        ))
    } else {
        Err(format!("GPU F32 reference {operation} failed: {detail}"))
    }
}

fn sq8_fp32_gpu_reference_c_string(value: &str, label: &str) -> Result<CString, String> {
    CString::new(value).map_err(|_| format!("GPU F32 reference {label} contains NUL"))
}

fn sq8_fp32_gpu_reference_c_string_from_array(value: &[c_char]) -> Result<String, String> {
    unsafe { CStr::from_ptr(value.as_ptr()) }
        .to_str()
        .map(str::to_owned)
        .map_err(|error| format!("GPU F32 reference device string is not UTF-8: {error}"))
}

/// Immutable admission and allocator information for the one allowed GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8Fp32GpuReferenceGfx1201DeviceInfo {
    pub total_global_mem_bytes: u64,
    pub free_global_mem_bytes: u64,
    pub name: String,
    pub gcn_arch_name: String,
    pub pci_bdf: String,
}

/// Standalone F32-reference HIP session.  It is intentionally !Send/!Sync by
/// construction: its current-device, stream, and hipBLAS ownership stays on
/// the creating host thread.
pub struct Sq8Fp32GpuReferenceGfx1201Session {
    raw: NonNull<RawSq8Fp32GpuReferenceGfx1201Session>,
    _not_send_or_sync: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Sq8Fp32GpuReferenceGfx1201Session {
    pub fn create(max_context: usize) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        sq8_fp32_gpu_reference_call("create", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_create(max_context, &mut raw, error, capacity)
        })?;
        let raw = NonNull::new(raw)
            .ok_or_else(|| "GPU F32 reference create returned a null session".to_string())?;
        Ok(Self {
            raw,
            _not_send_or_sync: std::marker::PhantomData,
        })
    }

    pub fn device_info(&mut self) -> Result<Sq8Fp32GpuReferenceGfx1201DeviceInfo, String> {
        let mut info = RawSq8Fp32GpuReferenceGfx1201DeviceInfo {
            total_global_mem_bytes: 0,
            free_global_mem_bytes: 0,
            name: [0; 128],
            gcn_arch_name: [0; 64],
            pci_bdf: [0; 32],
        };
        sq8_fp32_gpu_reference_call("device_info", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_device_info(
                self.raw.as_ptr(),
                &mut info,
                error,
                capacity,
            )
        })?;
        Ok(Sq8Fp32GpuReferenceGfx1201DeviceInfo {
            total_global_mem_bytes: info.total_global_mem_bytes,
            free_global_mem_bytes: info.free_global_mem_bytes,
            name: sq8_fp32_gpu_reference_c_string_from_array(&info.name)?,
            gcn_arch_name: sq8_fp32_gpu_reference_c_string_from_array(&info.gcn_arch_name)?,
            pci_bdf: sq8_fp32_gpu_reference_c_string_from_array(&info.pci_bdf)?,
        })
    }

    pub fn reserve_sq8_weight(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        scales_bf16: &[u8],
    ) -> Result<(), String> {
        let tensor_name = sq8_fp32_gpu_reference_c_string(tensor_name, "tensor name")?;
        sq8_fp32_gpu_reference_call("reserve SQ8 weight", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_reserve_sq8_weight(
                self.raw.as_ptr(),
                tensor_name.as_ptr(),
                rows,
                cols,
                scales_bf16.as_ptr().cast(),
                scales_bf16.len(),
                error,
                capacity,
            )
        })
    }

    pub fn upload_sq8_weight_chunk(
        &mut self,
        tensor_name: &str,
        offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), String> {
        let tensor_name = sq8_fp32_gpu_reference_c_string(tensor_name, "tensor name")?;
        sq8_fp32_gpu_reference_call("upload SQ8 weight chunk", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_upload_sq8_weight_chunk(
                self.raw.as_ptr(),
                tensor_name.as_ptr(),
                offset_bytes,
                bytes.as_ptr().cast(),
                bytes.len(),
                error,
                capacity,
            )
        })
    }

    pub fn reserve_bf16_tensor(&mut self, slot: &str, elements: usize) -> Result<(), String> {
        let slot = sq8_fp32_gpu_reference_c_string(slot, "BF16 slot")?;
        sq8_fp32_gpu_reference_call("reserve BF16 tensor", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_reserve_bf16_tensor(
                self.raw.as_ptr(),
                slot.as_ptr(),
                elements,
                error,
                capacity,
            )
        })
    }

    pub fn upload_bf16_tensor_chunk(
        &mut self,
        slot: &str,
        offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), String> {
        let slot = sq8_fp32_gpu_reference_c_string(slot, "BF16 slot")?;
        sq8_fp32_gpu_reference_call("upload BF16 tensor chunk", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_upload_bf16_tensor_chunk(
                self.raw.as_ptr(),
                slot.as_ptr(),
                offset_bytes,
                bytes.as_ptr().cast(),
                bytes.len(),
                error,
                capacity,
            )
        })
    }

    pub fn upload_layer_norms(
        &mut self,
        layer_index: usize,
        input_norm_bf16: &[u8],
        post_attention_norm_bf16: &[u8],
        q_norm_bf16: &[u8],
        k_norm_bf16: &[u8],
    ) -> Result<(), String> {
        sq8_fp32_gpu_reference_call("upload layer norms", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_upload_layer_norms(
                self.raw.as_ptr(),
                layer_index,
                input_norm_bf16.as_ptr().cast(),
                post_attention_norm_bf16.as_ptr().cast(),
                q_norm_bf16.as_ptr().cast(),
                k_norm_bf16.as_ptr().cast(),
                error,
                capacity,
            )
        })
    }

    pub fn upload_final_norm(&mut self, final_norm_bf16: &[u8]) -> Result<(), String> {
        sq8_fp32_gpu_reference_call("upload final norm", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_upload_final_norm(
                self.raw.as_ptr(),
                final_norm_bf16.as_ptr().cast(),
                error,
                capacity,
            )
        })
    }

    pub fn finalize_model(&mut self) -> Result<(), String> {
        sq8_fp32_gpu_reference_call("finalize model", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_finalize_model(self.raw.as_ptr(), error, capacity)
        })
    }

    pub fn forward(
        &mut self,
        token_id: u32,
        logits_f32: &mut [f32],
        final_hidden_f32: &mut [f32],
        layer_hidden_f32: &mut [f32],
    ) -> Result<(), String> {
        sq8_fp32_gpu_reference_call("forward", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_forward(
                self.raw.as_ptr(),
                token_id,
                logits_f32.as_mut_ptr(),
                logits_f32.len(),
                final_hidden_f32.as_mut_ptr(),
                final_hidden_f32.len(),
                layer_hidden_f32.as_mut_ptr(),
                layer_hidden_f32.len(),
                error,
                capacity,
            )
        })
    }

    pub fn reset(&mut self) -> Result<(), String> {
        sq8_fp32_gpu_reference_call("reset", |error, capacity| unsafe {
            ullm_sq8_fp32_gpu_reference_gfx1201_reset(self.raw.as_ptr(), error, capacity)
        })
    }
}

impl Drop for Sq8Fp32GpuReferenceGfx1201Session {
    fn drop(&mut self) {
        unsafe { ullm_sq8_fp32_gpu_reference_gfx1201_destroy(self.raw.as_ptr()) }
    }
}

/// Whether the standalone F32-reference HIP control was compiled in.
pub const fn sq8_fp32_gpu_reference_gfx1201_feature_enabled() -> bool {
    cfg!(feature = "rocm-fp32-reference-gfx1201")
}
