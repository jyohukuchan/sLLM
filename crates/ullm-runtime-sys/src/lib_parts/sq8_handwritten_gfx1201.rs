// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

unsafe extern "C" {
    fn ullm_runtime_sq8_handwritten_gfx1201_m1_projection_f32(
        quantized_activation_buffer: *const RawRuntimeBuffer,
        activation_scale_buffer: *const RawRuntimeBuffer,
        weight_buffer: *const RawRuntimeBuffer,
        weight_scale_buffer: *const RawRuntimeBuffer,
        m: usize,
        n: usize,
        k: usize,
        output_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sq8_handwritten_gfx1201_m1_resources(
        device_anchor: *const RawRuntimeBuffer,
        vgpr_per_thread: *mut u32,
        static_lds_bytes: *mut usize,
        local_bytes_per_thread: *mut usize,
        threads_per_block: *mut c_int,
        active_blocks_per_cu: *mut c_int,
    ) -> c_int;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sq8HandwrittenGfx1201ResourceInfo {
    /// HIP's compiled register allocation, in 32-bit registers per thread.
    pub vgpr_per_thread: u32,
    pub static_lds_bytes: usize,
    pub local_bytes_per_thread: usize,
    pub threads_per_block: i32,
    pub active_blocks_per_cu: i32,
}

/// Private M=1 WMMA feasibility path.  It intentionally accepts only the
/// exact Qwen3-14B SQ8_0 projection shapes and reuses the existing typed
/// canonical activation: OCP E4M3 raw bytes plus its F32 [1,128] activation
/// scales, alongside F32-expanded canonical BF16 [128,128] weight scales.
pub fn sq8_handwritten_gfx1201_m1_projection_f32(
    activation: &Sq8CkQuantizedActivation,
    weight: &RuntimeBuffer,
    weight_scales: &RuntimeBuffer,
    n: usize,
    output: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq8_handwritten_gfx1201_m1_projection_f32(
            activation.quantized_buffer().raw.as_ptr(),
            activation.scale_buffer().raw.as_ptr(),
            weight.raw.as_ptr(),
            weight_scales.raw.as_ptr(),
            activation.m(),
            n,
            activation.k(),
            output.raw.as_ptr(),
            stream,
        )
    })
}

pub fn sq8_handwritten_gfx1201_m1_resource_info(
    device_anchor: &RuntimeBuffer,
) -> Result<Sq8HandwrittenGfx1201ResourceInfo, String> {
    let mut vgpr_per_thread = 0_u32;
    let mut static_lds_bytes = 0_usize;
    let mut local_bytes_per_thread = 0_usize;
    let mut threads_per_block = 0_i32;
    let mut active_blocks_per_cu = 0_i32;
    status_to_result(unsafe {
        ullm_runtime_sq8_handwritten_gfx1201_m1_resources(
            device_anchor.raw.as_ptr(),
            &mut vgpr_per_thread,
            &mut static_lds_bytes,
            &mut local_bytes_per_thread,
            &mut threads_per_block,
            &mut active_blocks_per_cu,
        )
    })?;
    Ok(Sq8HandwrittenGfx1201ResourceInfo {
        vgpr_per_thread,
        static_lds_bytes,
        local_bytes_per_thread,
        threads_per_block,
        active_blocks_per_cu,
    })
}

#[cfg(test)]
mod handwritten_tests {
    #[test]
    fn canonical_m1_shape_table_has_only_the_seven_projection_families() {
        let shapes = [(5120, 5120), (1024, 5120), (17408, 5120), (5120, 17408)];
        assert!(shapes.iter().all(|(n, k)| n % 128 == 0 && k % 128 == 0));
        assert_eq!(shapes[0], (5120, 5120));
        assert_eq!(shapes[1], (1024, 5120));
        assert_eq!(shapes[2], (17408, 5120));
        assert_eq!(shapes[3], (5120, 17408));
    }
}
