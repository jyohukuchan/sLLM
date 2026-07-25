// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Isolated `SQ8_0` gfx942 A′ and B-control bindings.
//
// A′ accepts only FNUZ-prepacked opaque byte buffers.  Although the installed
// CK archive links through `f8_ocp_t`, this module intentionally exposes no
// raw-OCP-to-CK function.  The separate B control is the only entry point
// that accepts canonical OCP bytes, and it dequantizes them to BF16 before
// calling hipBLAS.

use std::ffi::CString;

const SQ8_GFX942_APRIME_IMPLEMENTATION_UNAVAILABLE: u32 = 0;
const SQ8_GFX942_APRIME_DEFAULT_TILE_16X128X128: u32 = 1;
const SQ8_GFX942_APRIME_KPADDING_TILE_16X128X256: u32 = 2;
const SQ8_GFX942_APRIME_DEFAULT_TILE_16X256X128: u32 = 3;
const SQ8_GFX942_APRIME_DEFAULT_TILE_16X128X256: u32 = 4;

/// Opaque CK XDL instance selected by the isolated `SQ8_0` gfx942 A′ table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8Gfx942AprimImplementation {
    DefaultTile16x128x128,
    KPaddingTile16x128x256,
    DefaultTile16x256x128,
    DefaultTile16x128x256,
}

impl Sq8Gfx942AprimImplementation {
    fn from_raw(raw: u32) -> Result<Self, String> {
        match raw {
            SQ8_GFX942_APRIME_DEFAULT_TILE_16X128X128 => Ok(Self::DefaultTile16x128x128),
            SQ8_GFX942_APRIME_KPADDING_TILE_16X128X256 => Ok(Self::KPaddingTile16x128x256),
            SQ8_GFX942_APRIME_DEFAULT_TILE_16X256X128 => Ok(Self::DefaultTile16x256x128),
            SQ8_GFX942_APRIME_DEFAULT_TILE_16X128X256 => Ok(Self::DefaultTile16x128x256),
            SQ8_GFX942_APRIME_IMPLEMENTATION_UNAVAILABLE => Err(
                "SQ8_0 gfx942 A′ runtime returned an unavailable implementation after success"
                    .to_string(),
            ),
            _ => Err(format!(
                "SQ8_0 gfx942 A′ runtime returned unknown implementation id {raw}"
            )),
        }
    }
}

unsafe extern "C" {
    fn ullm_runtime_sq8_ck_gfx942_aprime_arch_name_is_exact(gcn_arch_name: *const c_char) -> c_int;
    fn ullm_runtime_sq8_ck_gfx942_aprime_projection_fnuz_prepacked_f32(
        activation_fnuz_prepacked_buffer: *const RawRuntimeBuffer,
        activation_scale_f32_x2_buffer: *const RawRuntimeBuffer,
        weight_fnuz_prepacked_buffer: *const RawRuntimeBuffer,
        weight_scale_f32_x2_buffer: *const RawRuntimeBuffer,
        m: usize,
        n: usize,
        k: usize,
        workspace_bf16_buffer: *mut RawRuntimeBuffer,
        output_f32_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
        implementation: *mut u32,
    ) -> c_int;
    fn ullm_runtime_sq8_ck_gfx942_control_dequant_ocp_bf16_projection_f32(
        activation_ocp_buffer: *const RawRuntimeBuffer,
        activation_scale_f32_buffer: *const RawRuntimeBuffer,
        weight_ocp_buffer: *const RawRuntimeBuffer,
        weight_scale_f32_buffer: *const RawRuntimeBuffer,
        m: usize,
        n: usize,
        k: usize,
        activation_bf16_buffer: *mut RawRuntimeBuffer,
        weight_bf16_buffer: *mut RawRuntimeBuffer,
        output_f32_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
    fn ullm_runtime_sq8_ck_gfx942_aprime_fragment_probe_fnuz(
        a_fnuz_16x32_row_major_buffer: *const RawRuntimeBuffer,
        b_fnuz_32x16_column_major_buffer: *const RawRuntimeBuffer,
        matrix_f32_16x16_buffer: *mut RawRuntimeBuffer,
        fragment_f32_lane64x4_buffer: *mut RawRuntimeBuffer,
        stream: *mut RawRuntimeStream,
    ) -> c_int;
}

/// Returns true only for the exact gfx942 base token, optionally followed by
/// HIP target modifiers (`gfx942:sramecc+:xnack-`, for example).
///
/// This delegates to the same shared C++ predicate used by the HIP A′ and B
/// bodies.  Strings with embedded NUL bytes fail closed before crossing FFI.
pub fn is_exact_gfx942_gcn_arch_name(gcn_arch_name: &str) -> bool {
    let Ok(gcn_arch_name) = CString::new(gcn_arch_name) else {
        return false;
    };
    unsafe { ullm_runtime_sq8_ck_gfx942_aprime_arch_name_is_exact(gcn_arch_name.as_ptr()) == 1 }
}

/// Whether the isolated A′ feature was compiled into this binary.
pub const fn sq8_gfx942_aprime_feature_enabled() -> bool {
    cfg!(feature = "rocm-ck-gfx942-aprime")
}

/// Returns true only when the compiled prototype and the actual HIP device
/// architecture both select A′.  No major/minor approximation is used.
pub fn sq8_gfx942_aprime_is_selected_for_device(device: &DeviceInfo) -> bool {
    sq8_gfx942_aprime_feature_enabled()
        && device.backend == "hip"
        && is_exact_gfx942_gcn_arch_name(&device.gcn_arch_name)
}

/// Isolated copy of the measured Qwen3-14B `SQ8_0` shape table used by A′.
/// It intentionally does not consult the gfx1201 dispatcher.
pub fn sq8_gfx942_aprime_implementation_for_shape(
    m: usize,
    n: usize,
    k: usize,
) -> Option<Sq8Gfx942AprimImplementation> {
    if !matches!(m, 1 | 2 | 4 | 8 | 16 | 32 | 128) {
        return None;
    }
    match (n, k, m) {
        (5120, 5120, _) | (1024, 5120, _) => {
            Some(Sq8Gfx942AprimImplementation::DefaultTile16x128x128)
        }
        (17408, 5120, 128) => Some(Sq8Gfx942AprimImplementation::DefaultTile16x256x128),
        (17408, 5120, _) => Some(Sq8Gfx942AprimImplementation::KPaddingTile16x128x256),
        (5120, 17408, 128) => Some(Sq8Gfx942AprimImplementation::DefaultTile16x128x128),
        (5120, 17408, _) => Some(Sq8Gfx942AprimImplementation::DefaultTile16x128x256),
        _ => None,
    }
}

/// Returns the A′ BF16 workspace and F32 output sizes for one MxN projection.
pub fn sq8_gfx942_aprime_projection_buffer_bytes(
    m: usize,
    n: usize,
) -> Result<(usize, usize), String> {
    let elements = m
        .checked_mul(n)
        .ok_or_else(|| "SQ8_0 gfx942 A′ output element count overflows".to_string())?;
    let workspace = elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "SQ8_0 gfx942 A′ workspace byte size overflows".to_string())?;
    let output = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ8_0 gfx942 A′ output byte size overflows".to_string())?;
    Ok((workspace, output))
}

/// Returns B-control temporary BF16 and final F32 buffer sizes.
pub fn sq8_gfx942_control_buffer_bytes(
    m: usize,
    n: usize,
    k: usize,
) -> Result<(usize, usize, usize), String> {
    let activation_elements = m
        .checked_mul(k)
        .ok_or_else(|| "SQ8_0 gfx942 B activation element count overflows".to_string())?;
    let weight_elements = n
        .checked_mul(k)
        .ok_or_else(|| "SQ8_0 gfx942 B weight element count overflows".to_string())?;
    let output_elements = m
        .checked_mul(n)
        .ok_or_else(|| "SQ8_0 gfx942 B output element count overflows".to_string())?;
    let activation_bf16 = activation_elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "SQ8_0 gfx942 B activation BF16 bytes overflow".to_string())?;
    let weight_bf16 = weight_elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "SQ8_0 gfx942 B weight BF16 bytes overflow".to_string())?;
    let output_f32 = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ8_0 gfx942 B output F32 bytes overflow".to_string())?;
    Ok((activation_bf16, weight_bf16, output_f32))
}

/// Enqueues A′ through the installed `f8_ocp_t` ABI.
///
/// Both byte buffers are required to contain only the output of the CPU
/// `SQ8_0` OCP-to-FNUZ prepack oracle.  `*_scale_f32_x2_buffer` must likewise
/// contain that oracle's per-operand x2 compensation.  There is deliberately
/// no overload accepting raw OCP bytes for the CK route.
#[allow(clippy::too_many_arguments)]
pub fn sq8_gfx942_aprime_projection_fnuz_prepacked_f32(
    activation_fnuz_prepacked_buffer: &RuntimeBuffer,
    activation_scale_f32_x2_buffer: &RuntimeBuffer,
    weight_fnuz_prepacked_buffer: &RuntimeBuffer,
    weight_scale_f32_x2_buffer: &RuntimeBuffer,
    m: usize,
    n: usize,
    k: usize,
    workspace_bf16_buffer: &mut RuntimeBuffer,
    output_f32_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<Sq8Gfx942AprimImplementation, String> {
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    let mut implementation = SQ8_GFX942_APRIME_IMPLEMENTATION_UNAVAILABLE;
    status_to_result(unsafe {
        ullm_runtime_sq8_ck_gfx942_aprime_projection_fnuz_prepacked_f32(
            activation_fnuz_prepacked_buffer.raw.as_ptr(),
            activation_scale_f32_x2_buffer.raw.as_ptr(),
            weight_fnuz_prepacked_buffer.raw.as_ptr(),
            weight_scale_f32_x2_buffer.raw.as_ptr(),
            m,
            n,
            k,
            workspace_bf16_buffer.raw.as_ptr(),
            output_f32_buffer.raw.as_ptr(),
            stream,
            &mut implementation,
        )
    })?;
    Sq8Gfx942AprimImplementation::from_raw(implementation)
}

/// Enqueues the B correctness control: raw OCP bytes are directly dequantized
/// to BF16 and multiplied through hipBLAS.  It cannot enter A′ or FNUZ.
#[allow(clippy::too_many_arguments)]
pub fn sq8_gfx942_control_dequant_ocp_bf16_projection_f32(
    activation_ocp_buffer: &RuntimeBuffer,
    activation_scale_f32_buffer: &RuntimeBuffer,
    weight_ocp_buffer: &RuntimeBuffer,
    weight_scale_f32_buffer: &RuntimeBuffer,
    m: usize,
    n: usize,
    k: usize,
    activation_bf16_buffer: &mut RuntimeBuffer,
    weight_bf16_buffer: &mut RuntimeBuffer,
    output_f32_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq8_ck_gfx942_control_dequant_ocp_bf16_projection_f32(
            activation_ocp_buffer.raw.as_ptr(),
            activation_scale_f32_buffer.raw.as_ptr(),
            weight_ocp_buffer.raw.as_ptr(),
            weight_scale_f32_buffer.raw.as_ptr(),
            m,
            n,
            k,
            activation_bf16_buffer.raw.as_ptr(),
            weight_bf16_buffer.raw.as_ptr(),
            output_f32_buffer.raw.as_ptr(),
            stream,
        )
    })
}

/// Bytes required by the physical-only 16x16x32 fragment diagnostic.
pub const SQ8_GFX942_APRIME_FRAGMENT_A_BYTES: usize = 16 * 32;
pub const SQ8_GFX942_APRIME_FRAGMENT_B_BYTES: usize = 32 * 16;
pub const SQ8_GFX942_APRIME_FRAGMENT_MATRIX_F32_BYTES: usize = 16 * 16 * 4;
pub const SQ8_GFX942_APRIME_FRAGMENT_LANE_F32_BYTES: usize = 64 * 4 * 4;

/// Runs the physical-only FNUZ fragment dump.  It is a diagnostic trace, not
/// a production fragment-layout contract.
pub fn sq8_gfx942_aprime_fragment_probe_fnuz(
    a_fnuz_16x32_row_major_buffer: &RuntimeBuffer,
    b_fnuz_32x16_column_major_buffer: &RuntimeBuffer,
    matrix_f32_16x16_buffer: &mut RuntimeBuffer,
    fragment_f32_lane64x4_buffer: &mut RuntimeBuffer,
    stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let stream = stream.map_or(std::ptr::null_mut(), |stream| stream.raw.as_ptr());
    status_to_result(unsafe {
        ullm_runtime_sq8_ck_gfx942_aprime_fragment_probe_fnuz(
            a_fnuz_16x32_row_major_buffer.raw.as_ptr(),
            b_fnuz_32x16_column_major_buffer.raw.as_ptr(),
            matrix_f32_16x16_buffer.raw.as_ptr(),
            fragment_f32_lane64x4_buffer.raw.as_ptr(),
            stream,
        )
    })
}

#[cfg(test)]
mod sq8_ck_gfx942_aprime_tests {
    use super::*;

    #[test]
    fn exact_gfx942_selector_accepts_only_the_arch_token_and_modifiers() {
        for accepted in ["gfx942", "gfx942:sramecc+:xnack-", "gfx942:xnack+"] {
            assert!(is_exact_gfx942_gcn_arch_name(accepted), "{accepted}");
        }
        for rejected in [
            "",
            "gfx940",
            "gfx950",
            "gfx9420",
            "gfx942junk",
            "gfx942:",
            "gfx942:bogus",
            "gfx942:xnack++",
            "gfx942:xnack+:xnack-",
            "GFX942",
            " gfx942",
            "gfx942\0:xnack+",
        ] {
            assert!(!is_exact_gfx942_gcn_arch_name(rejected), "{rejected:?}");
        }
    }

    #[test]
    fn isolated_shape_table_preserves_all_qwen3_14b_projection_cases() {
        use Sq8Gfx942AprimImplementation::*;
        for m in [1, 2, 4, 8, 16, 32, 128] {
            assert_eq!(
                sq8_gfx942_aprime_implementation_for_shape(m, 5120, 5120),
                Some(DefaultTile16x128x128)
            );
            assert_eq!(
                sq8_gfx942_aprime_implementation_for_shape(m, 1024, 5120),
                Some(DefaultTile16x128x128)
            );
            assert_eq!(
                sq8_gfx942_aprime_implementation_for_shape(m, 17408, 5120),
                Some(if m == 128 {
                    DefaultTile16x256x128
                } else {
                    KPaddingTile16x128x256
                })
            );
            assert_eq!(
                sq8_gfx942_aprime_implementation_for_shape(m, 5120, 17408),
                Some(if m == 128 {
                    DefaultTile16x128x128
                } else {
                    DefaultTile16x128x256
                })
            );
        }
        assert_eq!(
            sq8_gfx942_aprime_implementation_for_shape(3, 5120, 5120),
            None
        );
        assert_eq!(
            sq8_gfx942_aprime_implementation_for_shape(1, 5121, 5120),
            None
        );
    }

    #[test]
    fn buffer_size_helpers_keep_bf16_and_f32_contracts_separate() {
        assert_eq!(
            sq8_gfx942_aprime_projection_buffer_bytes(16, 128),
            Ok((4096, 8192))
        );
        assert_eq!(
            sq8_gfx942_control_buffer_bytes(16, 128, 128),
            Ok((4096, 32768, 8192))
        );
    }

    #[test]
    fn gfx1201_public_abi_and_dispatch_source_remain_isolated_from_aprime() {
        // A′ is deliberately internal-only.  These source-level invariants
        // prevent a future refactor from silently adding gfx942 names to the
        // pre-existing public header or from routing it through the gfx1201
        // implementation body.
        let public_header = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/include/ullm_runtime.h"
        ));
        let gfx1201_dispatch = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/src/ullm_runtime_api_sq8_ck.inc"
        ));
        let gfx1201_body = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/src/sq8_ck_gfx1201.hip.cpp"
        ));
        assert!(!public_header.contains("gfx942"));
        assert!(!gfx1201_dispatch.contains("gfx942"));
        assert!(!gfx1201_body.contains("gfx942"));
    }

    #[cfg(not(feature = "rocm-ck-gfx942-aprime"))]
    #[test]
    fn feature_off_is_explicit_and_does_not_repurpose_gfx1201() {
        // Keep this test CPU-only: creating a RuntimeContext could select a
        // real HIP device.  The core wrapper contains a separate feature-off
        // error stub, while this binding cannot claim the feature is enabled.
        assert!(!sq8_gfx942_aprime_feature_enabled());
        assert!(!sq8_gfx942_aprime_is_selected_for_device(&DeviceInfo {
            device_id: 0,
            backend: "hip".to_string(),
            name: "synthetic".to_string(),
            total_global_mem: 0,
            compute_major: 9,
            compute_minor: 4,
            gcn_arch_name: "gfx942".to_string(),
            flags: 0,
        }));
    }
}
