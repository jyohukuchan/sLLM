// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU-only OCP E4M3FN to E4M3FNUZ prepack oracle for canonical `SQ8_0`.
//!
//! The canonical artifact stores raw OCP E4M3FN payload bytes and positive
//! BF16 `[128, 128]` dequantization multipliers.  gfx942 FP8 MFMA operands use
//! E4M3FNUZ instead.  For every finite OCP byte, the corresponding FNUZ value
//! is half the OCP value when the raw byte is retained, except that OCP's
//! negative zero (`0x80`) must be normalized to the only FNUZ zero (`0x00`).
//! A converted operand therefore requires an exactly doubled scale.

use crate::sq_canonical::Sq8CanonicalArtifact;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Default streaming chunk size for CPU-only artifact scans.
pub const SQ8_FNUZ_PREPACK_SCAN_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// How many OCP-to-FNUZ operands a scale quantity compensates.
///
/// A canonical weight scale uses [`OneConvertedOperand`].  The two-operand
/// form is useful for validating the MFMA pair-scale product: it is four times
/// the original product, not two times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FnuzScaleCompensation {
    OneConvertedOperand,
    TwoConvertedOperands,
}

impl FnuzScaleCompensation {
    /// Exact power-of-two multiplier applied to the scale quantity.
    pub const fn factor(self) -> f32 {
        match self {
            Self::OneConvertedOperand => 2.0,
            Self::TwoConvertedOperands => 4.0,
        }
    }
}

impl fmt::Display for FnuzScaleCompensation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OneConvertedOperand => formatter.write_str("one converted operand (x2)"),
            Self::TwoConvertedOperands => formatter.write_str("two converted operands (x4)"),
        }
    }
}

/// A finite OCP byte's FNUZ representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcpE4m3FnuzMapping {
    /// The raw bit pattern has the intended finite FNUZ meaning.
    Exact(u8),
    /// OCP negative zero has no FNUZ encoding and becomes positive zero.
    NegativeZeroNormalized,
}

impl OcpE4m3FnuzMapping {
    pub const fn byte(self) -> u8 {
        match self {
            Self::Exact(byte) => byte,
            Self::NegativeZeroNormalized => 0x00,
        }
    }
}

/// An OCP byte that has no admissible finite FNUZ prepack result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcpE4m3FnuzByteError {
    /// OCP E4M3FN reserves `0x7f` and `0xff` for NaN.
    OcpNaN { byte: u8 },
}

impl fmt::Display for OcpE4m3FnuzByteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OcpNaN { byte } => write!(
                formatter,
                "OCP E4M3FN byte 0x{byte:02x} is NaN and cannot be prepacked as FNUZ"
            ),
        }
    }
}

impl std::error::Error for OcpE4m3FnuzByteError {}

/// Maps one canonical OCP E4M3FN raw byte to its FNUZ raw byte.
///
/// The operation is deliberately fail-closed for OCP NaNs.  E4M3FN has no
/// infinity encodings, so those two NaNs are the complete non-finite input
/// set.  There is no finite OCP code other than negative zero that lacks an
/// FNUZ representation.
pub const fn map_ocp_e4m3fn_byte_to_fnuz(
    byte: u8,
) -> Result<OcpE4m3FnuzMapping, OcpE4m3FnuzByteError> {
    match byte {
        0x7f | 0xff => Err(OcpE4m3FnuzByteError::OcpNaN { byte }),
        0x80 => Ok(OcpE4m3FnuzMapping::NegativeZeroNormalized),
        _ => Ok(OcpE4m3FnuzMapping::Exact(byte)),
    }
}

/// Decodes a raw E4M3FNUZ byte using the ROCm FNUZ semantics.
///
/// FNUZ has exponent bias eight and reserves `0x80` as NaN.  Its minimum
/// positive subnormal is `2^-10` and its largest finite value is `240`.
pub fn fnuz_e4m3_to_f32(byte: u8) -> f32 {
    if byte == 0x80 {
        return f32::NAN;
    }
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        sign * (mantissa as f32) * 2.0_f32.powi(-10)
    } else {
        sign * (1.0 + (mantissa as f32) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 8)
    }
}

/// Converts a BF16 bit pattern to its exact `f32` value.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// A rejection from the BF16 scale range gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16ScaleTransformError {
    /// Canonical scales must be finite and strictly positive.
    InvalidSource { source_bits: u16 },
    /// The doubled or quadrupled result is not finite BF16.
    Overflow {
        source_bits: u16,
        compensation: FnuzScaleCompensation,
    },
    /// The result rounded to zero.  This cannot occur for a valid positive
    /// BF16 input multiplied by x2 or x4, but is guarded explicitly.
    Underflow {
        source_bits: u16,
        compensation: FnuzScaleCompensation,
    },
    /// A fail-closed guard for a result that cannot be represented exactly as
    /// BF16.  Power-of-two x2/x4 transforms of BF16 values should never take
    /// this branch.
    NonExactBf16 {
        source_bits: u16,
        result_f32_bits: u32,
        compensation: FnuzScaleCompensation,
    },
}

impl fmt::Display for Bf16ScaleTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSource { source_bits } => write!(
                formatter,
                "canonical BF16 scale 0x{source_bits:04x} is not finite and strictly positive"
            ),
            Self::Overflow {
                source_bits,
                compensation,
            } => write!(
                formatter,
                "canonical BF16 scale 0x{source_bits:04x} overflows under {compensation}"
            ),
            Self::Underflow {
                source_bits,
                compensation,
            } => write!(
                formatter,
                "canonical BF16 scale 0x{source_bits:04x} underflows under {compensation}"
            ),
            Self::NonExactBf16 {
                source_bits,
                result_f32_bits,
                compensation,
            } => write!(
                formatter,
                "canonical BF16 scale 0x{source_bits:04x} produces non-exact BF16 f32 bits 0x{result_f32_bits:08x} under {compensation}"
            ),
        }
    }
}

impl std::error::Error for Bf16ScaleTransformError {}

/// Applies the required x2 or x4 compensation to a canonical BF16 scale.
///
/// The transform is accepted only when the result remains a finite, strictly
/// positive, exactly representable BF16 value.  It never clamps or rounds.
pub fn prepack_bf16_scale_bits_for_fnuz(
    source_bits: u16,
    compensation: FnuzScaleCompensation,
) -> Result<u16, Bf16ScaleTransformError> {
    let source = bf16_bits_to_f32(source_bits);
    if !source.is_finite() || source <= 0.0 {
        return Err(Bf16ScaleTransformError::InvalidSource { source_bits });
    }

    let transformed = source * compensation.factor();
    if !transformed.is_finite() {
        return Err(Bf16ScaleTransformError::Overflow {
            source_bits,
            compensation,
        });
    }
    if transformed == 0.0 {
        return Err(Bf16ScaleTransformError::Underflow {
            source_bits,
            compensation,
        });
    }

    let result_f32_bits = transformed.to_bits();
    if result_f32_bits & 0xffff != 0 {
        return Err(Bf16ScaleTransformError::NonExactBf16 {
            source_bits,
            result_f32_bits,
            compensation,
        });
    }
    let result_bits = (result_f32_bits >> 16) as u16;
    let result = bf16_bits_to_f32(result_bits);
    if !result.is_finite() {
        return Err(Bf16ScaleTransformError::Overflow {
            source_bits,
            compensation,
        });
    }
    if result <= 0.0 {
        return Err(Bf16ScaleTransformError::Underflow {
            source_bits,
            compensation,
        });
    }
    Ok(result_bits)
}

/// An error while prepacking a canonical payload or its BF16 scale payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sq8FnuzPrepackError {
    PayloadNonFinite {
        offset: usize,
        byte: u8,
    },
    OddBf16ScalePayload {
        bytes: usize,
    },
    Scale {
        index: usize,
        source_bits: u16,
        error: Bf16ScaleTransformError,
    },
}

impl fmt::Display for Sq8FnuzPrepackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PayloadNonFinite { offset, byte } => write!(
                formatter,
                "canonical OCP E4M3FN payload contains non-finite byte 0x{byte:02x} at offset {offset}"
            ),
            Self::OddBf16ScalePayload { bytes } => write!(
                formatter,
                "canonical BF16 scale payload has odd byte length {bytes}"
            ),
            Self::Scale {
                index,
                source_bits,
                error,
            } => write!(
                formatter,
                "canonical BF16 scale {index} (0x{source_bits:04x}) cannot be FNUZ-prepacked: {error}"
            ),
        }
    }
}

impl std::error::Error for Sq8FnuzPrepackError {}

/// Prepacks raw OCP E4M3FN bytes into raw FNUZ bytes.
pub fn prepack_ocp_e4m3fn_payload_to_fnuz(payload: &[u8]) -> Result<Vec<u8>, Sq8FnuzPrepackError> {
    let mut result = Vec::with_capacity(payload.len());
    for (offset, byte) in payload.iter().copied().enumerate() {
        let mapping = map_ocp_e4m3fn_byte_to_fnuz(byte)
            .map_err(|_| Sq8FnuzPrepackError::PayloadNonFinite { offset, byte })?;
        result.push(mapping.byte());
    }
    Ok(result)
}

/// Prepacks little-endian BF16 scales with the requested FNUZ compensation.
pub fn prepack_bf16_scale_payload_for_fnuz(
    scales_bf16_le: &[u8],
    compensation: FnuzScaleCompensation,
) -> Result<Vec<u8>, Sq8FnuzPrepackError> {
    if !scales_bf16_le
        .len()
        .is_multiple_of(std::mem::size_of::<u16>())
    {
        return Err(Sq8FnuzPrepackError::OddBf16ScalePayload {
            bytes: scales_bf16_le.len(),
        });
    }
    let mut result = Vec::with_capacity(scales_bf16_le.len());
    for (index, bytes) in scales_bf16_le.chunks_exact(2).enumerate() {
        let source_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let transformed =
            prepack_bf16_scale_bits_for_fnuz(source_bits, compensation).map_err(|error| {
                Sq8FnuzPrepackError::Scale {
                    index,
                    source_bits,
                    error,
                }
            })?;
        result.extend_from_slice(&transformed.to_le_bytes());
    }
    Ok(result)
}

/// A derived, immutable FNUZ tensor representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8FnuzPrepackedTensor {
    pub payload: Vec<u8>,
    pub scales_bf16_le: Vec<u8>,
    pub scale_compensation: FnuzScaleCompensation,
}

/// Prepacks one canonical OCP payload and its BF16 scale payload together.
///
/// The function returns no partially usable result: a payload NaN or a failed
/// scale-range gate rejects the whole derived tensor.
pub fn prepack_sq8_ocp_e4m3fn_tensor_to_fnuz(
    payload: &[u8],
    scales_bf16_le: &[u8],
    compensation: FnuzScaleCompensation,
) -> Result<Sq8FnuzPrepackedTensor, Sq8FnuzPrepackError> {
    let payload = prepack_ocp_e4m3fn_payload_to_fnuz(payload)?;
    let scales_bf16_le = prepack_bf16_scale_payload_for_fnuz(scales_bf16_le, compensation)?;
    Ok(Sq8FnuzPrepackedTensor {
        payload,
        scales_bf16_le,
        scale_compensation: compensation,
    })
}

/// CPU scan result for a verified canonical `SQ8_0` artifact.
///
/// `byte_frequency` is indexed by the raw OCP byte value and always has 256
/// entries.  The scanner hashes the exact bytes it classifies against the
/// canonical manifest, so the reported frequency is tied to the artifact
/// identity rather than merely to a path.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sq8FnuzArtifactScan {
    pub format_id: String,
    pub artifact_content_sha256: String,
    pub tensor_count: u64,
    pub payload_bytes: u64,
    pub byte_frequency: Vec<u64>,
    pub ocp_negative_zero_count: u64,
    pub ocp_nan_0x7f_count: u64,
    pub ocp_nan_0xff_count: u64,
    pub finite_fnuz_unrepresentable_count: u64,
    pub scale_count: u64,
    pub invalid_bf16_scale_count: u64,
    pub scale_x2_overflow_count: u64,
    pub scale_x4_overflow_count: u64,
    pub scale_x2_underflow_count: u64,
    pub scale_x4_underflow_count: u64,
    pub scale_x2_non_exact_count: u64,
    pub scale_x4_non_exact_count: u64,
    pub min_positive_bf16_scale: Option<f32>,
    pub max_positive_bf16_scale: Option<f32>,
}

impl Sq8FnuzArtifactScan {
    /// True only when every scanned payload and scale can use this prepack.
    pub fn prepack_eligible(&self) -> bool {
        self.ocp_nan_0x7f_count == 0
            && self.ocp_nan_0xff_count == 0
            && self.finite_fnuz_unrepresentable_count == 0
            && self.invalid_bf16_scale_count == 0
            && self.scale_x2_overflow_count == 0
            && self.scale_x4_overflow_count == 0
            && self.scale_x2_underflow_count == 0
            && self.scale_x4_underflow_count == 0
            && self.scale_x2_non_exact_count == 0
            && self.scale_x4_non_exact_count == 0
    }
}

/// Scans every canonical payload byte and BF16 scale using CPU I/O only.
///
/// The supplied artifact has already passed its normal canonical validation.
/// This scan performs a second, independent hash-checked pass so it can report
/// complete frequencies and the FNUZ scale gate outcome for the same bytes.
pub fn scan_sq8_canonical_artifact_for_fnuz_prepack(
    artifact: &Sq8CanonicalArtifact,
    chunk_bytes: usize,
) -> Result<Sq8FnuzArtifactScan, String> {
    if chunk_bytes == 0 {
        return Err("SQ8 FNUZ artifact scan chunk_bytes must be greater than zero".to_string());
    }
    let manifest = artifact.manifest();
    let tensor_count = u64::try_from(manifest.quantized_tensors.len())
        .map_err(|_| "SQ8 FNUZ artifact tensor count does not fit u64".to_string())?;
    let mut report = Sq8FnuzArtifactScan {
        format_id: manifest.format_id.clone(),
        artifact_content_sha256: manifest.integrity.content_sha256.clone(),
        tensor_count,
        payload_bytes: 0,
        byte_frequency: vec![0; 256],
        ocp_negative_zero_count: 0,
        ocp_nan_0x7f_count: 0,
        ocp_nan_0xff_count: 0,
        finite_fnuz_unrepresentable_count: 0,
        scale_count: 0,
        invalid_bf16_scale_count: 0,
        scale_x2_overflow_count: 0,
        scale_x4_overflow_count: 0,
        scale_x2_underflow_count: 0,
        scale_x4_underflow_count: 0,
        scale_x2_non_exact_count: 0,
        scale_x4_non_exact_count: 0,
        min_positive_bf16_scale: None,
        max_positive_bf16_scale: None,
    };

    for pair in &manifest.quantized_tensors {
        let paths = artifact.tensor_payload_paths(&pair.name)?;
        scan_declared_file(
            &paths.weight,
            pair.weight.bytes,
            &pair.weight.sha256,
            chunk_bytes,
            |bytes| scan_ocp_payload_chunk(bytes, &mut report),
        )?;
        let scale_chunk_bytes = chunk_bytes.max(2) & !1;
        scan_declared_file(
            &paths.scale,
            pair.scale.bytes,
            &pair.scale.sha256,
            scale_chunk_bytes,
            |bytes| scan_bf16_scale_chunk(bytes, &mut report),
        )?;
    }

    if report.payload_bytes != manifest.storage.weight_payload_bytes {
        return Err(format!(
            "SQ8 FNUZ scan payload byte count mismatch: scanned={} manifest={}",
            report.payload_bytes, manifest.storage.weight_payload_bytes
        ));
    }
    let expected_scale_count = manifest.storage.scale_payload_bytes / 2;
    if report.scale_count != expected_scale_count {
        return Err(format!(
            "SQ8 FNUZ scan scale count mismatch: scanned={} manifest={expected_scale_count}",
            report.scale_count
        ));
    }
    Ok(report)
}

fn scan_ocp_payload_chunk(bytes: &[u8], report: &mut Sq8FnuzArtifactScan) -> Result<(), String> {
    report.payload_bytes =
        report
            .payload_bytes
            .checked_add(u64::try_from(bytes.len()).map_err(|_| {
                "SQ8 FNUZ scan payload chunk byte count does not fit u64".to_string()
            })?)
            .ok_or_else(|| "SQ8 FNUZ scan payload byte count overflows".to_string())?;
    for byte in bytes.iter().copied() {
        let frequency = report
            .byte_frequency
            .get_mut(usize::from(byte))
            .ok_or_else(|| "SQ8 FNUZ scan byte frequency index escaped 256 entries".to_string())?;
        *frequency = frequency
            .checked_add(1)
            .ok_or_else(|| "SQ8 FNUZ scan byte frequency overflows".to_string())?;
        match map_ocp_e4m3fn_byte_to_fnuz(byte) {
            Ok(OcpE4m3FnuzMapping::Exact(_)) => {}
            Ok(OcpE4m3FnuzMapping::NegativeZeroNormalized) => {
                report.ocp_negative_zero_count = report
                    .ocp_negative_zero_count
                    .checked_add(1)
                    .ok_or_else(|| "SQ8 FNUZ scan negative-zero count overflows".to_string())?;
            }
            Err(OcpE4m3FnuzByteError::OcpNaN { byte: 0x7f }) => {
                report.ocp_nan_0x7f_count = report
                    .ocp_nan_0x7f_count
                    .checked_add(1)
                    .ok_or_else(|| "SQ8 FNUZ scan 0x7f NaN count overflows".to_string())?;
            }
            Err(OcpE4m3FnuzByteError::OcpNaN { byte: 0xff }) => {
                report.ocp_nan_0xff_count = report
                    .ocp_nan_0xff_count
                    .checked_add(1)
                    .ok_or_else(|| "SQ8 FNUZ scan 0xff NaN count overflows".to_string())?;
            }
            Err(OcpE4m3FnuzByteError::OcpNaN { .. }) => {
                return Err("SQ8 FNUZ scan encountered an impossible OCP NaN code".to_string());
            }
        }
    }
    Ok(())
}

fn scan_bf16_scale_chunk(bytes: &[u8], report: &mut Sq8FnuzArtifactScan) -> Result<(), String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "SQ8 FNUZ scan received an odd BF16 scale chunk of {} bytes",
            bytes.len()
        ));
    }
    for raw in bytes.chunks_exact(2) {
        let source_bits = u16::from_le_bytes([raw[0], raw[1]]);
        report.scale_count = report
            .scale_count
            .checked_add(1)
            .ok_or_else(|| "SQ8 FNUZ scan scale count overflows".to_string())?;
        let source = bf16_bits_to_f32(source_bits);
        if !source.is_finite() || source <= 0.0 {
            report.invalid_bf16_scale_count = report
                .invalid_bf16_scale_count
                .checked_add(1)
                .ok_or_else(|| "SQ8 FNUZ scan invalid scale count overflows".to_string())?;
            continue;
        }
        report.min_positive_bf16_scale = Some(
            report
                .min_positive_bf16_scale
                .map_or(source, |current| current.min(source)),
        );
        report.max_positive_bf16_scale = Some(
            report
                .max_positive_bf16_scale
                .map_or(source, |current| current.max(source)),
        );
        count_scale_transform_error(
            prepack_bf16_scale_bits_for_fnuz(
                source_bits,
                FnuzScaleCompensation::OneConvertedOperand,
            )
            .err(),
            true,
            report,
        )?;
        count_scale_transform_error(
            prepack_bf16_scale_bits_for_fnuz(
                source_bits,
                FnuzScaleCompensation::TwoConvertedOperands,
            )
            .err(),
            false,
            report,
        )?;
    }
    Ok(())
}

fn count_scale_transform_error(
    error: Option<Bf16ScaleTransformError>,
    one_operand: bool,
    report: &mut Sq8FnuzArtifactScan,
) -> Result<(), String> {
    let Some(error) = error else {
        return Ok(());
    };
    match error {
        Bf16ScaleTransformError::InvalidSource { .. } => {
            return Err(
                "SQ8 FNUZ scale scanner classified a validated scale as invalid".to_string(),
            );
        }
        Bf16ScaleTransformError::Overflow { .. } => {
            let count = if one_operand {
                &mut report.scale_x2_overflow_count
            } else {
                &mut report.scale_x4_overflow_count
            };
            *count = count
                .checked_add(1)
                .ok_or_else(|| "SQ8 FNUZ scan scale overflow count overflows".to_string())?;
        }
        Bf16ScaleTransformError::Underflow { .. } => {
            let count = if one_operand {
                &mut report.scale_x2_underflow_count
            } else {
                &mut report.scale_x4_underflow_count
            };
            *count = count
                .checked_add(1)
                .ok_or_else(|| "SQ8 FNUZ scan scale underflow count overflows".to_string())?;
        }
        Bf16ScaleTransformError::NonExactBf16 { .. } => {
            let count = if one_operand {
                &mut report.scale_x2_non_exact_count
            } else {
                &mut report.scale_x4_non_exact_count
            };
            *count = count
                .checked_add(1)
                .ok_or_else(|| "SQ8 FNUZ scan non-exact scale count overflows".to_string())?;
        }
    }
    Ok(())
}

fn scan_declared_file(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    chunk_bytes: usize,
    mut consume: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let mut file = File::open(path).map_err(|error| {
        format!(
            "failed to open {} for SQ8 FNUZ scan: {error}",
            path.display()
        )
    })?;
    let opened_bytes = file
        .metadata()
        .map_err(|error| {
            format!(
                "failed to stat {} for SQ8 FNUZ scan: {error}",
                path.display()
            )
        })?
        .len();
    if opened_bytes != expected_bytes {
        return Err(format!(
            "SQ8 FNUZ scan byte length mismatch before reading {}: manifest={expected_bytes} file={opened_bytes}",
            path.display()
        ));
    }
    let buffer_len = usize::try_from(expected_bytes.min(chunk_bytes as u64))
        .map_err(|_| {
            format!(
                "SQ8 FNUZ scan chunk length does not fit usize: {}",
                path.display()
            )
        })?
        .max(1);
    let mut buffer = vec![0_u8; buffer_len];
    let mut remaining = expected_bytes;
    let mut digest = Sha256::new();
    while remaining > 0 {
        let read_len = usize::try_from(remaining.min(buffer.len() as u64)).map_err(|_| {
            format!(
                "SQ8 FNUZ scan read length does not fit usize: {}",
                path.display()
            )
        })?;
        file.read_exact(&mut buffer[..read_len]).map_err(|error| {
            format!(
                "failed to read {read_len} bytes from {} during SQ8 FNUZ scan: {error}",
                path.display()
            )
        })?;
        let chunk = &buffer[..read_len];
        digest.update(chunk);
        consume(chunk)?;
        remaining -= read_len as u64;
    }
    let mut trailing = [0_u8; 1];
    let trailing_bytes = file.read(&mut trailing).map_err(|error| {
        format!(
            "failed to check EOF for {} during SQ8 FNUZ scan: {error}",
            path.display()
        )
    })?;
    if trailing_bytes != 0 {
        return Err(format!(
            "SQ8 FNUZ scan found trailing data after {expected_bytes} bytes in {}",
            path.display()
        ));
    }
    let final_bytes = file
        .metadata()
        .map_err(|error| {
            format!(
                "failed to re-stat {} for SQ8 FNUZ scan: {error}",
                path.display()
            )
        })?
        .len();
    if final_bytes != expected_bytes {
        return Err(format!(
            "SQ8 FNUZ scan byte length changed for {}: before={opened_bytes} after={final_bytes} expected={expected_bytes}",
            path.display()
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "SQ8 FNUZ scan checksum mismatch for {}: manifest={expected_sha256} file={actual_sha256}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq::fp8_e4m3fn_to_f32;

    #[test]
    fn fnuz_decoder_has_the_header_special_cases() {
        assert_eq!(fnuz_e4m3_to_f32(0x00).to_bits(), 0.0_f32.to_bits());
        assert!(fnuz_e4m3_to_f32(0x80).is_nan());
        assert_eq!(fnuz_e4m3_to_f32(0x01), 2.0_f32.powi(-10));
        assert_eq!(fnuz_e4m3_to_f32(0x7e), 224.0);
        assert_eq!(fnuz_e4m3_to_f32(0x7f), 240.0);
    }

    #[test]
    fn all_finite_ocp_bytes_scale_to_their_fnuz_values() {
        let mut mapped = 0_u16;
        let mut rejected = Vec::new();
        for raw in 0_u8..=u8::MAX {
            match map_ocp_e4m3fn_byte_to_fnuz(raw) {
                Ok(mapping) => {
                    mapped += 1;
                    let ocp = fp8_e4m3fn_to_f32(raw);
                    let fnuz = fnuz_e4m3_to_f32(mapping.byte());
                    assert!(ocp.is_finite(), "0x{raw:02x}");
                    assert!(fnuz.is_finite(), "0x{raw:02x}");
                    assert_eq!(ocp, 2.0 * fnuz, "0x{raw:02x}");
                }
                Err(OcpE4m3FnuzByteError::OcpNaN { byte }) => rejected.push(byte),
            }
        }
        assert_eq!(mapped, 254);
        assert_eq!(rejected, [0x7f, 0xff]);
        assert_eq!(
            map_ocp_e4m3fn_byte_to_fnuz(0x80),
            Ok(OcpE4m3FnuzMapping::NegativeZeroNormalized)
        );
    }
}
