// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU preparation and reference math for the isolated `SQ8_0` gfx942 A′
//! prototype and its dequant-to-BF16/FP16 B control.
//!
//! This module is intentionally CPU-only.  It prepares FNUZ-derived buffers
//! through [`crate::sq8_fnuz_prepack`] and computes the expectations consumed
//! by the physical-gfx942 smoke test.  It never selects a local GPU.

use crate::sq::fp8_e4m3fn_to_f32;
use crate::sq8_fnuz_prepack::{
    FnuzScaleCompensation, bf16_bits_to_f32 as oracle_bf16_bits_to_f32, fnuz_e4m3_to_f32,
    prepack_f32_scale_payload_for_fnuz, prepack_ocp_e4m3fn_payload_to_fnuz,
    prepack_sq8_ocp_e4m3fn_tensor_to_fnuz,
};
use std::collections::{BTreeMap, BTreeSet};

pub const SQ8_0_BLOCK_K: usize = 128;
pub const SQ8_0_BLOCK_N: usize = 128;

/// The scale layout of an `SQ8_0` matrix operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8_0ScaleLayout {
    /// Activation A[M,K] has one scale for each row/K128 block.
    ActivationRowK128,
    /// Weight B[N,K] has one scale for each N128/K128 block.
    WeightBlock128x128,
}

/// Borrowed canonical OCP bytes and F32-expanded `SQ8_0` scales.
#[derive(Debug, Clone, Copy)]
pub struct Sq8_0OcpBlockScaledMatrix<'a> {
    payload: &'a [u8],
    scales: &'a [f32],
    rows: usize,
    cols: usize,
    scale_layout: Sq8_0ScaleLayout,
}

impl<'a> Sq8_0OcpBlockScaledMatrix<'a> {
    pub fn activation(
        payload: &'a [u8],
        scales: &'a [f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        Self::new(
            payload,
            scales,
            rows,
            cols,
            Sq8_0ScaleLayout::ActivationRowK128,
        )
    }

    pub fn weight(
        payload: &'a [u8],
        scales: &'a [f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        Self::new(
            payload,
            scales,
            rows,
            cols,
            Sq8_0ScaleLayout::WeightBlock128x128,
        )
    }

    fn new(
        payload: &'a [u8],
        scales: &'a [f32],
        rows: usize,
        cols: usize,
        scale_layout: Sq8_0ScaleLayout,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(SQ8_0_BLOCK_K) {
            return Err(
                "SQ8_0 block-scaled matrix requires nonzero rows and K divisible by 128".into(),
            );
        }
        if scale_layout == Sq8_0ScaleLayout::WeightBlock128x128
            && !rows.is_multiple_of(SQ8_0_BLOCK_N)
        {
            return Err("SQ8_0 block-scaled weight requires N divisible by 128".into());
        }
        let payload_len = rows
            .checked_mul(cols)
            .ok_or_else(|| "SQ8_0 payload element count overflows".to_string())?;
        if payload.len() != payload_len {
            return Err(format!(
                "SQ8_0 payload has {} bytes; expected {payload_len}",
                payload.len()
            ));
        }
        let scale_len = match scale_layout {
            Sq8_0ScaleLayout::ActivationRowK128 => rows
                .checked_mul(cols / SQ8_0_BLOCK_K)
                .ok_or_else(|| "SQ8_0 activation scale count overflows".to_string())?,
            Sq8_0ScaleLayout::WeightBlock128x128 => (rows / SQ8_0_BLOCK_N)
                .checked_mul(cols / SQ8_0_BLOCK_K)
                .ok_or_else(|| "SQ8_0 weight scale count overflows".to_string())?,
        };
        if scales.len() != scale_len {
            return Err(format!(
                "SQ8_0 scale count is {}; expected {scale_len}",
                scales.len()
            ));
        }
        if let Some((index, scale)) = scales
            .iter()
            .copied()
            .enumerate()
            .find(|(_, scale)| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(format!(
                "SQ8_0 scale {index} is not finite and strictly positive: {scale}"
            ));
        }
        Ok(Self {
            payload,
            scales,
            rows,
            cols,
            scale_layout,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub const fn scales(&self) -> &'a [f32] {
        self.scales
    }

    pub const fn scale_layout(&self) -> Sq8_0ScaleLayout {
        self.scale_layout
    }

    fn scale_index(&self, row: usize, col: usize) -> usize {
        match self.scale_layout {
            Sq8_0ScaleLayout::ActivationRowK128 => {
                row * (self.cols / SQ8_0_BLOCK_K) + col / SQ8_0_BLOCK_K
            }
            Sq8_0ScaleLayout::WeightBlock128x128 => {
                (row / SQ8_0_BLOCK_N) * (self.cols / SQ8_0_BLOCK_K) + col / SQ8_0_BLOCK_K
            }
        }
    }

    fn decoded_ocp_at(&self, row: usize, col: usize) -> Result<f32, String> {
        let raw = self.payload[row * self.cols + col];
        let value = fp8_e4m3fn_to_f32(raw);
        if !value.is_finite() {
            return Err(format!(
                "SQ8_0 canonical OCP payload contains non-finite byte 0x{raw:02x} at ({row},{col})"
            ));
        }
        Ok(value * self.scales[self.scale_index(row, col)])
    }
}

/// A FNUZ-derived opaque operand for A′.  Its bytes have no OCP semantic
/// meaning even though the installed CK archive's link ABI is `f8_ocp_t`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8_0FnuzPrepackedMatrix {
    pub payload: Vec<u8>,
    pub scales_f32_x2: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub scale_layout: Sq8_0ScaleLayout,
}

impl Sq8_0FnuzPrepackedMatrix {
    /// Uses the shared fail-closed byte and F32 scale oracle for exactly one
    /// converted operand.  Call this independently for A and B: each gets x2,
    /// making their product x4 inside the CK ABScale computation.
    pub fn from_ocp(source: Sq8_0OcpBlockScaledMatrix<'_>) -> Result<Self, String> {
        let payload = prepack_ocp_e4m3fn_payload_to_fnuz(source.payload)
            .map_err(|error| format!("SQ8_0 FNUZ payload prepack: {error}"))?;
        let scales_f32_x2 = prepack_f32_scale_payload_for_fnuz(
            source.scales,
            FnuzScaleCompensation::OneConvertedOperand,
        )
        .map_err(|error| format!("SQ8_0 FNUZ scale prepack: {error}"))?;
        Ok(Self {
            payload,
            scales_f32_x2,
            rows: source.rows,
            cols: source.cols,
            scale_layout: source.scale_layout,
        })
    }

    /// Builds an opaque A′ operand directly from an artifact-format OCP
    /// payload plus little-endian BF16 scales.  This path is deliberately
    /// routed through the original atomic byte-and-BF16 oracle rather than
    /// reconstructing its x2 transform here.  It is the weight-side entry
    /// point for canonical `SQ8_0` artifacts; dynamic activations use
    /// [`Self::from_ocp`] with their F32 scales.
    pub fn from_ocp_bf16_scales(
        payload: &[u8],
        scales_bf16_le: &[u8],
        rows: usize,
        cols: usize,
        scale_layout: Sq8_0ScaleLayout,
    ) -> Result<Self, String> {
        if !scales_bf16_le.len().is_multiple_of(2) {
            return Err(format!(
                "SQ8_0 BF16 scale payload has odd byte length {}",
                scales_bf16_le.len()
            ));
        }
        let canonical_scales_f32: Vec<f32> = scales_bf16_le
            .chunks_exact(2)
            .map(|bytes| oracle_bf16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect();
        let source = Sq8_0OcpBlockScaledMatrix::new(
            payload,
            &canonical_scales_f32,
            rows,
            cols,
            scale_layout,
        )?;
        let packed = prepack_sq8_ocp_e4m3fn_tensor_to_fnuz(
            source.payload,
            scales_bf16_le,
            FnuzScaleCompensation::OneConvertedOperand,
        )
        .map_err(|error| format!("SQ8_0 FNUZ BF16 prepack: {error}"))?;
        let scales_f32_x2 = packed
            .scales_bf16_le
            .chunks_exact(2)
            .map(|bytes| oracle_bf16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect();
        Ok(Self {
            payload: packed.payload,
            scales_f32_x2,
            rows: source.rows,
            cols: source.cols,
            scale_layout: source.scale_layout,
        })
    }

    fn decoded_fnuz_at(&self, row: usize, col: usize) -> Result<f32, String> {
        let raw = self.payload[row * self.cols + col];
        let value = fnuz_e4m3_to_f32(raw);
        if !value.is_finite() {
            return Err(format!(
                "SQ8_0 derived FNUZ payload contains non-finite byte 0x{raw:02x} at ({row},{col})"
            ));
        }
        let scale_index = match self.scale_layout {
            Sq8_0ScaleLayout::ActivationRowK128 => {
                row * (self.cols / SQ8_0_BLOCK_K) + col / SQ8_0_BLOCK_K
            }
            Sq8_0ScaleLayout::WeightBlock128x128 => {
                (row / SQ8_0_BLOCK_N) * (self.cols / SQ8_0_BLOCK_K) + col / SQ8_0_BLOCK_K
            }
        };
        Ok(value * self.scales_f32_x2[scale_index])
    }
}

/// Low-precision dequantization format used by B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8_0ControlDequantPrecision {
    Fp16,
    Bf16,
}

/// Computes the canonical OCP F32 reference C[M,N] = A[M,K] × B[N,K]^T.
pub fn sq8_0_ocp_reference_gemm(
    activation: Sq8_0OcpBlockScaledMatrix<'_>,
    weight: Sq8_0OcpBlockScaledMatrix<'_>,
) -> Result<Vec<f32>, String> {
    validate_gemm_operands(activation, weight)?;
    let mut output = vec![0.0_f32; activation.rows * weight.rows];
    for row in 0..activation.rows {
        for column in 0..weight.rows {
            let mut sum = 0.0_f32;
            for k in 0..activation.cols {
                sum += activation.decoded_ocp_at(row, k)? * weight.decoded_ocp_at(column, k)?;
            }
            output[row * weight.rows + column] = sum;
        }
    }
    Ok(output)
}

/// CPU model of the A′ byte/scaling contract.  It is deliberately a separate
/// decode from [`sq8_0_ocp_reference_gemm`], so the x2-per-operand / x4-pair
/// invariant is checked before any physical GPU is rented.
pub fn sq8_0_aprime_fnuz_reference_gemm(
    activation: &Sq8_0FnuzPrepackedMatrix,
    weight: &Sq8_0FnuzPrepackedMatrix,
) -> Result<Vec<f32>, String> {
    if activation.cols != weight.cols
        || activation.scale_layout != Sq8_0ScaleLayout::ActivationRowK128
        || weight.scale_layout != Sq8_0ScaleLayout::WeightBlock128x128
    {
        return Err("SQ8_0 A′ FNUZ operands have incompatible shapes or scale layouts".to_string());
    }
    let mut output = vec![0.0_f32; activation.rows * weight.rows];
    for row in 0..activation.rows {
        for column in 0..weight.rows {
            let mut sum = 0.0_f32;
            for k in 0..activation.cols {
                sum += activation.decoded_fnuz_at(row, k)? * weight.decoded_fnuz_at(column, k)?;
            }
            output[row * weight.rows + column] = sum;
        }
    }
    Ok(output)
}

/// CPU model of B: direct OCP decode followed by BF16 or FP16 rounding before
/// the F32 accumulation.  This path never invokes the FNUZ oracle.
pub fn sq8_0_control_dequant_gemm(
    activation: Sq8_0OcpBlockScaledMatrix<'_>,
    weight: Sq8_0OcpBlockScaledMatrix<'_>,
    precision: Sq8_0ControlDequantPrecision,
) -> Result<Vec<f32>, String> {
    validate_gemm_operands(activation, weight)?;
    let activation = dequantize_ocp_operand(activation, precision)?;
    let weight = dequantize_ocp_operand(weight, precision)?;
    let rows = activation.rows;
    let n = weight.rows;
    let k = activation.cols;
    let mut output = vec![0.0_f32; rows * n];
    for row in 0..rows {
        for column in 0..n {
            let mut sum = 0.0_f32;
            for index in 0..k {
                sum += activation.values[row * k + index] * weight.values[column * k + index];
            }
            output[row * n + column] = sum;
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq)]
struct DequantizedMatrix {
    values: Vec<f32>,
    rows: usize,
    cols: usize,
}

fn validate_gemm_operands(
    activation: Sq8_0OcpBlockScaledMatrix<'_>,
    weight: Sq8_0OcpBlockScaledMatrix<'_>,
) -> Result<(), String> {
    if activation.scale_layout != Sq8_0ScaleLayout::ActivationRowK128
        || weight.scale_layout != Sq8_0ScaleLayout::WeightBlock128x128
    {
        return Err(
            "SQ8_0 GEMM requires activation row-K128 and weight block-128x128 scales".into(),
        );
    }
    if activation.cols != weight.cols {
        return Err(format!(
            "SQ8_0 GEMM K mismatch: activation K={} weight K={}",
            activation.cols, weight.cols
        ));
    }
    Ok(())
}

fn dequantize_ocp_operand(
    matrix: Sq8_0OcpBlockScaledMatrix<'_>,
    precision: Sq8_0ControlDequantPrecision,
) -> Result<DequantizedMatrix, String> {
    let mut values = Vec::with_capacity(matrix.payload.len());
    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            let value = matrix.decoded_ocp_at(row, col)?;
            let rounded = match precision {
                Sq8_0ControlDequantPrecision::Fp16 => f16_bits_to_f32(f32_to_f16_rne_bits(value)?),
                Sq8_0ControlDequantPrecision::Bf16 => {
                    bf16_bits_to_f32(f32_to_bf16_rne_bits(value)?)
                }
            };
            values.push(rounded);
        }
    }
    Ok(DequantizedMatrix {
        values,
        rows: matrix.rows,
        cols: matrix.cols,
    })
}

fn f32_to_bf16_rne_bits(value: f32) -> Result<u16, String> {
    if !value.is_finite() {
        return Err(format!(
            "SQ8_0 B BF16 dequant result is non-finite: {value}"
        ));
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    Ok(((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16)
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_f16_rne_bits(value: f32) -> Result<u16, String> {
    if !value.is_finite() {
        return Err(format!(
            "SQ8_0 B FP16 dequant result is non-finite: {value}"
        ));
    }
    let sign = if value.is_sign_negative() {
        0x8000_u16
    } else {
        0_u16
    };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return Ok(sign);
    }
    if magnitude > 65504.0 {
        return Err(format!("SQ8_0 B FP16 dequant result overflows: {value}"));
    }
    if magnitude < 2.0_f32.powi(-14) {
        let scaled = magnitude * 2.0_f32.powi(24);
        let rounded = round_ties_even_nonnegative(scaled)?;
        return Ok(sign | rounded.min(0x03ff));
    }

    let bits = magnitude.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    let mut half_mantissa = (mantissa >> 13) as u16;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half_mantissa & 1 == 1) {
        half_mantissa += 1;
    }
    let mut half_exponent = exponent as u16;
    if half_mantissa == 0x0400 {
        half_mantissa = 0;
        half_exponent += 1;
    }
    if half_exponent >= 31 {
        return Err(format!("SQ8_0 B FP16 dequant result overflows: {value}"));
    }
    Ok(sign | (half_exponent << 10) | half_mantissa)
}

fn round_ties_even_nonnegative(value: f32) -> Result<u16, String> {
    if !value.is_finite() || value < 0.0 || value > f32::from(u16::MAX) {
        return Err("SQ8_0 FP16 subnormal rounding input is outside u16 range".to_string());
    }
    let floor = value.floor();
    let fraction = value - floor;
    let mut rounded = floor as u16;
    if fraction > 0.5 || (fraction == 0.5 && rounded & 1 == 1) {
        rounded += 1;
    }
    Ok(rounded)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    if exponent == 0 {
        return sign * f32::from(mantissa) * 2.0_f32.powi(-24);
    }
    sign * (1.0 + f32::from(mantissa) / 1024.0) * 2.0_f32.powi(i32::from(exponent) - 15)
}

/// A CPU-generated fixture for the one-wave physical fragment diagnostic.
/// `b_fnuz_32x16_column_major` is laid out exactly as rocWMMA's col-major B
/// input expects.  Every expected output is distinct, so runtime results can
/// infer `(lane, register) -> (row, column)` without encoding an unverified
/// fragment layout assumption into production code.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8_0FnuzFragmentProbeFixture {
    pub a_fnuz_16x32_row_major: Vec<u8>,
    pub b_fnuz_32x16_column_major: Vec<u8>,
    pub expected_matrix_f32_16x16: Vec<f32>,
}

pub fn sq8_0_fnuz_fragment_probe_fixture() -> Result<Sq8_0FnuzFragmentProbeFixture, String> {
    let finite_codes: Vec<u8> = (1_u8..0x7f).collect();
    let mut state = 0x51_82_a4_d3_u32;
    for _attempt in 0..4096 {
        let mut a = vec![0_u8; 16 * 32];
        let mut b = vec![0_u8; 32 * 16];
        for value in a.iter_mut().chain(b.iter_mut()) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = finite_codes[((state >> 16) as usize) % finite_codes.len()];
        }
        let expected = fragment_probe_expected_matrix(&a, &b)?;
        let unique = expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<BTreeSet<_>>();
        if unique.len() == expected.len() {
            return Ok(Sq8_0FnuzFragmentProbeFixture {
                a_fnuz_16x32_row_major: a,
                b_fnuz_32x16_column_major: b,
                expected_matrix_f32_16x16: expected,
            });
        }
    }
    Err("failed to construct a unique-output SQ8_0 FNUZ fragment fixture".to_string())
}

fn fragment_probe_expected_matrix(a: &[u8], b: &[u8]) -> Result<Vec<f32>, String> {
    if a.len() != 16 * 32 || b.len() != 32 * 16 {
        return Err("SQ8_0 fragment fixture has invalid input sizes".to_string());
    }
    let mut output = vec![0.0_f32; 16 * 16];
    for row in 0..16 {
        for column in 0..16 {
            let mut sum = 0.0_f32;
            for k in 0..32 {
                sum += fnuz_e4m3_to_f32(a[row * 32 + k]) * fnuz_e4m3_to_f32(b[column * 32 + k]);
            }
            if !sum.is_finite() {
                return Err("SQ8_0 fragment fixture produced a non-finite expectation".to_string());
            }
            output[row * 16 + column] = sum;
        }
    }
    Ok(output)
}

/// One observed raw accumulator register slot and its logical output location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sq8_0FragmentLaneCoordinate {
    pub lane: usize,
    pub register: usize,
    pub row: usize,
    pub column: usize,
}

/// Infers a diagnostic lane/register map from an observed logical matrix and
/// raw accumulator dump.  It makes no permanent rocWMMA layout assumption.
pub fn infer_sq8_0_fragment_lane_map(
    matrix_f32_16x16: &[f32],
    fragment_f32_lane64x4: &[f32],
) -> Result<Vec<Sq8_0FragmentLaneCoordinate>, String> {
    if matrix_f32_16x16.len() != 16 * 16 || fragment_f32_lane64x4.len() != 64 * 4 {
        return Err("SQ8_0 fragment diagnostic has invalid output sizes".to_string());
    }
    let mut locations = BTreeMap::new();
    for (index, value) in matrix_f32_16x16.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "SQ8_0 fragment matrix has non-finite value at {index}"
            ));
        }
        if locations
            .insert(value.to_bits(), (index / 16, index % 16))
            .is_some()
        {
            return Err(
                "SQ8_0 fragment matrix values are not unique; cannot infer a lane map".into(),
            );
        }
    }
    let mut mapping = Vec::with_capacity(64 * 4);
    for (index, value) in fragment_f32_lane64x4.iter().copied().enumerate() {
        let Some(&(row, column)) = locations.get(&value.to_bits()) else {
            return Err(format!(
                "SQ8_0 fragment slot {index} is not an exact value in the logical matrix"
            ));
        };
        mapping.push(Sq8_0FragmentLaneCoordinate {
            lane: index / 4,
            register: index % 4,
            row,
            column,
        });
    }
    mapping.sort_unstable();
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_matrices() -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>) {
        let mut activation = vec![0_u8; 2 * 128];
        let mut weight = vec![0_u8; 128 * 128];
        for (index, value) in activation.iter_mut().enumerate() {
            *value = match index % 5 {
                0 => 0x38, // OCP 1.0 / FNUZ 0.5
                1 => 0x40, // OCP 2.0 / FNUZ 1.0
                2 => 0x80, // OCP -0.0 -> FNUZ +0.0
                3 => 0xb8, // OCP -1.0 / FNUZ -0.5
                _ => 0x30, // OCP 0.5 / FNUZ 0.25
            };
        }
        for (index, value) in weight.iter_mut().enumerate() {
            *value = match index % 7 {
                0 => 0x38,
                1 => 0x40,
                2 => 0x30,
                3 => 0xb8,
                4 => 0x48,
                5 => 0x28,
                _ => 0x00,
            };
        }
        (activation, vec![0.5, 1.0], weight, vec![0.75])
    }

    #[test]
    fn aprime_cpu_oracle_applies_x2_to_each_operand_and_matches_ocp_reference() {
        let (activation_bytes, activation_scales, weight_bytes, weight_scales) = fixture_matrices();
        let activation =
            Sq8_0OcpBlockScaledMatrix::activation(&activation_bytes, &activation_scales, 2, 128)
                .unwrap();
        let weight =
            Sq8_0OcpBlockScaledMatrix::weight(&weight_bytes, &weight_scales, 128, 128).unwrap();
        let expected = sq8_0_ocp_reference_gemm(activation, weight).unwrap();
        let fnuz_activation = Sq8_0FnuzPrepackedMatrix::from_ocp(activation).unwrap();
        let fnuz_weight = Sq8_0FnuzPrepackedMatrix::from_ocp(weight).unwrap();
        assert_eq!(fnuz_activation.payload[2], 0x00);
        assert_eq!(fnuz_activation.scales_f32_x2, [1.0, 2.0]);
        assert_eq!(fnuz_weight.scales_f32_x2, [1.5]);
        assert_eq!(
            sq8_0_aprime_fnuz_reference_gemm(&fnuz_activation, &fnuz_weight).unwrap(),
            expected
        );
    }

    #[test]
    fn artifact_bf16_weight_prepack_uses_the_atomic_oracle_and_expands_losslessly() {
        let (_, _, weight_bytes, _) = fixture_matrices();
        let packed = Sq8_0FnuzPrepackedMatrix::from_ocp_bf16_scales(
            &weight_bytes,
            &0x3f40_u16.to_le_bytes(), // canonical BF16 0.75
            128,
            128,
            Sq8_0ScaleLayout::WeightBlock128x128,
        )
        .unwrap();
        assert_eq!(packed.payload[2], 0x30);
        assert_eq!(packed.scales_f32_x2, [1.5]);
    }

    #[test]
    fn b_control_bf16_and_fp16_match_the_exact_reference_for_representable_fixture() {
        let (activation_bytes, activation_scales, weight_bytes, weight_scales) = fixture_matrices();
        let activation =
            Sq8_0OcpBlockScaledMatrix::activation(&activation_bytes, &activation_scales, 2, 128)
                .unwrap();
        let weight =
            Sq8_0OcpBlockScaledMatrix::weight(&weight_bytes, &weight_scales, 128, 128).unwrap();
        let reference = sq8_0_ocp_reference_gemm(activation, weight).unwrap();
        assert_eq!(
            sq8_0_control_dequant_gemm(activation, weight, Sq8_0ControlDequantPrecision::Bf16)
                .unwrap(),
            reference
        );
        assert_eq!(
            sq8_0_control_dequant_gemm(activation, weight, Sq8_0ControlDequantPrecision::Fp16)
                .unwrap(),
            reference
        );
    }

    #[test]
    fn fp16_and_bf16_rounding_handle_normal_and_subnormal_inputs() {
        assert_eq!(f16_bits_to_f32(f32_to_f16_rne_bits(1.5).unwrap()), 1.5);
        assert_eq!(
            f16_bits_to_f32(f32_to_f16_rne_bits(2.0_f32.powi(-24)).unwrap()),
            2.0_f32.powi(-24)
        );
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_rne_bits(1.5).unwrap()), 1.5);
        assert!(f32_to_f16_rne_bits(65536.0).is_err());
    }

    #[test]
    fn fragment_fixture_is_unique_and_can_infer_an_identity_dump_without_a_layout_assumption() {
        let fixture = sq8_0_fnuz_fragment_probe_fixture().unwrap();
        assert_eq!(fixture.expected_matrix_f32_16x16.len(), 256);
        let map = infer_sq8_0_fragment_lane_map(
            &fixture.expected_matrix_f32_16x16,
            &fixture.expected_matrix_f32_16x16,
        )
        .unwrap();
        assert_eq!(map.len(), 256);
        assert_eq!(
            map[0],
            Sq8_0FragmentLaneCoordinate {
                lane: 0,
                register: 0,
                row: 0,
                column: 0
            }
        );
    }
}
