//! OCP Microscaling (MX) numeric contracts.
//!
//! MXFP4 is E2M1 values in blocks of 32 with one E8M0 scale. It is not
//! NVIDIA NVFP4, whose block size and hierarchical scale formats differ.

use crate::{decode_e2m1, decode_e4m3fn, encode_e2m1};
use std::fmt;
use std::thread;

pub const MX_BLOCK_SIZE: usize = 32;
const PARALLEL_QUANTIZATION_MIN_ELEMENTS: usize = 1 << 20;
const MAX_QUANTIZATION_WORKERS: usize = 32;

pub fn decode_e8m0(bits: u8) -> f32 {
    match bits {
        0 => f32::from_bits(0x0040_0000), // 2^-127
        255 => f32::NAN,
        exponent => f32::from_bits(u32::from(exponent) << 23),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MxElementFormat {
    E2M1,
    E3M2,
    E4M3Fn,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedMx {
    values: Vec<u8>,
    scales: Vec<u8>,
    rows: usize,
    columns: usize,
    format: MxElementFormat,
}

impl QuantizedMx {
    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn format(&self) -> MxElementFormat {
        self.format
    }

    pub fn dequantize(&self) -> Result<Vec<f32>, MxError> {
        decode_mx(
            &self.values,
            &self.scales,
            self.rows,
            self.columns,
            self.format,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MxError {
    Empty,
    ShapeOverflow,
    ColumnsNotBlockAligned { columns: usize },
    InputLength { expected: usize, actual: usize },
    ValueLength { expected: usize, actual: usize },
    ScaleLength { expected: usize, actual: usize },
}

impl fmt::Display for MxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("MX tensor dimensions must be non-zero"),
            Self::ShapeOverflow => formatter.write_str("MX tensor shape overflowed usize"),
            Self::ColumnsNotBlockAligned { columns } => write!(
                formatter,
                "MX W/A K dimension must be padded to a multiple of 32, got {columns}"
            ),
            Self::InputLength { expected, actual } => {
                write!(
                    formatter,
                    "MX input values: expected {expected}, got {actual}"
                )
            }
            Self::ValueLength { expected, actual } => {
                write!(
                    formatter,
                    "MX value bytes: expected {expected}, got {actual}"
                )
            }
            Self::ScaleLength { expected, actual } => {
                write!(
                    formatter,
                    "MX scale bytes: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for MxError {}

pub fn decode_mxfp4(
    packed_values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, MxError> {
    decode_mx(packed_values, scales, rows, columns, MxElementFormat::E2M1)
}

pub fn decode_mxfp8(
    values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, MxError> {
    decode_mx(values, scales, rows, columns, MxElementFormat::E4M3Fn)
}

/// Decode OCP MXFP6 E3M2 values packed as a little-endian six-bit stream.
/// The OCP numerical format does not prescribe a physical layout; sLLM's
/// resident contract stores each consecutive four values in three bytes.
pub fn decode_mxfp6(
    packed_values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, MxError> {
    decode_mx(packed_values, scales, rows, columns, MxElementFormat::E3M2)
}

/// Decode one OCP FP6 E3M2 code. Only the low six bits are significant.
pub fn decode_e3m2(bits: u8) -> f32 {
    let bits = bits & 0x3f;
    let sign: f32 = if bits & 0x20 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 2) & 0x07;
    let mantissa = bits & 0x03;
    if exponent == 0 {
        if mantissa == 0 {
            return if sign.is_sign_negative() { -0.0 } else { 0.0 };
        }
        return sign * f32::from(mantissa) * 2.0_f32.powi(-4);
    }
    sign * (1.0 + f32::from(mantissa) / 4.0) * 2.0_f32.powi(i32::from(exponent) - 3)
}

/// Encode OCP FP6 E3M2 using roundTiesToEven and finite saturation.
/// OCP FP6 has no Inf or NaN encoding; callers choose their NaN policy before
/// entering this scalar converter.
pub fn encode_e3m2(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x20 } else { 0 };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    if !magnitude.is_finite() || magnitude >= 28.0 {
        return sign | 0x1f;
    }
    sign | encode_e3m2_positive_rne(magnitude)
}

pub fn quantize_mxfp8_e4m3(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MxError> {
    quantize_mx(
        input,
        rows,
        columns,
        MxElementFormat::E4M3Fn,
        MxScalePolicy::Clipped,
    )
}

/// Quantize OCP MXFP8 E4M3 with a scale chosen to avoid finite E4M3
/// saturation.  The historical [`quantize_mxfp8_e4m3`] policy chooses the
/// lower power-of-two scale and consequently clips values above 448 times
/// that scale.  This diagnostic policy chooses the smallest E8M0 scale for
/// which every finite input in the block is representable without clipping.
///
/// NaN blocks and blocks containing infinity retain the historical handling;
/// this function only changes scale selection for finite, non-zero blocks.
pub fn quantize_mxfp8_e4m3_no_clipping_scale(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MxError> {
    quantize_mx(
        input,
        rows,
        columns,
        MxElementFormat::E4M3Fn,
        MxScalePolicy::NoClipping,
    )
}

pub fn quantize_mxfp6_e3m2(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MxError> {
    quantize_mx(
        input,
        rows,
        columns,
        MxElementFormat::E3M2,
        MxScalePolicy::Clipped,
    )
}

/// Quantize OCP MXFP6 E3M2 with the smallest E8M0 scale that does not clip
/// any finite element in the block.  This is the MXFP6 counterpart of
/// [`quantize_mxfp8_e4m3_no_clipping_scale`].
pub fn quantize_mxfp6_e3m2_no_clipping_scale(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MxError> {
    quantize_mx(
        input,
        rows,
        columns,
        MxElementFormat::E3M2,
        MxScalePolicy::NoClipping,
    )
}

/// Quantize OCP MXFP4 E2M1 with the historical sLLM codec rule: the lower
/// power-of-two E8M0 scale, bumped one step when the block maximum sits at or
/// above 1.75 times that power of two.  This mirrors the HIP
/// `mxfp4_even_scale_code` policy and exists as the reference for the new
/// best-of-two rule.
pub fn quantize_mxfp4_e2m1_even_scale(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MxError> {
    quantize_mx(
        input,
        rows,
        columns,
        MxElementFormat::E2M1,
        MxScalePolicy::EvenMxfp4,
    )
}

/// Quantize OCP MXFP4 E2M1 by trying the lower power-of-two E8M0 scale and
/// the next step, then keeping the one with the smaller block squared error.
/// Ties keep the lower scale.
pub fn quantize_mxfp4_e2m1_best_of_two(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MxError> {
    quantize_mx(
        input,
        rows,
        columns,
        MxElementFormat::E2M1,
        MxScalePolicy::BestOfTwo,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MxScalePolicy {
    /// Historical rule: always the lower power-of-two E8M0 scale.  Blocks
    /// whose maximum needs more range than the element format can represent
    /// saturate at the format's largest finite magnitude.
    Clipped,
    /// Smallest E8M0 scale that keeps every finite element representable.
    NoClipping,
    /// Try the [`Self::Clipped`] exponent and the next E8M0 step, and keep
    /// whichever gives the smaller block sum of squared error.  This is only
    /// used for element formats narrow enough that avoiding saturation can
    /// hurt the resolution of small values, i.e. E2M1.
    BestOfTwo,
    /// Historical sLLM MXFP4 codec rule: the lower power-of-two E8M0 scale,
    /// bumped one step when the block maximum is at or above 1.75 times that
    /// power of two.  Retained only as the reference for [`Self::BestOfTwo`].
    EvenMxfp4,
}

fn quantize_mx(
    input: &[f32],
    rows: usize,
    columns: usize,
    format: MxElementFormat,
    scale_policy: MxScalePolicy,
) -> Result<QuantizedMx, MxError> {
    if rows == 0 || columns == 0 {
        return Err(MxError::Empty);
    }
    if columns % MX_BLOCK_SIZE != 0 {
        return Err(MxError::ColumnsNotBlockAligned { columns });
    }
    let elements = rows.checked_mul(columns).ok_or(MxError::ShapeOverflow)?;
    if input.len() != elements {
        return Err(MxError::InputLength {
            expected: elements,
            actual: input.len(),
        });
    }
    let value_bytes = match format {
        MxElementFormat::E4M3Fn => elements,
        MxElementFormat::E3M2 => elements.checked_mul(3).ok_or(MxError::ShapeOverflow)? / 4,
        MxElementFormat::E2M1 => elements.div_ceil(2),
    };
    let blocks_per_row = columns / MX_BLOCK_SIZE;
    let value_bytes_per_row = match format {
        MxElementFormat::E4M3Fn => columns,
        MxElementFormat::E3M2 => columns * 3 / 4,
        MxElementFormat::E2M1 => columns.div_ceil(2),
    };
    let mut values = vec![0_u8; value_bytes];
    let mut scales = vec![0_u8; rows * blocks_per_row];
    let available_workers = thread::available_parallelism().map_or(1, usize::from);
    let workers = if elements >= PARALLEL_QUANTIZATION_MIN_ELEMENTS {
        available_workers.min(MAX_QUANTIZATION_WORKERS).min(rows)
    } else {
        1
    };
    let rows_per_worker = rows.div_ceil(workers);
    thread::scope(|scope| {
        for ((input_rows, value_rows), scale_rows) in input
            .chunks(rows_per_worker * columns)
            .zip(values.chunks_mut(rows_per_worker * value_bytes_per_row))
            .zip(scales.chunks_mut(rows_per_worker * blocks_per_row))
        {
            scope.spawn(move || {
                quantize_mx_rows(
                    input_rows,
                    value_rows,
                    scale_rows,
                    columns,
                    blocks_per_row,
                    format,
                    scale_policy,
                );
            });
        }
    });
    Ok(QuantizedMx {
        values,
        scales,
        rows,
        columns,
        format,
    })
}

fn quantize_mx_rows(
    input: &[f32],
    values: &mut [u8],
    scales: &mut [u8],
    columns: usize,
    blocks_per_row: usize,
    format: MxElementFormat,
    scale_policy: MxScalePolicy,
) {
    for row in 0..input.len() / columns {
        for block in 0..blocks_per_row {
            let start = row * columns + block * MX_BLOCK_SIZE;
            let source = &input[start..start + MX_BLOCK_SIZE];
            let has_nan = source.iter().any(|value| value.is_nan());
            let maximum = source
                .iter()
                .filter(|value| !value.is_nan())
                .fold(0.0_f32, |current, value| current.max(value.abs()));
            let element_power = match format {
                MxElementFormat::E4M3Fn => 8,
                MxElementFormat::E3M2 => 4,
                MxElementFormat::E2M1 => 2,
            };
            let scale_bits = if has_nan {
                255
            } else if maximum == 0.0 || maximum.is_infinite() {
                127
            } else {
                match scale_policy {
                    MxScalePolicy::Clipped => {
                        e8m0_bits(floor_log2(maximum).saturating_sub(element_power))
                    }
                    MxScalePolicy::NoClipping => {
                        // Start at floor(log2(max)) - element_power, then move
                        // up one E8M0 step while the exact ratio still exceeds
                        // the format's largest finite magnitude.  This is
                        // ceil(log2(max / max_finite)) without a lossy
                        // logarithm at the boundary.
                        let mut exponent = floor_log2(maximum).saturating_sub(element_power);
                        let bits = e8m0_bits(exponent);
                        if maximum / decode_e8m0(bits) > format_max_magnitude(format) {
                            exponent = exponent.saturating_add(1);
                        }
                        e8m0_bits(exponent)
                    }
                    MxScalePolicy::BestOfTwo => {
                        let exponent = floor_log2(maximum).saturating_sub(element_power);
                        let low = e8m0_bits(exponent);
                        let high = e8m0_bits(exponent.saturating_add(1));
                        // Ties keep the lower scale so the result is
                        // deterministic and stays close to the old rule.
                        if block_squared_error(source, decode_e8m0(high), format)
                            < block_squared_error(source, decode_e8m0(low), format)
                        {
                            high
                        } else {
                            low
                        }
                    }
                    MxScalePolicy::EvenMxfp4 => {
                        // The HIP codec rounds the exponent up once the block
                        // maximum reaches 1.75 * 2^floor(log2(max)).
                        let floor_exponent = floor_log2(maximum);
                        let mut exponent = floor_exponent.saturating_sub(element_power);
                        let unit = decode_e8m0(e8m0_bits(floor_exponent));
                        if maximum >= 1.75 * unit {
                            exponent = exponent.saturating_add(1);
                        }
                        e8m0_bits(exponent)
                    }
                }
            };
            scales[row * blocks_per_row + block] = scale_bits;
            let scale = decode_e8m0(scale_bits);
            match format {
                MxElementFormat::E4M3Fn => {
                    for (lane, source) in source.iter().enumerate() {
                        values[start + lane] = if scale.is_nan() {
                            0
                        } else {
                            encode_e4m3fn_rne(*source / scale)
                        };
                    }
                }
                MxElementFormat::E3M2 => {
                    let destination = (row * columns + block * MX_BLOCK_SIZE) * 3 / 4;
                    for group in 0..8 {
                        let mut packed = 0_u32;
                        for lane in 0..4 {
                            let code = if scale.is_nan() {
                                0
                            } else {
                                encode_e3m2(source[group * 4 + lane] / scale)
                            };
                            packed |= u32::from(code) << (lane * 6);
                        }
                        let bytes = packed.to_le_bytes();
                        values[destination + group * 3..destination + group * 3 + 3]
                            .copy_from_slice(&bytes[..3]);
                    }
                }
                MxElementFormat::E2M1 => {
                    let destination = start / 2;
                    for pair in 0..MX_BLOCK_SIZE / 2 {
                        let low = if scale.is_nan() {
                            0
                        } else {
                            encode_e2m1(source[pair * 2] / scale)
                        };
                        let high = if scale.is_nan() {
                            0
                        } else {
                            encode_e2m1(source[pair * 2 + 1] / scale)
                        };
                        values[destination + pair] = low | (high << 4);
                    }
                }
            }
        }
    }
}

const fn format_max_magnitude(format: MxElementFormat) -> f32 {
    match format {
        MxElementFormat::E4M3Fn => 448.0,
        MxElementFormat::E3M2 => 28.0,
        MxElementFormat::E2M1 => 6.0,
    }
}

fn e8m0_bits(exponent: i32) -> u8 {
    (exponent.clamp(-127, 127) + 127) as u8
}

/// Sum of squared error of one block quantized with `scale`, used to pick
/// between two candidate E8M0 scales.  The encode/decode pair matches the
/// writer so the chosen scale is the one the payload actually realizes.
fn block_squared_error(source: &[f32], scale: f32, format: MxElementFormat) -> f64 {
    if scale.is_nan() {
        return f64::INFINITY;
    }
    source
        .iter()
        .map(|value| {
            let decoded = match format {
                MxElementFormat::E4M3Fn => decode_e4m3fn(encode_e4m3fn_rne(*value / scale)) * scale,
                MxElementFormat::E3M2 => decode_e3m2(encode_e3m2(*value / scale)) * scale,
                MxElementFormat::E2M1 => decode_e2m1(encode_e2m1(*value / scale)) * scale,
            };
            let delta = f64::from(decoded) - f64::from(*value);
            delta * delta
        })
        .sum()
}

fn encode_e4m3fn_rne(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    if !magnitude.is_finite() || magnitude >= 448.0 {
        return sign | 0x7e;
    }
    let code = if magnitude < 2.0_f32.powi(-6) {
        (magnitude * 512.0).round_ties_even() as u8
    } else {
        let exponent = floor_log2(magnitude);
        let quantum = power_of_two(exponent - 3);
        let significand = (magnitude / quantum).round_ties_even() as i32;
        (exponent * 8 + 48 + significand).clamp(0, 0x7e) as u8
    };
    sign | code
}

fn encode_e3m2_positive_rne(magnitude: f32) -> u8 {
    if magnitude < 0.25 {
        return ((magnitude * 16.0).round_ties_even() as u8).min(4);
    }
    let exponent = floor_log2(magnitude);
    let quantum = power_of_two(exponent - 2);
    let significand = (magnitude / quantum).round_ties_even() as i32;
    (exponent * 4 + 8 + significand).clamp(0, 0x1f) as u8
}

fn power_of_two(exponent: i32) -> f32 {
    debug_assert!((-126..=127).contains(&exponent));
    f32::from_bits(((exponent + 127) as u32) << 23)
}

fn floor_log2(value: f32) -> i32 {
    let bits = value.to_bits() & 0x7fff_ffff;
    let exponent = ((bits >> 23) & 0xff) as i32;
    if exponent != 0 {
        exponent - 127
    } else {
        let mantissa = bits & 0x7f_ffff;
        (31 - mantissa.leading_zeros()) as i32 - 149
    }
}

fn decode_mx(
    values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
    format: MxElementFormat,
) -> Result<Vec<f32>, MxError> {
    if rows == 0 || columns == 0 {
        return Err(MxError::Empty);
    }
    let elements = rows.checked_mul(columns).ok_or(MxError::ShapeOverflow)?;
    let expected_values = match format {
        MxElementFormat::E2M1 => elements.div_ceil(2),
        MxElementFormat::E3M2 => rows
            .checked_mul(
                columns
                    .checked_mul(6)
                    .ok_or(MxError::ShapeOverflow)?
                    .div_ceil(8),
            )
            .ok_or(MxError::ShapeOverflow)?,
        MxElementFormat::E4M3Fn => elements,
    };
    if values.len() != expected_values {
        return Err(MxError::ValueLength {
            expected: expected_values,
            actual: values.len(),
        });
    }
    let blocks_per_row = columns.div_ceil(MX_BLOCK_SIZE);
    let expected_scales = rows
        .checked_mul(blocks_per_row)
        .ok_or(MxError::ShapeOverflow)?;
    if scales.len() != expected_scales {
        return Err(MxError::ScaleLength {
            expected: expected_scales,
            actual: scales.len(),
        });
    }
    let mut output = Vec::with_capacity(elements);
    for index in 0..elements {
        let row = index / columns;
        let column = index % columns;
        let block = row * blocks_per_row + column / MX_BLOCK_SIZE;
        let scale = decode_e8m0(scales[block]);
        if scale.is_nan() {
            output.push(f32::NAN);
            continue;
        }
        let element = match format {
            MxElementFormat::E2M1 => {
                let packed = values[index / 2];
                decode_e2m1(if index & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                })
            }
            MxElementFormat::E3M2 => {
                let packed_row_bytes = (columns * 6).div_ceil(8);
                let bit = column * 6;
                let byte = row * packed_row_bytes + bit / 8;
                let shift = bit % 8;
                let mut packed = u32::from(values[byte]);
                if byte + 1 < (row + 1) * packed_row_bytes {
                    packed |= u32::from(values[byte + 1]) << 8;
                }
                decode_e3m2(((packed >> shift) & 0x3f) as u8)
            }
            MxElementFormat::E4M3Fn => decode_e4m3fn(values[index]),
        };
        output.push(element * scale);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_e8m0_codes_have_the_ocp_special_boundary() {
        assert_eq!(decode_e8m0(0), 2.0_f32.powi(-127));
        for code in 1_u8..=254 {
            assert_eq!(decode_e8m0(code), 2.0_f32.powi(i32::from(code) - 127));
        }
        assert!(decode_e8m0(255).is_nan());
    }

    #[test]
    fn block32_and_odd_tail_are_distinct_from_nvfp4() {
        for columns in [31_usize, 32, 33] {
            let elements = 3 * columns;
            let values = vec![0x22; elements.div_ceil(2)];
            let scales = vec![127; 3 * columns.div_ceil(32)];
            let decoded = decode_mxfp4(&values, &scales, 3, columns).unwrap();
            assert_eq!(decoded, vec![1.0; elements]);
        }
    }

    #[test]
    fn nan_scale_propagates_to_the_whole_block_and_wrong_lengths_fail_closed() {
        assert!(
            decode_mxfp4(&[0; 16], &[255], 1, 32)
                .unwrap()
                .iter()
                .all(|value| value.is_nan())
        );
        assert!(matches!(
            decode_mxfp8(&[0; 31], &[127], 1, 32),
            Err(MxError::ValueLength { .. })
        ));
    }

    #[test]
    fn e3m2_boundaries_and_rne_match_ocp_table() {
        assert_eq!(decode_e3m2(0x01), 0.0625);
        assert_eq!(decode_e3m2(0x03), 0.1875);
        assert_eq!(decode_e3m2(0x04), 0.25);
        assert_eq!(decode_e3m2(0x1f), 28.0);
        assert_eq!(decode_e3m2(0x3f), -28.0);
        assert_eq!(encode_e3m2(0.03125), 0x00);
        assert_eq!(encode_e3m2(0.09375), 0x02);
        assert_eq!(encode_e3m2(100.0), 0x1f);
    }

    #[test]
    fn direct_rne_encoders_match_exhaustive_oracles_for_every_bf16_value() {
        fn exhaustive_e3m2(value: f32) -> u8 {
            let sign = if value.is_sign_negative() { 0x20 } else { 0 };
            let magnitude = value.abs();
            if magnitude == 0.0 {
                return sign;
            }
            if !magnitude.is_finite() || magnitude >= 28.0 {
                return sign | 0x1f;
            }
            let mut best = 0_u8;
            let mut best_error = f32::INFINITY;
            for candidate in 0_u8..=0x1f {
                let error = (decode_e3m2(candidate) - magnitude).abs();
                if error < best_error
                    || (error == best_error && candidate & 1 == 0 && best & 1 != 0)
                {
                    best = candidate;
                    best_error = error;
                }
            }
            sign | best
        }

        for bits in 0_u16..=u16::MAX {
            let value = f32::from_bits(u32::from(bits) << 16);
            assert_eq!(
                encode_e4m3fn_rne(value),
                crate::encode_e4m3fn(value),
                "E4M3 mismatch for BF16 bits 0x{bits:04x}"
            );
            assert_eq!(
                encode_e3m2(value),
                exhaustive_e3m2(value),
                "E3M2 mismatch for BF16 bits 0x{bits:04x}"
            );
        }
    }

    #[test]
    fn mxfp8_and_mxfp6_use_ocp_floor_power_scale_and_round_trip() {
        let mut input = vec![0.0_f32; 64];
        input[0] = 511.0;
        input[32] = 31.0;
        let fp8 = quantize_mxfp8_e4m3(&input, 1, 64).unwrap();
        assert_eq!(fp8.scales(), &[127, 123]);
        assert_eq!(fp8.values()[0], 0x7e);
        assert_eq!(fp8.dequantize().unwrap()[0], 448.0);

        let fp6 = quantize_mxfp6_e3m2(&input, 1, 64).unwrap();
        assert_eq!(fp6.scales(), &[131, 127]);
        let decoded = fp6.dequantize().unwrap();
        assert_eq!(decoded[0], 448.0);
        assert_eq!(decoded[32], 28.0);
    }

    #[test]
    fn mxfp8_no_clipping_scale_uses_smallest_scale_at_448_boundary() {
        let mut input = vec![0.0_f32; 128];
        input[0] = 448.0;
        input[32] = 449.0;
        input[64] = 896.0;
        input[96] = 0.5;
        let fp8 = quantize_mxfp8_e4m3_no_clipping_scale(&input, 1, 128).unwrap();

        // 448 itself is representable with scale 1.  The first value above
        // that boundary requires scale 2, as does 896.  For 0.5 the smallest
        // admissible scale is 2^-9 (E8M0 exponent field 118).
        assert_eq!(fp8.scales(), &[127, 128, 128, 118]);
        let decoded = fp8.dequantize().unwrap();
        assert_eq!(decoded[0], 448.0);
        assert!(decoded[32] <= 448.0 * 2.0);
        assert_eq!(decoded[64], 896.0);
        assert_eq!(decoded[96], 0.5);

        let mut half_scale_boundary = vec![0.0_f32; 64];
        half_scale_boundary[0] = 224.0;
        half_scale_boundary[32] = 225.0;
        let boundary = quantize_mxfp8_e4m3_no_clipping_scale(&half_scale_boundary, 1, 64).unwrap();
        assert_eq!(boundary.scales(), &[126, 127]);
    }

    #[test]
    fn mxfp8_no_clipping_scale_preserves_nonfinite_policy() {
        let nan = quantize_mxfp8_e4m3_no_clipping_scale(&[f32::NAN; 32], 1, 32).unwrap();
        assert_eq!(nan.scales(), &[255]);
        assert!(nan.dequantize().unwrap().iter().all(|value| value.is_nan()));

        let infinity = quantize_mxfp8_e4m3_no_clipping_scale(&[f32::INFINITY; 32], 1, 32).unwrap();
        assert_eq!(infinity.scales(), &[127]);
        assert!(
            infinity
                .dequantize()
                .unwrap()
                .iter()
                .all(|value| *value == 448.0)
        );

        let zeros = quantize_mxfp8_e4m3_no_clipping_scale(&[0.0; 32], 1, 32).unwrap();
        assert_eq!(zeros.scales(), &[127]);
        assert!(
            zeros
                .dequantize()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn mxfp8_no_clipping_scale_rejects_non_aligned_columns() {
        for columns in [31, 33] {
            assert_eq!(
                quantize_mxfp8_e4m3_no_clipping_scale(&vec![0.0; columns], 1, columns),
                Err(MxError::ColumnsNotBlockAligned { columns })
            );
        }
    }

    #[test]
    fn wa_codec_preserves_ocp_nan_scale_and_saturates_infinity() {
        for quantized in [
            quantize_mxfp8_e4m3(&[f32::NAN; 32], 1, 32).unwrap(),
            quantize_mxfp6_e3m2(&[f32::NAN; 32], 1, 32).unwrap(),
        ] {
            assert_eq!(quantized.scales(), &[255]);
            assert!(
                quantized
                    .dequantize()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan())
            );
        }

        let fp8 = quantize_mxfp8_e4m3(&[f32::INFINITY; 32], 1, 32).unwrap();
        assert_eq!(fp8.scales(), &[127]);
        assert!(
            fp8.dequantize()
                .unwrap()
                .iter()
                .all(|value| *value == 448.0)
        );
        let fp6 = quantize_mxfp6_e3m2(&[f32::NEG_INFINITY; 32], 1, 32).unwrap();
        assert_eq!(fp6.scales(), &[127]);
        assert!(
            fp6.dequantize()
                .unwrap()
                .iter()
                .all(|value| *value == -28.0)
        );
    }

    #[test]
    fn wa_codec_rejects_unpadded_k_on_both_sides_of_block_boundary() {
        for columns in [31, 33] {
            assert_eq!(
                quantize_mxfp6_e3m2(&vec![0.0; columns], 1, columns),
                Err(MxError::ColumnsNotBlockAligned { columns })
            );
        }
        assert!(quantize_mxfp6_e3m2(&[0.0; 32], 1, 32).is_ok());
    }

    fn quantize_with_policy(
        format: MxElementFormat,
        policy: MxScalePolicy,
        values: &[f32],
    ) -> QuantizedMx {
        quantize_mx(values, 1, 32, format, policy).unwrap()
    }

    fn block_with_max(format: MxElementFormat, maximum: f32) -> Vec<f32> {
        // Put the maximum in one lane and deterministic small values in the
        // rest so the block maximum is exactly `maximum`.
        let mut values = Vec::with_capacity(32);
        values.push(maximum);
        for lane in 1..32 {
            values.push((lane as f32) * maximum / 64.0);
        }
        let _ = format;
        values
    }

    #[test]
    fn no_clipping_scale_is_the_smallest_block_max_representable_scale() {
        let cases = [
            (MxElementFormat::E4M3Fn, 8, 448.0_f32),
            (MxElementFormat::E3M2, 4, 28.0_f32),
        ];
        // Boundaries: exactly the format max, just above it, the 1.5/1.75
        // mantissa points, a non-power-of-two, a subnormal-range maximum and
        // ordinary values on both sides of a power of two.
        let maxima = [
            1.0_f32, 1.25, 1.4999, 1.5, 1.75, 2.0, 3.0, 5.75, 28.0, 28.0001, 29.0, 64.0, 448.0,
            448.5, 512.0, 1024.0, 0.0001, 0.3,
        ];
        for (format, element_power, format_max) in cases {
            for maximum in maxima {
                let values = block_with_max(format, maximum);
                let clipped = quantize_with_policy(format, MxScalePolicy::Clipped, &values);
                let no_clip = quantize_with_policy(format, MxScalePolicy::NoClipping, &values);
                let clipped_exponent = i32::from(clipped.scales()[0]) - 127;
                let no_clip_exponent = i32::from(no_clip.scales()[0]) - 127;
                assert!(
                    no_clip_exponent == clipped_exponent
                        || no_clip_exponent == clipped_exponent + 1,
                    "no-clip must move at most one E8M0 step: {no_clip_exponent} vs {clipped_exponent}"
                );
                let unit = 2.0_f32.powi(no_clip_exponent);
                assert!(
                    maximum / unit <= format_max,
                    "no-clip scale {} saturates max {maximum}",
                    no_clip_exponent
                );
                // Every exponent below the chosen one saturates, and the
                // chosen one is at most one step above floor(log2(max))-P.
                let expected_floor = floor_log2(maximum).saturating_sub(element_power);
                assert!(
                    (no_clip_exponent - expected_floor.max(-127)).abs() <= 1,
                    "no-clip moved more than one step: {no_clip_exponent} from {expected_floor}"
                );
                for exponent in expected_floor.max(-126)..no_clip_exponent {
                    let lower = 2.0_f32.powi(exponent);
                    assert!(
                        maximum / lower > format_max,
                        "exponent {exponent} already avoids saturation for max {maximum}"
                    );
                }
            }
        }
    }

    #[test]
    fn mxfp4_even_scale_bumps_at_the_1_75_mantissa_threshold() {
        // floor(log2(max)) = 2 for all of these, element_power 2 -> scale 1.
        for (maximum, expected_exponent) in [(4.0_f32, 0_i32), (6.0, 0), (7.0, 1), (6.999, 0)] {
            let values = block_with_max(MxElementFormat::E2M1, maximum);
            let even =
                quantize_with_policy(MxElementFormat::E2M1, MxScalePolicy::EvenMxfp4, &values);
            assert_eq!(
                i32::from(even.scales()[0]) - 127,
                expected_exponent,
                "even rule for max {maximum}"
            );
        }
        // 1.75 * 2^2 = 7.0 is the bump point; 6.999 is not.
    }

    #[test]
    fn mxfp4_best_of_two_never_exceeds_floor_or_even_block_error() {
        let mut state = 0x1234_5678_u32;
        let mut maximum = 0.25_f32;
        while maximum < 32.0 {
            let mut values = Vec::with_capacity(32);
            for _ in 0..32 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
                values.push((unit * 2.0 - 1.0) * maximum);
            }
            let floor =
                quantize_with_policy(MxElementFormat::E2M1, MxScalePolicy::Clipped, &values);
            let even =
                quantize_with_policy(MxElementFormat::E2M1, MxScalePolicy::EvenMxfp4, &values);
            let best =
                quantize_with_policy(MxElementFormat::E2M1, MxScalePolicy::BestOfTwo, &values);
            let source_error = |quantized: &QuantizedMx| -> f64 {
                let decoded = quantized.dequantize().unwrap();
                decoded
                    .iter()
                    .zip(values.iter())
                    .map(|(q, w)| {
                        let delta = f64::from(*q) - f64::from(*w);
                        delta * delta
                    })
                    .sum()
            };
            let best_error = source_error(&best);
            assert!(best_error <= source_error(&floor) + 1e-30);
            assert!(best_error <= source_error(&even) + 1e-30);
            // The chosen scale is one of the two candidate steps.
            let floor_exponent = floor_log2(maximum).saturating_sub(2);
            let chosen = i32::from(best.scales()[0]) - 127;
            assert!(chosen == floor_exponent || chosen == floor_exponent + 1);
            maximum *= 1.37;
        }
    }

    #[test]
    fn special_blocks_keep_the_historical_handling() {
        let all_zero = quantize_mxfp4_e2m1_best_of_two(&[0.0; 32], 1, 32).unwrap();
        assert_eq!(all_zero.scales(), &[127]);
        assert!(
            all_zero
                .dequantize()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );

        let mut nan_block = [0.0_f32; 32];
        nan_block[5] = f32::NAN;
        let nan = quantize_mxfp4_e2m1_best_of_two(&nan_block, 1, 32).unwrap();
        assert_eq!(nan.scales(), &[255]);
        assert!(nan.dequantize().unwrap().iter().all(|value| value.is_nan()));

        let inf = quantize_mxfp4_e2m1_best_of_two(&[f32::INFINITY; 32], 1, 32).unwrap();
        assert_eq!(inf.scales(), &[127]);

        // Subnormal and tiny blocks still resolve to a finite E8M0 code.
        let tiny = quantize_mxfp4_e2m1_best_of_two(&[f32::MIN_POSITIVE; 32], 1, 32).unwrap();
        assert_ne!(tiny.scales()[0], 255);
    }
}
