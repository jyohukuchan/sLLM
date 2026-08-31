//! OCP Microscaling (MX) numeric contracts.
//!
//! MXFP4 is E2M1 values in blocks of 32 with one E8M0 scale. It is not
//! NVIDIA NVFP4, whose block size and hierarchical scale formats differ.

use crate::{decode_e2m1, decode_e4m3fn};
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
    quantize_mx(input, rows, columns, MxElementFormat::E4M3Fn)
}

pub fn quantize_mxfp6_e3m2(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MxError> {
    quantize_mx(input, rows, columns, MxElementFormat::E3M2)
}

fn quantize_mx(
    input: &[f32],
    rows: usize,
    columns: usize,
    format: MxElementFormat,
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
        MxElementFormat::E2M1 => unreachable!("MXFP4 encoding is artifact-recipe specific"),
    };
    let blocks_per_row = columns / MX_BLOCK_SIZE;
    let value_bytes_per_row = match format {
        MxElementFormat::E4M3Fn => columns,
        MxElementFormat::E3M2 => columns * 3 / 4,
        MxElementFormat::E2M1 => unreachable!(),
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
                MxElementFormat::E2M1 => unreachable!(),
            };
            let scale_bits = if has_nan {
                255
            } else if maximum == 0.0 || maximum.is_infinite() {
                127
            } else {
                (floor_log2(maximum)
                    .saturating_sub(element_power)
                    .clamp(-127, 127)
                    + 127) as u8
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
                MxElementFormat::E2M1 => unreachable!(),
            }
        }
    }
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
}
