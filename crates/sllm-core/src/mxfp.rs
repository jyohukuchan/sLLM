//! OCP Microscaling (MX) numeric contracts.
//!
//! MXFP4 is E2M1 values in blocks of 32 with one E8M0 scale. It is not
//! NVIDIA NVFP4, whose block size and hierarchical scale formats differ.

use crate::{decode_e2m1, decode_e4m3fn};
use std::fmt;

pub const MX_BLOCK_SIZE: usize = 32;

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
    E4M3Fn,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MxError {
    Empty,
    ShapeOverflow,
    ValueLength { expected: usize, actual: usize },
    ScaleLength { expected: usize, actual: usize },
    NonFiniteScale { block: usize },
}

impl fmt::Display for MxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("MX tensor dimensions must be non-zero"),
            Self::ShapeOverflow => formatter.write_str("MX tensor shape overflowed usize"),
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
            Self::NonFiniteScale { block } => write!(formatter, "MX block {block} has NaN scale"),
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
        if !scale.is_finite() {
            return Err(MxError::NonFiniteScale { block });
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
    fn nan_scale_and_wrong_lengths_fail_closed() {
        assert_eq!(
            decode_mxfp4(&[0; 16], &[255], 1, 32),
            Err(MxError::NonFiniteScale { block: 0 })
        );
        assert!(matches!(
            decode_mxfp8(&[0; 31], &[127], 1, 32),
            Err(MxError::ValueLength { .. })
        ));
    }
}
