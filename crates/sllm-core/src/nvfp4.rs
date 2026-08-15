//! NVIDIA NVFP4 weight numeric and provider contracts.
//!
//! The v1 inference encoding is E2M1 values with one OCP E4M3FN scale per
//! 16 consecutive K-axis values and one FP32 tensor scale. It is distinct
//! from MXFP4 and from Transformer Engine's training-only 2D recipe.

use std::fmt;

use crate::{decode_e4m3fn, encode_e4m3fn};

pub const NVFP4_BLOCK_SIZE: usize = 16;
pub const E2M1_MAX: f32 = 6.0;
pub const NVFP4_E4M3_MAX: f32 = 448.0;

const E2M1_POSITIVE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub fn decode_e2m1(bits: u8) -> f32 {
    let magnitude = E2M1_POSITIVE[usize::from(bits & 0x07)];
    if bits & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

pub fn encode_e2m1(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x08 } else { 0 };
    let magnitude = value.abs().min(E2M1_MAX);
    let mut best = 0_u8;
    let mut best_error = f32::INFINITY;
    for candidate in 0_u8..8 {
        let error = (E2M1_POSITIVE[usize::from(candidate)] - magnitude).abs();
        if error < best_error || (error == best_error && candidate & 1 == 0 && best & 1 != 0) {
            best = candidate;
            best_error = error;
        }
    }
    sign | best
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedNvfp4 {
    pub packed_values: Vec<u8>,
    pub block_scales: Vec<u8>,
    pub tensor_scale: f32,
    pub rows: usize,
    pub columns: usize,
}

impl QuantizedNvfp4 {
    pub fn blocks_per_row(&self) -> usize {
        self.columns.div_ceil(NVFP4_BLOCK_SIZE)
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let blocks_per_row = self.blocks_per_row();
        (0..self.rows * self.columns)
            .map(|index| {
                let row = index / self.columns;
                let column = index % self.columns;
                let byte = self.packed_values[index / 2];
                let code = if index & 1 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let scale = decode_e4m3fn(
                    self.block_scales[row * blocks_per_row + column / NVFP4_BLOCK_SIZE],
                );
                decode_e2m1(code) * scale * self.tensor_scale
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Nvfp4Error {
    EmptyMatrix,
    ShapeOverflow,
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteInput { index: usize },
    UnsupportedProvider,
}

impl fmt::Display for Nvfp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => formatter.write_str("NVFP4 matrix dimensions must be non-zero"),
            Self::ShapeOverflow => formatter.write_str("NVFP4 matrix shape overflowed usize"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "NVFP4 matrix length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonFiniteInput { index } => {
                write!(formatter, "NVFP4 source is non-finite at element {index}")
            }
            Self::UnsupportedProvider => formatter.write_str("NVFP4 provider is unsupported"),
        }
    }
}

impl std::error::Error for Nvfp4Error {}

pub fn quantize_nvfp4_weights(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedNvfp4, Nvfp4Error> {
    if rows == 0 || columns == 0 {
        return Err(Nvfp4Error::EmptyMatrix);
    }
    let expected = rows.checked_mul(columns).ok_or(Nvfp4Error::ShapeOverflow)?;
    if input.len() != expected {
        return Err(Nvfp4Error::LengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    if let Some((index, _)) = input
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Nvfp4Error::NonFiniteInput { index });
    }
    let global_amax = input
        .iter()
        .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
    let tensor_scale = if global_amax == 0.0 {
        1.0
    } else {
        global_amax / (NVFP4_E4M3_MAX * E2M1_MAX)
    };
    let blocks_per_row = columns.div_ceil(NVFP4_BLOCK_SIZE);
    let mut packed_values = vec![0_u8; expected.div_ceil(2)];
    let mut block_scales = Vec::with_capacity(rows * blocks_per_row);
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let start_column = block * NVFP4_BLOCK_SIZE;
            let end_column = (start_column + NVFP4_BLOCK_SIZE).min(columns);
            let start = row * columns + start_column;
            let end = row * columns + end_column;
            let block_amax = input[start..end]
                .iter()
                .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
            // Transformer Engine v2.18 computes the decode scale directly
            // from block_amax. A zero block therefore has a zero E4M3 scale;
            // a positive scale below E4M3's range may also round to zero and
            // canonically collapses that block to zero.
            let raw_scale = (block_amax / E2M1_MAX) / tensor_scale;
            let scale_bits = encode_e4m3fn(raw_scale);
            let decoded_scale = decode_e4m3fn(scale_bits);
            block_scales.push(scale_bits);
            for (offset, source) in input[start..end].iter().enumerate() {
                let index = start + offset;
                let code = if decoded_scale == 0.0 {
                    0
                } else {
                    encode_e2m1(*source / (decoded_scale * tensor_scale))
                };
                if index & 1 == 0 {
                    packed_values[index / 2] = code;
                } else {
                    packed_values[index / 2] |= code << 4;
                }
            }
        }
    }
    Ok(QuantizedNvfp4 {
        packed_values,
        block_scales,
        tensor_scale,
        rows,
        columns,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nvfp4Provider {
    Gfx1201PackedDequant,
    Gfx1030PackedDequant,
    ConvertedBf16,
}

pub fn select_nvfp4_provider(
    exact_gcn_arch: &str,
    packed_dequant_enabled: bool,
    converted_bf16_enabled: bool,
) -> Result<Nvfp4Provider, Nvfp4Error> {
    if packed_dequant_enabled {
        return match exact_gcn_arch {
            "gfx1201" => Ok(Nvfp4Provider::Gfx1201PackedDequant),
            "gfx1030" => Ok(Nvfp4Provider::Gfx1030PackedDequant),
            _ => Err(Nvfp4Error::UnsupportedProvider),
        };
    }
    if converted_bf16_enabled && matches!(exact_gcn_arch, "gfx1201" | "gfx1030") {
        return Ok(Nvfp4Provider::ConvertedBf16);
    }
    Err(Nvfp4Error::UnsupportedProvider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_e2m1_code_points_and_ties_are_exact() {
        let expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (bits, value) in expected.into_iter().enumerate() {
            assert_eq!(decode_e2m1(bits as u8), value);
            assert_eq!(decode_e2m1(bits as u8 | 8), -value);
            assert_eq!(encode_e2m1(value), bits as u8);
        }
        assert_eq!(encode_e2m1(0.25), 0);
        assert_eq!(encode_e2m1(0.75), 2);
        assert_eq!(encode_e2m1(5.0), 6);
    }

    #[test]
    fn block_boundaries_and_odd_tail_round_trip() {
        for columns in [15, 16, 17, 31, 32, 33] {
            let source = (0..2 * columns)
                .map(|index| (index as f32 - columns as f32) / 7.0)
                .collect::<Vec<_>>();
            let quantized = quantize_nvfp4_weights(&source, 2, columns).unwrap();
            assert_eq!(quantized.packed_values.len(), source.len().div_ceil(2));
            assert_eq!(quantized.block_scales.len(), 2 * columns.div_ceil(16));
            assert_eq!(quantized.dequantize().len(), source.len());
            if source.len() & 1 != 0 {
                assert_eq!(quantized.packed_values.last().unwrap() & 0xf0, 0);
            }
        }
    }

    #[test]
    fn all_zero_is_canonical_and_nonfinite_rejected() {
        let zero = quantize_nvfp4_weights(&[0.0; 17], 1, 17).unwrap();
        assert_eq!(zero.tensor_scale, 1.0);
        assert_eq!(zero.block_scales, vec![encode_e4m3fn(0.0); 2]);
        assert_eq!(zero.dequantize(), vec![0.0; 17]);
        assert_eq!(
            quantize_nvfp4_weights(&[0.0, f32::INFINITY], 1, 2),
            Err(Nvfp4Error::NonFiniteInput { index: 1 })
        );
    }

    #[test]
    fn underflowed_block_scale_canonically_collapses_to_zero() {
        let mut source = vec![f32::from_bits(0x0080_0000); 32];
        source[0] = 1.0;
        let quantized = quantize_nvfp4_weights(&source, 1, 32).unwrap();
        assert_ne!(quantized.block_scales[0], 0);
        assert_eq!(quantized.block_scales[1], 0);
        assert!(
            quantized.dequantize()[16..]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
}
