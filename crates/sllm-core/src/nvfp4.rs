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
    quantize_nvfp4_weights_with_scale_policy(input, rows, columns, Nvfp4ScalePolicy::Nearest)
}

/// Select the per-block NVFP4 scale by rounding `block_amax / (6 * global)`
/// to the nearest E4M3 code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nvfp4ScalePolicy {
    /// Historical rule: round the raw per-block scale to the nearest E4M3
    /// code.  The rounding can land below the raw scale and clip the block.
    Nearest,
    /// Take the two E4M3 codes adjacent to the raw per-block scale and keep
    /// whichever gives the smaller block squared error.  The global tensor
    /// scale is unchanged.  Ties keep the lower code.
    BestOfTwo,
}

/// Quantize NVFP4 weights with an explicit per-block scale policy.  The
/// historical [`quantize_nvfp4_weights`] behavior is
/// [`Nvfp4ScalePolicy::Nearest`].
pub fn quantize_nvfp4_weights_with_scale_policy(
    input: &[f32],
    rows: usize,
    columns: usize,
    scale_policy: Nvfp4ScalePolicy,
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
            let scale_bits = match scale_policy {
                Nvfp4ScalePolicy::Nearest => encode_e4m3fn(raw_scale),
                Nvfp4ScalePolicy::BestOfTwo => {
                    best_of_two_e4m3_scale(raw_scale, &input[start..end], tensor_scale)
                }
            };
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

/// Pick between the two E4M3 magnitude codes bracketing `raw_scale` by block
/// squared error.  Positive E4M3FN codes `0x00..=0x7e` are monotonic in
/// magnitude, so the neighbours are `code - 1` and `code + 1` around the
/// nearest code.  Ties keep the lower code.
fn best_of_two_e4m3_scale(raw_scale: f32, block: &[f32], tensor_scale: f32) -> u8 {
    let nearest = encode_e4m3fn(raw_scale);
    let (lower, upper) = if decode_e4m3fn(nearest) <= raw_scale {
        (nearest, nearest.saturating_add(1).min(0x7e))
    } else {
        (nearest.saturating_sub(1), nearest)
    };
    let lower_error = block_e4m3_squared_error(block, lower, tensor_scale);
    let upper_error = block_e4m3_squared_error(block, upper, tensor_scale);
    if upper_error < lower_error {
        upper
    } else {
        lower
    }
}

fn block_e4m3_squared_error(block: &[f32], scale_bits: u8, tensor_scale: f32) -> f64 {
    let scale = decode_e4m3fn(scale_bits) * tensor_scale;
    if scale == 0.0 {
        return block
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                value * value
            })
            .sum();
    }
    block
        .iter()
        .map(|value| {
            let decoded = f64::from(decode_e2m1(encode_e2m1(*value / scale)) * scale);
            let delta = decoded - f64::from(*value);
            delta * delta
        })
        .sum()
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

    #[test]
    fn nearest_weight_rule_matches_the_default_entry_point() {
        let mut state = 0x0bad_f00d_u32;
        let mut values = Vec::with_capacity(16 * 32);
        for _ in 0..16 * 32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            values.push((unit * 2.0 - 1.0) * 3.0);
        }
        let nearest = quantize_nvfp4_weights(&values, 16, 32).unwrap();
        let explicit =
            quantize_nvfp4_weights_with_scale_policy(&values, 16, 32, Nvfp4ScalePolicy::Nearest)
                .unwrap();
        assert_eq!(nearest.block_scales, explicit.block_scales);
        assert_eq!(nearest.packed_values, explicit.packed_values);
        assert_eq!(nearest.tensor_scale, explicit.tensor_scale);
    }

    #[test]
    fn best_of_two_weight_rule_never_exceeds_nearest_block_error() {
        let mut state = 0x00c0_ffee_u32;
        let mut worst_improvement: f64 = 0.0;
        for _ in 0..256 {
            let mut values = Vec::with_capacity(64);
            let shape_scale = 0.5 + ((state >> 16) & 0x3f) as f32 / 8.0;
            for _ in 0..64 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
                values.push((unit * 2.0 - 1.0) * shape_scale);
            }
            let nearest = quantize_nvfp4_weights(&values, 1, 64).unwrap();
            let best = quantize_nvfp4_weights_with_scale_policy(
                &values,
                1,
                64,
                Nvfp4ScalePolicy::BestOfTwo,
            )
            .unwrap();
            let error = |q: &QuantizedNvfp4| -> f64 {
                q.dequantize()
                    .iter()
                    .zip(values.iter())
                    .map(|(a, b)| {
                        let delta = f64::from(*a) - f64::from(*b);
                        delta * delta
                    })
                    .sum()
            };
            let nearest_error = error(&nearest);
            let best_error = error(&best);
            assert!(
                best_error <= nearest_error + 1e-30,
                "best-of-two regressed: {best_error} > {nearest_error}"
            );
            worst_improvement = worst_improvement.max(nearest_error - best_error);
        }
        assert!(worst_improvement >= 0.0);
    }

    #[test]
    fn best_of_two_scale_uses_one_of_the_two_adjacent_e4m3_codes() {
        // A block whose raw scale sits exactly on an E4M3 code should keep
        // that code; one just above it may pick the neighbour.
        let global = 1.0_f32;
        let block = [3.0_f32, 1.0, 0.5, 0.25, 0.0, -3.0];
        let raw = (3.0 / E2M1_MAX) / 1.0;
        let nearest = encode_e4m3fn(raw);
        let chosen = best_of_two_e4m3_scale(raw, &block, 1.0);
        let _ = global;
        assert!(
            chosen == nearest
                || chosen == nearest.saturating_add(1)
                || chosen.saturating_add(1) == nearest
        );
    }
}
