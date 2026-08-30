//! Host-independent reference codec for KV FP8 block16 state.
//!
//! Values are grouped along the innermost head-dimension axis. Every block
//! owns one E8M0 power-of-two scale and sixteen value bytes; partial tails are
//! zero padded so host and device state images have one canonical layout.

use std::fmt;

use crate::{
    KV_FP8_BLOCK_SIZE, KvFp8PhysicalVariant, decode_e4m3fn, decode_e4m3fnuz, decode_e8m0,
    encode_e4m3fn, encode_e4m3fnuz,
};

/// Largest finite OCP E5M2 magnitude (`0x7b`).
pub const E5M2_MAX: f32 = 57_344.0;

/// Decode one OCP E5M2 byte to FP32. Infinity and NaN encodings retain their
/// IEEE classifications; the block16 encoder itself never emits infinity.
pub fn decode_e5m2(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 2) & 0x1f;
    let mantissa = bits & 0x03;
    match (exponent, mantissa) {
        (0, 0) => {
            if bits & 0x80 == 0 {
                0.0
            } else {
                -0.0
            }
        }
        (0, _) => sign * f32::from(mantissa) * 2.0_f32.powi(-16),
        (0x1f, 0) => sign * f32::INFINITY,
        (0x1f, _) => f32::NAN,
        _ => sign * (1.0 + f32::from(mantissa) / 4.0) * 2.0_f32.powi(i32::from(exponent) - 15),
    }
}

/// Encode FP32 as OCP E5M2 with round-to-nearest-even. NaNs use canonical
/// positive `0x7f`; infinities and finite overflow saturate to signed max.
pub fn encode_e5m2(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    if !magnitude.is_finite() || magnitude >= E5M2_MAX {
        return sign | 0x7b;
    }

    // A small exhaustive reference table makes ties explicit and independent
    // of host conversion instructions or the current floating-point mode.
    let mut best = 0_u8;
    let mut best_error = f32::INFINITY;
    for candidate in 0_u8..=0x7b {
        let decoded = decode_e5m2(candidate);
        let error = (decoded - magnitude).abs();
        if error < best_error || (error == best_error && candidate & 1 == 0 && best & 1 != 0) {
            best = candidate;
            best_error = error;
        }
    }
    sign | best
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvFp8CodecError {
    Empty,
    ShapeOverflow,
    LengthMismatch { expected: usize, actual: usize },
    ScaleLengthMismatch { expected: usize, actual: usize },
    ValueLengthMismatch { expected: usize, actual: usize },
    NonFiniteScale { block: usize },
    NonCanonicalPadding { row: usize, column: usize },
}

impl fmt::Display for KvFp8CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("KV FP8 block16 dimensions must be non-zero"),
            Self::ShapeOverflow => formatter.write_str("KV FP8 block16 shape overflowed usize"),
            Self::LengthMismatch { expected, actual } => write!(
                formatter,
                "KV FP8 block16 input length mismatch: expected {expected}, got {actual}"
            ),
            Self::ScaleLengthMismatch { expected, actual } => write!(
                formatter,
                "KV FP8 block16 scale length mismatch: expected {expected}, got {actual}"
            ),
            Self::ValueLengthMismatch { expected, actual } => write!(
                formatter,
                "KV FP8 block16 value length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteScale { block } => {
                write!(formatter, "KV FP8 block16 block {block} has NaN E8M0 scale")
            }
            Self::NonCanonicalPadding { row, column } => write!(
                formatter,
                "KV FP8 block16 row {row} has nonzero padding at column {column}"
            ),
        }
    }
}

impl std::error::Error for KvFp8CodecError {}

/// Canonical encoded rows. `values` contains exactly sixteen bytes per scale,
/// including zero padding in the final block of a non-aligned row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedKvFp8Block16 {
    values: Vec<u8>,
    scales: Vec<u8>,
    rows: usize,
    columns: usize,
    physical_variant: KvFp8PhysicalVariant,
}

/// Closed scale-selector set used only by the Phase 54 research oracle.
///
/// Production block16 remains [`KvFp8ResearchScaleRecipe::Floor`]. The other
/// selectors deliberately do not define a public KV-state descriptor.
#[cfg(feature = "phase54-research")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvFp8ResearchScaleRecipe {
    Floor,
    Ceil,
    NearestEvenExponent,
    /// Reproduce standard MXFP8 block-32 scaling exactly by deriving one
    /// scale from each consecutive 32 logical lanes and storing that same
    /// scale in both child block-16 descriptors.
    Parent32Duplicate,
}

/// Per-block observability for Phase 54 attribution. Counts cover logical
/// lanes only; canonical storage padding is excluded.
#[cfg(feature = "phase54-research")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvFp8ResearchBlockStats {
    pub amax: f32,
    pub scale_bits: u8,
    pub scale_exponent: i16,
    pub finite_count: usize,
    pub encoded_zero_count: usize,
    pub underflow_to_zero_count: usize,
    pub saturation_count: usize,
}

impl QuantizedKvFp8Block16 {
    pub fn from_parts(
        values: Vec<u8>,
        scales: Vec<u8>,
        rows: usize,
        columns: usize,
        physical_variant: KvFp8PhysicalVariant,
    ) -> Result<Self, KvFp8CodecError> {
        if rows == 0 || columns == 0 {
            return Err(KvFp8CodecError::Empty);
        }
        let blocks_per_row = columns.div_ceil(KV_FP8_BLOCK_SIZE);
        let expected_scales = rows
            .checked_mul(blocks_per_row)
            .ok_or(KvFp8CodecError::ShapeOverflow)?;
        let expected_values = expected_scales
            .checked_mul(KV_FP8_BLOCK_SIZE)
            .ok_or(KvFp8CodecError::ShapeOverflow)?;
        if scales.len() != expected_scales {
            return Err(KvFp8CodecError::ScaleLengthMismatch {
                expected: expected_scales,
                actual: scales.len(),
            });
        }
        if values.len() != expected_values {
            return Err(KvFp8CodecError::ValueLengthMismatch {
                expected: expected_values,
                actual: values.len(),
            });
        }
        let tail = columns % KV_FP8_BLOCK_SIZE;
        if tail != 0 {
            for row in 0..rows {
                let base = (row * blocks_per_row + blocks_per_row - 1) * KV_FP8_BLOCK_SIZE;
                if let Some(offset) = values[base + tail..base + KV_FP8_BLOCK_SIZE]
                    .iter()
                    .position(|byte| *byte != 0)
                {
                    return Err(KvFp8CodecError::NonCanonicalPadding {
                        row,
                        column: columns + offset,
                    });
                }
            }
        }
        Ok(Self {
            values,
            scales,
            rows,
            columns,
            physical_variant,
        })
    }

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

    pub const fn physical_variant(&self) -> KvFp8PhysicalVariant {
        self.physical_variant
    }

    pub const fn blocks_per_row(&self) -> usize {
        self.columns.div_ceil(KV_FP8_BLOCK_SIZE)
    }

    pub const fn padded_columns(&self) -> usize {
        self.blocks_per_row() * KV_FP8_BLOCK_SIZE
    }

    /// Decode only logical lanes. Canonical storage padding is never exposed
    /// as an attention input.
    pub fn dequantize(&self) -> Result<Vec<f32>, KvFp8CodecError> {
        let blocks_per_row = self.blocks_per_row();
        let expected_scales = self
            .rows
            .checked_mul(blocks_per_row)
            .ok_or(KvFp8CodecError::ShapeOverflow)?;
        let expected_values = expected_scales
            .checked_mul(KV_FP8_BLOCK_SIZE)
            .ok_or(KvFp8CodecError::ShapeOverflow)?;
        if self.scales.len() != expected_scales {
            return Err(KvFp8CodecError::ScaleLengthMismatch {
                expected: expected_scales,
                actual: self.scales.len(),
            });
        }
        if self.values.len() != expected_values {
            return Err(KvFp8CodecError::ValueLengthMismatch {
                expected: expected_values,
                actual: self.values.len(),
            });
        }
        let logical_len = self
            .rows
            .checked_mul(self.columns)
            .ok_or(KvFp8CodecError::ShapeOverflow)?;
        let mut output = Vec::with_capacity(logical_len);
        for row in 0..self.rows {
            for column in 0..self.columns {
                let block_in_row = column / KV_FP8_BLOCK_SIZE;
                let block = row * blocks_per_row + block_in_row;
                let scale = decode_e8m0(self.scales[block]);
                if !scale.is_finite() {
                    return Err(KvFp8CodecError::NonFiniteScale { block });
                }
                let value_index = block * KV_FP8_BLOCK_SIZE + column % KV_FP8_BLOCK_SIZE;
                output.push(decode_value(self.values[value_index], self.physical_variant) * scale);
            }
        }
        Ok(output)
    }
}

/// Quantize row-major KV head rows with a deterministic E8M0 scale per sixteen
/// consecutive columns. The row axis may combine token, plane, and KV-head
/// indices; callers keep K and V in separate invocations/planes.
pub fn quantize_kv_fp8_block16(
    input: &[f32],
    rows: usize,
    columns: usize,
    physical_variant: KvFp8PhysicalVariant,
) -> Result<QuantizedKvFp8Block16, KvFp8CodecError> {
    if rows == 0 || columns == 0 {
        return Err(KvFp8CodecError::Empty);
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(KvFp8CodecError::ShapeOverflow)?;
    if input.len() != expected {
        return Err(KvFp8CodecError::LengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let blocks_per_row = columns.div_ceil(KV_FP8_BLOCK_SIZE);
    let block_count = rows
        .checked_mul(blocks_per_row)
        .ok_or(KvFp8CodecError::ShapeOverflow)?;
    let value_count = block_count
        .checked_mul(KV_FP8_BLOCK_SIZE)
        .ok_or(KvFp8CodecError::ShapeOverflow)?;
    let mut values = vec![0_u8; value_count];
    let mut scales = Vec::with_capacity(block_count);

    for row in 0..rows {
        for block_in_row in 0..blocks_per_row {
            let start_column = block_in_row * KV_FP8_BLOCK_SIZE;
            let end_column = (start_column + KV_FP8_BLOCK_SIZE).min(columns);
            let start = row * columns + start_column;
            let end = row * columns + end_column;
            let valid = &input[start..end];
            let all_zero = valid.iter().all(|value| *value == 0.0);
            let scale_bits = if all_zero {
                127 // unit scale and positive-zero payload is canonical
            } else {
                e8m0_scale_for_block(valid, physical_variant)
            };
            scales.push(scale_bits);
            if all_zero {
                continue;
            }
            let scale = decode_e8m0(scale_bits);
            let value_base = (row * blocks_per_row + block_in_row) * KV_FP8_BLOCK_SIZE;
            for (lane, source) in valid.iter().enumerate() {
                values[value_base + lane] = encode_value(*source / scale, physical_variant);
            }
        }
    }

    Ok(QuantizedKvFp8Block16 {
        values,
        scales,
        rows,
        columns,
        physical_variant,
    })
}

/// Quantize with an explicitly identified Phase 54 scale selector and return
/// per-block attribution counters. This API is absent from normal builds.
#[cfg(feature = "phase54-research")]
pub fn quantize_kv_fp8_block16_research(
    input: &[f32],
    rows: usize,
    columns: usize,
    physical_variant: KvFp8PhysicalVariant,
    recipe: KvFp8ResearchScaleRecipe,
) -> Result<(QuantizedKvFp8Block16, Vec<KvFp8ResearchBlockStats>), KvFp8CodecError> {
    if rows == 0 || columns == 0 {
        return Err(KvFp8CodecError::Empty);
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(KvFp8CodecError::ShapeOverflow)?;
    if input.len() != expected {
        return Err(KvFp8CodecError::LengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let blocks_per_row = columns.div_ceil(KV_FP8_BLOCK_SIZE);
    let block_count = rows
        .checked_mul(blocks_per_row)
        .ok_or(KvFp8CodecError::ShapeOverflow)?;
    let value_count = block_count
        .checked_mul(KV_FP8_BLOCK_SIZE)
        .ok_or(KvFp8CodecError::ShapeOverflow)?;
    let mut values = vec![0_u8; value_count];
    let mut scales = Vec::with_capacity(block_count);
    let mut stats = Vec::with_capacity(block_count);

    for row in 0..rows {
        for block_in_row in 0..blocks_per_row {
            let start_column = block_in_row * KV_FP8_BLOCK_SIZE;
            let end_column = (start_column + KV_FP8_BLOCK_SIZE).min(columns);
            let start = row * columns + start_column;
            let end = row * columns + end_column;
            let valid = &input[start..end];
            let scale_values = if recipe == KvFp8ResearchScaleRecipe::Parent32Duplicate {
                let parent_start_column = block_in_row / 2 * 32;
                let parent_end_column = (parent_start_column + 32).min(columns);
                &input[row * columns + parent_start_column..row * columns + parent_end_column]
            } else {
                valid
            };
            let all_zero = scale_values.iter().all(|value| *value == 0.0);
            let amax = finite_amax(scale_values);
            let scale_bits = if all_zero {
                127
            } else {
                e8m0_scale_for_block_with_recipe(scale_values, physical_variant, recipe)
            };
            scales.push(scale_bits);

            let mut block_stats = KvFp8ResearchBlockStats {
                amax,
                scale_bits,
                scale_exponent: i16::from(scale_bits) - 127,
                finite_count: valid.iter().filter(|value| value.is_finite()).count(),
                encoded_zero_count: valid.len(),
                underflow_to_zero_count: 0,
                saturation_count: 0,
            };
            if !all_zero {
                let scale = decode_e8m0(scale_bits);
                let value_base = (row * blocks_per_row + block_in_row) * KV_FP8_BLOCK_SIZE;
                block_stats.encoded_zero_count = 0;
                for (lane, source) in valid.iter().enumerate() {
                    let normalized = *source / scale;
                    let encoded = encode_value(normalized, physical_variant);
                    values[value_base + lane] = encoded;
                    let decoded = decode_value(encoded, physical_variant);
                    if decoded == 0.0 {
                        block_stats.encoded_zero_count += 1;
                        if source.is_finite() && *source != 0.0 {
                            block_stats.underflow_to_zero_count += 1;
                        }
                    }
                    let maximum = element_maximum(physical_variant);
                    if source.is_finite() && normalized.abs() > maximum && decoded.abs() == maximum
                    {
                        block_stats.saturation_count += 1;
                    }
                }
            }
            stats.push(block_stats);
        }
    }

    Ok((
        QuantizedKvFp8Block16 {
            values,
            scales,
            rows,
            columns,
            physical_variant,
        },
        stats,
    ))
}

/// Decode raw canonical value and E8M0 scale planes without depending on host
/// FP8 instructions. Nonzero tail padding and malformed plane lengths fail
/// closed before any logical values are returned.
pub fn decode_kv_fp8_block16(
    values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
    physical_variant: KvFp8PhysicalVariant,
) -> Result<Vec<f32>, KvFp8CodecError> {
    QuantizedKvFp8Block16::from_parts(
        values.to_vec(),
        scales.to_vec(),
        rows,
        columns,
        physical_variant,
    )?
    .dequantize()
}

fn e8m0_scale_for_block(values: &[f32], variant: KvFp8PhysicalVariant) -> u8 {
    let amax = finite_amax(values);
    if amax == 0.0 {
        return 127;
    }
    let element_power = match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn | KvFp8PhysicalVariant::E4M3FnuZ => 8,
        KvFp8PhysicalVariant::OcpE5M2 => 15,
    };
    let exponent = floor_log2(amax)
        .saturating_sub(element_power)
        .clamp(-127, 127);
    (exponent + 127) as u8
}

fn finite_amax(values: &[f32]) -> f32 {
    values.iter().fold(0.0_f32, |current, value| {
        if value.is_finite() {
            current.max(value.abs())
        } else {
            current
        }
    })
}

#[cfg(feature = "phase54-research")]
fn e8m0_scale_for_block_with_recipe(
    values: &[f32],
    variant: KvFp8PhysicalVariant,
    recipe: KvFp8ResearchScaleRecipe,
) -> u8 {
    let amax = finite_amax(values);
    if amax == 0.0 {
        return 127;
    }
    let element_power = match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn | KvFp8PhysicalVariant::E4M3FnuZ => 8,
        KvFp8PhysicalVariant::OcpE5M2 => 15,
    };
    let floor_maximum_exponent = floor_log2(amax);
    let selected_maximum_exponent = match recipe {
        KvFp8ResearchScaleRecipe::Floor => floor_maximum_exponent,
        KvFp8ResearchScaleRecipe::Ceil => {
            if amax == 2.0_f32.powi(floor_maximum_exponent) {
                floor_maximum_exponent
            } else {
                floor_maximum_exponent + 1
            }
        }
        KvFp8ResearchScaleRecipe::NearestEvenExponent => {
            let normalized = amax * 2.0_f32.powi(-floor_maximum_exponent);
            let floor_scale_exponent = floor_maximum_exponent - element_power;
            if normalized > std::f32::consts::SQRT_2
                || (normalized == std::f32::consts::SQRT_2 && floor_scale_exponent & 1 != 0)
            {
                floor_maximum_exponent + 1
            } else {
                floor_maximum_exponent
            }
        }
        KvFp8ResearchScaleRecipe::Parent32Duplicate => floor_maximum_exponent,
    };
    let exponent = selected_maximum_exponent
        .saturating_sub(element_power)
        .clamp(-127, 127);
    (exponent + 127) as u8
}

#[cfg(feature = "phase54-research")]
fn element_maximum(variant: KvFp8PhysicalVariant) -> f32 {
    match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn => 448.0,
        KvFp8PhysicalVariant::E4M3FnuZ => 240.0,
        KvFp8PhysicalVariant::OcpE5M2 => E5M2_MAX,
    }
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

fn encode_value(value: f32, variant: KvFp8PhysicalVariant) -> u8 {
    match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn => encode_e4m3fn(value),
        KvFp8PhysicalVariant::E4M3FnuZ => encode_e4m3fnuz(value),
        KvFp8PhysicalVariant::OcpE5M2 => encode_e5m2(value),
    }
}

fn decode_value(bits: u8, variant: KvFp8PhysicalVariant) -> f32 {
    match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn => decode_e4m3fn(bits),
        KvFp8PhysicalVariant::E4M3FnuZ => decode_e4m3fnuz(bits),
        KvFp8PhysicalVariant::OcpE5M2 => decode_e5m2(bits),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "phase54-research")]
    use crate::{KvCacheEncoding, KvMxfp8Descriptor, quantize_kv_mxfp8};

    fn block_scale(value: f32, variant: KvFp8PhysicalVariant) -> u8 {
        quantize_kv_fp8_block16(&[value], 1, 1, variant)
            .unwrap()
            .scales()[0]
    }

    #[test]
    fn e5m2_specials_subnormal_and_rne_are_canonical() {
        assert_eq!(encode_e5m2(f32::NAN), 0x7f);
        assert_eq!(encode_e5m2(f32::INFINITY), 0x7b);
        assert_eq!(encode_e5m2(f32::NEG_INFINITY), 0xfb);
        assert_eq!(encode_e5m2(0.0), 0x00);
        assert_eq!(encode_e5m2(-0.0), 0x80);
        assert_eq!(decode_e5m2(0x01), 2.0_f32.powi(-16));
        assert_eq!(decode_e5m2(0x7b), E5M2_MAX);
        // Midpoint between 1.0 (even code 0x3c) and 1.25 (odd 0x3d).
        assert_eq!(encode_e5m2(1.125), 0x3c);
        // Midpoint between odd 0x3d and even 0x3e rounds upward.
        assert_eq!(encode_e5m2(1.375), 0x3e);
    }

    #[test]
    fn special_values_follow_each_physical_variant() {
        for (variant, nan, positive_max, negative_max, negative_zero) in [
            (KvFp8PhysicalVariant::OcpE4M3Fn, 0x7f, 0x7e, 0xfe, 0x80),
            (KvFp8PhysicalVariant::E4M3FnuZ, 0x80, 0x7f, 0xff, 0x00),
            (KvFp8PhysicalVariant::OcpE5M2, 0x7f, 0x7b, 0xfb, 0x80),
        ] {
            let input = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
            let encoded = quantize_kv_fp8_block16(&input, 1, input.len(), variant).unwrap();
            assert_eq!(encoded.scales(), &[127]);
            assert_eq!(encoded.values()[0], nan);
            assert_eq!(encoded.values()[1], positive_max);
            assert_eq!(encoded.values()[2], negative_max);
            assert_eq!(encoded.values()[3], negative_zero);
            assert!(
                encoded.values()[input.len()..]
                    .iter()
                    .all(|byte| *byte == 0)
            );
        }
    }

    #[test]
    fn block_and_tail_shapes_have_canonical_padding() {
        for columns in [15_usize, 16, 17, 255, 256, 257] {
            let input = (0..2 * columns)
                .map(|index| (index as f32 % 31.0) - 15.0)
                .collect::<Vec<_>>();
            let encoded =
                quantize_kv_fp8_block16(&input, 2, columns, KvFp8PhysicalVariant::OcpE4M3Fn)
                    .unwrap();
            let blocks = columns.div_ceil(KV_FP8_BLOCK_SIZE);
            assert_eq!(encoded.scales().len(), 2 * blocks);
            assert_eq!(encoded.values().len(), 2 * blocks * KV_FP8_BLOCK_SIZE);
            let tail = columns % KV_FP8_BLOCK_SIZE;
            if tail != 0 {
                for row in 0..2 {
                    let block_start = (row * blocks + blocks - 1) * KV_FP8_BLOCK_SIZE;
                    assert!(
                        encoded.values()[block_start + tail..block_start + 16]
                            .iter()
                            .all(|byte| *byte == 0)
                    );
                }
            }
            assert_eq!(encoded.dequantize().unwrap().len(), input.len());
        }
    }

    #[cfg(feature = "phase54-research")]
    #[test]
    fn parent32_duplicate_reproduces_mxfp8_values_scales_and_decode() {
        const ROWS: usize = 2;
        for columns in [15_usize, 16, 17, 31, 32, 33, 255, 256, 257] {
            let mut input = (0..ROWS * columns)
                .map(|index| {
                    let column = index % columns;
                    let base = ((index * 37 + 11) % 113) as f32 - 56.0;
                    if column % 32 == 31 {
                        base * 7.0
                    } else {
                        base / 3.0
                    }
                })
                .collect::<Vec<_>>();
            if columns > 16 {
                for row in 0..ROWS {
                    for (index, value) in input[row * columns..row * columns + 16]
                        .iter_mut()
                        .enumerate()
                    {
                        *value = if index % 2 == 0 { -0.0 } else { 0.0 };
                    }
                    input[row * columns + 16] = 3.25 + row as f32;
                }
            }
            for (variant, encoding) in [
                (KvFp8PhysicalVariant::OcpE4M3Fn, KvCacheEncoding::Mxfp8E4),
                (KvFp8PhysicalVariant::OcpE5M2, KvCacheEncoding::Mxfp8E5),
            ] {
                let (block16, _) = quantize_kv_fp8_block16_research(
                    &input,
                    ROWS,
                    columns,
                    variant,
                    KvFp8ResearchScaleRecipe::Parent32Duplicate,
                )
                .unwrap();
                let mxfp8 = quantize_kv_mxfp8(
                    &input,
                    ROWS,
                    columns,
                    KvMxfp8Descriptor::new(encoding, variant).unwrap(),
                )
                .unwrap();

                assert_eq!(block16.dequantize().unwrap(), mxfp8.dequantize().unwrap());
                for row in 0..ROWS {
                    for column in 0..columns {
                        let block16_value = (row * block16.blocks_per_row()
                            + column / KV_FP8_BLOCK_SIZE)
                            * KV_FP8_BLOCK_SIZE
                            + column % KV_FP8_BLOCK_SIZE;
                        let mxfp8_value = (row * mxfp8.blocks_per_row()
                            + column / crate::KV_MXFP8_BLOCK_SIZE)
                            * crate::KV_MXFP8_BLOCK_SIZE
                            + column % crate::KV_MXFP8_BLOCK_SIZE;
                        assert_eq!(block16.values()[block16_value], mxfp8.values()[mxfp8_value]);
                    }
                    for child in 0..block16.blocks_per_row() {
                        let block16_scale = row * block16.blocks_per_row() + child;
                        let mxfp8_scale = row * mxfp8.blocks_per_row() + child / 2;
                        assert_eq!(block16.scales()[block16_scale], mxfp8.scales()[mxfp8_scale]);
                    }
                }
            }
        }
    }

    #[test]
    fn all_zero_blocks_use_unit_scale_and_positive_zero_payload() {
        for variant in [
            KvFp8PhysicalVariant::OcpE4M3Fn,
            KvFp8PhysicalVariant::E4M3FnuZ,
            KvFp8PhysicalVariant::OcpE5M2,
        ] {
            let input = [0.0, -0.0, 0.0, -0.0, 0.0];
            let encoded = quantize_kv_fp8_block16(&input, 1, input.len(), variant).unwrap();
            assert_eq!(encoded.scales(), &[127]);
            assert!(encoded.values().iter().all(|byte| *byte == 0));
        }
    }

    #[test]
    fn block16_scale_uses_floor_power_across_one_point_seven_five_mantissa() {
        let at = 1.75_f32;
        let below = f32::from_bits(at.to_bits() - 1);
        let above = f32::from_bits(at.to_bits() + 1);
        for (variant, expected_scale) in [
            (KvFp8PhysicalVariant::OcpE4M3Fn, 119),
            (KvFp8PhysicalVariant::E4M3FnuZ, 119),
            (KvFp8PhysicalVariant::OcpE5M2, 112),
        ] {
            assert_eq!(block_scale(below, variant), expected_scale);
            assert_eq!(block_scale(at, variant), expected_scale);
            assert_eq!(block_scale(above, variant), expected_scale);
        }
    }

    #[test]
    fn block16_standard_mx_vectors_keep_unit_scale_and_saturate_elements() {
        for (variant, amax, expected_value, expected_dequantized) in [
            (KvFp8PhysicalVariant::OcpE4M3Fn, 511.0, 0x7e, 448.0),
            (KvFp8PhysicalVariant::E4M3FnuZ, 511.0, 0x7f, 240.0),
            (KvFp8PhysicalVariant::OcpE5M2, 65_535.0, 0x7b, 57_344.0),
        ] {
            let encoded = quantize_kv_fp8_block16(&[amax], 1, 1, variant).unwrap();
            assert_eq!(encoded.scales(), &[127]);
            assert_eq!(encoded.values()[0], expected_value);
            assert_eq!(encoded.dequantize().unwrap(), vec![expected_dequantized]);
        }
    }

    #[test]
    fn block16_scale_clamps_low_and_covers_f32_exponent_limits() {
        for variant in [
            KvFp8PhysicalVariant::OcpE4M3Fn,
            KvFp8PhysicalVariant::E4M3FnuZ,
        ] {
            assert_eq!(block_scale(f32::from_bits(1), variant), 0);
            assert_eq!(block_scale(2.0_f32.powi(-119), variant), 0);
            assert_eq!(
                block_scale(f32::from_bits(2.0_f32.powi(-118).to_bits() - 1), variant),
                0
            );
            assert_eq!(block_scale(2.0_f32.powi(-118), variant), 1);
            assert_eq!(block_scale(f32::MAX, variant), 246);
        }

        let variant = KvFp8PhysicalVariant::OcpE5M2;
        assert_eq!(block_scale(f32::from_bits(1), variant), 0);
        assert_eq!(block_scale(2.0_f32.powi(-112), variant), 0);
        assert_eq!(
            block_scale(f32::from_bits(2.0_f32.powi(-111).to_bits() - 1), variant),
            0
        );
        assert_eq!(block_scale(2.0_f32.powi(-111), variant), 1);
        assert_eq!(block_scale(f32::MAX, variant), 239);
    }

    #[test]
    fn raw_decode_rejects_malformed_planes_and_noncanonical_tail() {
        let encoded =
            quantize_kv_fp8_block16(&[1.0; 17], 1, 17, KvFp8PhysicalVariant::OcpE4M3Fn).unwrap();
        assert_eq!(
            decode_kv_fp8_block16(
                encoded.values(),
                encoded.scales(),
                1,
                17,
                KvFp8PhysicalVariant::OcpE4M3Fn,
            )
            .unwrap()
            .len(),
            17
        );
        let mut bad_padding = encoded.values().to_vec();
        bad_padding[17] = 1;
        assert_eq!(
            decode_kv_fp8_block16(
                &bad_padding,
                encoded.scales(),
                1,
                17,
                KvFp8PhysicalVariant::OcpE4M3Fn,
            ),
            Err(KvFp8CodecError::NonCanonicalPadding { row: 0, column: 17 })
        );
        assert!(matches!(
            decode_kv_fp8_block16(
                &encoded.values()[..31],
                encoded.scales(),
                1,
                17,
                KvFp8PhysicalVariant::OcpE4M3Fn,
            ),
            Err(KvFp8CodecError::ValueLengthMismatch { .. })
        ));
    }

    #[test]
    fn e8m0_scale_uses_valid_lanes_and_covers_exponent_boundaries() {
        let tiny =
            quantize_kv_fp8_block16(&[f32::from_bits(1)], 1, 1, KvFp8PhysicalVariant::OcpE4M3Fn)
                .unwrap();
        assert_eq!(tiny.scales(), &[0]);

        let largest =
            quantize_kv_fp8_block16(&[f32::MAX], 1, 1, KvFp8PhysicalVariant::OcpE4M3Fn).unwrap();
        assert_eq!(largest.scales(), &[246]);
        assert_ne!(largest.values()[0], 0);

        let tail =
            quantize_kv_fp8_block16(&[1.0; 17], 1, 17, KvFp8PhysicalVariant::OcpE5M2).unwrap();
        assert_eq!(tail.scales().len(), 2);
        assert!(tail.values()[17..32].iter().all(|byte| *byte == 0));

        let maximum_finite_scale = QuantizedKvFp8Block16 {
            values: {
                let mut values = vec![0; KV_FP8_BLOCK_SIZE];
                values[0] = encode_e4m3fn(1.0);
                values
            },
            scales: vec![254],
            rows: 1,
            columns: 1,
            physical_variant: KvFp8PhysicalVariant::OcpE4M3Fn,
        };
        assert_eq!(
            maximum_finite_scale.dequantize().unwrap(),
            vec![2.0_f32.powi(127)]
        );

        let nan_scale = QuantizedKvFp8Block16 {
            scales: vec![255],
            ..maximum_finite_scale
        };
        assert_eq!(
            nan_scale.dequantize(),
            Err(KvFp8CodecError::NonFiniteScale { block: 0 })
        );
    }

    #[cfg(feature = "phase54-research")]
    #[test]
    fn phase54_floor_is_byte_exact_with_production() {
        for variant in [
            KvFp8PhysicalVariant::OcpE4M3Fn,
            KvFp8PhysicalVariant::E4M3FnuZ,
            KvFp8PhysicalVariant::OcpE5M2,
        ] {
            for columns in [15_usize, 16, 17, 255, 256, 257] {
                let mut input = (0..2 * columns)
                    .map(|index| {
                        let sign = if index & 1 == 0 { 1.0 } else { -1.0 };
                        sign * ((index % 37 + 1) as f32) * 2.0_f32.powi((index % 11) as i32 - 5)
                    })
                    .collect::<Vec<_>>();
                input[0] = -0.0;
                input[1] = f32::NAN;
                input[columns] = f32::INFINITY;
                let production = quantize_kv_fp8_block16(&input, 2, columns, variant).unwrap();
                let (research, stats) = quantize_kv_fp8_block16_research(
                    &input,
                    2,
                    columns,
                    variant,
                    KvFp8ResearchScaleRecipe::Floor,
                )
                .unwrap();
                assert_eq!(research, production);
                assert_eq!(stats.len(), 2 * columns.div_ceil(KV_FP8_BLOCK_SIZE));
            }
        }
    }

    #[cfg(feature = "phase54-research")]
    #[test]
    fn phase54_scale_selectors_cover_power_and_geometric_boundaries() {
        let variant = KvFp8PhysicalVariant::OcpE5M2;
        let scale = |value, recipe| {
            quantize_kv_fp8_block16_research(&[value], 1, 1, variant, recipe)
                .unwrap()
                .0
                .scales()[0]
        };
        let power = 2.0_f32.powi(7);
        let below_power = f32::from_bits(power.to_bits() - 1);
        let above_power = f32::from_bits(power.to_bits() + 1);
        assert_eq!(scale(power, KvFp8ResearchScaleRecipe::Floor), 119);
        assert_eq!(scale(power, KvFp8ResearchScaleRecipe::Ceil), 119);
        assert_eq!(scale(below_power, KvFp8ResearchScaleRecipe::Ceil), 119);
        assert_eq!(scale(above_power, KvFp8ResearchScaleRecipe::Ceil), 120);

        let midpoint = std::f32::consts::SQRT_2 * power;
        let below_midpoint = f32::from_bits(midpoint.to_bits() - 1);
        let above_midpoint = f32::from_bits(midpoint.to_bits() + 1);
        assert_eq!(
            scale(
                below_midpoint,
                KvFp8ResearchScaleRecipe::NearestEvenExponent
            ),
            119
        );
        // The lower scale exponent is -8 (even), so the exact tie stays low.
        assert_eq!(
            scale(midpoint, KvFp8ResearchScaleRecipe::NearestEvenExponent),
            119
        );
        assert_eq!(
            scale(
                above_midpoint,
                KvFp8ResearchScaleRecipe::NearestEvenExponent
            ),
            120
        );
        // At the next power, the lower scale exponent is -7 (odd), so tie goes high.
        assert_eq!(
            scale(
                std::f32::consts::SQRT_2 * 2.0_f32.powi(8),
                KvFp8ResearchScaleRecipe::NearestEvenExponent
            ),
            121
        );
    }

    #[cfg(feature = "phase54-research")]
    #[test]
    fn phase54_stats_exclude_padding_and_canonicalize_signed_zero() {
        let mut input = vec![-0.0; 17];
        input[16] = f32::from_bits(1);
        let (encoded, stats) = quantize_kv_fp8_block16_research(
            &input,
            1,
            input.len(),
            KvFp8PhysicalVariant::OcpE5M2,
            KvFp8ResearchScaleRecipe::NearestEvenExponent,
        )
        .unwrap();
        assert!(encoded.values()[..16].iter().all(|byte| *byte == 0));
        assert_eq!(encoded.scales()[0], 127);
        assert_eq!(stats[0].finite_count, 16);
        assert_eq!(stats[0].encoded_zero_count, 16);
        assert_eq!(stats[0].underflow_to_zero_count, 0);
        assert_eq!(stats[1].finite_count, 1);
        assert_eq!(stats.len(), 2);
    }
}
