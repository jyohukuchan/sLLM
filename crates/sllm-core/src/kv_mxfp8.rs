//! Host-independent standard OCP MXFP8 KV codec.
//!
//! This is deliberately separate from the Phase 53 block16 codec. OCP MX
//! v1.0 section 6.3 selects `X = floor_power_of_two(amax) / P`, where `P` is
//! 256 for E4M3 and 32768 for E5M2, then applies RNE and saturation to Pi.

use std::fmt;

use crate::{
    KV_MXFP8_BLOCK_SIZE, KvFp8PhysicalVariant, KvMxfp8Descriptor, decode_e4m3fn, decode_e5m2,
    decode_e8m0, encode_e4m3fn, encode_e5m2,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvMxfp8CodecError {
    Empty,
    ShapeOverflow,
    LengthMismatch { expected: usize, actual: usize },
    ScaleLengthMismatch { expected: usize, actual: usize },
    ValueLengthMismatch { expected: usize, actual: usize },
    NonFiniteScale { block: usize },
    NonCanonicalPadding { row: usize, column: usize },
}

impl fmt::Display for KvMxfp8CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("KV MXFP8 dimensions must be non-zero"),
            Self::ShapeOverflow => formatter.write_str("KV MXFP8 shape overflowed usize"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "KV MXFP8 input: expected {expected}, got {actual}"
                )
            }
            Self::ScaleLengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "KV MXFP8 scales: expected {expected}, got {actual}"
                )
            }
            Self::ValueLengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "KV MXFP8 values: expected {expected}, got {actual}"
                )
            }
            Self::NonFiniteScale { block } => {
                write!(formatter, "KV MXFP8 block {block} has NaN E8M0 scale")
            }
            Self::NonCanonicalPadding { row, column } => write!(
                formatter,
                "KV MXFP8 row {row} has nonzero padding at column {column}"
            ),
        }
    }
}

impl std::error::Error for KvMxfp8CodecError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedKvMxfp8 {
    values: Vec<u8>,
    scales: Vec<u8>,
    rows: usize,
    columns: usize,
    descriptor: KvMxfp8Descriptor,
}

impl QuantizedKvMxfp8 {
    pub fn from_parts(
        values: Vec<u8>,
        scales: Vec<u8>,
        rows: usize,
        columns: usize,
        descriptor: KvMxfp8Descriptor,
    ) -> Result<Self, KvMxfp8CodecError> {
        validate_parts(&values, &scales, rows, columns)?;
        Ok(Self {
            values,
            scales,
            rows,
            columns,
            descriptor,
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

    pub const fn descriptor(&self) -> KvMxfp8Descriptor {
        self.descriptor
    }

    pub const fn blocks_per_row(&self) -> usize {
        self.columns.div_ceil(KV_MXFP8_BLOCK_SIZE)
    }

    pub fn dequantize(&self) -> Result<Vec<f32>, KvMxfp8CodecError> {
        decode_kv_mxfp8(
            &self.values,
            &self.scales,
            self.rows,
            self.columns,
            self.descriptor,
        )
    }
}

pub fn quantize_kv_mxfp8(
    input: &[f32],
    rows: usize,
    columns: usize,
    descriptor: KvMxfp8Descriptor,
) -> Result<QuantizedKvMxfp8, KvMxfp8CodecError> {
    quantize_kv_mxfp8_with_policy(
        input,
        rows,
        columns,
        descriptor,
        KvMxfp8ScalePolicy::Clipped,
    )
}

/// E8M0 scale selection for the standard OCP MXFP8 KV codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvMxfp8ScalePolicy {
    /// OCP MX v1.0 section 6.3: `floor_power_of_two(amax) / P`.  Blocks whose
    /// maximum sits above 1.75 times that power of two saturate.
    Clipped,
    /// Smallest E8M0 scale that keeps the finite block maximum representable.
    NoClipping,
}

/// Quantize standard MXFP8 KV blocks with an explicit scale policy.  The
/// default [`quantize_kv_mxfp8`] keeps the historical [`KvMxfp8ScalePolicy::Clipped`]
/// rule so existing models are unchanged.
pub fn quantize_kv_mxfp8_with_policy(
    input: &[f32],
    rows: usize,
    columns: usize,
    descriptor: KvMxfp8Descriptor,
    scale_policy: KvMxfp8ScalePolicy,
) -> Result<QuantizedKvMxfp8, KvMxfp8CodecError> {
    if rows == 0 || columns == 0 {
        return Err(KvMxfp8CodecError::Empty);
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(KvMxfp8CodecError::ShapeOverflow)?;
    if input.len() != expected {
        return Err(KvMxfp8CodecError::LengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let blocks_per_row = columns.div_ceil(KV_MXFP8_BLOCK_SIZE);
    let block_count = rows
        .checked_mul(blocks_per_row)
        .ok_or(KvMxfp8CodecError::ShapeOverflow)?;
    let value_count = block_count
        .checked_mul(KV_MXFP8_BLOCK_SIZE)
        .ok_or(KvMxfp8CodecError::ShapeOverflow)?;
    let mut values = vec![0_u8; value_count];
    let mut scales = Vec::with_capacity(block_count);

    for row in 0..rows {
        for block_in_row in 0..blocks_per_row {
            let start_column = block_in_row * KV_MXFP8_BLOCK_SIZE;
            let end_column = (start_column + KV_MXFP8_BLOCK_SIZE).min(columns);
            let start = row * columns + start_column;
            let end = row * columns + end_column;
            let valid = &input[start..end];
            let all_zero = valid.iter().all(|value| *value == 0.0);
            let scale_bits = if all_zero {
                127
            } else {
                standard_mx_scale(valid, descriptor.physical_variant(), scale_policy)
            };
            scales.push(scale_bits);
            if all_zero {
                continue;
            }
            let scale = decode_e8m0(scale_bits);
            let value_base = (row * blocks_per_row + block_in_row) * KV_MXFP8_BLOCK_SIZE;
            for (lane, source) in valid.iter().enumerate() {
                values[value_base + lane] =
                    encode_value(*source / scale, descriptor.physical_variant());
            }
        }
    }
    QuantizedKvMxfp8::from_parts(values, scales, rows, columns, descriptor)
}

pub fn decode_kv_mxfp8(
    values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
    descriptor: KvMxfp8Descriptor,
) -> Result<Vec<f32>, KvMxfp8CodecError> {
    validate_parts(values, scales, rows, columns)?;
    let blocks_per_row = columns.div_ceil(KV_MXFP8_BLOCK_SIZE);
    let logical_len = rows
        .checked_mul(columns)
        .ok_or(KvMxfp8CodecError::ShapeOverflow)?;
    let mut output = Vec::with_capacity(logical_len);
    for row in 0..rows {
        for column in 0..columns {
            let block = row * blocks_per_row + column / KV_MXFP8_BLOCK_SIZE;
            let scale = decode_e8m0(scales[block]);
            if !scale.is_finite() {
                return Err(KvMxfp8CodecError::NonFiniteScale { block });
            }
            let value_index = block * KV_MXFP8_BLOCK_SIZE + column % KV_MXFP8_BLOCK_SIZE;
            output.push(decode_value(values[value_index], descriptor.physical_variant()) * scale);
        }
    }
    Ok(output)
}

fn validate_parts(
    values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<(), KvMxfp8CodecError> {
    if rows == 0 || columns == 0 {
        return Err(KvMxfp8CodecError::Empty);
    }
    let blocks_per_row = columns.div_ceil(KV_MXFP8_BLOCK_SIZE);
    let expected_scales = rows
        .checked_mul(blocks_per_row)
        .ok_or(KvMxfp8CodecError::ShapeOverflow)?;
    let expected_values = expected_scales
        .checked_mul(KV_MXFP8_BLOCK_SIZE)
        .ok_or(KvMxfp8CodecError::ShapeOverflow)?;
    if scales.len() != expected_scales {
        return Err(KvMxfp8CodecError::ScaleLengthMismatch {
            expected: expected_scales,
            actual: scales.len(),
        });
    }
    if values.len() != expected_values {
        return Err(KvMxfp8CodecError::ValueLengthMismatch {
            expected: expected_values,
            actual: values.len(),
        });
    }
    let tail = columns % KV_MXFP8_BLOCK_SIZE;
    if tail != 0 {
        for row in 0..rows {
            let base = (row * blocks_per_row + blocks_per_row - 1) * KV_MXFP8_BLOCK_SIZE;
            if let Some(offset) = values[base + tail..base + KV_MXFP8_BLOCK_SIZE]
                .iter()
                .position(|byte| *byte != 0)
            {
                return Err(KvMxfp8CodecError::NonCanonicalPadding {
                    row,
                    column: columns + offset,
                });
            }
        }
    }
    Ok(())
}

fn standard_mx_scale(
    values: &[f32],
    variant: KvFp8PhysicalVariant,
    scale_policy: KvMxfp8ScalePolicy,
) -> u8 {
    let amax = values.iter().fold(0.0_f32, |current, value| {
        if value.is_finite() {
            current.max(value.abs())
        } else {
            current
        }
    });
    if amax == 0.0 {
        return 127;
    }
    let (element_power, format_max) = match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn => (8, 448.0_f32),
        KvFp8PhysicalVariant::OcpE5M2 => (15, 57344.0_f32),
        KvFp8PhysicalVariant::E4M3FnuZ => unreachable!("standard MXFP8 excludes FNUZ"),
    };
    let mut exponent = floor_log2(amax).saturating_sub(element_power);
    if scale_policy == KvMxfp8ScalePolicy::NoClipping {
        let bits = (exponent.clamp(-127, 127) + 127) as u8;
        if amax / decode_e8m0(bits) > format_max {
            exponent = exponent.saturating_add(1);
        }
    }
    (exponent.clamp(-127, 127) + 127) as u8
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
        KvFp8PhysicalVariant::OcpE5M2 => encode_e5m2(value),
        KvFp8PhysicalVariant::E4M3FnuZ => unreachable!("standard MXFP8 excludes FNUZ"),
    }
}

fn decode_value(bits: u8, variant: KvFp8PhysicalVariant) -> f32 {
    match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn => decode_e4m3fn(bits),
        KvFp8PhysicalVariant::OcpE5M2 => decode_e5m2(bits),
        KvFp8PhysicalVariant::E4M3FnuZ => unreachable!("standard MXFP8 excludes FNUZ"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvCacheEncoding;

    fn e4() -> KvMxfp8Descriptor {
        KvMxfp8Descriptor::new(KvCacheEncoding::Mxfp8E4, KvFp8PhysicalVariant::OcpE4M3Fn).unwrap()
    }

    fn e5() -> KvMxfp8Descriptor {
        KvMxfp8Descriptor::new(KvCacheEncoding::Mxfp8E5, KvFp8PhysicalVariant::OcpE5M2).unwrap()
    }

    #[test]
    fn standard_scale_uses_floor_power_not_block16_overflow_avoidance() {
        let e4_encoded = quantize_kv_mxfp8(&[511.0], 1, 1, e4()).unwrap();
        assert_eq!(e4_encoded.scales(), &[127]);
        assert_eq!(e4_encoded.values()[0], 0x7e); // SAT to 448

        let e5_encoded = quantize_kv_mxfp8(&[65_535.0], 1, 1, e5()).unwrap();
        assert_eq!(e5_encoded.scales(), &[127]);
        assert_eq!(e5_encoded.values()[0], 0x7b); // SAT to 57344

        assert_eq!(encode_e4m3fn(1.0625), 0x38); // RNE tie to even
        assert_eq!(encode_e5m2(1.125), 0x3c);
    }

    #[test]
    fn standard_block32_shapes_and_tail_padding_are_exact() {
        for columns in [15_usize, 16, 17, 31, 32, 33, 255, 256, 257] {
            let input = (0..2 * columns)
                .map(|index| (index as f32 % 47.0) - 23.0)
                .collect::<Vec<_>>();
            for descriptor in [e4(), e5()] {
                let encoded = quantize_kv_mxfp8(&input, 2, columns, descriptor).unwrap();
                let blocks = columns.div_ceil(KV_MXFP8_BLOCK_SIZE);
                assert_eq!(encoded.scales().len(), 2 * blocks);
                assert_eq!(encoded.values().len(), 2 * blocks * KV_MXFP8_BLOCK_SIZE);
                assert_eq!(encoded.dequantize().unwrap().len(), input.len());
                let tail = columns % KV_MXFP8_BLOCK_SIZE;
                if tail != 0 {
                    for row in 0..2 {
                        let base = (row * blocks + blocks - 1) * KV_MXFP8_BLOCK_SIZE;
                        assert!(
                            encoded.values()[base + tail..base + KV_MXFP8_BLOCK_SIZE]
                                .iter()
                                .all(|byte| *byte == 0)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn special_values_and_all_zero_are_canonical() {
        for (descriptor, nan, max, negative_max) in
            [(e4(), 0x7f, 0x7e, 0xfe), (e5(), 0x7f, 0x7b, 0xfb)]
        {
            let special = quantize_kv_mxfp8(
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0],
                1,
                4,
                descriptor,
            )
            .unwrap();
            assert_eq!(special.scales(), &[127]);
            assert_eq!(&special.values()[..4], &[nan, max, negative_max, 0x80]);
            assert!(special.values()[4..].iter().all(|byte| *byte == 0));

            let zero = quantize_kv_mxfp8(&[0.0, -0.0], 1, 2, descriptor).unwrap();
            assert_eq!(zero.scales(), &[127]);
            assert!(zero.values().iter().all(|byte| *byte == 0));
        }
    }

    #[test]
    fn scale_exponent_and_noncanonical_raw_planes_fail_closed() {
        let tiny = quantize_kv_mxfp8(&[f32::from_bits(1)], 1, 1, e4()).unwrap();
        assert_eq!(tiny.scales(), &[0]);
        let mut bad = tiny.values().to_vec();
        bad[1] = 1;
        assert_eq!(
            decode_kv_mxfp8(&bad, tiny.scales(), 1, 1, e4()),
            Err(KvMxfp8CodecError::NonCanonicalPadding { row: 0, column: 1 })
        );
        assert_eq!(
            decode_kv_mxfp8(&tiny.values, &[255], 1, 1, e4()),
            Err(KvMxfp8CodecError::NonFiniteScale { block: 0 })
        );
    }

    #[test]
    fn no_clipping_kv_scale_is_the_smallest_non_saturating_e8m0_step() {
        // E4M3 finite max is 448, E5M2 is 57344.  Sweep maxima around both
        // boundaries and assert the no-clip block never saturates while the
        // clipped rule does at the mantissa > 1.75 points.
        for (descriptor, format_max) in [(e4(), 448.0_f32), (e5(), 57344.0_f32)] {
            for maximum in [
                1.0_f32, 1.75, 2.0, 3.0, 448.0, 448.5, 512.0, 57344.0, 60000.0, 0.013,
            ] {
                let mut block = vec![maximum];
                block.resize(32, maximum / 4.0);
                let clipped = quantize_kv_mxfp8(&block, 1, 32, descriptor).unwrap();
                let no_clip = quantize_kv_mxfp8_with_policy(
                    &block,
                    1,
                    32,
                    descriptor,
                    KvMxfp8ScalePolicy::NoClipping,
                )
                .unwrap();
                let clipped_exponent = i32::from(clipped.scales()[0]) - 127;
                let no_clip_exponent = i32::from(no_clip.scales()[0]) - 127;
                // The new rule moves at most one E8M0 step up.
                assert!(
                    no_clip_exponent == clipped_exponent
                        || no_clip_exponent == clipped_exponent + 1,
                    "no-clip moved more than one step: {no_clip_exponent} from {clipped_exponent}"
                );
                // The chosen scale never saturates the block maximum.
                let unit = decode_e8m0(no_clip.scales()[0]);
                assert!(
                    maximum / unit <= format_max,
                    "no-clip KV scale saturates max {maximum} above {format_max}"
                );
                if no_clip_exponent > clipped_exponent {
                    // Moving up is only allowed when the lower step saturates.
                    let lower = decode_e8m0(clipped.scales()[0]);
                    assert!(
                        maximum / lower > format_max,
                        "no-clip moved up without saturation for max {maximum}"
                    );
                }
            }
        }
    }
}
