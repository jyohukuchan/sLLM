//! Backend-neutral FP8 numeric, scale, and provider contracts.
//!
//! Phase 10 uses OCP E4M3FN values with separately resident FP32 scales.  The
//! helpers here deliberately operate on numeric values rather than byte casts,
//! so Phase 11 can add CDNA3 FNUZ conversion without treating the encodings as
//! interchangeable.

use std::fmt;

/// Largest finite OCP E4M3FN magnitude (`0x7e`).
pub const E4M3FN_MAX: f32 = 448.0;

/// Decode one OCP E4M3FN byte to FP32.
pub fn decode_e4m3fn(bits: u8) -> f32 {
    let sign: f32 = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        if mantissa == 0 {
            return if sign.is_sign_negative() { -0.0 } else { 0.0 };
        }
        return sign * f32::from(mantissa) * 2.0_f32.powi(-9);
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
}

/// Encode FP32 as OCP E4M3FN using round-to-nearest-even and finite
/// saturation. NaNs use the canonical positive NaN byte (`0x7f`); infinities
/// saturate to the signed finite maximum.
pub fn encode_e4m3fn(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    if !magnitude.is_finite() || magnitude >= E4M3FN_MAX {
        return sign | 0x7e;
    }

    // There are only 127 non-negative finite encodings.  Exhaustive nearest
    // selection keeps this reference implementation straightforward and makes
    // it an independent oracle for accelerated converter/kernel paths.
    let mut best = 0_u8;
    let mut best_error = f32::INFINITY;
    for candidate in 0_u8..=0x7e {
        let decoded = decode_e4m3fn(candidate);
        let error = (decoded - magnitude).abs();
        if error < best_error || (error == best_error && candidate & 1 == 0 && best & 1 != 0) {
            best = candidate;
            best_error = error;
        }
    }
    sign | best
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedFp8 {
    pub values: Vec<u8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub columns: usize,
    pub block_size: Option<usize>,
}

impl QuantizedFp8 {
    pub fn dequantize(&self) -> Vec<f32> {
        let blocks_per_row = self
            .block_size
            .map_or(1, |block| self.columns.div_ceil(block));
        self.values
            .iter()
            .enumerate()
            .map(|(index, bits)| {
                let row = index / self.columns;
                let column = index % self.columns;
                let scale_index = self
                    .block_size
                    .map_or(row, |block| row * blocks_per_row + column / block);
                decode_e4m3fn(*bits) * self.scales[scale_index]
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Fp8Error {
    EmptyMatrix,
    ShapeOverflow,
    LengthMismatch { expected: usize, actual: usize },
    ZeroBlockSize,
    NonFiniteInput { index: usize },
}

impl fmt::Display for Fp8Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => formatter.write_str("FP8 matrix dimensions must be non-zero"),
            Self::ShapeOverflow => formatter.write_str("FP8 matrix shape overflowed usize"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "FP8 matrix length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ZeroBlockSize => formatter.write_str("FP8 block size must be non-zero"),
            Self::NonFiniteInput { index } => {
                write!(
                    formatter,
                    "FP8 quantization input is non-finite at element {index}"
                )
            }
        }
    }
}

impl std::error::Error for Fp8Error {}

/// Quantize a row-major matrix with one FP32 scale per row.  This is the Phase
/// 10 production contract because ROCm 7.14 hipBLASLt supports outer-vector
/// scaling on gfx1201 while its advertised vec128 scale mode is not supported.
pub fn quantize_e4m3fn_outer_rows(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedFp8, Fp8Error> {
    quantize_e4m3fn(input, rows, columns, None)
}

/// Quantize a row-major matrix using consecutive K-axis blocks.  Phase 10
/// retains this format for converter/oracle interoperability, but does not
/// claim hipBLASLt native execution for it.
pub fn quantize_e4m3fn_k_blocks(
    input: &[f32],
    rows: usize,
    columns: usize,
    block_size: usize,
) -> Result<QuantizedFp8, Fp8Error> {
    if block_size == 0 {
        return Err(Fp8Error::ZeroBlockSize);
    }
    quantize_e4m3fn(input, rows, columns, Some(block_size))
}

fn quantize_e4m3fn(
    input: &[f32],
    rows: usize,
    columns: usize,
    block_size: Option<usize>,
) -> Result<QuantizedFp8, Fp8Error> {
    if rows == 0 || columns == 0 {
        return Err(Fp8Error::EmptyMatrix);
    }
    let expected = rows.checked_mul(columns).ok_or(Fp8Error::ShapeOverflow)?;
    if input.len() != expected {
        return Err(Fp8Error::LengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    if let Some((index, _)) = input
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Fp8Error::NonFiniteInput { index });
    }

    let blocks_per_row = block_size.map_or(1, |block| columns.div_ceil(block));
    let mut values = vec![0_u8; input.len()];
    let mut scales = Vec::with_capacity(rows * blocks_per_row);
    for row in 0..rows {
        for block_index in 0..blocks_per_row {
            let start_column = block_size.map_or(0, |block| block_index * block);
            let end_column =
                block_size.map_or(columns, |block| (start_column + block).min(columns));
            let start = row * columns + start_column;
            let end = row * columns + end_column;
            let amax = input[start..end]
                .iter()
                .fold(0.0_f32, |current, value| current.max(value.abs()));
            let scale = if amax == 0.0 { 1.0 } else { amax / E4M3FN_MAX };
            scales.push(scale);
            for (destination, source) in values[start..end].iter_mut().zip(&input[start..end]) {
                *destination = encode_e4m3fn(*source / scale);
            }
        }
    }
    Ok(QuantizedFp8 {
        values,
        scales,
        rows,
        columns,
        block_size,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8Provider {
    Gfx1201HipBlasLtOuterVector,
    Gfx1030ByteDecodeEmulation,
    Gfx1030ConvertedBf16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fp8ProviderRequest<'a> {
    pub exact_gcn_arch: &'a str,
    pub runtime_fp8_capable: bool,
    pub hipblaslt_solution_supported: bool,
    pub outer_vector_scales: bool,
    pub m: u64,
    pub k: u64,
    pub n: u64,
    pub aligned_16_bytes: bool,
    pub workspace_available: bool,
    pub allow_gfx1030_bf16_conversion: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8ProviderRejection(pub &'static str);

impl fmt::Display for Fp8ProviderRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for Fp8ProviderRejection {}

/// Fail-closed Phase 10 provider selection.  Runtime execution errors are not
/// accepted here as a reason to retry a different provider.
pub fn select_fp8_provider(
    request: Fp8ProviderRequest<'_>,
) -> Result<Fp8Provider, Fp8ProviderRejection> {
    if request.m == 0 || request.k == 0 || request.n == 0 {
        return Err(Fp8ProviderRejection(
            "FP8 matrix dimensions must be non-zero",
        ));
    }
    match request.exact_gcn_arch {
        "gfx1201" => {
            if !request.runtime_fp8_capable {
                return Err(Fp8ProviderRejection(
                    "gfx1201 runtime does not report FP8 capability",
                ));
            }
            if !request.outer_vector_scales {
                return Err(Fp8ProviderRejection(
                    "gfx1201 native provider requires outer-vector FP32 scales",
                ));
            }
            if !request.aligned_16_bytes {
                return Err(Fp8ProviderRejection(
                    "gfx1201 native provider requires 16-byte-aligned bindings",
                ));
            }
            if !request.workspace_available {
                return Err(Fp8ProviderRejection(
                    "gfx1201 native provider workspace is unavailable",
                ));
            }
            if !request.hipblaslt_solution_supported {
                return Err(Fp8ProviderRejection(
                    "hipBLASLt returned no supported FP8 solution for this shape",
                ));
            }
            Ok(Fp8Provider::Gfx1201HipBlasLtOuterVector)
        }
        "gfx1030" if request.allow_gfx1030_bf16_conversion => Ok(Fp8Provider::Gfx1030ConvertedBf16),
        "gfx1030" => Ok(Fp8Provider::Gfx1030ByteDecodeEmulation),
        _ => Err(Fp8ProviderRejection(
            "Phase 10 FP8 provider requires exact gfx1201 or gfx1030",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_special_values_and_signed_zero_are_explicit() {
        assert_eq!(encode_e4m3fn(0.0), 0x00);
        assert_eq!(encode_e4m3fn(-0.0), 0x80);
        assert_eq!(encode_e4m3fn(f32::INFINITY), 0x7e);
        assert_eq!(encode_e4m3fn(f32::NEG_INFINITY), 0xfe);
        assert_eq!(encode_e4m3fn(f32::NAN), 0x7f);
        assert_eq!(decode_e4m3fn(0x01), 2.0_f32.powi(-9));
        assert_eq!(decode_e4m3fn(0x7e), E4M3FN_MAX);
        assert!(decode_e4m3fn(0xff).is_nan());
    }

    #[test]
    fn e4m3fn_every_finite_byte_round_trips() {
        for bits in 0_u8..=u8::MAX {
            let value = decode_e4m3fn(bits);
            if value.is_nan() {
                continue;
            }
            assert_eq!(encode_e4m3fn(value), bits, "byte 0x{bits:02x}");
        }
    }

    #[test]
    fn k_block_boundaries_127_128_129_are_independent() {
        for columns in [127, 128, 129] {
            let input: Vec<f32> = (0..columns)
                .map(|column| (column as f32 - 63.0) / 13.0)
                .collect();
            let quantized = quantize_e4m3fn_k_blocks(&input, 1, columns, 128).unwrap();
            assert_eq!(quantized.scales.len(), columns.div_ceil(128));
            assert_eq!(quantized.values.len(), columns);
            let restored = quantized.dequantize();
            assert_eq!(restored.len(), input.len());
            assert!(restored.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn non_finite_converter_input_is_rejected_before_artifact_creation() {
        assert_eq!(
            quantize_e4m3fn_outer_rows(&[1.0, f32::INFINITY], 1, 2),
            Err(Fp8Error::NonFiniteInput { index: 1 })
        );
    }

    #[test]
    fn provider_selection_is_exact_and_fail_closed() {
        let request = Fp8ProviderRequest {
            exact_gcn_arch: "gfx1201",
            runtime_fp8_capable: true,
            hipblaslt_solution_supported: true,
            outer_vector_scales: true,
            m: 1,
            k: 4096,
            n: 4096,
            aligned_16_bytes: true,
            workspace_available: true,
            allow_gfx1030_bf16_conversion: false,
        };
        assert_eq!(
            select_fp8_provider(request),
            Ok(Fp8Provider::Gfx1201HipBlasLtOuterVector)
        );
        assert!(
            select_fp8_provider(Fp8ProviderRequest {
                hipblaslt_solution_supported: false,
                ..request
            })
            .is_err()
        );
        assert_eq!(
            select_fp8_provider(Fp8ProviderRequest {
                exact_gcn_arch: "gfx1030",
                runtime_fp8_capable: false,
                hipblaslt_solution_supported: false,
                workspace_available: false,
                ..request
            }),
            Ok(Fp8Provider::Gfx1030ByteDecodeEmulation)
        );
    }
}
