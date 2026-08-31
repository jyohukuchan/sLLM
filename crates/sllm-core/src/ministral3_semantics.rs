//! Model-free Ministral 3 text semantics.
//!
//! The reviewed checkpoint uses YaRN for every rotary dimension and applies
//! the Llama 4 position-dependent query scale separately. Keeping both
//! operations typed prevents long-context requests from using plain RoPE.

use std::error::Error;
use std::fmt;

pub const MINISTRAL3_HEAD_DIM: usize = 128;
pub const MINISTRAL3_ROTARY_PAIRS: usize = MINISTRAL3_HEAD_DIM / 2;
pub const MINISTRAL3_ROPE_THETA: f32 = 1_000_000.0;
pub const MINISTRAL3_YARN_FACTOR: f32 = 16.0;
pub const MINISTRAL3_ORIGINAL_CONTEXT: u32 = 16_384;
pub const MINISTRAL3_MAX_CONTEXT: u32 = 262_144;
pub const MINISTRAL3_YARN_BETA_FAST: f64 = 32.0;
pub const MINISTRAL3_YARN_BETA_SLOW: f64 = 1.0;
pub const MINISTRAL3_LLAMA4_SCALING_BETA: f32 = 0.1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3SemanticError {
    PositionOutsideContext { position: u32, context: u32 },
    InvalidHeadLength { expected: usize, actual: usize },
    NonFiniteInput { index: usize },
    NonFiniteResult,
}

impl fmt::Display for Ministral3SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionOutsideContext { position, context } => write!(
                formatter,
                "Ministral 3 position {position} is outside context {context}"
            ),
            Self::InvalidHeadLength { expected, actual } => write!(
                formatter,
                "Ministral 3 head length differs: expected {expected}, got {actual}"
            ),
            Self::NonFiniteInput { index } => {
                write!(formatter, "Ministral 3 head value {index} is non-finite")
            }
            Self::NonFiniteResult => formatter.write_str("Ministral 3 RoPE result is non-finite"),
        }
    }
}

impl Error for Ministral3SemanticError {}

fn correction_dimension(rotations: f64) -> f64 {
    let dimension = MINISTRAL3_HEAD_DIM as f64;
    let original_context = f64::from(MINISTRAL3_ORIGINAL_CONTEXT);
    dimension * (original_context / (rotations * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * f64::from(MINISTRAL3_ROPE_THETA).ln())
}

/// Inclusive ramp endpoints after the official default floor/ceil truncation.
pub fn ministral3_yarn_correction_range() -> (usize, usize) {
    let low = correction_dimension(MINISTRAL3_YARN_BETA_FAST)
        .floor()
        .max(0.0) as usize;
    let high = correction_dimension(MINISTRAL3_YARN_BETA_SLOW)
        .ceil()
        .min((MINISTRAL3_HEAD_DIM - 1) as f64) as usize;
    (low, high)
}

/// Return the 64 FP32 inverse frequencies used by the fixed checkpoint.
pub fn ministral3_yarn_inverse_frequencies() -> [f32; MINISTRAL3_ROTARY_PAIRS] {
    let (low, high) = ministral3_yarn_correction_range();
    let mut frequencies = [0.0_f32; MINISTRAL3_ROTARY_PAIRS];
    for (index, frequency) in frequencies.iter_mut().enumerate() {
        let dimension = (index * 2) as f32;
        let positional_frequency = MINISTRAL3_ROPE_THETA.powf(dimension / 128.0);
        let extrapolated = 1.0 / positional_frequency;
        let interpolated = 1.0 / (MINISTRAL3_YARN_FACTOR * positional_frequency);
        let ramp = (((index as f32) - (low as f32)) / ((high - low) as f32)).clamp(0.0, 1.0);
        let extrapolation_factor = 1.0 - ramp;
        *frequency =
            interpolated * (1.0 - extrapolation_factor) + extrapolated * extrapolation_factor;
    }
    frequencies
}

/// Position-dependent scale applied to Q after rotary embedding.
pub fn ministral3_query_scale(position: u32) -> Result<f32, Ministral3SemanticError> {
    if position >= MINISTRAL3_MAX_CONTEXT {
        return Err(Ministral3SemanticError::PositionOutsideContext {
            position,
            context: MINISTRAL3_MAX_CONTEXT,
        });
    }
    let block = position / MINISTRAL3_ORIGINAL_CONTEXT;
    let scale = 1.0 + MINISTRAL3_LLAMA4_SCALING_BETA * ((block + 1) as f32).ln();
    if !scale.is_finite() {
        return Err(Ministral3SemanticError::NonFiniteResult);
    }
    Ok(scale)
}

/// Apply the official-GGUF adjacent-pair YaRN RoPE and the Ministral
/// query-only long-position scale. The GGUF Q/K weights already contain the
/// head permutation that converts the source split-half layout to adjacent
/// pairs. Pass `scale_query = false` for K. V is not rotary transformed.
pub fn ministral3_apply_yarn_rope(
    head: &[f32],
    position: u32,
    scale_query: bool,
) -> Result<[f32; MINISTRAL3_HEAD_DIM], Ministral3SemanticError> {
    if head.len() != MINISTRAL3_HEAD_DIM {
        return Err(Ministral3SemanticError::InvalidHeadLength {
            expected: MINISTRAL3_HEAD_DIM,
            actual: head.len(),
        });
    }
    if position >= MINISTRAL3_MAX_CONTEXT {
        return Err(Ministral3SemanticError::PositionOutsideContext {
            position,
            context: MINISTRAL3_MAX_CONTEXT,
        });
    }
    if let Some(index) = head.iter().position(|value| !value.is_finite()) {
        return Err(Ministral3SemanticError::NonFiniteInput { index });
    }

    let inverse = ministral3_yarn_inverse_frequencies();
    let query_scale = if scale_query {
        ministral3_query_scale(position)?
    } else {
        1.0
    };
    let mut output = [0.0_f32; MINISTRAL3_HEAD_DIM];
    for (pair, &inverse_frequency) in inverse.iter().enumerate() {
        let angle = (position as f32) * inverse_frequency;
        let cosine = angle.cos();
        let sine = angle.sin();
        let first_index = pair * 2;
        let second_index = first_index + 1;
        let first = head[first_index];
        let second = head[second_index];
        output[first_index] = (first * cosine - second * sine) * query_scale;
        output[second_index] = (second * cosine + first * sine) * query_scale;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Ministral3SemanticError::NonFiniteResult);
    }
    Ok(output)
}

/// Fixed 4:1 Q-head to KV-head grouping for 32 Q and 8 KV heads.
pub fn ministral3_kv_head_for_query(query_head: u32) -> Option<u32> {
    (query_head < 32).then_some(query_head / 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:e}, got {actual:e}"
        );
    }

    #[test]
    #[allow(clippy::excessive_precision)] // Preserve the independently computed FP32 oracle literals.
    fn yarn_frequency_range_and_reference_points_match_fp32_oracle() {
        assert_eq!(ministral3_yarn_correction_range(), (20, 37));
        let frequencies = ministral3_yarn_inverse_frequencies();
        for (index, expected) in [
            (0, 1.0),
            (19, 0.0165481716),
            (20, 0.0133352149),
            (21, 0.0101534640),
            (36, 0.0000496113498),
            (37, 0.0000212388004),
            (38, 0.0000171151223),
            (63, 0.0000000775586102),
        ] {
            close(
                frequencies[index],
                expected,
                expected.abs() * 2.0e-6 + 1.0e-12,
            );
        }
    }

    #[test]
    fn query_scale_changes_only_at_original_context_blocks() {
        assert_eq!(
            ministral3_query_scale(0).unwrap().to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(
            ministral3_query_scale(16_383).unwrap().to_bits(),
            1.0_f32.to_bits()
        );
        close(ministral3_query_scale(16_384).unwrap(), 1.0693147, 1.0e-7);
        close(ministral3_query_scale(32_768).unwrap(), 1.1098613, 1.0e-7);
        close(ministral3_query_scale(262_143).unwrap(), 1.2772589, 1.0e-7);
        assert!(ministral3_query_scale(262_144).is_err());
    }

    #[test]
    fn gguf_adjacent_rotary_query_and_key_are_separate() {
        let mut head = [0.0_f32; MINISTRAL3_HEAD_DIM];
        head[0] = 1.0;
        head[1] = 2.0;
        let query = ministral3_apply_yarn_rope(&head, 16_384, true).unwrap();
        let key = ministral3_apply_yarn_rope(&head, 16_384, false).unwrap();
        let scale = ministral3_query_scale(16_384).unwrap();
        close(query[0], key[0] * scale, 2.0e-6);
        close(query[1], key[1] * scale, 2.0e-6);
        assert!(ministral3_apply_yarn_rope(&head[..127], 0, true).is_err());
        head[17] = f32::NAN;
        assert!(matches!(
            ministral3_apply_yarn_rope(&head, 0, false),
            Err(Ministral3SemanticError::NonFiniteInput { index: 17 })
        ));
    }

    #[test]
    fn gqa_mapping_covers_both_sides() {
        assert_eq!(ministral3_kv_head_for_query(0), Some(0));
        assert_eq!(ministral3_kv_head_for_query(3), Some(0));
        assert_eq!(ministral3_kv_head_for_query(4), Some(1));
        assert_eq!(ministral3_kv_head_for_query(31), Some(7));
        assert_eq!(ministral3_kv_head_for_query(32), None);
    }
}
