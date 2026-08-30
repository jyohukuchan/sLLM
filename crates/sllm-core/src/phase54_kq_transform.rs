//! Phase 54 research-only fixed K/Q transform.
//!
//! The transform is deliberately a permutation, rather than a learned or
//! request-dependent operation.  It transposes the sixteen block16 groups of
//! sixteen head-dimension lanes.  Applying the same permutation to Q and K
//! preserves their dot product while changing only the grouping seen by the
//! K block16 codec.  V, O, and all scale selection remain untouched.

use std::fmt;

/// Stable semantic label for the Phase 54 K/Q intervention.  This is not a
/// KV descriptor or a production state encoding name.
pub const PHASE54_KQ_TRANSFORM_SEMANTICS: &str = "kq-fixed-permutation/transpose16x16-v1";

/// Exact environment variable selecting the research transform.
pub const PHASE54_KQ_TRANSFORM_ENV: &str = "SLLM_PHASE54_KQ_TRANSFORM";

/// The reviewed Qwen full-attention head dimension.
pub const PHASE54_KQ_TRANSFORM_HEAD_DIM: usize = 256;

/// Reviewed full-attention layers in Qwen3.5-4B.  The transform is a closed
/// all-full-layer candidate; arbitrary layer selection is intentionally not
/// accepted by this module.
pub const PHASE54_KQ_TRANSFORM_LAYERS: &[u32] = &[3, 7, 11, 15, 19, 23, 27, 31];

/// Canonical transform specification bytes.  Field order is part of the
/// identity and must not be changed without incrementing the transform
/// version.
pub const PHASE54_KQ_TRANSFORM_CANONICAL: &str = "{\"algorithm\":\"transpose16x16\",\"head_dim\":256,\"layers\":[3,7,11,15,19,23,27,31],\"mapping\":\"out[i]=in[16*(i%16)+floor(i/16)]\",\"plane\":\"K\",\"q_companion\":true,\"recipe\":\"StandardMxFloorPowerV1\",\"version\":\"v1\"}";

/// SHA-256 of [`PHASE54_KQ_TRANSFORM_CANONICAL`].
pub const PHASE54_KQ_TRANSFORM_DIGEST: &str =
    "sha256:806cc66a1135d36fe594c96c78b1329efb955f94a30e9664c20e3d0e41c0cef6";

/// Closed target set for the first Phase 54 K/Q candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase54KqTransformTarget {
    Gfx1030,
    Gfx1201,
}

impl Phase54KqTransformTarget {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Gfx1030 => "gfx1030",
            Self::Gfx1201 => "gfx1201",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "gfx1030" => Some(Self::Gfx1030),
            "gfx1201" => Some(Self::Gfx1201),
            _ => None,
        }
    }
}

/// Research-only K/Q transform selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase54KqTransformMode {
    Off,
    Transpose16x16AllFull,
}

impl Phase54KqTransformMode {
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Transpose16x16AllFull)
    }

    pub const fn identity_tag(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Transpose16x16AllFull => "transpose16x16-all-full",
        }
    }

    pub const fn semantics(self) -> &'static str {
        PHASE54_KQ_TRANSFORM_SEMANTICS
    }

    pub const fn digest(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Transpose16x16AllFull => Some(PHASE54_KQ_TRANSFORM_DIGEST),
        }
    }

    pub fn parse(value: &str) -> Result<Self, Phase54KqTransformError> {
        match value {
            "off" => Ok(Self::Off),
            "transpose16x16-all-full" => Ok(Self::Transpose16x16AllFull),
            _ => Err(Phase54KqTransformError::InvalidSelector(format!(
                "unknown Phase 54 K/Q transform selector {value:?}"
            ))),
        }
    }
}

/// Request-local, target-bound transform configuration.  A disabled config
/// intentionally carries no target, so unset research controls are inert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Phase54KqTransformConfig {
    mode: Phase54KqTransformMode,
    target: Option<Phase54KqTransformTarget>,
}

impl Phase54KqTransformConfig {
    pub const fn off() -> Self {
        Self {
            mode: Phase54KqTransformMode::Off,
            target: None,
        }
    }

    /// Parse [`PHASE54_KQ_TRANSFORM_ENV`].  An unset variable is the normal
    /// no-intervention path; any present unknown value fails closed.
    pub fn from_env(expected_target: Option<&str>) -> Result<Self, Phase54KqTransformError> {
        let mode = match std::env::var_os(PHASE54_KQ_TRANSFORM_ENV) {
            None => Phase54KqTransformMode::Off,
            Some(value) => {
                let value = value.to_str().ok_or_else(|| {
                    Phase54KqTransformError::InvalidSelector(format!(
                        "{PHASE54_KQ_TRANSFORM_ENV} is not valid UTF-8"
                    ))
                })?;
                Phase54KqTransformMode::parse(value)?
            }
        };
        Self::for_mode(mode, expected_target)
    }

    /// Bind an enabled transform to the exact target set.  Unknown or absent
    /// targets are rejected rather than silently selecting a variant.
    pub fn for_mode(
        mode: Phase54KqTransformMode,
        expected_target: Option<&str>,
    ) -> Result<Self, Phase54KqTransformError> {
        let target = if mode.is_enabled() {
            let value = expected_target.ok_or_else(|| {
                Phase54KqTransformError::UnsupportedTarget("<missing>".to_owned())
            })?;
            Some(
                Phase54KqTransformTarget::parse(value)
                    .ok_or_else(|| Phase54KqTransformError::UnsupportedTarget(value.to_owned()))?,
            )
        } else {
            None
        };
        Ok(Self { mode, target })
    }

    pub const fn mode(self) -> Phase54KqTransformMode {
        self.mode
    }

    pub const fn target(self) -> Option<Phase54KqTransformTarget> {
        self.target
    }

    pub const fn is_enabled(self) -> bool {
        self.mode.is_enabled()
    }

    pub const fn semantics(self) -> &'static str {
        self.mode.semantics()
    }

    pub const fn digest(self) -> Option<&'static str> {
        self.mode.digest()
    }

    pub fn applies_layer(self, layer: u32) -> bool {
        self.is_enabled() && PHASE54_KQ_TRANSFORM_LAYERS.contains(&layer)
    }
}

impl Default for Phase54KqTransformConfig {
    fn default() -> Self {
        Self::off()
    }
}

/// Host-side transform failures are explicit: research execution must never
/// silently fall back to the original tensor after an invalid readback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Phase54KqTransformError {
    InvalidSelector(String),
    UnsupportedTarget(String),
    InvalidShape(String),
    OddByteLength { actual: usize },
    NonFiniteInput { index: usize },
}

impl fmt::Display for Phase54KqTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSelector(reason) => formatter.write_str(reason),
            Self::UnsupportedTarget(target) => {
                write!(
                    formatter,
                    "Phase 54 K/Q transform target {target:?} is unsupported"
                )
            }
            Self::InvalidShape(reason) => formatter.write_str(reason),
            Self::OddByteLength { actual } => {
                write!(formatter, "BF16 host tensor byte length {actual} is odd")
            }
            Self::NonFiniteInput { index } => {
                write!(
                    formatter,
                    "BF16 host tensor contains non-finite value at {index}"
                )
            }
        }
    }
}

impl std::error::Error for Phase54KqTransformError {}

/// Transpose permutation index for a 256-lane row.  The mapping is an
/// involution: applying it twice returns the original lane.
pub const fn transpose16x16_index(index: usize) -> usize {
    (index % 16) * 16 + index / 16
}

/// Transform finite BF16 rows in place logically (the returned vector owns a
/// new row-major payload).  `columns` must be exactly the reviewed 256-lane
/// Qwen head dimension; accepting another shape could produce an identity
/// that does not match the candidate digest.
pub fn transpose_bf16_words(
    words: &[u16],
    rows: usize,
    columns: usize,
) -> Result<Vec<u16>, Phase54KqTransformError> {
    if rows == 0 || columns != PHASE54_KQ_TRANSFORM_HEAD_DIM {
        return Err(Phase54KqTransformError::InvalidShape(format!(
            "transpose16x16 requires non-zero rows and {PHASE54_KQ_TRANSFORM_HEAD_DIM} columns"
        )));
    }
    let expected = rows.checked_mul(columns).ok_or_else(|| {
        Phase54KqTransformError::InvalidShape("K/Q transform shape overflowed".to_owned())
    })?;
    if words.len() != expected {
        return Err(Phase54KqTransformError::InvalidShape(format!(
            "K/Q transform words are {}, expected {expected}",
            words.len()
        )));
    }
    for (index, bits) in words.iter().copied().enumerate() {
        if bits & 0x7f80 == 0x7f80 {
            return Err(Phase54KqTransformError::NonFiniteInput { index });
        }
    }

    let mut output = vec![0_u16; expected];
    for row in 0..rows {
        let base = row * columns;
        for column in 0..columns {
            output[base + column] = words[base + transpose16x16_index(column)];
        }
    }
    Ok(output)
}

/// Byte-oriented wrapper used by the host readback path in Qwen execution.
pub fn transpose_bf16_bytes(
    bytes: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<u8>, Phase54KqTransformError> {
    if bytes.len() % 2 != 0 {
        return Err(Phase54KqTransformError::OddByteLength {
            actual: bytes.len(),
        });
    }
    let words = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    let transformed = transpose_bf16_words(&words, rows, columns)?;
    let mut output = Vec::with_capacity(bytes.len());
    for word in transformed {
        output.extend_from_slice(&word.to_le_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        let upper = bits >> 16;
        let lower = bits & 0xffff;
        (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
    }

    fn as_f64(bits: u16) -> f64 {
        f64::from(f32::from_bits(u32::from(bits) << 16))
    }

    fn dot(left: &[u16], right: &[u16]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| as_f64(*left) * as_f64(*right))
            .sum()
    }

    #[test]
    fn selector_and_target_are_typed_and_fail_closed() {
        assert_eq!(
            Phase54KqTransformMode::parse("off").unwrap(),
            Phase54KqTransformMode::Off
        );
        assert_eq!(
            Phase54KqTransformMode::parse("transpose16x16-all-full").unwrap(),
            Phase54KqTransformMode::Transpose16x16AllFull
        );
        assert!(Phase54KqTransformMode::parse("bogus").is_err());
        assert_eq!(
            Phase54KqTransformConfig::for_mode(
                Phase54KqTransformMode::Transpose16x16AllFull,
                Some("gfx1030")
            )
            .unwrap()
            .target(),
            Some(Phase54KqTransformTarget::Gfx1030)
        );
        assert!(
            Phase54KqTransformConfig::for_mode(
                Phase54KqTransformMode::Transpose16x16AllFull,
                Some("gfx942")
            )
            .is_err()
        );
    }

    #[test]
    fn permutation_is_bijective_and_involutive() {
        let mut seen = [false; PHASE54_KQ_TRANSFORM_HEAD_DIM];
        for index in 0..PHASE54_KQ_TRANSFORM_HEAD_DIM {
            let mapped = transpose16x16_index(index);
            assert!(mapped < PHASE54_KQ_TRANSFORM_HEAD_DIM);
            assert!(!seen[mapped]);
            seen[mapped] = true;
            assert_eq!(transpose16x16_index(mapped), index);
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn transform_preserves_qk_dot_product() {
        let query = (0..PHASE54_KQ_TRANSFORM_HEAD_DIM)
            .map(|index| bf16((index as f32 - 127.0) / 32.0))
            .collect::<Vec<_>>();
        let key = (0..PHASE54_KQ_TRANSFORM_HEAD_DIM)
            .map(|index| bf16(((index * 13 % 97) as f32 - 48.0) / 16.0))
            .collect::<Vec<_>>();
        let transformed_query =
            transpose_bf16_words(&query, 1, PHASE54_KQ_TRANSFORM_HEAD_DIM).unwrap();
        let transformed_key = transpose_bf16_words(&key, 1, PHASE54_KQ_TRANSFORM_HEAD_DIM).unwrap();
        assert_eq!(dot(&query, &key), dot(&transformed_query, &transformed_key));
    }

    #[test]
    fn shape_and_finite_contracts_are_fail_closed() {
        assert!(matches!(
            transpose_bf16_words(&[0; PHASE54_KQ_TRANSFORM_HEAD_DIM], 1, 255),
            Err(Phase54KqTransformError::InvalidShape(_))
        ));
        assert!(matches!(
            transpose_bf16_bytes(&[0], 1, PHASE54_KQ_TRANSFORM_HEAD_DIM),
            Err(Phase54KqTransformError::OddByteLength { actual: 1 })
        ));
        let mut non_finite = vec![0_u16; PHASE54_KQ_TRANSFORM_HEAD_DIM];
        non_finite[17] = 0x7f80;
        assert!(matches!(
            transpose_bf16_words(&non_finite, 1, PHASE54_KQ_TRANSFORM_HEAD_DIM),
            Err(Phase54KqTransformError::NonFiniteInput { index: 17 })
        ));
    }

    #[test]
    fn digest_preimage_is_stable() {
        assert_eq!(
            PHASE54_KQ_TRANSFORM_CANONICAL,
            "{\"algorithm\":\"transpose16x16\",\"head_dim\":256,\"layers\":[3,7,11,15,19,23,27,31],\"mapping\":\"out[i]=in[16*(i%16)+floor(i/16)]\",\"plane\":\"K\",\"q_companion\":true,\"recipe\":\"StandardMxFloorPowerV1\",\"version\":\"v1\"}"
        );
        assert_eq!(
            PHASE54_KQ_TRANSFORM_DIGEST,
            "sha256:806cc66a1135d36fe594c96c78b1329efb955f94a30e9664c20e3d0e41c0cef6"
        );
    }
}
