//! Phase 54 research-only V/attention-output preserving permutation.
//!
//! Transposing each value head changes block16 grouping without changing the
//! model: attention weights are scalar per KV position, so the same
//! permutation appears in the attention output and the self-inverse companion
//! restores it before the sigmoid gate and O projection.

use std::fmt;

/// Stable research semantics. This is not a public KV descriptor identity.
pub const PHASE54_VO_TRANSFORM_SEMANTICS: &str = "vo-fixed-permutation/transpose16x16-layer19-v1";

/// Exact environment variable selecting the V/output intervention.
pub const PHASE54_VO_TRANSFORM_ENV: &str = "SLLM_PHASE54_VO_TRANSFORM";

/// Exact selector value for the original reviewed V/output candidate.
pub const PHASE54_VO_TRANSFORM_SELECTOR: &str = "transpose16x16-v-layer19-output-inverse";

/// Exact selector value for the reviewed two-layer V/output candidate.
pub const PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR: &str =
    "transpose16x16-v-layers19-31-output-inverse";

/// Host path used by the research implementation.
pub const PHASE54_VO_TRANSFORM_BACKEND: &str = "host-readback-bf16-permute-upload-v1";

pub const PHASE54_VO_TRANSFORM_HEAD_DIM: usize = 256;
pub const PHASE54_VO_TRANSFORM_KV_HEADS: usize = 4;
pub const PHASE54_VO_TRANSFORM_Q_HEADS: usize = 16;
pub const PHASE54_VO_TRANSFORM_LAYERS: &[u32] = &[19];
pub const PHASE54_VO_TRANSFORM_LAYERS_19_31_LAYERS: &[u32] = &[19, 31];

/// Canonical specification bytes. Field order is part of the identity.
pub const PHASE54_VO_TRANSFORM_CANONICAL: &str = "{\"algorithm\":\"transpose16x16\",\"head_dim\":256,\"inverse\":\"self\",\"kv_heads\":4,\"layers\":[19],\"mapping\":\"out[i]=in[16*(i%16)+floor(i/16)]\",\"output_stage\":\"post-attention-pre-sigmoid-gate\",\"planes\":[\"V\",\"attention_output\"],\"q_heads\":16,\"recipe\":\"StandardMxFloorPowerV1\",\"version\":\"v1\"}";

pub const PHASE54_VO_TRANSFORM_DIGEST: &str =
    "sha256:7da862b274ac32124e4a7b2550ed947fe865140ddaa2fd940a89ebfa9d8c4ad4";

pub const PHASE54_VO_TRANSFORM_LAYERS_19_31_SEMANTICS: &str =
    "vo-fixed-permutation/transpose16x16-layers19-31-v1";

/// Canonical specification bytes for the two-layer candidate. Field order is
/// part of the identity.
pub const PHASE54_VO_TRANSFORM_LAYERS_19_31_CANONICAL: &str = "{\"algorithm\":\"transpose16x16\",\"head_dim\":256,\"inverse\":\"self\",\"kv_heads\":4,\"layers\":[19,31],\"mapping\":\"out[i]=in[16*(i%16)+floor(i/16)]\",\"output_stage\":\"post-attention-pre-sigmoid-gate\",\"planes\":[\"V\",\"attention_output\"],\"q_heads\":16,\"recipe\":\"StandardMxFloorPowerV1\",\"version\":\"v1\"}";

pub const PHASE54_VO_TRANSFORM_LAYERS_19_31_DIGEST: &str =
    "sha256:5439e11e91b4c2acfd060fb1ec4d8f5fee2f1244e28c3e6588f2202fbe8e9a74";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase54VoTransformTarget {
    Gfx1030,
    Gfx1201,
}

impl Phase54VoTransformTarget {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase54VoTransformMode {
    Off,
    Transpose16x16VLayer19OutputInverse,
    Transpose16x16VLayers19And31OutputInverse,
}

impl Phase54VoTransformMode {
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub const fn identity_tag(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Transpose16x16VLayer19OutputInverse => PHASE54_VO_TRANSFORM_SELECTOR,
            Self::Transpose16x16VLayers19And31OutputInverse => {
                PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR
            }
        }
    }

    pub const fn semantics(self) -> &'static str {
        match self {
            Self::Off | Self::Transpose16x16VLayer19OutputInverse => PHASE54_VO_TRANSFORM_SEMANTICS,
            Self::Transpose16x16VLayers19And31OutputInverse => {
                PHASE54_VO_TRANSFORM_LAYERS_19_31_SEMANTICS
            }
        }
    }

    pub const fn digest(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Transpose16x16VLayer19OutputInverse => Some(PHASE54_VO_TRANSFORM_DIGEST),
            Self::Transpose16x16VLayers19And31OutputInverse => {
                Some(PHASE54_VO_TRANSFORM_LAYERS_19_31_DIGEST)
            }
        }
    }

    pub const fn backend(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Transpose16x16VLayer19OutputInverse
            | Self::Transpose16x16VLayers19And31OutputInverse => Some(PHASE54_VO_TRANSFORM_BACKEND),
        }
    }

    pub const fn layers(self) -> &'static [u32] {
        match self {
            Self::Off => &[],
            Self::Transpose16x16VLayer19OutputInverse => PHASE54_VO_TRANSFORM_LAYERS,
            Self::Transpose16x16VLayers19And31OutputInverse => {
                PHASE54_VO_TRANSFORM_LAYERS_19_31_LAYERS
            }
        }
    }

    pub fn parse(value: &str) -> Result<Self, Phase54VoTransformError> {
        match value {
            "off" => Ok(Self::Off),
            PHASE54_VO_TRANSFORM_SELECTOR => Ok(Self::Transpose16x16VLayer19OutputInverse),
            PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR => {
                Ok(Self::Transpose16x16VLayers19And31OutputInverse)
            }
            _ => Err(Phase54VoTransformError::InvalidSelector(format!(
                "unknown Phase 54 V/output transform selector {value:?}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Phase54VoTransformConfig {
    mode: Phase54VoTransformMode,
    target: Option<Phase54VoTransformTarget>,
}

impl Phase54VoTransformConfig {
    pub const fn off() -> Self {
        Self {
            mode: Phase54VoTransformMode::Off,
            target: None,
        }
    }

    pub fn from_env(expected_target: Option<&str>) -> Result<Self, Phase54VoTransformError> {
        let mode = match std::env::var_os(PHASE54_VO_TRANSFORM_ENV) {
            None => Phase54VoTransformMode::Off,
            Some(value) => {
                let value = value.to_str().ok_or_else(|| {
                    Phase54VoTransformError::InvalidSelector(format!(
                        "{PHASE54_VO_TRANSFORM_ENV} is not valid UTF-8"
                    ))
                })?;
                Phase54VoTransformMode::parse(value)?
            }
        };
        Self::for_mode(mode, expected_target)
    }

    pub fn for_mode(
        mode: Phase54VoTransformMode,
        expected_target: Option<&str>,
    ) -> Result<Self, Phase54VoTransformError> {
        let target = if mode.is_enabled() {
            let value = expected_target.ok_or_else(|| {
                Phase54VoTransformError::UnsupportedTarget("<missing>".to_owned())
            })?;
            Some(
                Phase54VoTransformTarget::parse(value)
                    .ok_or_else(|| Phase54VoTransformError::UnsupportedTarget(value.to_owned()))?,
            )
        } else {
            None
        };
        Ok(Self { mode, target })
    }

    pub const fn mode(self) -> Phase54VoTransformMode {
        self.mode
    }

    pub const fn target(self) -> Option<Phase54VoTransformTarget> {
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

    pub const fn backend(self) -> Option<&'static str> {
        self.mode.backend()
    }

    pub fn applies_value_layer(self, layer: u32) -> bool {
        self.mode.layers().contains(&layer)
    }

    pub fn applies_output_layer(self, layer: u32) -> bool {
        self.mode.layers().contains(&layer)
    }
}

impl Default for Phase54VoTransformConfig {
    fn default() -> Self {
        Self::off()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Phase54VoTransformError {
    InvalidSelector(String),
    UnsupportedTarget(String),
}

impl fmt::Display for Phase54VoTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSelector(reason) => formatter.write_str(reason),
            Self::UnsupportedTarget(target) => write!(
                formatter,
                "Phase 54 V/output transform target {target:?} is unsupported"
            ),
        }
    }
}

impl std::error::Error for Phase54VoTransformError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase54_kq_transform::{transpose_bf16_words, transpose16x16_index};

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        let upper = bits >> 16;
        let lower = bits & 0xffff;
        (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
    }

    fn as_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    #[test]
    fn selector_target_and_layer_roles_are_closed() {
        let mode = Phase54VoTransformMode::parse(PHASE54_VO_TRANSFORM_SELECTOR).unwrap();
        assert_eq!(
            mode,
            Phase54VoTransformMode::Transpose16x16VLayer19OutputInverse
        );
        assert_eq!(mode.identity_tag(), PHASE54_VO_TRANSFORM_SELECTOR);
        assert!(Phase54VoTransformMode::parse("transpose16x16-v-o-layer19").is_err());
        assert!(Phase54VoTransformMode::parse("transpose16x16-all-full").is_err());
        let config = Phase54VoTransformConfig::for_mode(mode, Some("gfx1030")).unwrap();
        assert_eq!(config.target(), Some(Phase54VoTransformTarget::Gfx1030));
        assert!(config.applies_value_layer(19));
        assert!(config.applies_output_layer(19));
        for layer in [0, 3, 18, 20, 31] {
            assert!(!config.applies_value_layer(layer));
            assert!(!config.applies_output_layer(layer));
        }
        assert!(Phase54VoTransformConfig::for_mode(mode, Some("gfx942")).is_err());
        assert_eq!(Phase54VoTransformConfig::off().target(), None);
    }

    #[test]
    fn two_layer_selector_identity_and_layer_roles_are_closed() {
        let mode =
            Phase54VoTransformMode::parse(PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR).unwrap();
        assert_eq!(
            mode,
            Phase54VoTransformMode::Transpose16x16VLayers19And31OutputInverse
        );
        assert_eq!(
            mode.identity_tag(),
            "transpose16x16-v-layers19-31-output-inverse"
        );
        assert_eq!(mode.layers(), &[19, 31]);
        assert_eq!(
            mode.semantics(),
            "vo-fixed-permutation/transpose16x16-layers19-31-v1"
        );
        assert_eq!(
            mode.digest(),
            Some("sha256:5439e11e91b4c2acfd060fb1ec4d8f5fee2f1244e28c3e6588f2202fbe8e9a74")
        );
        assert_eq!(mode.backend(), Some(PHASE54_VO_TRANSFORM_BACKEND));

        let config = Phase54VoTransformConfig::for_mode(mode, Some("gfx1201")).unwrap();
        for layer in [19, 31] {
            assert!(config.applies_value_layer(layer));
            assert!(config.applies_output_layer(layer));
        }
        for layer in [0, 18, 20, 30] {
            assert!(!config.applies_value_layer(layer));
            assert!(!config.applies_output_layer(layer));
        }
        assert!(
            Phase54VoTransformMode::parse("transpose16x16-v-layers19-30-output-inverse").is_err()
        );
    }

    #[test]
    fn canonical_identity_and_backend_are_exact() {
        assert_eq!(
            PHASE54_VO_TRANSFORM_CANONICAL,
            "{\"algorithm\":\"transpose16x16\",\"head_dim\":256,\"inverse\":\"self\",\"kv_heads\":4,\"layers\":[19],\"mapping\":\"out[i]=in[16*(i%16)+floor(i/16)]\",\"output_stage\":\"post-attention-pre-sigmoid-gate\",\"planes\":[\"V\",\"attention_output\"],\"q_heads\":16,\"recipe\":\"StandardMxFloorPowerV1\",\"version\":\"v1\"}"
        );
        assert_eq!(
            PHASE54_VO_TRANSFORM_DIGEST,
            "sha256:7da862b274ac32124e4a7b2550ed947fe865140ddaa2fd940a89ebfa9d8c4ad4"
        );
        let mode = Phase54VoTransformMode::Transpose16x16VLayer19OutputInverse;
        assert_eq!(mode.semantics(), PHASE54_VO_TRANSFORM_SEMANTICS);
        assert_eq!(mode.digest(), Some(PHASE54_VO_TRANSFORM_DIGEST));
        assert_eq!(mode.backend(), Some(PHASE54_VO_TRANSFORM_BACKEND));

        assert_eq!(
            PHASE54_VO_TRANSFORM_LAYERS_19_31_CANONICAL,
            "{\"algorithm\":\"transpose16x16\",\"head_dim\":256,\"inverse\":\"self\",\"kv_heads\":4,\"layers\":[19,31],\"mapping\":\"out[i]=in[16*(i%16)+floor(i/16)]\",\"output_stage\":\"post-attention-pre-sigmoid-gate\",\"planes\":[\"V\",\"attention_output\"],\"q_heads\":16,\"recipe\":\"StandardMxFloorPowerV1\",\"version\":\"v1\"}"
        );
        assert_eq!(
            PHASE54_VO_TRANSFORM_LAYERS_19_31_DIGEST,
            "sha256:5439e11e91b4c2acfd060fb1ec4d8f5fee2f1244e28c3e6588f2202fbe8e9a74"
        );
    }

    #[test]
    fn value_permutation_and_output_companion_preserve_attention_result() {
        let first = (0..PHASE54_VO_TRANSFORM_HEAD_DIM)
            .map(|lane| bf16((lane as f32 - 127.0) / 32.0))
            .collect::<Vec<_>>();
        let second = (0..PHASE54_VO_TRANSFORM_HEAD_DIM)
            .map(|lane| bf16(((lane * 7 % 113) as f32 - 56.0) / 16.0))
            .collect::<Vec<_>>();
        let first_permuted =
            transpose_bf16_words(&first, 1, PHASE54_VO_TRANSFORM_HEAD_DIM).unwrap();
        let second_permuted =
            transpose_bf16_words(&second, 1, PHASE54_VO_TRANSFORM_HEAD_DIM).unwrap();
        let mix = |left: &[u16], right: &[u16]| {
            left.iter()
                .zip(right)
                .map(|(left, right)| bf16(as_f32(*left) * 0.25 + as_f32(*right) * 0.75))
                .collect::<Vec<_>>()
        };
        let original_output = mix(&first, &second);
        let permuted_attention_output = mix(&first_permuted, &second_permuted);
        let restored =
            transpose_bf16_words(&permuted_attention_output, 1, PHASE54_VO_TRANSFORM_HEAD_DIM)
                .unwrap();
        assert_eq!(restored, original_output);
        for lane in 0..PHASE54_VO_TRANSFORM_HEAD_DIM {
            assert_eq!(transpose16x16_index(transpose16x16_index(lane)), lane);
        }
    }

    #[test]
    fn reviewed_value_and_output_shapes_share_the_exact_row_permutation() {
        for (tokens, heads) in [
            (1, PHASE54_VO_TRANSFORM_KV_HEADS),
            (3, PHASE54_VO_TRANSFORM_KV_HEADS),
            (1, PHASE54_VO_TRANSFORM_Q_HEADS),
            (3, PHASE54_VO_TRANSFORM_Q_HEADS),
        ] {
            let rows = tokens * heads;
            let words = (0..rows * PHASE54_VO_TRANSFORM_HEAD_DIM)
                .map(|index| bf16((index % 127) as f32 - 63.0))
                .collect::<Vec<_>>();
            let transformed =
                transpose_bf16_words(&words, rows, PHASE54_VO_TRANSFORM_HEAD_DIM).unwrap();
            let restored =
                transpose_bf16_words(&transformed, rows, PHASE54_VO_TRANSFORM_HEAD_DIM).unwrap();
            assert_eq!(restored, words);
        }
    }
}
