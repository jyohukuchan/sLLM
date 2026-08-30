//! Phase 54 research-only KV attribution surrogate.
//!
//! The production KV ABI stores one encoding for both K and V planes.  This
//! module deliberately does not change that ABI.  Instead, a research build
//! can round-trip a selected BF16 input plane through the existing block16
//! host codec immediately before an FP16-state append.  The resulting bytes
//! are uploaded back to the original BF16 workspace tensor, so the normal
//! append and attention state contracts remain unchanged.

use std::fmt;

use crate::{
    KvFp8CodecError, KvFp8PhysicalVariant, KvFp8ResearchBlockStats, KvFp8ResearchScaleRecipe,
    quantize_kv_fp8_block16_research,
};

/// Stable semantic label for the Phase 54 surrogate.  This is intentionally
/// not a production KV descriptor or state encoding name.
pub const PHASE54_KV_ATTRIBUTION_SEMANTICS: &str = "fp16-state/block16-roundtrip";

/// Reviewed full-attention layers available to the bounded attribution
/// control. The selector is intentionally a closed enum: arbitrary layers
/// cannot silently acquire research semantics.
pub const PHASE54_KV_ATTRIBUTION_ALLOWED_LAYERS: &[u32] = &[3, 7, 11, 15, 19, 23, 27, 31];

/// Environment variable carrying the selected reviewed full-attention layer.
pub const PHASE54_KV_ATTRIBUTION_LAYER_ENV: &str = "SLLM_PHASE54_KV_ATTRIBUTION_LAYER";

/// Compatibility alias for callers that used the original layer-three
/// prototype. New attribution reports must carry their runtime `layer`.
pub const PHASE54_KV_ATTRIBUTION_FIXED_LAYER: u32 = 3;

pub fn is_allowed_layer(layer: u32) -> bool {
    PHASE54_KV_ATTRIBUTION_ALLOWED_LAYERS.contains(&layer)
}

pub fn parse_layer(value: &str) -> Result<u32, Phase54KvAttributionError> {
    let layer = value.parse::<u32>().map_err(|_| {
        Phase54KvAttributionError::InvalidLayer(format!(
            "{PHASE54_KV_ATTRIBUTION_LAYER_ENV} must be an unsigned layer index"
        ))
    })?;
    if !is_allowed_layer(layer) {
        return Err(Phase54KvAttributionError::InvalidLayer(format!(
            "{PHASE54_KV_ATTRIBUTION_LAYER_ENV} layer {layer} is not in the reviewed set"
        )));
    }
    Ok(layer)
}

fn layer_from_env() -> Result<u32, Phase54KvAttributionError> {
    let value = std::env::var_os(PHASE54_KV_ATTRIBUTION_LAYER_ENV).ok_or_else(|| {
        Phase54KvAttributionError::InvalidLayer(format!(
            "{PHASE54_KV_ATTRIBUTION_LAYER_ENV} is required when attribution is enabled"
        ))
    })?;
    let value = value.to_str().ok_or_else(|| {
        Phase54KvAttributionError::InvalidLayer(format!(
            "{PHASE54_KV_ATTRIBUTION_LAYER_ENV} is not valid UTF-8"
        ))
    })?;
    parse_layer(value)
}

/// Research-only K/V plane selector.  The normal build does not expose this
/// enum or any intervention path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase54KvAttributionMode {
    /// Keep the production input untouched.
    Off,
    /// Round-trip only the key plane.
    KeyOnly,
    /// Round-trip only the value plane.
    ValueOnly,
    /// Round-trip key and value independently.
    KeyAndValue,
}

impl Phase54KvAttributionMode {
    /// Short selector spellings used by attribution reports.
    #[allow(non_upper_case_globals)]
    pub const KOnly: Self = Self::KeyOnly;
    #[allow(non_upper_case_globals)]
    pub const VOnly: Self = Self::ValueOnly;
    #[allow(non_upper_case_globals)]
    pub const KAndV: Self = Self::KeyAndValue;

    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub const fn transforms_key(self) -> bool {
        matches!(self, Self::KeyOnly | Self::KeyAndValue)
    }

    pub const fn transforms_value(self) -> bool {
        matches!(self, Self::ValueOnly | Self::KeyAndValue)
    }

    pub const fn identity_tag(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::KeyOnly => "key-only",
            Self::ValueOnly => "value-only",
            Self::KeyAndValue => "key-and-value",
        }
    }

    /// Parse the exact research environment selector.  An unset variable is
    /// the normal no-intervention path; any present unknown value fails closed.
    pub fn from_env() -> Result<Self, Phase54KvAttributionError> {
        match std::env::var_os("SLLM_PHASE54_KV_ATTRIBUTION") {
            None => Ok(Self::Off),
            Some(value) => {
                let value = value.to_str().ok_or_else(|| {
                    Phase54KvAttributionError::InvalidSelector(
                        "SLLM_PHASE54_KV_ATTRIBUTION is not valid UTF-8".to_owned(),
                    )
                })?;
                Self::parse(value)
            }
        }
    }

    pub fn parse(value: &str) -> Result<Self, Phase54KvAttributionError> {
        match value {
            "off" => Ok(Self::Off),
            "key-only" => Ok(Self::KeyOnly),
            "value-only" => Ok(Self::ValueOnly),
            "key-and-value" => Ok(Self::KeyAndValue),
            _ => Err(Phase54KvAttributionError::InvalidSelector(format!(
                "unknown Phase 54 KV attribution selector {value:?}"
            ))),
        }
    }
}

/// Compatibility alias emphasizing that the mode is a typed selector.
pub type Phase54KvAttributionSelector = Phase54KvAttributionMode;

/// Request-local Phase 54 configuration. The selector, resolved target
/// format, selected reviewed layer, and fixed recipe travel together so a
/// report cannot claim a different dispatch than the intervention that ran.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Phase54KvAttributionConfig {
    mode: Phase54KvAttributionMode,
    layer: Option<u32>,
    physical_variant: Option<KvFp8PhysicalVariant>,
    recipe: KvFp8ResearchScaleRecipe,
}

impl Phase54KvAttributionConfig {
    pub const fn off() -> Self {
        Self {
            mode: Phase54KvAttributionMode::Off,
            layer: None,
            physical_variant: None,
            recipe: KvFp8ResearchScaleRecipe::Floor,
        }
    }

    pub fn from_env(target: Option<&str>) -> Result<Self, Phase54KvAttributionError> {
        let mode = Phase54KvAttributionMode::from_env()?;
        let layer = mode.is_enabled().then(layer_from_env).transpose()?;
        Self::for_mode_at_layer(mode, target, layer)
    }

    /// Compatibility constructor for the original layer-three prototype.
    /// New callers should use [`Self::for_mode_at_layer`] so the selected
    /// layer is explicit in the research identity.
    pub fn for_mode(
        mode: Phase54KvAttributionMode,
        target: Option<&str>,
    ) -> Result<Self, Phase54KvAttributionError> {
        let layer = mode
            .is_enabled()
            .then_some(PHASE54_KV_ATTRIBUTION_FIXED_LAYER);
        Self::for_mode_at_layer(mode, target, layer)
    }

    pub fn for_mode_at_layer(
        mode: Phase54KvAttributionMode,
        target: Option<&str>,
        layer: Option<u32>,
    ) -> Result<Self, Phase54KvAttributionError> {
        let layer = if mode.is_enabled() {
            let layer = layer.ok_or_else(|| {
                Phase54KvAttributionError::InvalidLayer(format!(
                    "{PHASE54_KV_ATTRIBUTION_LAYER_ENV} is required when attribution is enabled"
                ))
            })?;
            if !is_allowed_layer(layer) {
                return Err(Phase54KvAttributionError::InvalidLayer(format!(
                    "{PHASE54_KV_ATTRIBUTION_LAYER_ENV} layer {layer} is not in the reviewed set"
                )));
            }
            Some(layer)
        } else {
            None
        };
        let physical_variant = if mode.is_enabled() {
            let target = target.ok_or_else(|| {
                Phase54KvAttributionError::UnsupportedTarget("<missing>".to_owned())
            })?;
            Some(physical_variant_for_target(target)?)
        } else {
            None
        };
        Ok(Self {
            mode,
            layer,
            physical_variant,
            recipe: KvFp8ResearchScaleRecipe::Floor,
        })
    }

    pub const fn mode(self) -> Phase54KvAttributionMode {
        self.mode
    }

    pub const fn layer(self) -> Option<u32> {
        self.layer
    }

    pub const fn physical_variant(self) -> Option<KvFp8PhysicalVariant> {
        self.physical_variant
    }

    pub const fn recipe(self) -> KvFp8ResearchScaleRecipe {
        self.recipe
    }

    pub const fn semantics(self) -> &'static str {
        PHASE54_KV_ATTRIBUTION_SEMANTICS
    }
}

impl Default for Phase54KvAttributionConfig {
    fn default() -> Self {
        Self::off()
    }
}

/// Host-side failures are explicit because an attribution run must never
/// silently fall back to the unmodified tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Phase54KvAttributionError {
    InvalidSelector(String),
    InvalidLayer(String),
    UnsupportedTarget(String),
    InvalidShape(String),
    OddByteLength { actual: usize },
    NonFiniteInput { index: usize },
    NonFiniteOutput { index: usize },
    Codec(KvFp8CodecError),
}

impl fmt::Display for Phase54KvAttributionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSelector(reason) => formatter.write_str(reason),
            Self::InvalidLayer(reason) => formatter.write_str(reason),
            Self::UnsupportedTarget(target) => {
                write!(
                    formatter,
                    "Phase 54 attribution target {target:?} is unsupported"
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
            Self::NonFiniteOutput { index } => {
                write!(
                    formatter,
                    "block16 roundtrip produced non-finite value at {index}"
                )
            }
            Self::Codec(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Phase54KvAttributionError {}

impl From<KvFp8CodecError> for Phase54KvAttributionError {
    fn from(error: KvFp8CodecError) -> Self {
        Self::Codec(error)
    }
}

/// Resolve the fixed target-to-format mapping used by the first attribution
/// control.  Unknown targets are rejected instead of being silently treated
/// as E4 or E5.
pub fn physical_variant_for_target(
    target: &str,
) -> Result<KvFp8PhysicalVariant, Phase54KvAttributionError> {
    match target {
        "gfx1030" => Ok(KvFp8PhysicalVariant::OcpE5M2),
        "gfx1201" => Ok(KvFp8PhysicalVariant::OcpE4M3Fn),
        _ => Err(Phase54KvAttributionError::UnsupportedTarget(
            target.to_owned(),
        )),
    }
}

/// Convert one contiguous row-major BF16 tensor through independent block16
/// quantize/dequantize and return BF16 bytes suitable for re-upload.  `rows`
/// and `columns` must preserve the head-dimension axis as `columns` so every
/// block is exactly sixteen consecutive head-dimension values.
pub fn roundtrip_bf16_plane(
    bytes: &[u8],
    rows: usize,
    columns: usize,
    physical_variant: KvFp8PhysicalVariant,
    recipe: KvFp8ResearchScaleRecipe,
) -> Result<(Vec<u8>, Vec<KvFp8ResearchBlockStats>), Phase54KvAttributionError> {
    if rows == 0 || columns == 0 {
        return Err(Phase54KvAttributionError::InvalidShape(
            "BF16 roundtrip shape must be non-zero".to_owned(),
        ));
    }
    if bytes.len() % 2 != 0 {
        return Err(Phase54KvAttributionError::OddByteLength {
            actual: bytes.len(),
        });
    }
    let expected_words = rows.checked_mul(columns).ok_or_else(|| {
        Phase54KvAttributionError::InvalidShape("BF16 roundtrip shape overflowed".to_owned())
    })?;
    let expected_bytes = expected_words.checked_mul(2).ok_or_else(|| {
        Phase54KvAttributionError::InvalidShape("BF16 roundtrip byte count overflowed".to_owned())
    })?;
    if bytes.len() != expected_bytes {
        return Err(Phase54KvAttributionError::InvalidShape(format!(
            "BF16 roundtrip bytes are {}, expected {expected_bytes}",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(expected_words);
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(Phase54KvAttributionError::NonFiniteInput { index });
        }
        values.push(value);
    }
    let (encoded, stats) =
        quantize_kv_fp8_block16_research(&values, rows, columns, physical_variant, recipe)?;
    let reconstructed = encoded
        .dequantize()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if !value.is_finite() {
                return Err(Phase54KvAttributionError::NonFiniteOutput { index });
            }
            Ok(f32_to_bf16_rne(value))
        })
        .collect::<Result<Vec<_>, Phase54KvAttributionError>>()?;
    let mut output = Vec::with_capacity(expected_bytes);
    for bits in reconstructed {
        output.extend_from_slice(&bits.to_le_bytes());
    }
    Ok((output, stats))
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return if bits & 0x007f_ffff == 0 {
            (bits >> 16) as u16
        } else {
            ((bits >> 16) as u16) | 0x0040
        };
    }
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f32_to_bf16_rne(*value).to_le_bytes())
            .collect()
    }

    #[test]
    fn selector_is_typed_and_fail_closed() {
        assert_eq!(
            Phase54KvAttributionMode::parse("off").unwrap(),
            Phase54KvAttributionMode::Off
        );
        assert_eq!(
            Phase54KvAttributionMode::parse("key-only").unwrap(),
            Phase54KvAttributionMode::KeyOnly
        );
        assert_eq!(
            Phase54KvAttributionMode::parse("value-only").unwrap(),
            Phase54KvAttributionMode::ValueOnly
        );
        assert_eq!(
            Phase54KvAttributionMode::parse("key-and-value").unwrap(),
            Phase54KvAttributionMode::KeyAndValue
        );
        assert!(Phase54KvAttributionMode::parse("bogus").is_err());
    }

    #[test]
    fn target_mapping_is_exact() {
        assert_eq!(
            physical_variant_for_target("gfx1030").unwrap(),
            KvFp8PhysicalVariant::OcpE5M2
        );
        assert_eq!(
            physical_variant_for_target("gfx1201").unwrap(),
            KvFp8PhysicalVariant::OcpE4M3Fn
        );
        assert!(physical_variant_for_target("gfx942").is_err());
    }

    #[test]
    fn attribution_layer_set_is_closed() {
        for layer in PHASE54_KV_ATTRIBUTION_ALLOWED_LAYERS {
            assert!(is_allowed_layer(*layer));
            assert_eq!(parse_layer(&layer.to_string()).unwrap(), *layer);
            let config = Phase54KvAttributionConfig::for_mode_at_layer(
                Phase54KvAttributionMode::KeyOnly,
                Some("gfx1030"),
                Some(*layer),
            )
            .unwrap();
            assert_eq!(config.layer(), Some(*layer));
        }
        for layer in [0, 1, 2, 4, 8, 12, 32, 255] {
            assert!(!is_allowed_layer(layer));
            assert!(parse_layer(&layer.to_string()).is_err());
            assert!(
                Phase54KvAttributionConfig::for_mode_at_layer(
                    Phase54KvAttributionMode::KeyOnly,
                    Some("gfx1030"),
                    Some(layer),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn off_mode_has_no_layer_or_target_requirement() {
        let config = Phase54KvAttributionConfig::for_mode_at_layer(
            Phase54KvAttributionMode::Off,
            None,
            None,
        )
        .unwrap();
        assert_eq!(config.layer(), None);
        assert_eq!(config.physical_variant(), None);
    }

    #[test]
    fn non_aligned_bf16_roundtrip_preserves_shape_and_finiteness() {
        let source = bf16_bytes(&[
            -3.25, -1.0, -0.25, 0.0, 0.125, 0.75, 1.5, 3.0, 7.0, 0.03125, 0.5, 2.0, -2.5, 1.25,
            0.0, 4.0, 8.0, -6.0,
        ]);
        let (output, stats) = roundtrip_bf16_plane(
            &source,
            2,
            9,
            KvFp8PhysicalVariant::OcpE5M2,
            KvFp8ResearchScaleRecipe::Floor,
        )
        .unwrap();
        assert_eq!(output.len(), source.len());
        assert_eq!(stats.len(), 2);
        assert!(
            output
                .chunks_exact(2)
                .all(|pair| (u16::from_le_bytes([pair[0], pair[1]]) & 0x7f80) != 0x7f80)
        );
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let source = bf16_bytes(&[1.0, f32::NAN]);
        assert!(matches!(
            roundtrip_bf16_plane(
                &source,
                1,
                2,
                KvFp8PhysicalVariant::OcpE4M3Fn,
                KvFp8ResearchScaleRecipe::Floor,
            ),
            Err(Phase54KvAttributionError::NonFiniteInput { index: 1 })
        ));
    }
}
