//! Versioned, fail-closed KV cache selection policy.
//!
//! Omission resolves to standard OCP MXFP8 E4M3 on the initial AMD target set.
//! The retired block16 experiment is rejected at this boundary. Explicit FP16
//! remains available as the rollback path.

use std::fmt;

use crate::{
    KvCacheEncoding, KvFp8Block16Descriptor, KvFp8PhysicalVariant, KvMxfp8Descriptor,
    QWEN35_4B_FINGERPRINT,
};

pub const KV_CACHE_SELECTION_POLICY_VERSION_V1: u32 = 1;
pub const KV_CACHE_SELECTION_POLICY_VERSION_V2: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheSelectionRequest<'a> {
    pub requested: Option<KvCacheEncoding>,
    pub exact_target: &'a str,
    pub model_fingerprint: &'a str,
    pub dense_text: bool,
    pub full_attention: bool,
    pub single_gpu: bool,
    pub head_dim: usize,
}

impl<'a> KvCacheSelectionRequest<'a> {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        requested: Option<KvCacheEncoding>,
        exact_target: &'a str,
        model_fingerprint: &'a str,
        dense_text: bool,
        full_attention: bool,
        single_gpu: bool,
        head_dim: usize,
    ) -> Self {
        Self {
            requested,
            exact_target,
            model_fingerprint,
            dense_text,
            full_attention,
            single_gpu,
            head_dim,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheSelectionSource {
    Explicit,
    Mxfp8E4Default,
    ModelFixedFp16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheSelection {
    requested: Option<KvCacheEncoding>,
    resolved: KvCacheEncoding,
    source: KvCacheSelectionSource,
    reason: &'static str,
    block16_descriptor: Option<KvFp8Block16Descriptor>,
    mxfp8_descriptor: Option<KvMxfp8Descriptor>,
    physical_variant: Option<KvFp8PhysicalVariant>,
    validated_exact_target: Option<&'static str>,
    validated_model_fingerprint: Option<&'static str>,
}

impl KvCacheSelection {
    pub const fn requested(self) -> Option<KvCacheEncoding> {
        self.requested
    }

    pub const fn resolved(self) -> KvCacheEncoding {
        self.resolved
    }

    pub const fn source(self) -> KvCacheSelectionSource {
        self.source
    }

    pub const fn reason(self) -> &'static str {
        self.reason
    }

    pub const fn block16_descriptor(self) -> Option<KvFp8Block16Descriptor> {
        self.block16_descriptor
    }

    pub const fn mxfp8_descriptor(self) -> Option<KvMxfp8Descriptor> {
        self.mxfp8_descriptor
    }

    pub const fn physical_variant(self) -> Option<KvFp8PhysicalVariant> {
        self.physical_variant
    }

    pub const fn validated_exact_target(self) -> Option<&'static str> {
        self.validated_exact_target
    }

    pub const fn validated_model_fingerprint(self) -> Option<&'static str> {
        self.validated_model_fingerprint
    }

    pub const fn policy_version(self) -> u32 {
        KV_CACHE_SELECTION_POLICY_VERSION_V2
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCacheSelectionError {
    UnsupportedExplicit {
        requested: KvCacheEncoding,
        exact_target: String,
        reason: &'static str,
    },
    UnsupportedDefaultTarget {
        exact_target: String,
    },
}

impl fmt::Display for KvCacheSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedExplicit {
                requested,
                exact_target,
                reason,
            } => write!(
                formatter,
                "explicit KV encoding {} is unsupported for exact target {exact_target}: {reason}",
                requested.canonical_name()
            ),
            Self::UnsupportedDefaultTarget { exact_target } => write!(
                formatter,
                "default KV encoding kv-mxfp8-e4 is unsupported for exact target {exact_target}"
            ),
        }
    }
}

impl std::error::Error for KvCacheSelectionError {}

pub fn resolve_kv_cache_selection(
    request: KvCacheSelectionRequest<'_>,
) -> Result<KvCacheSelection, KvCacheSelectionError> {
    let Some(requested) = request.requested else {
        if !is_default_mxfp8_e4_scope(request) {
            return Ok(KvCacheSelection {
                requested: None,
                resolved: KvCacheEncoding::Fp16,
                source: KvCacheSelectionSource::ModelFixedFp16,
                reason: "this model lane retains its reviewed fixed FP16 KV recipe",
                block16_descriptor: None,
                mxfp8_descriptor: None,
                physical_variant: None,
                validated_exact_target: None,
                validated_model_fingerprint: None,
            });
        }
        if !supports_default_mxfp8_e4(request.exact_target) {
            return Err(KvCacheSelectionError::UnsupportedDefaultTarget {
                exact_target: request.exact_target.to_owned(),
            });
        }
        let descriptor =
            KvMxfp8Descriptor::new(KvCacheEncoding::Mxfp8E4, KvFp8PhysicalVariant::OcpE4M3Fn)
                .expect("standard MXFP8 E4 has one canonical OCP physical variant");
        return Ok(KvCacheSelection {
            requested: None,
            resolved: KvCacheEncoding::Mxfp8E4,
            source: KvCacheSelectionSource::Mxfp8E4Default,
            reason: "standard OCP MXFP8 E4M3 is the configured default KV encoding",
            block16_descriptor: None,
            mxfp8_descriptor: Some(descriptor),
            physical_variant: Some(KvFp8PhysicalVariant::OcpE4M3Fn),
            validated_exact_target: canonical_exact_target(request.exact_target),
            validated_model_fingerprint: Some(QWEN35_4B_FINGERPRINT),
        });
    };

    if requested.is_kv_fp8_block16() {
        return Err(unsupported(
            requested,
            request.exact_target,
            "the block16 KV experiment has been retired; use kv-mxfp8-e4 or explicit fp16",
        ));
    }

    if !requested.is_kv_mxfp8() {
        return Ok(KvCacheSelection {
            requested: Some(requested),
            resolved: requested,
            source: KvCacheSelectionSource::Explicit,
            reason: "existing explicit KV encoding retained",
            block16_descriptor: None,
            mxfp8_descriptor: None,
            physical_variant: None,
            validated_exact_target: None,
            validated_model_fingerprint: None,
        });
    }

    if requested == KvCacheEncoding::Mxfp8E4 {
        if !is_default_mxfp8_e4_scope(request) {
            return Err(unsupported(
                requested,
                request.exact_target,
                "standard OCP MXFP8 E4M3 is currently scoped to the reviewed Qwen3.5-4B dense text lane",
            ));
        }
        if !supports_default_mxfp8_e4(request.exact_target) {
            return Err(unsupported(
                requested,
                request.exact_target,
                "standard OCP MXFP8 E4M3 is not enabled for this target",
            ));
        }
        let descriptor = KvMxfp8Descriptor::new(requested, KvFp8PhysicalVariant::OcpE4M3Fn)
            .expect("standard MXFP8 E4 has one canonical OCP physical variant");
        return Ok(KvCacheSelection {
            requested: Some(requested),
            resolved: requested,
            source: KvCacheSelectionSource::Explicit,
            reason: "explicit standard OCP MXFP8 E4M3 request validated for the target",
            block16_descriptor: None,
            mxfp8_descriptor: Some(descriptor),
            physical_variant: Some(KvFp8PhysicalVariant::OcpE4M3Fn),
            validated_exact_target: canonical_exact_target(request.exact_target),
            validated_model_fingerprint: Some(QWEN35_4B_FINGERPRINT),
        });
    }

    if request.model_fingerprint != QWEN35_4B_FINGERPRINT {
        return Err(unsupported(
            requested,
            request.exact_target,
            "low-bit KV is currently scoped to the reviewed Qwen3.5-4B model lock",
        ));
    }
    if !request.dense_text || !request.full_attention || !request.single_gpu {
        return Err(unsupported(
            requested,
            request.exact_target,
            "low-bit KV requires dense text, full attention, and one GPU",
        ));
    }
    if request.head_dim != 256 {
        return Err(unsupported(
            requested,
            request.exact_target,
            "low-bit KV is currently scoped to KV head dimension 256",
        ));
    }

    let (physical_variant, validated_exact_target) = match (requested, request.exact_target) {
        (KvCacheEncoding::Mxfp8E5, "gfx1030") => (KvFp8PhysicalVariant::OcpE5M2, "gfx1030"),
        _ => {
            return Err(unsupported(
                requested,
                request.exact_target,
                "logical encoding does not match the exact target's physical FP8 contract",
            ));
        }
    };
    let block16_descriptor = None;
    let mxfp8_descriptor = requested.is_kv_mxfp8().then(|| {
        KvMxfp8Descriptor::new(requested, physical_variant)
            .expect("selector target table contains compatible OCP MXFP8 pairs")
    });
    Ok(KvCacheSelection {
        requested: Some(requested),
        resolved: requested,
        source: KvCacheSelectionSource::Explicit,
        reason: "explicit low-bit KV request validated against exact target, model, and shape",
        block16_descriptor,
        mxfp8_descriptor,
        physical_variant: Some(physical_variant),
        validated_exact_target: Some(validated_exact_target),
        validated_model_fingerprint: Some(QWEN35_4B_FINGERPRINT),
    })
}

fn supports_default_mxfp8_e4(exact_target: &str) -> bool {
    matches!(
        exact_target,
        "gfx1030" | "gfx1201" | "gfx942" | "gfx942:sramecc+:xnack-"
    )
}

fn is_default_mxfp8_e4_scope(request: KvCacheSelectionRequest<'_>) -> bool {
    request.model_fingerprint == QWEN35_4B_FINGERPRINT
        && request.dense_text
        && request.full_attention
        && request.single_gpu
        && request.head_dim == 256
}

fn canonical_exact_target(exact_target: &str) -> Option<&'static str> {
    match exact_target {
        "gfx1030" => Some("gfx1030"),
        "gfx1201" => Some("gfx1201"),
        "gfx942" => Some("gfx942"),
        "gfx942:sramecc+:xnack-" => Some("gfx942:sramecc+:xnack-"),
        _ => None,
    }
}

fn unsupported(
    requested: KvCacheEncoding,
    exact_target: &str,
    reason: &'static str,
) -> KvCacheSelectionError {
    KvCacheSelectionError::UnsupportedExplicit {
        requested,
        exact_target: exact_target.to_owned(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(requested: Option<KvCacheEncoding>, target: &str) -> KvCacheSelectionRequest<'_> {
        KvCacheSelectionRequest::new(
            requested,
            target,
            QWEN35_4B_FINGERPRINT,
            true,
            true,
            true,
            256,
        )
    }

    #[test]
    fn omission_is_standard_mxfp8_e4_on_supported_targets() {
        for target in ["gfx942:sramecc+:xnack-", "gfx942", "gfx1201", "gfx1030"] {
            let selection = resolve_kv_cache_selection(request(None, target)).unwrap();
            assert_eq!(selection.requested(), None);
            assert_eq!(selection.resolved(), KvCacheEncoding::Mxfp8E4);
            assert_eq!(selection.source(), KvCacheSelectionSource::Mxfp8E4Default);
            assert_eq!(
                selection.physical_variant(),
                Some(KvFp8PhysicalVariant::OcpE4M3Fn)
            );
            assert_eq!(selection.validated_exact_target(), Some(target));
            assert_eq!(
                selection.validated_model_fingerprint(),
                Some(QWEN35_4B_FINGERPRINT)
            );
            assert_eq!(
                selection.policy_version(),
                KV_CACHE_SELECTION_POLICY_VERSION_V2
            );
        }
        assert!(resolve_kv_cache_selection(request(None, "unknown")).is_err());
    }

    #[test]
    fn explicit_standard_mxfp8_e4_uses_ocp_bytes_on_every_supported_target() {
        for target in ["gfx942:sramecc+:xnack-", "gfx942", "gfx1201", "gfx1030"] {
            let encoding = KvCacheEncoding::Mxfp8E4;
            let selection = resolve_kv_cache_selection(request(Some(encoding), target)).unwrap();
            assert_eq!(selection.resolved(), encoding);
            assert_eq!(selection.source(), KvCacheSelectionSource::Explicit);
            assert_eq!(
                selection.physical_variant(),
                Some(KvFp8PhysicalVariant::OcpE4M3Fn)
            );
            assert_eq!(selection.validated_exact_target(), Some(target));
            assert_eq!(
                selection.validated_model_fingerprint(),
                Some(QWEN35_4B_FINGERPRINT)
            );
            assert_eq!(selection.block16_descriptor(), None);
            assert_eq!(
                selection.mxfp8_descriptor().unwrap().physical_variant(),
                KvFp8PhysicalVariant::OcpE4M3Fn
            );
        }
    }

    #[test]
    fn explicit_block16_is_retired_for_every_target() {
        for encoding in [
            KvCacheEncoding::Fp8E4M3Block16,
            KvCacheEncoding::Fp8E5M2Block16,
        ] {
            for target in ["gfx942:sramecc+:xnack-", "gfx1201", "gfx1030"] {
                let error = resolve_kv_cache_selection(request(Some(encoding), target))
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("retired"));
            }
        }
    }

    #[test]
    fn mxfp8_e5_remains_a_scoped_explicit_comparison() {
        let selection =
            resolve_kv_cache_selection(request(Some(KvCacheEncoding::Mxfp8E5), "gfx1030")).unwrap();
        assert_eq!(
            selection.physical_variant(),
            Some(KvFp8PhysicalVariant::OcpE5M2)
        );
        assert!(
            resolve_kv_cache_selection(request(Some(KvCacheEncoding::Mxfp8E5), "gfx1201",))
                .is_err()
        );
    }

    #[test]
    fn non_dense_or_other_model_lanes_keep_their_fixed_fp16_recipe() {
        let mut other = request(None, "gfx1201");
        other.model_fingerprint = "sha256:other";
        let selection = resolve_kv_cache_selection(other).unwrap();
        assert_eq!(selection.resolved(), KvCacheEncoding::Fp16);
        assert_eq!(selection.source(), KvCacheSelectionSource::ModelFixedFp16);

        let mut moe = request(None, "gfx1201");
        moe.dense_text = false;
        let selection = resolve_kv_cache_selection(moe).unwrap();
        assert_eq!(selection.resolved(), KvCacheEncoding::Fp16);
        assert_eq!(selection.source(), KvCacheSelectionSource::ModelFixedFp16);
    }

    #[test]
    fn existing_explicit_encodings_keep_their_prior_meaning() {
        for encoding in [
            KvCacheEncoding::Fp16,
            KvCacheEncoding::Fp8E4M3Fn,
            KvCacheEncoding::Fp8E4M3FnStatic,
            KvCacheEncoding::Nvfp4,
        ] {
            let selection = resolve_kv_cache_selection(KvCacheSelectionRequest::new(
                Some(encoding),
                "unknown",
                "other-model",
                false,
                false,
                false,
                17,
            ))
            .unwrap();
            assert_eq!(selection.resolved(), encoding);
            assert_eq!(selection.source(), KvCacheSelectionSource::Explicit);
        }
    }
}
