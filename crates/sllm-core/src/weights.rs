//! Host-only Qwen3.5 weight registry and deterministic load-plan construction.
//!
//! The builder consumes descriptors that have already passed model-lock and
//! safetensors validation. It never reads tensor payloads and deliberately does
//! not duplicate the model parser, shape catalog, cache hasher, or range reader.

use crate::model::{
    LayerType, LockedFile, ModelLock, TensorDType, TensorDescriptor, VerifiedCache,
    reviewed_qwen35_spec,
};
use crate::{
    BufferRange, ExecutionQueue, ExecutionSession, ExecutionState, GgufRecipeEncoding,
    GgufTensorBinding, GgufTensorType, VerifiedDerivedGguf, VerifiedGguf,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

pub const WEIGHT_LOAD_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
/// Logical consumer layer reserved for the single Qwen3.5 MTP decoder block.
/// Keeping it outside the text decoder's `0..32` namespace makes combined
/// plans one-to-one without leaking an MTP-specific key type into executors.
pub const QWEN35_MTP_CONSUMER_LAYER: u64 = 32;

const PLAN_DOMAIN: &[u8] = b"sLLM-weight-load-plan-v1\0";
const QWEN_SCHEMA_VERSION: &str = "model-lock-v1";
#[cfg(test)]
const QWEN_REPO_ID: &str = crate::model::QWEN35_4B_REPO_ID;
#[cfg(test)]
const QWEN_REVISION: &str = crate::model::QWEN35_4B_REVISION;
#[cfg(test)]
const QWEN_FINGERPRINT: &str = crate::model::QWEN35_4B_FINGERPRINT;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightPlanError(String);

impl WeightPlanError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for WeightPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid weight load plan: {}", self.0)
    }
}

impl std::error::Error for WeightPlanError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightUploadError(String);

impl WeightUploadError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for WeightUploadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid verified weight upload: {}", self.0)
    }
}

impl std::error::Error for WeightUploadError {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WeightClassification {
    Required,
    ConfigConditional,
    KnownUnconsumed,
}

impl WeightClassification {
    fn tag(self) -> u8 {
        match self {
            Self::Required => 1,
            Self::ConfigConditional => 2,
            Self::KnownUnconsumed => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WeightConsumer {
    EmbeddingAndTiedOutput,
    Embedding,
    OutputProjection,
    FinalNorm,
    InputNorm,
    PostAttentionNorm,
    MlpGate,
    MlpUp,
    MlpDown,
    GdnInProjQkv,
    GdnInProjZ,
    GdnInProjB,
    GdnInProjA,
    GdnConv1d,
    GdnALog,
    GdnDtBias,
    GdnNorm,
    GdnOutProj,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    AttentionQNorm,
    AttentionKNorm,
    PreFeedforwardNorm,
    PostFeedforwardNorm,
    LayerScalar,
    AttentionKAndV,
    MtpFusion,
    MtpEmbeddingNorm,
    MtpHiddenNorm,
    MtpFinalNorm,
    MoeRouter,
    MoeLayerBlob,
    Gemma4MoePreFeedforwardNorm2,
    Gemma4MoePostFeedforwardNorm1,
    Gemma4MoePostFeedforwardNorm2,
    Gemma4MoeRouterScale,
    Gemma4MoeRouterPerExpertScale,
    Gemma4MoeLayerBlob,
    Gemma4MtpPreProjection,
    Gemma4MtpPostProjection,
}

impl WeightConsumer {
    fn tag(self) -> u8 {
        match self {
            Self::EmbeddingAndTiedOutput => 1,
            Self::FinalNorm => 2,
            Self::InputNorm => 3,
            Self::PostAttentionNorm => 4,
            Self::MlpGate => 5,
            Self::MlpUp => 6,
            Self::MlpDown => 7,
            Self::GdnInProjQkv => 8,
            Self::GdnInProjZ => 9,
            Self::GdnInProjB => 10,
            Self::GdnInProjA => 11,
            Self::GdnConv1d => 12,
            Self::GdnALog => 13,
            Self::GdnDtBias => 14,
            Self::GdnNorm => 15,
            Self::GdnOutProj => 16,
            Self::AttentionQ => 17,
            Self::AttentionK => 18,
            Self::AttentionV => 19,
            Self::AttentionO => 20,
            Self::AttentionQNorm => 21,
            Self::AttentionKNorm => 22,
            Self::Embedding => 23,
            Self::OutputProjection => 24,
            Self::PreFeedforwardNorm => 25,
            Self::PostFeedforwardNorm => 26,
            Self::LayerScalar => 27,
            Self::AttentionKAndV => 28,
            Self::MtpFusion => 29,
            Self::MtpEmbeddingNorm => 30,
            Self::MtpHiddenNorm => 31,
            Self::MtpFinalNorm => 32,
            Self::MoeRouter => 33,
            Self::MoeLayerBlob => 34,
            Self::Gemma4MoePreFeedforwardNorm2 => 35,
            Self::Gemma4MoePostFeedforwardNorm1 => 36,
            Self::Gemma4MoePostFeedforwardNorm2 => 37,
            Self::Gemma4MoeRouterScale => 38,
            Self::Gemma4MoeRouterPerExpertScale => 39,
            Self::Gemma4MoeLayerBlob => 40,
            Self::Gemma4MtpPreProjection => 41,
            Self::Gemma4MtpPostProjection => 42,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct WeightConsumerKey {
    pub layer: Option<u64>,
    pub role: WeightConsumer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightLoadChunk {
    pub source_offset: u64,
    pub destination_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightLoadEntry {
    pub tensor_name: String,
    pub classification: WeightClassification,
    pub consumer: Option<WeightConsumerKey>,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub source_file: String,
    pub locked_file_size: u64,
    pub locked_file_sha256: String,
    pub source_range: [u64; 2],
    pub destination_start: Option<u64>,
    pub chunks: Vec<WeightLoadChunk>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightLoadPlan {
    pub schema_version: String,
    pub repo_id: String,
    pub resolved_revision: String,
    pub lock_fingerprint: String,
    pub tied_embeddings: bool,
    pub chunk_size: u64,
    pub total_destination_bytes: u64,
    pub entries: Vec<WeightLoadEntry>,
    digest: [u8; 32],
}

pub(crate) struct VerifiedWeightPlanMetadata {
    pub schema_version: String,
    pub repo_id: String,
    pub resolved_revision: String,
    pub lock_fingerprint: String,
    pub tied_embeddings: bool,
    pub chunk_size: u64,
    pub total_destination_bytes: u64,
}

impl WeightLoadPlan {
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn digest_hex(&self) -> String {
        let mut output = String::with_capacity(7 + self.digest.len() * 2);
        output.push_str("sha256:");
        for byte in self.digest {
            use std::fmt::Write as _;
            write!(output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }

    fn recompute_digest(&self) -> Result<[u8; 32], WeightPlanError> {
        digest_plan(
            &PlanDigestHeader {
                schema_version: &self.schema_version,
                repo_id: &self.repo_id,
                resolved_revision: &self.resolved_revision,
                fingerprint: &self.lock_fingerprint,
                tied_embeddings: self.tied_embeddings,
                chunk_size: self.chunk_size,
                total_destination_bytes: self.total_destination_bytes,
            },
            &self.entries,
        )
    }

    pub(crate) fn has_valid_digest(&self) -> Result<bool, WeightPlanError> {
        Ok(self.recompute_digest()? == self.digest)
    }

    pub(crate) fn from_verified_entries(
        metadata: VerifiedWeightPlanMetadata,
        entries: Vec<WeightLoadEntry>,
    ) -> Result<Self, WeightPlanError> {
        let mut plan = Self {
            schema_version: metadata.schema_version,
            repo_id: metadata.repo_id,
            resolved_revision: metadata.resolved_revision,
            lock_fingerprint: metadata.lock_fingerprint,
            tied_embeddings: metadata.tied_embeddings,
            chunk_size: metadata.chunk_size,
            total_destination_bytes: metadata.total_destination_bytes,
            entries,
            digest: [0; 32],
        };
        plan.digest = plan.recompute_digest()?;
        Ok(plan)
    }
}

/// A single verified tensor upload request.
///
/// `destination` is the exact tensor-sized target range. Plan-global offsets
/// are translated relative to that range, so callers may use either a packed
/// model allocation or a tensor-specific allocation without changing the
/// source plan.
pub struct WeightUploadRequest<'a> {
    pub plan: &'a WeightLoadPlan,
    pub expected_plan_digest: [u8; 32],
    pub cache: &'a VerifiedCache,
    pub tensor_name: &'a str,
    pub expected_dtype: TensorDType,
    pub session: &'a ExecutionSession,
    pub queue: &'a ExecutionQueue,
    pub destination: BufferRange,
    pub completion_timeout: Duration,
}

pub struct GgufWeightUploadRequest<'a> {
    pub plan: &'a WeightLoadPlan,
    pub expected_plan_digest: [u8; 32],
    pub source: &'a VerifiedGgufWeightSource,
    pub tensor_name: &'a str,
    pub expected_dtype: TensorDType,
    pub session: &'a ExecutionSession,
    pub queue: &'a ExecutionQueue,
    pub destination: BufferRange,
    pub completion_timeout: Duration,
}

pub struct VerifiedGgufWeightSource {
    lock_fingerprint: String,
    file_sha256: String,
    descriptors: BTreeMap<String, TensorDescriptor>,
    recipe_bindings: BTreeMap<String, GgufTensorBinding>,
    gguf: VerifiedGguf,
}

pub struct VerifiedGgufGemmaSource {
    lock_fingerprint: String,
    repository: String,
    resolved_revision: String,
    file_sha256: String,
    tensors: BTreeMap<String, crate::QuantizedTensorDescriptor>,
    kv_scales: BTreeMap<u32, crate::StaticFp8KvScale>,
    gguf: VerifiedGguf,
}

impl VerifiedGgufGemmaSource {
    pub fn gguf(&self) -> &VerifiedGguf {
        &self.gguf
    }

    pub fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }

    pub fn repository(&self) -> &str {
        &self.repository
    }

    pub fn resolved_revision(&self) -> &str {
        &self.resolved_revision
    }

    pub fn tensor(&self, name: &str) -> Option<&crate::QuantizedTensorDescriptor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &crate::QuantizedTensorDescriptor> {
        self.tensors.values()
    }

    pub fn recipe_digest(&self) -> &str {
        &self
            .gguf
            .extension()
            .expect("verified Gemma GGUF has an extension")
            .recipe_sha256
    }

    pub fn kv_scale(&self, layer: u32) -> Option<crate::StaticFp8KvScale> {
        self.kv_scales.get(&layer).copied()
    }

    pub fn resident_bytes(&self, logical_name: &str) -> Result<Vec<u8>, WeightPlanError> {
        let descriptor = self
            .tensor(logical_name)
            .ok_or_else(|| WeightPlanError::invalid("GGUF Gemma tensor is absent"))?;
        let value = self
            .gguf
            .tensor(&descriptor.source_name)
            .ok_or_else(|| WeightPlanError::invalid("GGUF Gemma value tensor is absent"))?;
        let length = usize::try_from(value.byte_length())
            .map_err(|_| WeightPlanError::invalid("GGUF Gemma value is too large"))?;
        let encoded = self
            .gguf
            .read_tensor_range(&descriptor.source_name, 0, length)
            .map_err(|error| WeightPlanError::invalid(error.to_string()))?;
        let mut output = Vec::new();
        match descriptor.encoding {
            crate::QuantizedTensorEncoding::UnquantizedBf16 => return Ok(encoded),
            crate::QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale => {
                output = encoded;
                let scale = descriptor
                    .scale_planes
                    .iter()
                    .find(|plane| plane.role == crate::ScalePlaneRole::WeightChannel)
                    .ok_or_else(|| WeightPlanError::invalid("GGUF Gemma FP8 scale is absent"))?;
                let scale_info = self.gguf.tensor(&scale.source_name).ok_or_else(|| {
                    WeightPlanError::invalid("GGUF Gemma FP8 scale tensor is absent")
                })?;
                let scale_length = usize::try_from(scale_info.byte_length())
                    .map_err(|_| WeightPlanError::invalid("GGUF Gemma FP8 scale is too large"))?;
                let bytes = self
                    .gguf
                    .read_tensor_range(&scale.source_name, 0, scale_length)
                    .map_err(|error| WeightPlanError::invalid(error.to_string()))?;
                if bytes.len() % 2 != 0 {
                    return Err(WeightPlanError::invalid(
                        "GGUF Gemma FP8 BF16 scale byte count is odd",
                    ));
                }
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    output.extend_from_slice(&f32::from_bits(u32::from(bits) << 16).to_le_bytes());
                }
            }
            crate::QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer => {
                if encoded.len() % 36 != 0 {
                    return Err(WeightPlanError::invalid(
                        "GGUF NVFP4 standard block byte count differs",
                    ));
                }
                output.reserve(encoded.len());
                for block in encoded.chunks_exact(36) {
                    for standard in block[4..].chunks_exact(8) {
                        append_adjacent_nibbles(standard, 16, &mut output);
                    }
                }
                for block in encoded.chunks_exact(36) {
                    output.extend_from_slice(&block[..4]);
                }
                while output.len() & 3 != 0 {
                    output.push(0);
                }
                for role in [
                    crate::ScalePlaneRole::WeightOuter,
                    crate::ScalePlaneRole::InputOuter,
                ] {
                    let scale = descriptor
                        .scale_planes
                        .iter()
                        .find(|plane| plane.role == role)
                        .ok_or_else(|| {
                            WeightPlanError::invalid("GGUF Gemma outer scale is absent")
                        })?;
                    let scale_info = self.gguf.tensor(&scale.source_name).ok_or_else(|| {
                        WeightPlanError::invalid("GGUF Gemma outer scale tensor is absent")
                    })?;
                    let scale_length = usize::try_from(scale_info.byte_length()).map_err(|_| {
                        WeightPlanError::invalid("GGUF Gemma outer scale is too large")
                    })?;
                    let bytes = self
                        .gguf
                        .read_tensor_range(&scale.source_name, 0, scale_length)
                        .map_err(|error| WeightPlanError::invalid(error.to_string()))?;
                    if bytes.len() != 4 {
                        return Err(WeightPlanError::invalid(
                            "GGUF Gemma outer scale is not one F32",
                        ));
                    }
                    output.extend_from_slice(&bytes);
                }
            }
            _ => {
                return Err(WeightPlanError::invalid(
                    "GGUF Gemma tensor has an unsupported encoding",
                ));
            }
        }
        Ok(output)
    }

    pub fn build_weight_load_plan(
        &self,
        lock: &crate::Gemma4ModelLock,
    ) -> Result<WeightLoadPlan, WeightPlanError> {
        if self.lock_fingerprint != lock.fingerprint() {
            return Err(WeightPlanError::invalid(
                "GGUF Gemma source and lock fingerprints differ",
            ));
        }
        build_gguf_gemma_plan(lock, self)
    }
}

fn append_adjacent_nibbles(standard: &[u8], elements: usize, output: &mut Vec<u8>) {
    let half = elements / 2;
    debug_assert_eq!(standard.len(), half);
    for adjacent in (0..elements).step_by(2) {
        let code = |index: usize| {
            if index < half {
                standard[index] & 0x0f
            } else {
                standard[index - half] >> 4
            }
        };
        output.push(code(adjacent) | code(adjacent + 1) << 4);
    }
}

impl VerifiedGgufWeightSource {
    pub fn gguf(&self) -> &VerifiedGguf {
        &self.gguf
    }

    pub fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }

    pub fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, crate::GgufError> {
        self.gguf.read_tensor_range(tensor_name, offset, length)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.descriptors.get(name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &TensorDescriptor> {
        self.descriptors.values()
    }

    pub fn recipe_binding(&self, logical_name: &str) -> Option<&GgufTensorBinding> {
        self.recipe_bindings.get(logical_name)
    }

    pub fn has_fp8_recipe(&self) -> bool {
        !self.recipe_bindings.is_empty()
            && self.recipe_bindings.values().all(|binding| {
                matches!(
                    binding.encoding,
                    GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale
                        | GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale
                )
            })
    }

    pub fn has_mx_weight_activation_recipe(&self) -> bool {
        !self.recipe_bindings.is_empty()
            && self.recipe_bindings.values().all(|binding| {
                matches!(
                    binding.encoding,
                    GgufRecipeEncoding::Mxfp8E4m3Block32E8m0
                        | GgufRecipeEncoding::Mxfp6E3m2Block32E8m0
                )
            })
    }

    pub fn has_quantized_linear_recipe(&self) -> bool {
        !self.recipe_bindings.is_empty()
            && self.recipe_bindings.values().all(|binding| {
                matches!(
                    binding.encoding,
                    GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale
                        | GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale
                        | GgufRecipeEncoding::Nvfp4E2m1Block16E4m3fnF32Outer
                        | GgufRecipeEncoding::Mxfp4E2m1Block32E8m0
                        | GgufRecipeEncoding::Mxfp8E4m3Block32E8m0
                        | GgufRecipeEncoding::Mxfp6E3m2Block32E8m0
                )
            })
    }

    /// True when a recipe contains more than one resident family (for
    /// example Qwen3.8's NVFP4 MLP plus FP8 attention/lm_head).  Unbound
    /// BF16 tensors are intentionally allowed; they are the expected mixed
    /// precision remainder.
    pub fn has_mixed_quantized_linear_recipe(&self) -> bool {
        let families: BTreeSet<_> = self
            .recipe_bindings
            .values()
            .map(|binding| match binding.encoding {
                GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale
                | GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale => "fp8",
                GgufRecipeEncoding::Nvfp4E2m1Block16E4m3fnF32Outer => "nvfp4",
                GgufRecipeEncoding::Mxfp4E2m1Block32E8m0 => "mxfp4",
                GgufRecipeEncoding::Mxfp8E4m3Block32E8m0 => "mxfp8",
                GgufRecipeEncoding::Mxfp6E3m2Block32E8m0 => "mxfp6",
            })
            .collect();
        families.len() > 1
    }

    pub fn mx_weight_activation_encoding_name(&self) -> Option<&'static str> {
        if !self.has_mx_weight_activation_recipe() {
            return None;
        }
        let mut encodings = self
            .recipe_bindings
            .values()
            .map(|binding| binding.encoding);
        let first = encodings.next()?;
        if encodings.any(|encoding| encoding != first) {
            return Some("mixed-mxfp-invalid");
        }
        Some(match first {
            GgufRecipeEncoding::Mxfp8E4m3Block32E8m0 => "mxfp8-e4m3-w8a8",
            GgufRecipeEncoding::Mxfp6E3m2Block32E8m0 => "mxfp6-e3m2-w6a6",
            _ => unreachable!("MX recipe predicate excluded other encodings"),
        })
    }

    pub fn recipe_digest(&self) -> Option<&str> {
        self.gguf
            .extension()
            .map(|extension| extension.recipe_sha256.as_str())
    }

    pub fn build_qwen_weight_load_plan(
        &self,
        lock: &ModelLock,
        selection: QwenComponentSelection,
    ) -> Result<WeightLoadPlan, WeightPlanError> {
        if self.lock_fingerprint != lock.fingerprint() {
            return Err(WeightPlanError::invalid(
                "GGUF source and Qwen lock fingerprints differ",
            ));
        }
        let locked_file = LockedFile {
            path: self.gguf.path().display().to_string(),
            size_bytes: self.gguf.file_size(),
            sha256: self
                .file_sha256
                .strip_prefix("sha256:")
                .ok_or_else(|| WeightPlanError::invalid("GGUF SHA-256 prefix differs"))?
                .to_owned(),
            git_blob: String::new(),
            source_page_url: String::new(),
            download_url: String::new(),
            lfs_oid: None,
        };
        if self.recipe_bindings.is_empty() {
            let locked_files = BTreeMap::from([(locked_file.path.as_str(), &locked_file)]);
            let mut plan = build_qwen_component_weight_load_plan_inner(
                lock,
                self.descriptors.values(),
                selection,
                &locked_files,
            )?;
            plan.schema_version = "gguf-model-plan-v1".to_owned();
            plan.digest = plan.recompute_digest()?;
            Ok(plan)
        } else {
            build_qwen_gguf_quantized_plan(lock, &self.descriptors, selection, &locked_file)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightUploadReceipt {
    pub plan_digest: [u8; 32],
    pub tensor_name: String,
    pub dtype: TensorDType,
    pub source_range: [u64; 2],
    pub destination_offset: u64,
    pub byte_length: u64,
    pub chunks_uploaded: usize,
    pub peak_host_staging_bytes: u64,
}

pub(crate) trait WeightRangeSource {
    fn lock_fingerprint(&self) -> &str;
    fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor>;
    fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, WeightUploadError>;
}

impl WeightRangeSource for VerifiedCache {
    fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor> {
        self.tensor(tensor_name)
    }

    fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, WeightUploadError> {
        self.read_tensor_range(tensor_name, offset, length)
            .map_err(|error| WeightUploadError::invalid(error.to_string()))
    }
}

impl WeightRangeSource for VerifiedGgufWeightSource {
    fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor> {
        self.descriptors.get(tensor_name)
    }

    fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, WeightUploadError> {
        self.gguf
            .read_tensor_range(tensor_name, offset, length)
            .map_err(|error| WeightUploadError::invalid(error.to_string()))
    }
}

/// Upload one load-plan entry through the backend-neutral transfer API.
///
/// Only one plan chunk is resident in host staging memory at a time. A failed
/// call yields no receipt; callers must discard the partially written target.
pub fn upload_verified_weight(
    request: WeightUploadRequest<'_>,
) -> Result<WeightUploadReceipt, WeightUploadError> {
    if request.completion_timeout.is_zero() {
        return Err(WeightUploadError::invalid(
            "completion timeout must be non-zero",
        ));
    }
    if request.queue.session_id() != request.session.id()
        || request.destination.buffer().session_id() != request.session.id()
    {
        return Err(WeightUploadError::invalid(
            "queue and destination must belong to the upload session",
        ));
    }
    let max_transfer_bytes = request
        .session
        .max_transfer_bytes()
        .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
    let destination = request.destination.clone();
    upload_weight_from_source(
        request.plan,
        request.expected_plan_digest,
        request.cache,
        request.tensor_name,
        request.expected_dtype,
        destination.offset_bytes(),
        destination.size_bytes(),
        max_transfer_bytes,
        |relative_offset, bytes| {
            let absolute_offset = destination
                .offset_bytes()
                .checked_add(relative_offset)
                .ok_or_else(|| WeightUploadError::invalid("destination offset overflow"))?;
            let range = destination
                .buffer()
                .range(
                    absolute_offset,
                    u64::try_from(bytes.len()).map_err(|_| {
                        WeightUploadError::invalid("upload byte length does not fit u64")
                    })?,
                )
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
            let mut transfer = request
                .session
                .upload(request.queue, range, bytes)
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
            match transfer
                .wait(request.completion_timeout)
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?
            {
                ExecutionState::Success => Ok(()),
                ExecutionState::Pending => Err(WeightUploadError::invalid(
                    "weight upload remained pending after wait",
                )),
                ExecutionState::Failure => {
                    Err(WeightUploadError::invalid("weight upload reported failure"))
                }
            }
        },
    )
}

pub fn upload_verified_gguf_weight(
    request: GgufWeightUploadRequest<'_>,
) -> Result<WeightUploadReceipt, WeightUploadError> {
    if request.completion_timeout.is_zero() {
        return Err(WeightUploadError::invalid(
            "completion timeout must be non-zero",
        ));
    }
    if request.queue.session_id() != request.session.id()
        || request.destination.buffer().session_id() != request.session.id()
    {
        return Err(WeightUploadError::invalid(
            "queue and destination must belong to the upload session",
        ));
    }
    let max_transfer_bytes = request
        .session
        .max_transfer_bytes()
        .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
    let destination = request.destination.clone();
    upload_weight_from_source(
        request.plan,
        request.expected_plan_digest,
        request.source,
        request.tensor_name,
        request.expected_dtype,
        destination.offset_bytes(),
        destination.size_bytes(),
        max_transfer_bytes,
        |relative_offset, bytes| {
            let absolute_offset = destination
                .offset_bytes()
                .checked_add(relative_offset)
                .ok_or_else(|| WeightUploadError::invalid("destination offset overflow"))?;
            let range = destination
                .buffer()
                .range(
                    absolute_offset,
                    u64::try_from(bytes.len()).map_err(|_| {
                        WeightUploadError::invalid("upload byte length does not fit u64")
                    })?,
                )
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
            let mut transfer = request
                .session
                .upload(request.queue, range, bytes)
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
            match transfer
                .wait(request.completion_timeout)
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?
            {
                ExecutionState::Success => Ok(()),
                ExecutionState::Pending => Err(WeightUploadError::invalid(
                    "weight upload remained pending after wait",
                )),
                ExecutionState::Failure => {
                    Err(WeightUploadError::invalid("weight upload reported failure"))
                }
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_weight_from_source<S, F>(
    plan: &WeightLoadPlan,
    expected_plan_digest: [u8; 32],
    source: &S,
    tensor_name: &str,
    expected_dtype: TensorDType,
    destination_offset: u64,
    destination_size: u64,
    max_transfer_bytes: u64,
    mut upload: F,
) -> Result<WeightUploadReceipt, WeightUploadError>
where
    S: WeightRangeSource,
    F: FnMut(u64, Arc<[u8]>) -> Result<(), WeightUploadError>,
{
    let recomputed = plan
        .recompute_digest()
        .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
    if recomputed != *plan.digest() || expected_plan_digest != *plan.digest() {
        return Err(WeightUploadError::invalid(
            "weight plan identity or content digest differs",
        ));
    }
    if source.lock_fingerprint() != plan.lock_fingerprint {
        return Err(WeightUploadError::invalid(
            "verified cache fingerprint differs from the weight plan",
        ));
    }
    if max_transfer_bytes == 0 {
        return Err(WeightUploadError::invalid(
            "backend transfer limit must be non-zero",
        ));
    }
    let entry = plan
        .entries
        .binary_search_by(|entry| entry.tensor_name.as_str().cmp(tensor_name))
        .ok()
        .and_then(|index| plan.entries.get(index))
        .ok_or_else(|| WeightUploadError::invalid("tensor target is absent from the load plan"))?;
    if entry.classification == WeightClassification::KnownUnconsumed
        || entry.destination_start.is_none()
        || entry.chunks.is_empty()
    {
        return Err(WeightUploadError::invalid(
            "tensor target is not a loadable main-text weight",
        ));
    }
    if entry.dtype != expected_dtype {
        return Err(WeightUploadError::invalid(
            "tensor dtype differs from the requested upload dtype",
        ));
    }
    let descriptor = source
        .tensor(tensor_name)
        .ok_or_else(|| WeightUploadError::invalid("tensor is absent from the verified cache"))?;
    if descriptor.tensor_name != entry.tensor_name
        || descriptor.source_file != entry.source_file
        || descriptor.dtype != entry.dtype
        || descriptor.shape != entry.shape
        || descriptor.absolute_byte_range != entry.source_range
    {
        return Err(WeightUploadError::invalid(
            "verified tensor descriptor differs from the load-plan entry",
        ));
    }
    let source_size = entry.source_range[1]
        .checked_sub(entry.source_range[0])
        .ok_or_else(|| WeightUploadError::invalid("tensor source range underflow"))?;
    if source_size == 0 || destination_size != source_size {
        return Err(WeightUploadError::invalid(
            "destination range must exactly match the tensor byte range",
        ));
    }
    let plan_destination = entry
        .destination_start
        .expect("loadable entry has destination");
    let mut expected_relative = 0_u64;
    let mut peak_host_staging_bytes = 0_u64;
    for chunk in &entry.chunks {
        if chunk.byte_length == 0
            || chunk.byte_length > plan.chunk_size
            || chunk.byte_length > max_transfer_bytes
        {
            return Err(WeightUploadError::invalid(
                "weight chunk exceeds a non-zero plan/backend transfer bound",
            ));
        }
        let source_relative = chunk
            .source_offset
            .checked_sub(entry.source_range[0])
            .ok_or_else(|| WeightUploadError::invalid("chunk source precedes tensor range"))?;
        let destination_relative = chunk
            .destination_offset
            .checked_sub(plan_destination)
            .ok_or_else(|| WeightUploadError::invalid("chunk destination precedes tensor range"))?;
        if source_relative != expected_relative || destination_relative != expected_relative {
            return Err(WeightUploadError::invalid(
                "weight chunks are not contiguous source/destination peers",
            ));
        }
        let next = expected_relative
            .checked_add(chunk.byte_length)
            .ok_or_else(|| WeightUploadError::invalid("weight chunk range overflow"))?;
        if next > source_size {
            return Err(WeightUploadError::invalid(
                "weight chunk exceeds the tensor range",
            ));
        }
        let length = usize::try_from(chunk.byte_length)
            .map_err(|_| WeightUploadError::invalid("weight chunk length does not fit usize"))?;
        let bytes = source.read_tensor_range(tensor_name, source_relative, length)?;
        if bytes.len() != length {
            return Err(WeightUploadError::invalid(
                "verified range reader returned a short or long chunk",
            ));
        }
        peak_host_staging_bytes = peak_host_staging_bytes.max(chunk.byte_length);
        upload(destination_relative, Arc::<[u8]>::from(bytes))?;
        expected_relative = next;
    }
    if expected_relative != source_size {
        return Err(WeightUploadError::invalid(
            "weight chunks do not cover the exact tensor range",
        ));
    }
    Ok(WeightUploadReceipt {
        plan_digest: expected_plan_digest,
        tensor_name: entry.tensor_name.clone(),
        dtype: entry.dtype,
        source_range: entry.source_range,
        destination_offset,
        byte_length: source_size,
        chunks_uploaded: entry.chunks.len(),
        peak_host_staging_bytes,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QwenComponentSelection {
    pub text: bool,
    pub vision: bool,
    pub mtp: bool,
}

impl QwenComponentSelection {
    pub const TEXT_ONLY: Self = Self {
        text: true,
        vision: false,
        mtp: false,
    };
    pub const TEXT_AND_MTP: Self = Self {
        text: true,
        vision: false,
        mtp: true,
    };
    pub const ALL: Self = Self {
        text: true,
        vision: true,
        mtp: true,
    };
    pub const MTP_ONLY: Self = Self {
        text: false,
        vision: false,
        mtp: true,
    };
}

/// Build the fixed Qwen3.5 load plan from already-verified descriptors.
///
/// The legacy entry point keeps the text-only contract. Component-enabled
/// callers use [`build_qwen_component_weight_load_plan`] so selected vision or
/// MTP tensors become required and receive exact destination ranges.
pub fn build_weight_load_plan<'a>(
    lock: &ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
) -> Result<WeightLoadPlan, WeightPlanError> {
    build_qwen_component_weight_load_plan(lock, descriptors, QwenComponentSelection::TEXT_ONLY)
}

pub fn build_qwen_component_weight_load_plan<'a>(
    lock: &ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
    selection: QwenComponentSelection,
) -> Result<WeightLoadPlan, WeightPlanError> {
    let locked_files = locked_file_map(lock)?;
    build_qwen_component_weight_load_plan_inner(lock, descriptors, selection, &locked_files)
}

fn build_qwen_component_weight_load_plan_inner<'a>(
    lock: &ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
    selection: QwenComponentSelection,
    locked_files: &BTreeMap<&str, &LockedFile>,
) -> Result<WeightLoadPlan, WeightPlanError> {
    validate_fixed_lock(lock)?;
    let mut by_name = BTreeMap::new();
    for descriptor in descriptors {
        if by_name
            .insert(descriptor.tensor_name.as_str(), descriptor)
            .is_some()
        {
            return Err(WeightPlanError::invalid(format!(
                "duplicate tensor descriptor: {}",
                descriptor.tensor_name
            )));
        }
    }

    let architecture = &lock.model.architecture;
    let config = &architecture.text_config;
    let mut selected_consumers = BTreeSet::new();
    if selection.text {
        selected_consumers.extend(expected_consumers(
            &config.layer_types,
            config.tie_word_embeddings,
        ));
    }
    if selection.mtp {
        selected_consumers.extend(expected_mtp_consumers());
        selected_consumers.insert(WeightConsumerKey {
            layer: None,
            role: WeightConsumer::EmbeddingAndTiedOutput,
        });
    }
    let mut observed_consumers = BTreeSet::new();
    let mut vision_count = 0_u64;
    let mut mtp_count = 0_u64;
    let mut entries = Vec::with_capacity(by_name.len());
    let mut destination_cursor = 0_u64;

    for descriptor in by_name.values() {
        validate_descriptor(descriptor, locked_files)?;
        let (mut classification, mut consumer) = classify_descriptor(
            descriptor,
            &config.layer_types,
            &architecture.vision.tensor_prefix,
            &architecture.mtp.tensor_prefix,
            config.tie_word_embeddings,
        )?;
        let is_vision = descriptor
            .tensor_name
            .starts_with(&architecture.vision.tensor_prefix);
        let is_mtp = descriptor
            .tensor_name
            .starts_with(&architecture.mtp.tensor_prefix);
        let is_shared_embedding =
            descriptor.tensor_name == "model.language_model.embed_tokens.weight";
        if !selection.text && !is_vision && !is_mtp {
            classification = WeightClassification::KnownUnconsumed;
            consumer = None;
        }
        if selection.mtp && is_shared_embedding {
            classification = WeightClassification::Required;
            consumer = Some(WeightConsumerKey {
                layer: None,
                role: WeightConsumer::EmbeddingAndTiedOutput,
            });
        }
        if is_vision {
            vision_count = vision_count
                .checked_add(1)
                .ok_or_else(|| WeightPlanError::invalid("vision tensor count overflow"))?;
            if selection.vision {
                classification = WeightClassification::Required;
            }
        } else if is_mtp {
            mtp_count = mtp_count
                .checked_add(1)
                .ok_or_else(|| WeightPlanError::invalid("MTP tensor count overflow"))?;
            if selection.mtp {
                classification = WeightClassification::Required;
                consumer = Some(classify_mtp_consumer(&descriptor.tensor_name)?);
            }
        }
        if classification == WeightClassification::Required {
            if let Some(key) = consumer {
                if !observed_consumers.insert(key) {
                    return Err(WeightPlanError::invalid(format!(
                        "duplicate weight consumer: {key:?}"
                    )));
                }
            }
        }

        let locked = locked_files
            .get(descriptor.source_file.as_str())
            .expect("descriptor source was validated");
        let (destination_start, chunks) = if classification == WeightClassification::KnownUnconsumed
        {
            (None, Vec::new())
        } else {
            let destination_start = destination_cursor;
            let chunks = split_chunks(descriptor, destination_start)?;
            destination_cursor = destination_cursor
                .checked_add(descriptor.byte_size)
                .ok_or_else(|| WeightPlanError::invalid("destination size overflow"))?;
            (Some(destination_start), chunks)
        };
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.tensor_name.clone(),
            classification,
            consumer,
            dtype: descriptor.dtype,
            shape: descriptor.shape.clone(),
            source_file: descriptor.source_file.clone(),
            locked_file_size: locked.size_bytes,
            locked_file_sha256: locked.sha256.clone(),
            source_range: descriptor.absolute_byte_range,
            destination_start,
            chunks,
        });
    }

    if observed_consumers != selected_consumers {
        let missing: Vec<_> = selected_consumers
            .difference(&observed_consumers)
            .copied()
            .collect();
        let unexpected: Vec<_> = observed_consumers
            .difference(&selected_consumers)
            .copied()
            .collect();
        return Err(WeightPlanError::invalid(format!(
            "weight consumer set differs: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    if vision_count != architecture.vision.tensor_count
        || mtp_count != architecture.mtp.tensor_count
    {
        return Err(WeightPlanError::invalid(format!(
            "known-unconsumed tensor count differs: vision={vision_count}/{}, mtp={mtp_count}/{}",
            architecture.vision.tensor_count, architecture.mtp.tensor_count
        )));
    }

    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version: &lock.schema_version,
            repo_id: &lock.model.repo_id,
            resolved_revision: &lock.model.resolved_revision,
            fingerprint: lock.fingerprint(),
            tied_embeddings: config.tie_word_embeddings,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: lock.schema_version.clone(),
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: config.tie_word_embeddings,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

fn classify_mtp_consumer(name: &str) -> Result<WeightConsumerKey, WeightPlanError> {
    let (layer, role) = match name {
        "mtp.fc.weight" => (None, WeightConsumer::MtpFusion),
        "mtp.pre_fc_norm_embedding.weight" => (None, WeightConsumer::MtpEmbeddingNorm),
        "mtp.pre_fc_norm_hidden.weight" => (None, WeightConsumer::MtpHiddenNorm),
        "mtp.norm.weight" => (None, WeightConsumer::MtpFinalNorm),
        "mtp.layers.0.input_layernorm.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::InputNorm)
        }
        "mtp.layers.0.post_attention_layernorm.weight" => (
            Some(QWEN35_MTP_CONSUMER_LAYER),
            WeightConsumer::PostAttentionNorm,
        ),
        "mtp.layers.0.mlp.gate_proj.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::MlpGate)
        }
        "mtp.layers.0.mlp.up_proj.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::MlpUp)
        }
        "mtp.layers.0.mlp.down_proj.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::MlpDown)
        }
        "mtp.layers.0.self_attn.q_proj.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::AttentionQ)
        }
        "mtp.layers.0.self_attn.k_proj.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::AttentionK)
        }
        "mtp.layers.0.self_attn.v_proj.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::AttentionV)
        }
        "mtp.layers.0.self_attn.o_proj.weight" => {
            (Some(QWEN35_MTP_CONSUMER_LAYER), WeightConsumer::AttentionO)
        }
        "mtp.layers.0.self_attn.q_norm.weight" => (
            Some(QWEN35_MTP_CONSUMER_LAYER),
            WeightConsumer::AttentionQNorm,
        ),
        "mtp.layers.0.self_attn.k_norm.weight" => (
            Some(QWEN35_MTP_CONSUMER_LAYER),
            WeightConsumer::AttentionKNorm,
        ),
        _ => {
            return Err(WeightPlanError::invalid(format!(
                "unknown component-enabled MTP tensor: {name}"
            )));
        }
    };
    Ok(WeightConsumerKey { layer, role })
}

fn expected_mtp_consumers() -> BTreeSet<WeightConsumerKey> {
    use WeightConsumer::*;
    [
        WeightConsumerKey {
            layer: None,
            role: MtpFusion,
        },
        WeightConsumerKey {
            layer: None,
            role: MtpEmbeddingNorm,
        },
        WeightConsumerKey {
            layer: None,
            role: MtpHiddenNorm,
        },
        WeightConsumerKey {
            layer: None,
            role: MtpFinalNorm,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: InputNorm,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: PostAttentionNorm,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: MlpGate,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: MlpUp,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: MlpDown,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: AttentionQ,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: AttentionK,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: AttentionV,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: AttentionO,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: AttentionQNorm,
        },
        WeightConsumerKey {
            layer: Some(QWEN35_MTP_CONSUMER_LAYER),
            role: AttentionKNorm,
        },
    ]
    .into_iter()
    .collect()
}

/// Thin wrapper that binds the plan to the verified cache fingerprint.
pub fn build_verified_weight_load_plan(
    lock: &ModelLock,
    cache: &VerifiedCache,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if cache.lock_fingerprint != lock.fingerprint() {
        return Err(WeightPlanError::invalid(
            "verified cache fingerprint differs from the model lock",
        ));
    }
    build_weight_load_plan(lock, cache.tensors())
}

pub fn build_verified_qwen_component_weight_load_plan(
    lock: &ModelLock,
    cache: &VerifiedCache,
    selection: QwenComponentSelection,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if cache.lock_fingerprint != lock.fingerprint() {
        return Err(WeightPlanError::invalid(
            "verified cache fingerprint differs from the model lock",
        ));
    }
    build_qwen_component_weight_load_plan(lock, cache.tensors(), selection)
}

pub fn build_verified_gguf_qwen_weight_load_plan(
    lock: &ModelLock,
    verified: VerifiedDerivedGguf,
    selection: QwenComponentSelection,
) -> Result<(VerifiedGgufWeightSource, WeightLoadPlan), WeightPlanError> {
    if verified.gguf.architecture() != "qwen35"
        || !verified
            .lock
            .source_lock_fingerprints
            .iter()
            .any(|fingerprint| fingerprint == lock.fingerprint())
    {
        return Err(WeightPlanError::invalid(
            "GGUF is not the reviewed Qwen3.5 semantic identity",
        ));
    }
    let recipe_bindings: BTreeMap<_, _> = verified
        .gguf
        .extension()
        .map(|extension| {
            extension
                .recipe
                .bindings
                .iter()
                .cloned()
                .map(|binding| (binding.logical_tensor.clone(), binding))
                .collect()
        })
        .unwrap_or_default();
    let logical_shapes: BTreeMap<_, _> = verified
        .gguf
        .extension()
        .map(|extension| {
            extension
                .recipe
                .logical_shapes
                .iter()
                .map(|binding| (binding.tensor.as_str(), binding.logical_shape.as_slice()))
                .collect()
        })
        .unwrap_or_default();
    if recipe_bindings.values().any(|binding| {
        !matches!(
            binding.encoding,
            GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale
                | GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale
                | GgufRecipeEncoding::Mxfp8E4m3Block32E8m0
                | GgufRecipeEncoding::Mxfp6E3m2Block32E8m0
                | GgufRecipeEncoding::Nvfp4E2m1Block16E4m3fnF32Outer
        )
    }) {
        return Err(WeightPlanError::invalid(
            "Qwen3.5 dense GGUF has an unsupported quantization recipe",
        ));
    }
    let value_bindings: BTreeMap<_, _> = recipe_bindings
        .values()
        .map(|binding| (binding.value_tensor.as_str(), binding))
        .collect();
    if value_bindings.len() != recipe_bindings.len() {
        return Err(WeightPlanError::invalid(
            "Qwen3.5 dense GGUF recipe value tensors are not one-to-one",
        ));
    }
    let scale_names: BTreeSet<_> = recipe_bindings
        .values()
        .flat_map(|binding| binding.scales.iter().map(|scale| scale.tensor.as_str()))
        .collect();
    let source_file = verified.gguf.path().display().to_string();
    let file_sha256 = verified
        .lock
        .output
        .sha256
        .strip_prefix("sha256:")
        .ok_or_else(|| WeightPlanError::invalid("GGUF SHA-256 prefix differs"))?
        .to_owned();
    let locked_file = LockedFile {
        path: source_file.clone(),
        size_bytes: verified.gguf.file_size(),
        sha256: file_sha256,
        git_blob: String::new(),
        source_page_url: String::new(),
        download_url: String::new(),
        lfs_oid: None,
    };
    let locked_files = BTreeMap::from([(locked_file.path.as_str(), &locked_file)]);
    let mut descriptors = BTreeMap::new();
    for tensor in verified.gguf.tensors() {
        if scale_names.contains(tensor.name.as_str()) {
            continue;
        }
        let dtype = match tensor.tensor_type {
            GgufTensorType::Bf16 => TensorDType::Bf16,
            GgufTensorType::F16 => TensorDType::F16,
            GgufTensorType::F32 => TensorDType::F32,
            GgufTensorType::I8Carrier if value_bindings.contains_key(tensor.name.as_str()) => {
                TensorDType::U8
            }
            _ => {
                return Err(WeightPlanError::invalid(format!(
                    "Qwen GGUF tensor has no supported semantic binding: {}",
                    tensor.name
                )));
            }
        };
        let recipe_binding = value_bindings.get(tensor.name.as_str()).copied();
        let shape = if let Some(binding) = recipe_binding {
            binding.logical_shape.clone()
        } else if let Some(shape) = logical_shapes.get(tensor.name.as_str()) {
            shape.to_vec()
        } else {
            let mut shape = tensor.dimensions.clone();
            shape.reverse();
            shape
        };
        let relative_end = tensor
            .relative_offset
            .checked_add(tensor.byte_length())
            .ok_or_else(|| WeightPlanError::invalid("GGUF relative tensor range overflows"))?;
        let descriptor = TensorDescriptor {
            tensor_name: recipe_binding.map_or_else(
                || tensor.name.clone(),
                |binding| binding.logical_tensor.clone(),
            ),
            source_file: source_file.clone(),
            dtype,
            shape,
            header_length_field_bytes: 0,
            header_length_bytes: verified.gguf.data_offset(),
            data_buffer_start: verified.gguf.data_offset(),
            data_offset_basis: "gguf-v3-tensor-data".to_owned(),
            data_offsets: [tensor.relative_offset, relative_end],
            absolute_byte_range: tensor.absolute_range,
            byte_size: tensor.byte_length(),
        };
        if descriptors
            .insert(descriptor.tensor_name.clone(), descriptor)
            .is_some()
        {
            return Err(WeightPlanError::invalid("duplicate GGUF tensor descriptor"));
        }
    }
    let plan = if recipe_bindings.is_empty() {
        let mut plan = build_qwen_component_weight_load_plan_inner(
            lock,
            descriptors.values(),
            selection,
            &locked_files,
        )?;
        plan.schema_version = "gguf-model-plan-v1".to_owned();
        plan.digest = plan.recompute_digest()?;
        plan
    } else {
        build_qwen_gguf_quantized_plan(lock, &descriptors, selection, &locked_file)?
    };
    let source = VerifiedGgufWeightSource {
        lock_fingerprint: lock.fingerprint().to_owned(),
        file_sha256: verified.lock.output.sha256,
        descriptors,
        recipe_bindings,
        gguf: verified.gguf,
    };
    Ok((source, plan))
}

/// Build the reviewed Qwen3.5-27B semantic load plan for the exact Unsloth
/// Qwen3.8 mixed NVFP4 source.  The artifact owns the physical safetensors
/// ranges; this plan deliberately records logical resident dtype/shape while
/// the Qwen38 provision source performs the value/scale packing at upload.
pub fn build_qwen38_nvfp4_weight_load_plan(
    lock: &ModelLock,
    artifact: &crate::VerifiedUnslothQwen38Nvfp4,
) -> Result<WeightLoadPlan, WeightPlanError> {
    validate_fixed_lock(lock)?;
    let spec = reviewed_qwen35_spec(lock)
        .ok_or_else(|| WeightPlanError::invalid("Qwen3.5 lock is not reviewed"))?;
    if spec.repo_id != crate::model::QWEN35_27B_REPO_ID
        || artifact.repository() != crate::UNSLOTH_QWEN38_NVFP4_REPOSITORY
    {
        return Err(WeightPlanError::invalid(
            "Qwen3.8 NVFP4 requires the reviewed Qwen3.5-27B semantic lock",
        ));
    }
    let expected = expected_consumers(
        &lock.model.architecture.text_config.layer_types,
        lock.model.architecture.text_config.tie_word_embeddings,
    );
    let model_file = artifact
        .root()
        .join("model.safetensors")
        .display()
        .to_string();
    let mtp_file = artifact
        .root()
        .join("model_mtp.safetensors")
        .display()
        .to_string();
    let mut observed = BTreeSet::new();
    let mut entries = Vec::with_capacity(artifact.tensors().len());
    let mut destination_cursor = 0_u64;
    for descriptor in artifact.tensors() {
        let dtype = if descriptor.logical_name.ends_with(".A_log")
            || descriptor
                .logical_name
                .ends_with(".linear_attn.norm.weight")
        {
            TensorDType::F32
        } else {
            TensorDType::Bf16
        };
        let synthetic = TensorDescriptor {
            tensor_name: descriptor.logical_name.clone(),
            source_file: if descriptor.source_name.starts_with("mtp.") {
                mtp_file.clone()
            } else {
                model_file.clone()
            },
            dtype,
            shape: descriptor.logical_shape.clone(),
            header_length_field_bytes: 8,
            header_length_bytes: 0,
            data_buffer_start: 0,
            data_offset_basis: "safetensors-data-buffer".to_owned(),
            data_offsets: descriptor.value_range,
            absolute_byte_range: descriptor.value_range,
            byte_size: descriptor.value_range[1]
                .checked_sub(descriptor.value_range[0])
                .ok_or_else(|| WeightPlanError::invalid("Qwen3.8 source range underflows"))?,
        };
        let (classification, consumer) = classify_descriptor(
            &synthetic,
            &lock.model.architecture.text_config.layer_types,
            &lock.model.architecture.vision.tensor_prefix,
            &lock.model.architecture.mtp.tensor_prefix,
            lock.model.architecture.text_config.tie_word_embeddings,
        )?;
        if let Some(key) = consumer {
            if classification != WeightClassification::Required || !observed.insert(key) {
                return Err(WeightPlanError::invalid(format!(
                    "Qwen3.8 consumer is not one-to-one: {}",
                    descriptor.logical_name
                )));
            }
        }
        let destination_start = if classification == WeightClassification::Required {
            let start = destination_cursor;
            let logical_bytes = descriptor
                .logical_shape
                .iter()
                .try_fold(dtype_width(dtype), |bytes, extent| {
                    bytes.checked_mul(*extent)
                })
                .ok_or_else(|| WeightPlanError::invalid("Qwen3.8 logical byte size overflows"))?;
            destination_cursor = destination_cursor
                .checked_add(logical_bytes)
                .ok_or_else(|| WeightPlanError::invalid("Qwen3.8 destination size overflows"))?;
            Some(start)
        } else {
            None
        };
        let (source_file, locked_size, locked_sha) = if descriptor.source_name.starts_with("mtp.") {
            (
                mtp_file.clone(),
                crate::UNSLOTH_QWEN38_NVFP4_MTP_SIZE,
                crate::UNSLOTH_QWEN38_NVFP4_MTP_SHA256.to_owned(),
            )
        } else {
            (
                model_file.clone(),
                crate::UNSLOTH_QWEN38_NVFP4_MODEL_SIZE,
                crate::UNSLOTH_QWEN38_NVFP4_MODEL_SHA256.to_owned(),
            )
        };
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.logical_name.clone(),
            classification,
            consumer,
            dtype,
            shape: descriptor.logical_shape.clone(),
            source_file,
            locked_file_size: locked_size,
            locked_file_sha256: locked_sha,
            source_range: descriptor.value_range,
            destination_start,
            chunks: Vec::new(),
        });
    }
    if observed != expected || entries.len() as u64 != spec.indexed_tensor_count {
        return Err(WeightPlanError::invalid(format!(
            "Qwen3.8 consumer/count coverage differs: consumers={}/{}, entries={}/{}",
            observed.len(),
            expected.len(),
            entries.len(),
            spec.indexed_tensor_count
        )));
    }
    let schema_version = "qwen38-nvfp4-model-plan-v1";
    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version,
            repo_id: &lock.model.repo_id,
            resolved_revision: &lock.model.resolved_revision,
            fingerprint: lock.fingerprint(),
            tied_embeddings: lock.model.architecture.text_config.tie_word_embeddings,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: schema_version.to_owned(),
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: lock.model.architecture.text_config.tie_word_embeddings,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

fn dtype_width(dtype: TensorDType) -> u64 {
    match dtype {
        TensorDType::Bf16 | TensorDType::F16 => 2,
        TensorDType::F32 | TensorDType::I32 => 4,
        TensorDType::I64 => 8,
        TensorDType::U8 => 1,
    }
}

fn build_qwen_gguf_quantized_plan(
    lock: &ModelLock,
    descriptors: &BTreeMap<String, TensorDescriptor>,
    selection: QwenComponentSelection,
    locked_file: &LockedFile,
) -> Result<WeightLoadPlan, WeightPlanError> {
    validate_fixed_lock(lock)?;
    let architecture = &lock.model.architecture;
    let config = &architecture.text_config;
    let mut selected_consumers = BTreeSet::new();
    if selection.text {
        selected_consumers.extend(expected_consumers(
            &config.layer_types,
            config.tie_word_embeddings,
        ));
    }
    if selection.mtp {
        selected_consumers.extend(expected_mtp_consumers());
        selected_consumers.insert(WeightConsumerKey {
            layer: None,
            role: WeightConsumer::EmbeddingAndTiedOutput,
        });
    }
    let mut observed_consumers = BTreeSet::new();
    let mut vision_count = 0_u64;
    let mut mtp_count = 0_u64;
    let mut entries = Vec::with_capacity(descriptors.len());
    let mut destination_cursor = 0_u64;
    for descriptor in descriptors.values() {
        let (mut classification, mut consumer) = classify_descriptor(
            descriptor,
            &config.layer_types,
            &architecture.vision.tensor_prefix,
            &architecture.mtp.tensor_prefix,
            config.tie_word_embeddings,
        )?;
        let is_vision = descriptor
            .tensor_name
            .starts_with(&architecture.vision.tensor_prefix);
        let is_mtp = descriptor
            .tensor_name
            .starts_with(&architecture.mtp.tensor_prefix);
        let is_shared_embedding =
            descriptor.tensor_name == "model.language_model.embed_tokens.weight";
        if !selection.text && !is_vision && !is_mtp {
            classification = WeightClassification::KnownUnconsumed;
            consumer = None;
        }
        if selection.mtp && is_shared_embedding {
            classification = WeightClassification::Required;
            consumer = Some(WeightConsumerKey {
                layer: None,
                role: WeightConsumer::EmbeddingAndTiedOutput,
            });
        }
        if is_vision {
            vision_count = vision_count
                .checked_add(1)
                .ok_or_else(|| WeightPlanError::invalid("vision tensor count overflow"))?;
            if selection.vision {
                classification = WeightClassification::Required;
            }
        } else if is_mtp {
            mtp_count = mtp_count
                .checked_add(1)
                .ok_or_else(|| WeightPlanError::invalid("MTP tensor count overflow"))?;
            if selection.mtp {
                classification = WeightClassification::Required;
                consumer = Some(classify_mtp_consumer(&descriptor.tensor_name)?);
            }
        }
        if classification == WeightClassification::Required {
            if let Some(key) = consumer {
                if !observed_consumers.insert(key) {
                    return Err(WeightPlanError::invalid(format!(
                        "duplicate GGUF weight consumer: {key:?}"
                    )));
                }
            }
        }
        let quantized = descriptor.dtype == TensorDType::U8;
        let logical_bytes = if quantized {
            descriptor
                .shape
                .iter()
                .try_fold(2_u64, |bytes, dimension| bytes.checked_mul(*dimension))
                .ok_or_else(|| WeightPlanError::invalid("GGUF logical tensor bytes overflow"))?
        } else {
            descriptor.byte_size
        };
        let destination_start = if classification == WeightClassification::KnownUnconsumed {
            None
        } else {
            let start = destination_cursor;
            destination_cursor = destination_cursor
                .checked_add(logical_bytes)
                .ok_or_else(|| WeightPlanError::invalid("GGUF destination size overflow"))?;
            Some(start)
        };
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.tensor_name.clone(),
            classification,
            consumer,
            dtype: if quantized {
                TensorDType::Bf16
            } else {
                descriptor.dtype
            },
            shape: descriptor.shape.clone(),
            source_file: locked_file.path.clone(),
            locked_file_size: locked_file.size_bytes,
            locked_file_sha256: locked_file.sha256.clone(),
            source_range: descriptor.absolute_byte_range,
            destination_start,
            chunks: Vec::new(),
        });
    }
    if observed_consumers != selected_consumers
        || vision_count != architecture.vision.tensor_count
        || mtp_count != architecture.mtp.tensor_count
        || entries.len() as u64 != lock.model.tensor_contract.indexed_tensor_count
    {
        return Err(WeightPlanError::invalid(format!(
            "quantized Qwen GGUF consumer/count contract differs: consumers={}/{}, vision={vision_count}/{}, mtp={mtp_count}/{}, all={}/{}",
            observed_consumers.len(),
            selected_consumers.len(),
            architecture.vision.tensor_count,
            architecture.mtp.tensor_count,
            entries.len(),
            lock.model.tensor_contract.indexed_tensor_count
        )));
    }
    let schema_version = "gguf-quantized-model-plan-v1";
    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version,
            repo_id: &lock.model.repo_id,
            resolved_revision: &lock.model.resolved_revision,
            fingerprint: lock.fingerprint(),
            tied_embeddings: config.tie_word_embeddings,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: schema_version.to_owned(),
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: config.tie_word_embeddings,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

pub fn build_verified_gguf_gemma_weight_load_plan(
    lock: &crate::Gemma4ModelLock,
    verified: VerifiedDerivedGguf,
) -> Result<(VerifiedGgufGemmaSource, WeightLoadPlan), WeightPlanError> {
    if verified.gguf.architecture() != "gemma4"
        || !verified
            .lock
            .source_lock_fingerprints
            .iter()
            .any(|fingerprint| fingerprint == lock.fingerprint())
    {
        return Err(WeightPlanError::invalid(
            "GGUF is not the reviewed Gemma 4 semantic identity",
        ));
    }
    let extension = verified
        .gguf
        .extension()
        .ok_or_else(|| WeightPlanError::invalid("Gemma mixed GGUF recipe is absent"))?;
    let known_unconsumed: BTreeSet<_> = extension
        .recipe
        .known_unconsumed_tensors
        .iter()
        .map(String::as_str)
        .collect();
    let scale_names: BTreeSet<_> = extension
        .recipe
        .bindings
        .iter()
        .flat_map(|binding| binding.scales.iter().map(|scale| scale.tensor.as_str()))
        .collect();
    let bindings: BTreeMap<_, _> = extension
        .recipe
        .bindings
        .iter()
        .map(|binding| (binding.logical_tensor.as_str(), binding))
        .collect();
    let mut tensors = BTreeMap::new();
    for tensor in verified.gguf.tensors() {
        if scale_names.contains(tensor.name.as_str()) {
            continue;
        }
        let (role, encoding, logical_shape, source_name, scale_planes) =
            if let Some(binding) = bindings.get(tensor.name.as_str()) {
                let role = parse_quantized_role(&binding.role)?;
                let encoding = match binding.encoding {
                    GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale => {
                        crate::QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale
                    }
                    GgufRecipeEncoding::Nvfp4E2m1Block16E4m3fnF32Outer => {
                        crate::QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer
                    }
                    _ => {
                        return Err(WeightPlanError::invalid(
                            "Gemma GGUF recipe contains an unsupported encoding",
                        ));
                    }
                };
                let mut planes = Vec::with_capacity(binding.scales.len());
                for scale in &binding.scales {
                    let info = verified.gguf.tensor(&scale.tensor).ok_or_else(|| {
                        WeightPlanError::invalid("Gemma GGUF scale tensor is absent")
                    })?;
                    let role = match scale.role {
                        crate::GgufScaleRole::Channel => crate::ScalePlaneRole::WeightChannel,
                        crate::GgufScaleRole::Outer => crate::ScalePlaneRole::WeightOuter,
                        crate::GgufScaleRole::Input => crate::ScalePlaneRole::InputOuter,
                        crate::GgufScaleRole::Block => crate::ScalePlaneRole::WeightBlock,
                    };
                    let mut shape = info.dimensions.clone();
                    shape.reverse();
                    planes.push(crate::QuantizedScalePlane {
                        role,
                        source_name: scale.tensor.clone(),
                        dtype: match info.tensor_type {
                            GgufTensorType::Bf16 => "BF16",
                            GgufTensorType::F32 => "F32",
                            _ => "GGUF_BLOCK",
                        }
                        .to_owned(),
                        shape,
                        source_range: info.absolute_range,
                        reciprocal: false,
                    });
                }
                (
                    role,
                    encoding,
                    binding.logical_shape.clone(),
                    binding.value_tensor.clone(),
                    planes,
                )
            } else {
                if tensor.tensor_type != GgufTensorType::Bf16 {
                    return Err(WeightPlanError::invalid(format!(
                        "unbound Gemma GGUF tensor is not BF16: {}",
                        tensor.name
                    )));
                }
                let mut shape = tensor.dimensions.clone();
                shape.reverse();
                let role = if known_unconsumed.contains(tensor.name.as_str()) {
                    crate::QuantizedTensorRole::KnownUnconsumed
                } else if tensor.name == "model.language_model.embed_tokens.weight" {
                    crate::QuantizedTensorRole::Embedding
                } else if tensor.name.contains("norm") {
                    crate::QuantizedTensorRole::Normalization
                } else {
                    crate::QuantizedTensorRole::Scalar
                };
                (
                    role,
                    crate::QuantizedTensorEncoding::UnquantizedBf16,
                    shape,
                    tensor.name.clone(),
                    Vec::new(),
                )
            };
        let descriptor = crate::QuantizedTensorDescriptor {
            logical_name: tensor.name.clone(),
            source_name,
            role,
            encoding,
            logical_shape: logical_shape.clone(),
            value_dtype: format!("gguf-type-{}", tensor.tensor_type.raw()),
            value_shape: logical_shape,
            value_range: tensor.absolute_range,
            scale_planes,
        };
        if tensors
            .insert(descriptor.logical_name.clone(), descriptor)
            .is_some()
        {
            return Err(WeightPlanError::invalid(
                "Gemma GGUF contains a duplicate logical tensor",
            ));
        }
    }
    let kv_scales = extension
        .recipe
        .static_fp8_kv
        .iter()
        .map(|scale| {
            (
                scale.layer,
                crate::StaticFp8KvScale {
                    key_decode_scale_bf16: scale.key_decode_scale_bf16,
                    value_decode_scale_bf16: scale.value_decode_scale_bf16,
                },
            )
        })
        .collect();
    let source = VerifiedGgufGemmaSource {
        lock_fingerprint: lock.fingerprint().to_owned(),
        repository: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        file_sha256: verified.lock.output.sha256,
        tensors,
        kv_scales,
        gguf: verified.gguf,
    };
    let plan = build_gguf_gemma_plan(lock, &source)?;
    Ok((source, plan))
}

fn parse_quantized_role(role: &str) -> Result<crate::QuantizedTensorRole, WeightPlanError> {
    match role {
        "embedding" => Ok(crate::QuantizedTensorRole::Embedding),
        "attention-projection" => Ok(crate::QuantizedTensorRole::AttentionProjection),
        "mlp-projection" => Ok(crate::QuantizedTensorRole::MlpProjection),
        "normalization" => Ok(crate::QuantizedTensorRole::Normalization),
        "scalar" => Ok(crate::QuantizedTensorRole::Scalar),
        "known-unconsumed" => Ok(crate::QuantizedTensorRole::KnownUnconsumed),
        _ => Err(WeightPlanError::invalid(
            "Gemma GGUF recipe has an unknown tensor role",
        )),
    }
}

fn build_gguf_gemma_plan(
    lock: &crate::Gemma4ModelLock,
    source: &VerifiedGgufGemmaSource,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if lock.schema_version != "model-lock-v2"
        || !crate::gemma4::is_reviewed_gemma4_identity(lock)
        || !lock.model.architecture.text.tie_word_embeddings
        || lock.model.architecture.text.layer_types != crate::gemma4::reviewed_layer_schedule()
    {
        return Err(WeightPlanError::invalid(
            "model identity is not the reviewed Gemma 4 dense contract",
        ));
    }
    let expected_consumers = expected_gemma4_consumers();
    let mut observed_consumers = BTreeSet::new();
    let mut audio_count = 0_u64;
    let mut vision_count = 0_u64;
    let mut entries = Vec::with_capacity(source.tensors.len());
    let mut destination_cursor = 0_u64;
    let source_file = source.gguf.path().display().to_string();
    let source_sha = source
        .file_sha256
        .strip_prefix("sha256:")
        .ok_or_else(|| WeightPlanError::invalid("GGUF SHA-256 prefix differs"))?;
    for descriptor in source.tensors.values() {
        let (classification, consumer) = classify_gemma4_name(&descriptor.logical_name)?;
        if let Some(key) = consumer {
            if !observed_consumers.insert(key) {
                return Err(WeightPlanError::invalid(format!(
                    "duplicate GGUF Gemma weight consumer: {key:?}"
                )));
            }
        }
        if classification == WeightClassification::KnownUnconsumed {
            if descriptor.logical_name.starts_with("model.embed_audio.") {
                audio_count = audio_count
                    .checked_add(1)
                    .ok_or_else(|| WeightPlanError::invalid("audio tensor count overflow"))?;
            } else {
                vision_count = vision_count
                    .checked_add(1)
                    .ok_or_else(|| WeightPlanError::invalid("vision tensor count overflow"))?;
            }
        }
        let logical_bytes = descriptor
            .logical_shape
            .iter()
            .try_fold(2_u64, |bytes, dimension| bytes.checked_mul(*dimension))
            .ok_or_else(|| WeightPlanError::invalid("logical tensor bytes overflow"))?;
        let destination_start = if classification == WeightClassification::KnownUnconsumed {
            None
        } else {
            let start = destination_cursor;
            destination_cursor = destination_cursor
                .checked_add(logical_bytes)
                .ok_or_else(|| WeightPlanError::invalid("destination size overflow"))?;
            Some(start)
        };
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.logical_name.clone(),
            classification,
            consumer,
            dtype: TensorDType::Bf16,
            shape: descriptor.logical_shape.clone(),
            source_file: source_file.clone(),
            locked_file_size: source.gguf.file_size(),
            locked_file_sha256: source_sha.to_owned(),
            source_range: descriptor.value_range,
            destination_start,
            chunks: Vec::new(),
        });
    }
    if observed_consumers != expected_consumers
        || audio_count != lock.model.architecture.audio.tensor_count
        || vision_count != lock.model.architecture.vision.tensor_count
        || entries.len() as u64 != lock.model.tensor_contract.tensor_count
    {
        return Err(WeightPlanError::invalid(format!(
            "GGUF Gemma consumer/count contract differs: consumers={}/{}, audio={audio_count}/{}, vision={vision_count}/{}, all={}/{}",
            observed_consumers.len(),
            expected_consumers.len(),
            lock.model.architecture.audio.tensor_count,
            lock.model.architecture.vision.tensor_count,
            entries.len(),
            lock.model.tensor_contract.tensor_count
        )));
    }
    let schema_version = "gguf-quantized-model-plan-v1";
    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version,
            repo_id: &lock.model.repo_id,
            resolved_revision: &lock.model.resolved_revision,
            fingerprint: lock.fingerprint(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: schema_version.to_owned(),
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: true,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

/// Build the exact Gemma 4 text-only load plan from verified direct-file
/// descriptors. Audio and vision tensors remain represented as
/// known-unconsumed entries and consume no device destination space.
pub fn build_gemma4_weight_load_plan<'a>(
    lock: &crate::gemma4::Gemma4ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if lock.schema_version != "model-lock-v2"
        || !crate::gemma4::is_reviewed_gemma4_identity(lock)
        || !lock.model.architecture.text.tie_word_embeddings
        || lock.model.architecture.text.layer_types != crate::gemma4::reviewed_layer_schedule()
    {
        return Err(WeightPlanError::invalid(
            "model identity is not the reviewed Gemma 4 dense contract",
        ));
    }
    let locked_files = locked_file_map_from_files(&lock.model.files)?;
    let mut by_name = BTreeMap::new();
    for descriptor in descriptors {
        if by_name
            .insert(descriptor.tensor_name.as_str(), descriptor)
            .is_some()
        {
            return Err(WeightPlanError::invalid(format!(
                "duplicate tensor descriptor: {}",
                descriptor.tensor_name
            )));
        }
    }

    let expected_consumers = expected_gemma4_consumers();
    let mut observed_consumers = BTreeSet::new();
    let mut audio_count = 0_u64;
    let mut vision_count = 0_u64;
    let mut entries = Vec::with_capacity(by_name.len());
    let mut destination_cursor = 0_u64;
    for descriptor in by_name.values() {
        validate_descriptor(descriptor, &locked_files)?;
        let (classification, consumer) = classify_gemma4_descriptor(descriptor)?;
        if let Some(key) = consumer {
            if !observed_consumers.insert(key) {
                return Err(WeightPlanError::invalid(format!(
                    "duplicate weight consumer: {key:?}"
                )));
            }
        }
        if classification == WeightClassification::KnownUnconsumed {
            if descriptor.tensor_name.starts_with("model.embed_audio.") {
                audio_count = audio_count
                    .checked_add(1)
                    .ok_or_else(|| WeightPlanError::invalid("audio tensor count overflow"))?;
            } else {
                vision_count = vision_count
                    .checked_add(1)
                    .ok_or_else(|| WeightPlanError::invalid("vision tensor count overflow"))?;
            }
        }
        let locked = locked_files
            .get(descriptor.source_file.as_str())
            .expect("descriptor source was validated");
        let (destination_start, chunks) = if classification == WeightClassification::KnownUnconsumed
        {
            (None, Vec::new())
        } else {
            let destination_start = destination_cursor;
            let chunks = split_chunks(descriptor, destination_start)?;
            destination_cursor = destination_cursor
                .checked_add(descriptor.byte_size)
                .ok_or_else(|| WeightPlanError::invalid("destination size overflow"))?;
            (Some(destination_start), chunks)
        };
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.tensor_name.clone(),
            classification,
            consumer,
            dtype: descriptor.dtype,
            shape: descriptor.shape.clone(),
            source_file: descriptor.source_file.clone(),
            locked_file_size: locked.size_bytes,
            locked_file_sha256: locked.sha256.clone(),
            source_range: descriptor.absolute_byte_range,
            destination_start,
            chunks,
        });
    }
    if observed_consumers != expected_consumers {
        let missing: Vec<_> = expected_consumers
            .difference(&observed_consumers)
            .copied()
            .collect();
        let unexpected: Vec<_> = observed_consumers
            .difference(&expected_consumers)
            .copied()
            .collect();
        return Err(WeightPlanError::invalid(format!(
            "Gemma 4 weight consumer set differs: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    if audio_count != lock.model.architecture.audio.tensor_count
        || vision_count != lock.model.architecture.vision.tensor_count
        || entries.len() as u64 != lock.model.tensor_contract.tensor_count
    {
        return Err(WeightPlanError::invalid(format!(
            "Gemma 4 known-unconsumed/count contract differs: audio={audio_count}/{}, vision={vision_count}/{}, all={}/{}",
            lock.model.architecture.audio.tensor_count,
            lock.model.architecture.vision.tensor_count,
            entries.len(),
            lock.model.tensor_contract.tensor_count
        )));
    }
    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version: &lock.schema_version,
            repo_id: &lock.model.repo_id,
            resolved_revision: &lock.model.resolved_revision,
            fingerprint: lock.fingerprint(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: lock.schema_version.clone(),
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: true,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

pub fn build_verified_gemma4_weight_load_plan(
    lock: &crate::gemma4::Gemma4ModelLock,
    cache: &VerifiedCache,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if cache.lock_fingerprint != lock.fingerprint() {
        return Err(WeightPlanError::invalid(
            "verified cache fingerprint differs from the model lock",
        ));
    }
    build_gemma4_weight_load_plan(lock, cache.tensors())
}

/// Build the exact lossless BF16 resident plan for the reviewed Gemma 4 MTP
/// assistant. The assistant is a separately resident draft model: its layer
/// namespace is therefore `0..4` and never aliases the 48-layer target plan.
pub fn build_gemma4_mtp_weight_load_plan<'a>(
    lock: &crate::gemma4_mtp::Gemma4MtpModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if lock.schema_version != "gemma4-mtp-model-lock-v1"
        || lock.model.repo_id != crate::gemma4_mtp::GEMMA4_MTP_REPO_ID
        || lock.model.resolved_revision != crate::gemma4_mtp::GEMMA4_MTP_REVISION
        || lock.fingerprint() != crate::gemma4_mtp::GEMMA4_MTP_FINGERPRINT
        || lock.model.architecture.layer_types != crate::gemma4_mtp::reviewed_mtp_layer_schedule()
        || lock.model.architecture.use_ordered_embeddings
        || lock.model.architecture.own_kv_projection_tensor_count != 0
    {
        return Err(WeightPlanError::invalid(
            "model identity is not the reviewed Gemma 4 MTP assistant contract",
        ));
    }
    let locked_files = locked_file_map_from_files(&lock.model.files)?;
    let mut by_name = BTreeMap::new();
    for descriptor in descriptors {
        if by_name
            .insert(descriptor.tensor_name.as_str(), descriptor)
            .is_some()
        {
            return Err(WeightPlanError::invalid(format!(
                "duplicate tensor descriptor: {}",
                descriptor.tensor_name
            )));
        }
    }

    let expected_consumers = expected_gemma4_mtp_consumers();
    let mut observed_consumers = BTreeSet::new();
    let mut entries = Vec::with_capacity(by_name.len());
    let mut destination_cursor = 0_u64;
    for descriptor in by_name.values() {
        validate_descriptor(descriptor, &locked_files)?;
        if descriptor.dtype != TensorDType::Bf16 {
            return Err(WeightPlanError::invalid(format!(
                "Gemma 4 MTP tensor is not BF16: {}",
                descriptor.tensor_name
            )));
        }
        let consumer = classify_gemma4_mtp_name(&descriptor.tensor_name)?;
        if !observed_consumers.insert(consumer) {
            return Err(WeightPlanError::invalid(format!(
                "duplicate Gemma 4 MTP weight consumer: {consumer:?}"
            )));
        }
        let locked = locked_files
            .get(descriptor.source_file.as_str())
            .expect("descriptor source was validated");
        let destination_start = destination_cursor;
        let chunks = split_chunks(descriptor, destination_start)?;
        destination_cursor = destination_cursor
            .checked_add(descriptor.byte_size)
            .ok_or_else(|| WeightPlanError::invalid("MTP destination size overflow"))?;
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.tensor_name.clone(),
            classification: WeightClassification::Required,
            consumer: Some(consumer),
            dtype: descriptor.dtype,
            shape: descriptor.shape.clone(),
            source_file: descriptor.source_file.clone(),
            locked_file_size: locked.size_bytes,
            locked_file_sha256: locked.sha256.clone(),
            source_range: descriptor.absolute_byte_range,
            destination_start: Some(destination_start),
            chunks,
        });
    }
    if observed_consumers != expected_consumers
        || entries.len() as u64 != crate::gemma4_mtp::GEMMA4_MTP_TENSOR_COUNT
        || destination_cursor
            != crate::gemma4_mtp::GEMMA4_MTP_MODEL_BYTES
                - crate::gemma4_mtp::GEMMA4_MTP_HEADER_BYTES
                - 8
    {
        return Err(WeightPlanError::invalid(format!(
            "Gemma 4 MTP consumer/count/resident byte contract differs: consumers={}/{}, tensors={}/{}, bytes={destination_cursor}",
            observed_consumers.len(),
            expected_consumers.len(),
            entries.len(),
            crate::gemma4_mtp::GEMMA4_MTP_TENSOR_COUNT,
        )));
    }
    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version: &lock.schema_version,
            repo_id: &lock.model.repo_id,
            resolved_revision: &lock.model.resolved_revision,
            fingerprint: lock.fingerprint(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: lock.schema_version.clone(),
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: true,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

pub fn build_verified_gemma4_mtp_weight_load_plan<S>(
    lock: &crate::gemma4_mtp::Gemma4MtpModelLock,
    source: &S,
) -> Result<WeightLoadPlan, WeightPlanError>
where
    S: crate::gemma4_mtp::Gemma4MtpWeightSource + ?Sized,
{
    if source.lock_fingerprint() != lock.fingerprint()
        || source.target_fingerprint() != lock.target_fingerprint()
    {
        return Err(WeightPlanError::invalid(
            "verified Gemma 4 MTP source identity differs from the assistant pair lock",
        ));
    }
    build_gemma4_mtp_weight_load_plan(lock, source.tensors().values())
}

/// Build the Gemma execution topology directly from the verified first-class
/// Unsloth artifact. Source ranges describe encoded values; they are never
/// presented as BF16 upload ranges. The resulting plan remains bound to the
/// reviewed Gemma architecture lock while its repository/revision identify
/// the actual low-bit source.
pub fn build_unsloth_gemma4_nvfp4_weight_load_plan(
    lock: &crate::gemma4::Gemma4ModelLock,
    artifact: &crate::VerifiedUnslothGemma4Nvfp4,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if lock.schema_version != "model-lock-v2"
        || !crate::gemma4::is_reviewed_gemma4_identity(lock)
        || !lock.model.architecture.text.tie_word_embeddings
        || lock.model.architecture.text.layer_types != crate::gemma4::reviewed_layer_schedule()
    {
        return Err(WeightPlanError::invalid(
            "model identity is not the reviewed Gemma 4 dense contract",
        ));
    }
    let mut descriptors = BTreeMap::new();
    for descriptor in artifact.tensors() {
        if descriptors
            .insert(descriptor.logical_name.as_str(), descriptor)
            .is_some()
        {
            return Err(WeightPlanError::invalid(
                "quantized artifact contains a duplicate logical tensor",
            ));
        }
    }
    let expected_consumers = expected_gemma4_consumers();
    let mut observed_consumers = BTreeSet::new();
    let mut audio_count = 0_u64;
    let mut vision_count = 0_u64;
    let mut entries = Vec::with_capacity(descriptors.len());
    let mut destination_cursor = 0_u64;
    for descriptor in descriptors.values() {
        let (classification, consumer) = classify_gemma4_name(&descriptor.logical_name)?;
        if let Some(key) = consumer {
            if !observed_consumers.insert(key) {
                return Err(WeightPlanError::invalid(format!(
                    "duplicate quantized weight consumer: {key:?}"
                )));
            }
        }
        if classification == WeightClassification::KnownUnconsumed {
            if descriptor.logical_name.starts_with("model.embed_audio.") {
                audio_count = audio_count
                    .checked_add(1)
                    .ok_or_else(|| WeightPlanError::invalid("audio tensor count overflow"))?;
            } else {
                vision_count = vision_count
                    .checked_add(1)
                    .ok_or_else(|| WeightPlanError::invalid("vision tensor count overflow"))?;
            }
        }
        let logical_bytes = descriptor
            .logical_shape
            .iter()
            .try_fold(2_u64, |bytes, dimension| bytes.checked_mul(*dimension))
            .ok_or_else(|| WeightPlanError::invalid("logical tensor bytes overflow"))?;
        let destination_start = if classification == WeightClassification::KnownUnconsumed {
            None
        } else {
            let start = destination_cursor;
            destination_cursor = destination_cursor
                .checked_add(logical_bytes)
                .ok_or_else(|| WeightPlanError::invalid("destination size overflow"))?;
            Some(start)
        };
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.logical_name.clone(),
            classification,
            consumer,
            dtype: TensorDType::Bf16,
            shape: descriptor.logical_shape.clone(),
            source_file: "model.safetensors".to_owned(),
            locked_file_size: crate::UNSLOTH_GEMMA4_NVFP4_MODEL_SIZE,
            locked_file_sha256: crate::UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256.to_owned(),
            source_range: descriptor.value_range,
            destination_start,
            chunks: Vec::new(),
        });
    }
    if observed_consumers != expected_consumers
        || audio_count != lock.model.architecture.audio.tensor_count
        || vision_count != lock.model.architecture.vision.tensor_count
        || entries.len() as u64 != lock.model.tensor_contract.tensor_count
    {
        return Err(WeightPlanError::invalid(format!(
            "quantized Gemma consumer/count contract differs: consumers={}/{}, audio={audio_count}/{}, vision={vision_count}/{}, all={}/{}",
            observed_consumers.len(),
            expected_consumers.len(),
            lock.model.architecture.audio.tensor_count,
            lock.model.architecture.vision.tensor_count,
            entries.len(),
            lock.model.tensor_contract.tensor_count
        )));
    }
    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version: "quantized-model-plan-v1",
            repo_id: artifact.repository(),
            resolved_revision: artifact.resolved_revision(),
            fingerprint: lock.fingerprint(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: "quantized-model-plan-v1".to_owned(),
        repo_id: artifact.repository().to_owned(),
        resolved_revision: artifact.resolved_revision().to_owned(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: true,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

fn validate_fixed_lock(lock: &ModelLock) -> Result<(), WeightPlanError> {
    let config = &lock.model.architecture.text_config;
    if lock.schema_version != QWEN_SCHEMA_VERSION || reviewed_qwen35_spec(lock).is_none() {
        return Err(WeightPlanError::invalid(
            "model identity is not a reviewed Qwen3.5 dense contract",
        ));
    }
    if config.num_hidden_layers
        != u64::try_from(config.layer_types.len())
            .map_err(|_| WeightPlanError::invalid("layer count does not fit u64"))?
    {
        return Err(WeightPlanError::invalid(
            "layer schedule length differs from num_hidden_layers",
        ));
    }
    Ok(())
}

fn locked_file_map(lock: &ModelLock) -> Result<BTreeMap<&str, &LockedFile>, WeightPlanError> {
    locked_file_map_from_files(&lock.model.files)
}

fn locked_file_map_from_files(
    locked_files: &[LockedFile],
) -> Result<BTreeMap<&str, &LockedFile>, WeightPlanError> {
    let mut files = BTreeMap::new();
    for file in locked_files {
        if file.sha256.len() != 64
            || !file
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(WeightPlanError::invalid(format!(
                "locked file SHA-256 is not lowercase hex: {}",
                file.path
            )));
        }
        if files.insert(file.path.as_str(), file).is_some() {
            return Err(WeightPlanError::invalid(format!(
                "duplicate locked file: {}",
                file.path
            )));
        }
    }
    Ok(files)
}

fn classify_gemma4_descriptor(
    descriptor: &TensorDescriptor,
) -> Result<(WeightClassification, Option<WeightConsumerKey>), WeightPlanError> {
    classify_gemma4_name(descriptor.tensor_name.as_str())
}

fn classify_gemma4_name(
    name: &str,
) -> Result<(WeightClassification, Option<WeightConsumerKey>), WeightPlanError> {
    let top_level = match name {
        "model.language_model.embed_tokens.weight" => Some(WeightConsumer::EmbeddingAndTiedOutput),
        "model.language_model.norm.weight" => Some(WeightConsumer::FinalNorm),
        _ => None,
    };
    if let Some(role) = top_level {
        return Ok((
            WeightClassification::Required,
            Some(WeightConsumerKey { layer: None, role }),
        ));
    }
    if name.starts_with("model.embed_audio.")
        || name.starts_with("model.embed_vision.")
        || name.starts_with("model.vision_embedder.")
    {
        return Ok((WeightClassification::KnownUnconsumed, None));
    }
    const LAYER_PREFIX: &str = "model.language_model.layers.";
    let remainder = name
        .strip_prefix(LAYER_PREFIX)
        .ok_or_else(|| WeightPlanError::invalid(format!("unknown Gemma 4 tensor name: {name}")))?;
    let (layer_text, suffix) = remainder.split_once('.').ok_or_else(|| {
        WeightPlanError::invalid(format!("malformed Gemma 4 layer tensor: {name}"))
    })?;
    let layer = layer_text
        .parse::<u64>()
        .map_err(|_| WeightPlanError::invalid(format!("invalid layer index: {name}")))?;
    if layer.to_string() != layer_text {
        return Err(WeightPlanError::invalid(format!(
            "layer index is not canonical decimal: {name}"
        )));
    }
    let layer_type = crate::gemma4::reviewed_layer_schedule()
        .get(
            usize::try_from(layer)
                .map_err(|_| WeightPlanError::invalid("layer index does not fit usize"))?,
        )
        .copied()
        .ok_or_else(|| WeightPlanError::invalid(format!("layer index is out of range: {layer}")))?;
    let common = match suffix {
        "input_layernorm.weight" => Some(WeightConsumer::InputNorm),
        "post_attention_layernorm.weight" => Some(WeightConsumer::PostAttentionNorm),
        "pre_feedforward_layernorm.weight" => Some(WeightConsumer::PreFeedforwardNorm),
        "post_feedforward_layernorm.weight" => Some(WeightConsumer::PostFeedforwardNorm),
        "layer_scalar" => Some(WeightConsumer::LayerScalar),
        "mlp.gate_proj.weight" => Some(WeightConsumer::MlpGate),
        "mlp.up_proj.weight" => Some(WeightConsumer::MlpUp),
        "mlp.down_proj.weight" => Some(WeightConsumer::MlpDown),
        "self_attn.q_proj.weight" => Some(WeightConsumer::AttentionQ),
        "self_attn.o_proj.weight" => Some(WeightConsumer::AttentionO),
        "self_attn.q_norm.weight" => Some(WeightConsumer::AttentionQNorm),
        "self_attn.k_norm.weight" => Some(WeightConsumer::AttentionKNorm),
        "self_attn.k_proj.weight" => Some(match layer_type {
            crate::gemma4::Gemma4LayerType::SlidingAttention => WeightConsumer::AttentionK,
            crate::gemma4::Gemma4LayerType::FullAttention => WeightConsumer::AttentionKAndV,
        }),
        "self_attn.v_proj.weight"
            if layer_type == crate::gemma4::Gemma4LayerType::SlidingAttention =>
        {
            Some(WeightConsumer::AttentionV)
        }
        _ => None,
    };
    let role = common.ok_or_else(|| {
        WeightPlanError::invalid(format!(
            "tensor suffix is invalid for its Gemma 4 layer class: {name}"
        ))
    })?;
    Ok((
        WeightClassification::Required,
        Some(WeightConsumerKey {
            layer: Some(layer),
            role,
        }),
    ))
}

fn expected_gemma4_consumers() -> BTreeSet<WeightConsumerKey> {
    let mut expected = BTreeSet::from([
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::EmbeddingAndTiedOutput,
        },
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::FinalNorm,
        },
    ]);
    for (layer, layer_type) in crate::gemma4::reviewed_layer_schedule()
        .into_iter()
        .enumerate()
    {
        let layer = u64::try_from(layer).expect("Gemma 4 layer index fits u64");
        for role in [
            WeightConsumer::InputNorm,
            WeightConsumer::PostAttentionNorm,
            WeightConsumer::PreFeedforwardNorm,
            WeightConsumer::PostFeedforwardNorm,
            WeightConsumer::LayerScalar,
            WeightConsumer::MlpGate,
            WeightConsumer::MlpUp,
            WeightConsumer::MlpDown,
            WeightConsumer::AttentionQ,
            WeightConsumer::AttentionO,
            WeightConsumer::AttentionQNorm,
            WeightConsumer::AttentionKNorm,
        ] {
            expected.insert(WeightConsumerKey {
                layer: Some(layer),
                role,
            });
        }
        match layer_type {
            crate::gemma4::Gemma4LayerType::SlidingAttention => {
                for role in [WeightConsumer::AttentionK, WeightConsumer::AttentionV] {
                    expected.insert(WeightConsumerKey {
                        layer: Some(layer),
                        role,
                    });
                }
            }
            crate::gemma4::Gemma4LayerType::FullAttention => {
                expected.insert(WeightConsumerKey {
                    layer: Some(layer),
                    role: WeightConsumer::AttentionKAndV,
                });
            }
        }
    }
    expected
}

fn classify_gemma4_mtp_name(name: &str) -> Result<WeightConsumerKey, WeightPlanError> {
    let top_level = match name {
        "model.embed_tokens.weight" => Some(WeightConsumer::EmbeddingAndTiedOutput),
        "model.norm.weight" => Some(WeightConsumer::FinalNorm),
        "pre_projection.weight" => Some(WeightConsumer::Gemma4MtpPreProjection),
        "post_projection.weight" => Some(WeightConsumer::Gemma4MtpPostProjection),
        _ => None,
    };
    if let Some(role) = top_level {
        return Ok(WeightConsumerKey { layer: None, role });
    }
    const PREFIX: &str = "model.layers.";
    let remainder = name
        .strip_prefix(PREFIX)
        .ok_or_else(|| WeightPlanError::invalid(format!("unknown Gemma 4 MTP tensor: {name}")))?;
    let (layer_text, suffix) = remainder.split_once('.').ok_or_else(|| {
        WeightPlanError::invalid(format!("malformed Gemma 4 MTP layer tensor: {name}"))
    })?;
    let layer = layer_text
        .parse::<u64>()
        .map_err(|_| WeightPlanError::invalid(format!("invalid MTP layer index: {name}")))?;
    if layer.to_string() != layer_text
        || usize::try_from(layer)
            .ok()
            .is_none_or(|index| index >= crate::gemma4_mtp::reviewed_mtp_layer_schedule().len())
    {
        return Err(WeightPlanError::invalid(format!(
            "Gemma 4 MTP layer index is noncanonical or out of range: {name}"
        )));
    }
    let role = match suffix {
        "input_layernorm.weight" => WeightConsumer::InputNorm,
        "post_attention_layernorm.weight" => WeightConsumer::PostAttentionNorm,
        "pre_feedforward_layernorm.weight" => WeightConsumer::PreFeedforwardNorm,
        "post_feedforward_layernorm.weight" => WeightConsumer::PostFeedforwardNorm,
        "layer_scalar" => WeightConsumer::LayerScalar,
        "mlp.gate_proj.weight" => WeightConsumer::MlpGate,
        "mlp.up_proj.weight" => WeightConsumer::MlpUp,
        "mlp.down_proj.weight" => WeightConsumer::MlpDown,
        "self_attn.q_proj.weight" => WeightConsumer::AttentionQ,
        "self_attn.q_norm.weight" => WeightConsumer::AttentionQNorm,
        "self_attn.o_proj.weight" => WeightConsumer::AttentionO,
        _ => {
            return Err(WeightPlanError::invalid(format!(
                "tensor suffix is invalid for Gemma 4 MTP: {name}"
            )));
        }
    };
    Ok(WeightConsumerKey {
        layer: Some(layer),
        role,
    })
}

fn expected_gemma4_mtp_consumers() -> BTreeSet<WeightConsumerKey> {
    let mut expected = BTreeSet::from([
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::EmbeddingAndTiedOutput,
        },
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::FinalNorm,
        },
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::Gemma4MtpPreProjection,
        },
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::Gemma4MtpPostProjection,
        },
    ]);
    for layer in 0..4_u64 {
        for role in [
            WeightConsumer::InputNorm,
            WeightConsumer::PostAttentionNorm,
            WeightConsumer::PreFeedforwardNorm,
            WeightConsumer::PostFeedforwardNorm,
            WeightConsumer::LayerScalar,
            WeightConsumer::MlpGate,
            WeightConsumer::MlpUp,
            WeightConsumer::MlpDown,
            WeightConsumer::AttentionQ,
            WeightConsumer::AttentionQNorm,
            WeightConsumer::AttentionO,
        ] {
            expected.insert(WeightConsumerKey {
                layer: Some(layer),
                role,
            });
        }
    }
    expected
}

fn validate_descriptor(
    descriptor: &TensorDescriptor,
    locked_files: &BTreeMap<&str, &LockedFile>,
) -> Result<(), WeightPlanError> {
    let [start, end] = descriptor.absolute_byte_range;
    if start >= end || descriptor.byte_size == 0 {
        return Err(WeightPlanError::invalid(format!(
            "tensor has an empty or reversed range: {}",
            descriptor.tensor_name
        )));
    }
    let span = end
        .checked_sub(start)
        .ok_or_else(|| WeightPlanError::invalid("tensor source range underflow"))?;
    if span != descriptor.byte_size {
        return Err(WeightPlanError::invalid(format!(
            "tensor range size differs from byte_size: {}",
            descriptor.tensor_name
        )));
    }
    let locked = locked_files
        .get(descriptor.source_file.as_str())
        .ok_or_else(|| {
            WeightPlanError::invalid(format!(
                "tensor source is not a locked file: {}",
                descriptor.source_file
            ))
        })?;
    if end > locked.size_bytes {
        return Err(WeightPlanError::invalid(format!(
            "tensor range exceeds locked source file: {}",
            descriptor.tensor_name
        )));
    }
    Ok(())
}

fn classify_descriptor(
    descriptor: &TensorDescriptor,
    layer_types: &[LayerType],
    vision_prefix: &str,
    mtp_prefix: &str,
    tied_embeddings: bool,
) -> Result<(WeightClassification, Option<WeightConsumerKey>), WeightPlanError> {
    let name = descriptor.tensor_name.as_str();
    let top_level = match name {
        "model.language_model.embed_tokens.weight" => Some(if tied_embeddings {
            WeightConsumer::EmbeddingAndTiedOutput
        } else {
            WeightConsumer::Embedding
        }),
        "model.language_model.norm.weight" => Some(WeightConsumer::FinalNorm),
        "lm_head.weight" if !tied_embeddings => Some(WeightConsumer::OutputProjection),
        "lm_head.weight" | "model.language_model.lm_head.weight" => {
            return Err(WeightPlanError::invalid(
                "independent lm_head contradicts tied embeddings",
            ));
        }
        _ => None,
    };
    if let Some(role) = top_level {
        return Ok((
            WeightClassification::Required,
            Some(WeightConsumerKey { layer: None, role }),
        ));
    }
    if name.starts_with(vision_prefix) || name.starts_with(mtp_prefix) {
        return Ok((WeightClassification::KnownUnconsumed, None));
    }

    const LAYER_PREFIX: &str = "model.language_model.layers.";
    let remainder = name.strip_prefix(LAYER_PREFIX).ok_or_else(|| {
        WeightPlanError::invalid(format!("unknown tensor name: {}", descriptor.tensor_name))
    })?;
    let (layer_text, suffix) = remainder.split_once('.').ok_or_else(|| {
        WeightPlanError::invalid(format!(
            "malformed layer tensor: {}",
            descriptor.tensor_name
        ))
    })?;
    let layer = layer_text.parse::<u64>().map_err(|_| {
        WeightPlanError::invalid(format!("invalid layer index: {}", descriptor.tensor_name))
    })?;
    if layer.to_string() != layer_text {
        return Err(WeightPlanError::invalid(format!(
            "layer index is not canonical decimal: {}",
            descriptor.tensor_name
        )));
    }
    let layer_index = usize::try_from(layer)
        .map_err(|_| WeightPlanError::invalid("layer index does not fit usize"))?;
    let layer_type = layer_types
        .get(layer_index)
        .ok_or_else(|| WeightPlanError::invalid(format!("layer index is out of range: {layer}")))?;
    let common = match suffix {
        "input_layernorm.weight" => Some(WeightConsumer::InputNorm),
        "post_attention_layernorm.weight" => Some(WeightConsumer::PostAttentionNorm),
        "mlp.gate_proj.weight" => Some(WeightConsumer::MlpGate),
        "mlp.up_proj.weight" => Some(WeightConsumer::MlpUp),
        "mlp.down_proj.weight" => Some(WeightConsumer::MlpDown),
        _ => None,
    };
    let role = if let Some(role) = common {
        role
    } else {
        match (layer_type, suffix) {
            (LayerType::LinearAttention, "linear_attn.in_proj_qkv.weight") => {
                WeightConsumer::GdnInProjQkv
            }
            (LayerType::LinearAttention, "linear_attn.in_proj_z.weight") => {
                WeightConsumer::GdnInProjZ
            }
            (LayerType::LinearAttention, "linear_attn.in_proj_b.weight") => {
                WeightConsumer::GdnInProjB
            }
            (LayerType::LinearAttention, "linear_attn.in_proj_a.weight") => {
                WeightConsumer::GdnInProjA
            }
            (LayerType::LinearAttention, "linear_attn.conv1d.weight") => WeightConsumer::GdnConv1d,
            (LayerType::LinearAttention, "linear_attn.A_log") => WeightConsumer::GdnALog,
            (LayerType::LinearAttention, "linear_attn.dt_bias") => WeightConsumer::GdnDtBias,
            (LayerType::LinearAttention, "linear_attn.norm.weight") => WeightConsumer::GdnNorm,
            (LayerType::LinearAttention, "linear_attn.out_proj.weight") => {
                WeightConsumer::GdnOutProj
            }
            (LayerType::FullAttention, "self_attn.q_proj.weight") => WeightConsumer::AttentionQ,
            (LayerType::FullAttention, "self_attn.k_proj.weight") => WeightConsumer::AttentionK,
            (LayerType::FullAttention, "self_attn.v_proj.weight") => WeightConsumer::AttentionV,
            (LayerType::FullAttention, "self_attn.o_proj.weight") => WeightConsumer::AttentionO,
            (LayerType::FullAttention, "self_attn.q_norm.weight") => WeightConsumer::AttentionQNorm,
            (LayerType::FullAttention, "self_attn.k_norm.weight") => WeightConsumer::AttentionKNorm,
            _ => {
                return Err(WeightPlanError::invalid(format!(
                    "tensor suffix is invalid for its layer class: {}",
                    descriptor.tensor_name
                )));
            }
        }
    };
    Ok((
        WeightClassification::Required,
        Some(WeightConsumerKey {
            layer: Some(layer),
            role,
        }),
    ))
}

fn expected_consumers(
    layer_types: &[LayerType],
    tied_embeddings: bool,
) -> BTreeSet<WeightConsumerKey> {
    let embedding_role = if tied_embeddings {
        WeightConsumer::EmbeddingAndTiedOutput
    } else {
        WeightConsumer::Embedding
    };
    let mut expected = BTreeSet::from([
        WeightConsumerKey {
            layer: None,
            role: embedding_role,
        },
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::FinalNorm,
        },
    ]);
    if !tied_embeddings {
        expected.insert(WeightConsumerKey {
            layer: None,
            role: WeightConsumer::OutputProjection,
        });
    }
    for (layer, layer_type) in layer_types.iter().enumerate() {
        let layer = u64::try_from(layer).expect("Qwen layer index fits u64");
        for role in [
            WeightConsumer::InputNorm,
            WeightConsumer::PostAttentionNorm,
            WeightConsumer::MlpGate,
            WeightConsumer::MlpUp,
            WeightConsumer::MlpDown,
        ] {
            expected.insert(WeightConsumerKey {
                layer: Some(layer),
                role,
            });
        }
        let roles: &[WeightConsumer] = match layer_type {
            LayerType::LinearAttention => &[
                WeightConsumer::GdnInProjQkv,
                WeightConsumer::GdnInProjZ,
                WeightConsumer::GdnInProjB,
                WeightConsumer::GdnInProjA,
                WeightConsumer::GdnConv1d,
                WeightConsumer::GdnALog,
                WeightConsumer::GdnDtBias,
                WeightConsumer::GdnNorm,
                WeightConsumer::GdnOutProj,
            ],
            LayerType::FullAttention => &[
                WeightConsumer::AttentionQ,
                WeightConsumer::AttentionK,
                WeightConsumer::AttentionV,
                WeightConsumer::AttentionO,
                WeightConsumer::AttentionQNorm,
                WeightConsumer::AttentionKNorm,
            ],
        };
        for &role in roles {
            expected.insert(WeightConsumerKey {
                layer: Some(layer),
                role,
            });
        }
    }
    expected
}

fn split_chunks(
    descriptor: &TensorDescriptor,
    destination_start: u64,
) -> Result<Vec<WeightLoadChunk>, WeightPlanError> {
    let mut chunks = Vec::new();
    let mut relative = 0_u64;
    while relative < descriptor.byte_size {
        let remaining = descriptor
            .byte_size
            .checked_sub(relative)
            .ok_or_else(|| WeightPlanError::invalid("chunk remaining underflow"))?;
        let byte_length = remaining.min(WEIGHT_LOAD_CHUNK_BYTES);
        let source_offset = descriptor
            .absolute_start()
            .checked_add(relative)
            .ok_or_else(|| WeightPlanError::invalid("chunk source offset overflow"))?;
        let destination_offset = destination_start
            .checked_add(relative)
            .ok_or_else(|| WeightPlanError::invalid("chunk destination offset overflow"))?;
        chunks.push(WeightLoadChunk {
            source_offset,
            destination_offset,
            byte_length,
        });
        relative = relative
            .checked_add(byte_length)
            .ok_or_else(|| WeightPlanError::invalid("chunk cursor overflow"))?;
    }
    Ok(chunks)
}

struct PlanDigestHeader<'a> {
    schema_version: &'a str,
    repo_id: &'a str,
    resolved_revision: &'a str,
    fingerprint: &'a str,
    tied_embeddings: bool,
    chunk_size: u64,
    total_destination_bytes: u64,
}

fn digest_plan(
    header: &PlanDigestHeader<'_>,
    entries: &[WeightLoadEntry],
) -> Result<[u8; 32], WeightPlanError> {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(PLAN_DOMAIN);
    put_string(&mut encoded, header.schema_version)?;
    put_string(&mut encoded, header.repo_id)?;
    put_string(&mut encoded, header.resolved_revision)?;
    put_string(&mut encoded, header.fingerprint)?;
    encoded.push(u8::from(header.tied_embeddings));
    put_u64(&mut encoded, header.chunk_size);
    put_u64(
        &mut encoded,
        u64::try_from(entries.len())
            .map_err(|_| WeightPlanError::invalid("entry count does not fit u64"))?,
    );
    put_u64(&mut encoded, header.total_destination_bytes);
    for entry in entries {
        put_string(&mut encoded, &entry.tensor_name)?;
        encoded.push(entry.classification.tag());
        match entry.consumer.and_then(|consumer| consumer.layer) {
            None => encoded.push(0),
            Some(layer) => {
                encoded.push(1);
                put_u64(&mut encoded, layer);
            }
        }
        encoded.push(entry.consumer.map_or(0, |consumer| consumer.role.tag()));
        encoded.push(dtype_tag(entry.dtype));
        put_u64(
            &mut encoded,
            u64::try_from(entry.shape.len())
                .map_err(|_| WeightPlanError::invalid("shape rank does not fit u64"))?,
        );
        for &dimension in &entry.shape {
            put_u64(&mut encoded, dimension);
        }
        put_string(&mut encoded, &entry.source_file)?;
        put_u64(&mut encoded, entry.locked_file_size);
        put_string(&mut encoded, &entry.locked_file_sha256)?;
        put_u64(&mut encoded, entry.source_range[0]);
        put_u64(&mut encoded, entry.source_range[1]);
        put_u64(&mut encoded, entry.destination_start.unwrap_or(0));
        put_u64(
            &mut encoded,
            u64::try_from(entry.chunks.len())
                .map_err(|_| WeightPlanError::invalid("chunk count does not fit u64"))?,
        );
        for chunk in &entry.chunks {
            put_u64(&mut encoded, chunk.source_offset);
            put_u64(&mut encoded, chunk.destination_offset);
            put_u64(&mut encoded, chunk.byte_length);
        }
    }
    Ok(Sha256::digest(encoded).into())
}

fn put_string(output: &mut Vec<u8>, value: &str) -> Result<(), WeightPlanError> {
    put_u64(
        output,
        u64::try_from(value.len())
            .map_err(|_| WeightPlanError::invalid("string length does not fit u64"))?,
    );
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn dtype_tag(dtype: TensorDType) -> u8 {
    match dtype {
        TensorDType::Bf16 => 1,
        TensorDType::F16 => 2,
        TensorDType::F32 => 3,
        TensorDType::I32 => 4,
        TensorDType::I64 => 5,
        TensorDType::U8 => 6,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn standard_nvfp4_blocks_restore_the_exact_adjacent_source_bytes() {
        let source: Vec<u8> = (0..32).map(|index| (index * 19 + 7) as u8).collect();
        let scales = [0x31, 0x42, 0x53, 0x64];
        let standard = crate::repack_nvfp4_standard(&source, &scales, 1, 64).unwrap();
        let mut restored = Vec::new();
        for subblock in standard[4..].chunks_exact(8) {
            append_adjacent_nibbles(subblock, 16, &mut restored);
        }
        assert_eq!(restored, source);
    }

    #[test]
    fn gemma4_plan_classifies_exact_text_and_unconsumed_catalog() {
        let lock = crate::gemma4::parse_gemma4_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-bf16.json"
        ))
        .expect("reviewed Gemma 4 lock parses");
        let catalog = crate::gemma4::expected_gemma4_tensor_catalog()
            .expect("reviewed Gemma 4 catalog derives");
        let plan = build_gemma4_weight_load_plan(&lock, catalog.values())
            .expect("reviewed Gemma 4 load plan builds");
        assert_eq!(plan.entries.len(), 677);
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| entry.classification == WeightClassification::KnownUnconsumed)
                .count(),
            11
        );
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| entry.destination_start.is_some())
                .count(),
            666
        );
        assert_eq!(plan.total_destination_bytes, 23_814_700_640);
        let consumer = |name: &str| {
            plan.entries
                .iter()
                .find(|entry| entry.tensor_name == name)
                .and_then(|entry| entry.consumer)
                .map(|consumer| consumer.role)
        };
        assert_eq!(
            consumer("model.language_model.layers.5.self_attn.k_proj.weight"),
            Some(WeightConsumer::AttentionKAndV)
        );
        assert_eq!(
            consumer("model.language_model.layers.6.self_attn.k_proj.weight"),
            Some(WeightConsumer::AttentionK)
        );
        assert_eq!(
            consumer("model.language_model.layers.6.self_attn.v_proj.weight"),
            Some(WeightConsumer::AttentionV)
        );
        assert_eq!(
            consumer("model.language_model.layers.47.post_feedforward_layernorm.weight"),
            Some(WeightConsumer::PostFeedforwardNorm)
        );
    }

    #[test]
    fn gemma4_plan_accepts_the_exact_instruction_tuned_lock() {
        let lock = crate::gemma4::parse_gemma4_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-it-bf16.json"
        ))
        .expect("reviewed Gemma 4 IT lock parses");
        let catalog = crate::gemma4::expected_gemma4_tensor_catalog()
            .expect("reviewed Gemma 4 catalog derives");
        let plan = build_gemma4_weight_load_plan(&lock, catalog.values())
            .expect("reviewed Gemma 4 IT load plan builds");
        assert_eq!(plan.repo_id, crate::gemma4::GEMMA4_12B_IT_REPO_ID);
        assert_eq!(
            plan.lock_fingerprint,
            crate::gemma4::GEMMA4_12B_IT_FINGERPRINT
        );
        assert_eq!(plan.total_destination_bytes, 23_814_700_640);
    }

    #[test]
    fn gemma4_mtp_plan_closes_the_exact_query_only_catalog() {
        let lock = crate::gemma4_mtp::parse_gemma4_mtp_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json"
        ))
        .expect("reviewed Gemma 4 MTP lock parses");
        let catalog = crate::gemma4_mtp::expected_gemma4_mtp_tensor_catalog()
            .expect("reviewed Gemma 4 MTP catalog derives");
        let plan = build_gemma4_mtp_weight_load_plan(&lock, catalog.values())
            .expect("reviewed Gemma 4 MTP load plan builds");
        assert_eq!(plan.entries.len(), 48);
        assert_eq!(plan.total_destination_bytes, 845_713_928);
        let consumer = |name: &str| {
            plan.entries
                .iter()
                .find(|entry| entry.tensor_name == name)
                .and_then(|entry| entry.consumer)
                .map(|consumer| consumer.role)
        };
        assert_eq!(
            consumer("pre_projection.weight"),
            Some(WeightConsumer::Gemma4MtpPreProjection)
        );
        assert_eq!(
            consumer("post_projection.weight"),
            Some(WeightConsumer::Gemma4MtpPostProjection)
        );
        assert_eq!(
            consumer("model.layers.3.self_attn.q_proj.weight"),
            Some(WeightConsumer::AttentionQ)
        );
        assert!(plan.entries.iter().all(|entry| {
            !matches!(
                entry.consumer.map(|consumer| consumer.role),
                Some(
                    WeightConsumer::AttentionK
                        | WeightConsumer::AttentionV
                        | WeightConsumer::AttentionKAndV
                )
            )
        }));
    }

    #[test]
    fn gemma4_mtp_plan_rejects_missing_extra_kv_and_wrong_dtype() {
        let lock = crate::gemma4_mtp::parse_gemma4_mtp_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json"
        ))
        .expect("reviewed Gemma 4 MTP lock parses");
        let mut catalog = crate::gemma4_mtp::expected_gemma4_mtp_tensor_catalog()
            .expect("reviewed Gemma 4 MTP catalog derives");
        let original = catalog
            .remove("model.layers.0.layer_scalar")
            .expect("boundary tensor exists");
        assert!(build_gemma4_mtp_weight_load_plan(&lock, catalog.values()).is_err());

        catalog.insert(original.tensor_name.clone(), original.clone());
        let mut unexpected_kv = original.clone();
        unexpected_kv.tensor_name = "model.layers.0.self_attn.k_proj.weight".to_owned();
        catalog.insert(unexpected_kv.tensor_name.clone(), unexpected_kv);
        assert!(build_gemma4_mtp_weight_load_plan(&lock, catalog.values()).is_err());

        catalog.remove("model.layers.0.self_attn.k_proj.weight");
        let descriptor = catalog
            .get_mut("model.layers.0.layer_scalar")
            .expect("restored tensor exists");
        descriptor.dtype = TensorDType::F32;
        assert!(build_gemma4_mtp_weight_load_plan(&lock, catalog.values()).is_err());
    }

    #[test]
    fn gemma4_plan_rejects_missing_extra_and_full_layer_v_weight() {
        let lock = crate::gemma4::parse_gemma4_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-bf16.json"
        ))
        .expect("reviewed Gemma 4 lock parses");
        let mut catalog = crate::gemma4::expected_gemma4_tensor_catalog()
            .expect("reviewed Gemma 4 catalog derives");
        let missing = catalog
            .remove("model.language_model.layers.0.layer_scalar")
            .expect("boundary tensor exists");
        assert!(build_gemma4_weight_load_plan(&lock, catalog.values()).is_err());
        catalog.insert(missing.tensor_name.clone(), missing.clone());
        let mut invalid = missing;
        invalid.tensor_name = "model.language_model.layers.5.self_attn.v_proj.weight".to_owned();
        catalog.insert(invalid.tensor_name.clone(), invalid);
        assert!(build_gemma4_weight_load_plan(&lock, catalog.values()).is_err());
    }

    struct FakeWeightSource {
        fingerprint: String,
        descriptor: TensorDescriptor,
        bytes: Vec<u8>,
        reads: Cell<usize>,
    }

    impl WeightRangeSource for FakeWeightSource {
        fn lock_fingerprint(&self) -> &str {
            &self.fingerprint
        }

        fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor> {
            (tensor_name == self.descriptor.tensor_name).then_some(&self.descriptor)
        }

        fn read_tensor_range(
            &self,
            tensor_name: &str,
            offset: u64,
            length: usize,
        ) -> Result<Vec<u8>, WeightUploadError> {
            self.reads.set(self.reads.get() + 1);
            if tensor_name != self.descriptor.tensor_name {
                return Err(WeightUploadError::invalid("unexpected fake tensor"));
            }
            let start = usize::try_from(offset)
                .map_err(|_| WeightUploadError::invalid("fake offset overflow"))?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| WeightUploadError::invalid("fake range overflow"))?;
            self.bytes
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| WeightUploadError::invalid("fake range is out of bounds"))
        }
    }

    fn upload_fixture() -> (WeightLoadPlan, FakeWeightSource) {
        let byte_size = WEIGHT_LOAD_CHUNK_BYTES + 1;
        let tensor_name = "model.language_model.norm.weight";
        let source_file = "model-00001-of-00002.safetensors";
        let descriptor = TensorDescriptor {
            tensor_name: tensor_name.to_owned(),
            source_file: source_file.to_owned(),
            dtype: TensorDType::Bf16,
            shape: vec![byte_size],
            header_length_field_bytes: 8,
            header_length_bytes: 9,
            data_buffer_start: 17,
            data_offset_basis: "safetensors-data-buffer".to_owned(),
            data_offsets: [0, byte_size],
            absolute_byte_range: [17, 17 + byte_size],
            byte_size,
        };
        let chunks = split_chunks(&descriptor, 91).unwrap();
        let entry = WeightLoadEntry {
            tensor_name: tensor_name.to_owned(),
            classification: WeightClassification::Required,
            consumer: Some(WeightConsumerKey {
                layer: None,
                role: WeightConsumer::FinalNorm,
            }),
            dtype: TensorDType::Bf16,
            shape: descriptor.shape.clone(),
            source_file: source_file.to_owned(),
            locked_file_size: descriptor.absolute_end(),
            locked_file_sha256: "0".repeat(64),
            source_range: descriptor.absolute_byte_range,
            destination_start: Some(91),
            chunks,
        };
        let mut plan = WeightLoadPlan {
            schema_version: QWEN_SCHEMA_VERSION.to_owned(),
            repo_id: QWEN_REPO_ID.to_owned(),
            resolved_revision: QWEN_REVISION.to_owned(),
            lock_fingerprint: QWEN_FINGERPRINT.to_owned(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: 91 + byte_size,
            entries: vec![entry],
            digest: [0; 32],
        };
        plan.digest = plan.recompute_digest().unwrap();
        let bytes = (0..byte_size)
            .map(|offset| ((offset * 37 + 11) % 251) as u8)
            .collect();
        (
            plan,
            FakeWeightSource {
                fingerprint: QWEN_FINGERPRINT.to_owned(),
                descriptor,
                bytes,
                reads: Cell::new(0),
            },
        )
    }

    #[test]
    fn canonical_wire_vector_matches_the_reader_contract() {
        let entry = WeightLoadEntry {
            tensor_name: "x".to_owned(),
            classification: WeightClassification::Required,
            consumer: Some(WeightConsumerKey {
                layer: None,
                role: WeightConsumer::EmbeddingAndTiedOutput,
            }),
            dtype: TensorDType::Bf16,
            shape: vec![3],
            source_file: "model-00001-of-00002.safetensors".to_owned(),
            locked_file_size: 20,
            locked_file_sha256: "0".repeat(64),
            source_range: [17, 20],
            destination_start: Some(0),
            chunks: vec![WeightLoadChunk {
                source_offset: 17,
                destination_offset: 0,
                byte_length: 3,
            }],
        };
        let mut encoded = Vec::new();
        encoded.extend_from_slice(PLAN_DOMAIN);
        put_string(&mut encoded, QWEN_SCHEMA_VERSION).unwrap();
        put_string(&mut encoded, QWEN_REPO_ID).unwrap();
        put_string(&mut encoded, QWEN_REVISION).unwrap();
        put_string(&mut encoded, QWEN_FINGERPRINT).unwrap();
        encoded.push(1);
        put_u64(&mut encoded, WEIGHT_LOAD_CHUNK_BYTES);
        put_u64(&mut encoded, 1);
        put_u64(&mut encoded, 3);
        put_string(&mut encoded, &entry.tensor_name).unwrap();
        encoded.push(entry.classification.tag());
        encoded.push(0);
        encoded.push(entry.consumer.unwrap().role.tag());
        encoded.push(dtype_tag(entry.dtype));
        put_u64(&mut encoded, 1);
        put_u64(&mut encoded, 3);
        put_string(&mut encoded, &entry.source_file).unwrap();
        put_u64(&mut encoded, entry.locked_file_size);
        put_string(&mut encoded, &entry.locked_file_sha256).unwrap();
        put_u64(&mut encoded, 17);
        put_u64(&mut encoded, 20);
        put_u64(&mut encoded, 0);
        put_u64(&mut encoded, 1);
        put_u64(&mut encoded, 17);
        put_u64(&mut encoded, 0);
        put_u64(&mut encoded, 3);
        assert_eq!(encoded.len(), 426);
        let digest = Sha256::digest(encoded);
        assert_eq!(
            format!("{digest:x}"),
            "a8ee5d60d9dffeac3020d1b3833677eb66e50958a53412f1ccb1e9c09c174fef"
        );
        let expected_digest: [u8; 32] = digest.into();
        assert_eq!(
            digest_plan(
                &PlanDigestHeader {
                    schema_version: QWEN_SCHEMA_VERSION,
                    repo_id: QWEN_REPO_ID,
                    resolved_revision: QWEN_REVISION,
                    fingerprint: QWEN_FINGERPRINT,
                    tied_embeddings: true,
                    chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
                    total_destination_bytes: 3,
                },
                &[entry],
            )
            .unwrap(),
            expected_digest
        );
    }

    #[test]
    fn chunk_boundaries_are_exact_and_nonzero() {
        for (bytes, expected) in [
            (1, vec![1]),
            (3, vec![3]),
            (17, vec![17]),
            (
                WEIGHT_LOAD_CHUNK_BYTES - 1,
                vec![WEIGHT_LOAD_CHUNK_BYTES - 1],
            ),
            (WEIGHT_LOAD_CHUNK_BYTES, vec![WEIGHT_LOAD_CHUNK_BYTES]),
            (
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                vec![WEIGHT_LOAD_CHUNK_BYTES, 1],
            ),
        ] {
            let descriptor = TensorDescriptor {
                tensor_name: "x".to_owned(),
                source_file: "x.safetensors".to_owned(),
                dtype: TensorDType::Bf16,
                shape: vec![bytes],
                header_length_field_bytes: 8,
                header_length_bytes: 1,
                data_buffer_start: 9,
                data_offset_basis: "safetensors-data-buffer".to_owned(),
                data_offsets: [0, bytes],
                absolute_byte_range: [17, 17 + bytes],
                byte_size: bytes,
            };
            let chunks = split_chunks(&descriptor, 3).unwrap();
            assert_eq!(
                chunks
                    .iter()
                    .map(|chunk| chunk.byte_length)
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(chunks.iter().all(|chunk| chunk.byte_length > 0));
        }
    }

    #[test]
    fn upload_bridge_reads_and_submits_one_exact_chunk_at_a_time() {
        let (plan, source) = upload_fixture();
        let mut calls = 0_usize;
        let receipt = upload_weight_from_source(
            &plan,
            *plan.digest(),
            &source,
            "model.language_model.norm.weight",
            TensorDType::Bf16,
            7,
            WEIGHT_LOAD_CHUNK_BYTES + 1,
            WEIGHT_LOAD_CHUNK_BYTES,
            |relative_offset, bytes| {
                let start = usize::try_from(relative_offset).unwrap();
                assert_eq!(bytes.as_ref(), &source.bytes[start..start + bytes.len()]);
                assert_eq!(relative_offset, calls as u64 * WEIGHT_LOAD_CHUNK_BYTES);
                calls += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(source.reads.get(), 2);
        assert_eq!(receipt.chunks_uploaded, 2);
        assert_eq!(receipt.byte_length, WEIGHT_LOAD_CHUNK_BYTES + 1);
        assert_eq!(receipt.destination_offset, 7);
        assert_eq!(receipt.peak_host_staging_bytes, WEIGHT_LOAD_CHUNK_BYTES);
    }

    #[test]
    fn upload_bridge_rejects_identity_target_dtype_and_range_before_read() {
        let (plan, source) = upload_fixture();
        let invoke = |plan: &WeightLoadPlan,
                      expected_digest: [u8; 32],
                      source: &FakeWeightSource,
                      tensor_name: &str,
                      dtype: TensorDType,
                      destination_size: u64| {
            upload_weight_from_source(
                plan,
                expected_digest,
                source,
                tensor_name,
                dtype,
                7,
                destination_size,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("invalid request must not submit a transfer"),
            )
        };

        let mut wrong_digest = *plan.digest();
        wrong_digest[0] ^= 1;
        assert!(
            invoke(
                &plan,
                wrong_digest,
                &source,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
            )
            .is_err()
        );
        assert!(
            invoke(
                &plan,
                *plan.digest(),
                &source,
                "model.language_model.missing.weight",
                TensorDType::Bf16,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
            )
            .is_err()
        );
        assert!(
            invoke(
                &plan,
                *plan.digest(),
                &source,
                "model.language_model.norm.weight",
                TensorDType::F32,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
            )
            .is_err()
        );
        assert!(
            invoke(
                &plan,
                *plan.digest(),
                &source,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                WEIGHT_LOAD_CHUNK_BYTES,
            )
            .is_err()
        );
        assert_eq!(source.reads.get(), 0);
    }

    #[test]
    fn upload_bridge_rejects_mutated_plan_cache_and_descriptor_before_read() {
        let (plan, source) = upload_fixture();
        let mut mutated_plan = plan.clone();
        mutated_plan.entries[0].chunks[0].byte_length -= 1;
        assert!(
            upload_weight_from_source(
                &mutated_plan,
                *plan.digest(),
                &source,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                7,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("mutated plan must not submit"),
            )
            .is_err()
        );

        let (_, mut wrong_cache) = upload_fixture();
        wrong_cache.fingerprint = format!("sha256:{}", "1".repeat(64));
        assert!(
            upload_weight_from_source(
                &plan,
                *plan.digest(),
                &wrong_cache,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                7,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("wrong cache must not submit"),
            )
            .is_err()
        );

        let (_, mut wrong_descriptor) = upload_fixture();
        wrong_descriptor.descriptor.absolute_byte_range[0] += 1;
        assert!(
            upload_weight_from_source(
                &plan,
                *plan.digest(),
                &wrong_descriptor,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                7,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("wrong descriptor must not submit"),
            )
            .is_err()
        );
        assert_eq!(source.reads.get(), 0);
        assert_eq!(wrong_cache.reads.get(), 0);
        assert_eq!(wrong_descriptor.reads.get(), 0);
    }
}
