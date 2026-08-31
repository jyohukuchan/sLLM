//! Write-disabled GGUF catalog dry run for the reviewed DiffusionGemma artifact.
//!
//! `diffusion-gemma` is an sLLM foundation key.  It is deliberately not
//! represented as an upstream-approved GGUF architecture: the fixed llama.cpp
//! reference does not define DiffusionGemma, and open draft PR #24423 is used
//! only to cross-check its provisional spelling.  This module accepts no output
//! path or writer, creates no `GgufWritePlan`, chooses no file type or tensor
//! dtype, and never claims that an unmaterialized tensor header or payload was
//! inspected.

use crate::gguf::{GgufArray, GgufValue};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DIFFUSION_GEMMA_GGUF_ARCHITECTURE: &str = "diffusion-gemma";
pub const DIFFUSION_GEMMA_REPOSITORY: &str = "google/diffusiongemma-26B-A4B-it";
pub const DIFFUSION_GEMMA_REVISION: &str = "f7f5b7f5fa82ffc52addd066915886d497f5517b";
pub const DIFFUSION_GEMMA_LICENSE: &str = "Apache-2.0";
pub const DIFFUSION_GEMMA_CONFIG_SHA256: &str =
    "13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506";
pub const DIFFUSION_GEMMA_INDEX_SHA256: &str =
    "6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a";
pub const DIFFUSION_GEMMA_SOURCE_TENSOR_COUNT: usize = 1_047;
pub const DIFFUSION_GEMMA_SOURCE_TEXT_TENSOR_COUNT: usize = 657;
pub const DIFFUSION_GEMMA_SOURCE_VISION_TENSOR_COUNT: usize = 355;
pub const DIFFUSION_GEMMA_SOURCE_PROJECTOR_TENSOR_COUNT: usize = 1;
pub const DIFFUSION_GEMMA_SOURCE_DIFFUSION_TENSOR_COUNT: usize = 34;
pub const DIFFUSION_GEMMA_PACKED_EXPERT_SOURCE_TENSOR_COUNT: usize = 60;
pub const DIFFUSION_GEMMA_PHYSICAL_CANDIDATE_COUNT: usize = 1_047;
pub const DIFFUSION_GEMMA_INDEX_ADVERTISED_PAYLOAD_BYTES: u64 = 51_647_562_456;
pub const DIFFUSION_GEMMA_INDEX_ADVERTISED_PARAMETERS: u64 = 25_823_778_864;
pub const DIFFUSION_GEMMA_SHARD_COUNT: usize = 11;
pub const DIFFUSION_GEMMA_GGUF_PASS_SCOPE: &str =
    "exact-config-index-catalog-only-no-headers-no-payload-no-write-no-filetype";
pub const DIFFUSION_GEMMA_GGUF_MAPPING_SERIALIZATION: &str =
    "utf8-tsv-v1:source,shard,family,typed-role,foundation-output;lf-rows";
pub const DIFFUSION_GEMMA_LLAMA_CPP_REFERENCE_COMMIT: &str =
    "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70";
pub const DIFFUSION_GEMMA_LLAMA_CPP_DRAFT_PR: &str =
    "https://github.com/ggml-org/llama.cpp/pull/24423";
pub const DIFFUSION_GEMMA_TRANSFORMERS_REFERENCE_COMMIT: &str =
    "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c";

/// SHA-256 of [`DIFFUSION_GEMMA_GGUF_MAPPING_SERIALIZATION`] for the fixed
/// official 1,047-row index.  This covers names and dry-run decisions, not
/// tensor headers or payload bytes.
pub const DIFFUSION_GEMMA_GGUF_MAPPING_SHA256: &str =
    "65387f7371dc95795e3b96c745cbbd7d9a16d4aa9e925c1de9589f4abde2ae3b";

const TEXT_LAYER_COUNT: u8 = 30;
const VISION_LAYER_COUNT: u8 = 27;
const EXPERT_COUNT: u16 = 128;
const SELECTED_EXPERT_COUNT: u32 = 8;
const FULL_ATTENTION_LAYERS: [u8; 5] = [5, 11, 17, 23, 29];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaGgufFoundationError {
    Invalid(String),
}

impl fmt::Display for DiffusionGemmaGgufFoundationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(
                formatter,
                "invalid DiffusionGemma GGUF foundation catalog: {message}"
            ),
        }
    }
}

impl std::error::Error for DiffusionGemmaGgufFoundationError {}

fn invalid(message: impl Into<String>) -> DiffusionGemmaGgufFoundationError {
    DiffusionGemmaGgufFoundationError::Invalid(message.into())
}

/// The authority of the architecture spelling used by this dry run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaGgufArchitectureAuthority {
    /// Local write-disabled key, cross-checked against an open draft PR only.
    SllmWriteDisabledFoundationNotUpstreamApproved,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaArtifactFamily {
    Text,
    Vision,
    Projector,
    DiffusionSpecific,
}

impl DiffusionGemmaArtifactFamily {
    pub const fn canonical(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Projector => "projector",
            Self::DiffusionSpecific => "diffusion-specific",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaAttentionRole {
    Query,
    QueryNorm,
    Key,
    KeyNorm,
    Value,
    Output,
}

impl DiffusionGemmaAttentionRole {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::QueryNorm => "query-norm",
            Self::Key => "key",
            Self::KeyNorm => "key-norm",
            Self::Value => "value",
            Self::Output => "output",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaProjection {
    Gate,
    Up,
    Down,
}

impl DiffusionGemmaProjection {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaPackedExpertProjection {
    GateUp,
    Down,
}

impl DiffusionGemmaPackedExpertProjection {
    const ALL: [Self; 2] = [Self::GateUp, Self::Down];

    const fn canonical(self) -> &'static str {
        match self {
            Self::GateUp => "gate-up",
            Self::Down => "down",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaTextLayerRole {
    AttentionNorm,
    PostAttentionNorm,
    PreFeedForwardNorm,
    PreFeedForwardNorm2,
    PostFeedForwardNorm,
    PostFeedForwardNorm1,
    PostFeedForwardNorm2,
    LayerOutputScale,
    Attention(DiffusionGemmaAttentionRole),
    DenseMlp(DiffusionGemmaProjection),
    Router,
    RouterScale,
    RouterPerExpertScale,
    PackedExperts(DiffusionGemmaPackedExpertProjection),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaVisionLayerRole {
    AttentionNorm,
    PostAttentionNorm,
    PreFeedForwardNorm,
    PostFeedForwardNorm,
    Attention(DiffusionGemmaAttentionRole),
    Mlp(DiffusionGemmaProjection),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaSelfConditioningRole {
    PreNorm,
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiffusionGemmaTensorRole {
    TokenEmbedding,
    OutputNorm,
    TextLayer {
        layer: u8,
        role: DiffusionGemmaTextLayerRole,
    },
    VisionPatchEmbedding,
    VisionPositionEmbedding,
    VisionStandardizationBias,
    VisionStandardizationScale,
    VisionLayer {
        layer: u8,
        role: DiffusionGemmaVisionLayerRole,
    },
    ImageEmbeddingProjection,
    EncoderLayerOutputScale {
        layer: u8,
    },
    SelfConditioning(DiffusionGemmaSelfConditioningRole),
}

impl DiffusionGemmaTensorRole {
    fn canonical(self) -> String {
        match self {
            Self::TokenEmbedding => "text-root:token-embedding".to_owned(),
            Self::OutputNorm => "text-root:output-norm".to_owned(),
            Self::TextLayer { layer, role } => {
                let role = match role {
                    DiffusionGemmaTextLayerRole::AttentionNorm => "attention-norm".to_owned(),
                    DiffusionGemmaTextLayerRole::PostAttentionNorm => {
                        "post-attention-norm".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::PreFeedForwardNorm => {
                        "pre-feed-forward-norm".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::PreFeedForwardNorm2 => {
                        "pre-feed-forward-norm-2".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::PostFeedForwardNorm => {
                        "post-feed-forward-norm".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::PostFeedForwardNorm1 => {
                        "post-feed-forward-norm-1".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::PostFeedForwardNorm2 => {
                        "post-feed-forward-norm-2".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::LayerOutputScale => {
                        "layer-output-scale".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::Attention(role) => {
                        format!("attention:{}", role.canonical())
                    }
                    DiffusionGemmaTextLayerRole::DenseMlp(projection) => {
                        format!("dense-mlp:{}", projection.canonical())
                    }
                    DiffusionGemmaTextLayerRole::Router => "router".to_owned(),
                    DiffusionGemmaTextLayerRole::RouterScale => "router-scale".to_owned(),
                    DiffusionGemmaTextLayerRole::RouterPerExpertScale => {
                        "router-per-expert-scale".to_owned()
                    }
                    DiffusionGemmaTextLayerRole::PackedExperts(projection) => {
                        format!("packed-experts:{}", projection.canonical())
                    }
                };
                format!("text-layer:{layer}:{role}")
            }
            Self::VisionPatchEmbedding => "vision-root:patch-embedding".to_owned(),
            Self::VisionPositionEmbedding => "vision-root:position-embedding".to_owned(),
            Self::VisionStandardizationBias => "vision-root:standardization-bias".to_owned(),
            Self::VisionStandardizationScale => "vision-root:standardization-scale".to_owned(),
            Self::VisionLayer { layer, role } => {
                let role = match role {
                    DiffusionGemmaVisionLayerRole::AttentionNorm => "attention-norm".to_owned(),
                    DiffusionGemmaVisionLayerRole::PostAttentionNorm => {
                        "post-attention-norm".to_owned()
                    }
                    DiffusionGemmaVisionLayerRole::PreFeedForwardNorm => {
                        "pre-feed-forward-norm".to_owned()
                    }
                    DiffusionGemmaVisionLayerRole::PostFeedForwardNorm => {
                        "post-feed-forward-norm".to_owned()
                    }
                    DiffusionGemmaVisionLayerRole::Attention(role) => {
                        format!("attention:{}", role.canonical())
                    }
                    DiffusionGemmaVisionLayerRole::Mlp(projection) => {
                        format!("mlp:{}", projection.canonical())
                    }
                };
                format!("vision-layer:{layer}:{role}")
            }
            Self::ImageEmbeddingProjection => "projector:image-embedding".to_owned(),
            Self::EncoderLayerOutputScale { layer } => {
                format!("diffusion:encoder-layer:{layer}:output-scale")
            }
            Self::SelfConditioning(role) => format!(
                "diffusion:self-conditioning:{}",
                match role {
                    DiffusionGemmaSelfConditioningRole::PreNorm => "pre-norm",
                    DiffusionGemmaSelfConditioningRole::Gate => "gate",
                    DiffusionGemmaSelfConditioningRole::Up => "up",
                    DiffusionGemmaSelfConditioningRole::Down => "down",
                }
            ),
        }
    }

    const fn packed_experts(self) -> Option<(u8, DiffusionGemmaPackedExpertProjection)> {
        match self {
            Self::TextLayer {
                layer,
                role: DiffusionGemmaTextLayerRole::PackedExperts(projection),
            } => Some((layer, projection)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaSourceTensorMapping {
    pub source_name: String,
    pub artifact_family: DiffusionGemmaArtifactFamily,
    pub tensor_role: DiffusionGemmaTensorRole,
    /// sLLM foundation candidate only; not an upstream-approved GGUF mapping.
    pub foundation_output_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaGgufCatalogRow {
    pub source_name: String,
    pub source_shard: String,
    pub artifact_family: DiffusionGemmaArtifactFamily,
    pub tensor_role: DiffusionGemmaTensorRole,
    pub foundation_output_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaPackedExpertAxisPlan {
    pub layer: u8,
    pub projection: DiffusionGemmaPackedExpertProjection,
    pub source_name: String,
    pub foundation_output_name: String,
    /// Logical packed-axis order required after header verification: `0..=127`.
    pub numeric_expert_order: Vec<u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaUnresolvedPayloadRequirement {
    PackedExpertAxisHeaderVerification,
    FullAttentionCombinedKeyValueHeaderReview,
    VisionQueryKeyRopeAxisPermutation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaGgufFoundationCatalogPlan {
    pub architecture_authority: DiffusionGemmaGgufArchitectureAuthority,
    pub target_metadata: BTreeMap<String, GgufValue>,
    pub source_rows: Vec<DiffusionGemmaGgufCatalogRow>,
    pub packed_expert_axes: Vec<DiffusionGemmaPackedExpertAxisPlan>,
    pub unresolved_payload_requirements: Vec<DiffusionGemmaUnresolvedPayloadRequirement>,
    pub source_tensor_count: usize,
    pub source_text_tensor_count: usize,
    pub source_vision_tensor_count: usize,
    pub source_projector_tensor_count: usize,
    pub source_diffusion_tensor_count: usize,
    pub packed_expert_source_tensor_count: usize,
    pub physical_candidate_count: usize,
    pub mapping_sha256: String,
    pub upstream_gguf_architecture_approved: bool,
    pub production_loadable: bool,
    pub payload_headers_verified: bool,
    pub payload_bytes_verified: bool,
    pub payload_transforms_executed: bool,
    pub dtype_conversion_executed: bool,
    pub quantization_executed: bool,
    pub writable_gguf_plan: bool,
    pub output_path: Option<String>,
    pub output_file_type: Option<u32>,
    pub output_payload_bytes: Option<u64>,
    pub pass_scope: &'static str,
}

fn parse_index(
    value: &str,
    label: &str,
    upper_exclusive: u8,
) -> Result<u8, DiffusionGemmaGgufFoundationError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid(format!(
            "{label} is not canonical decimal: {value}"
        )));
    }
    let parsed = value
        .parse::<u8>()
        .map_err(|_| invalid(format!("{label} is invalid: {value}")))?;
    if parsed >= upper_exclusive {
        return Err(invalid(format!("{label} is out of range: {parsed}")));
    }
    Ok(parsed)
}

fn layer_and_suffix<'a>(
    source_name: &'a str,
    prefix: &str,
    upper_exclusive: u8,
    label: &str,
) -> Result<Option<(u8, &'a str)>, DiffusionGemmaGgufFoundationError> {
    let Some(remainder) = source_name.strip_prefix(prefix) else {
        return Ok(None);
    };
    let (layer, suffix) = remainder
        .split_once('.')
        .ok_or_else(|| invalid(format!("malformed {label} tensor: {source_name}")))?;
    Ok(Some((parse_index(layer, label, upper_exclusive)?, suffix)))
}

fn text_layer_mapping(
    source_name: &str,
    layer: u8,
    suffix: &str,
) -> Result<DiffusionGemmaSourceTensorMapping, DiffusionGemmaGgufFoundationError> {
    let (role, output) = match suffix {
        "input_layernorm.weight" => (
            DiffusionGemmaTextLayerRole::AttentionNorm,
            format!("blk.{layer}.attn_norm.weight"),
        ),
        "post_attention_layernorm.weight" => (
            DiffusionGemmaTextLayerRole::PostAttentionNorm,
            format!("blk.{layer}.post_attention_norm.weight"),
        ),
        "pre_feedforward_layernorm.weight" => (
            DiffusionGemmaTextLayerRole::PreFeedForwardNorm,
            format!("blk.{layer}.ffn_norm.weight"),
        ),
        "pre_feedforward_layernorm_2.weight" => (
            DiffusionGemmaTextLayerRole::PreFeedForwardNorm2,
            format!("blk.{layer}.pre_ffw_norm_2.weight"),
        ),
        "post_feedforward_layernorm.weight" => (
            DiffusionGemmaTextLayerRole::PostFeedForwardNorm,
            format!("blk.{layer}.post_ffw_norm.weight"),
        ),
        "post_feedforward_layernorm_1.weight" => (
            DiffusionGemmaTextLayerRole::PostFeedForwardNorm1,
            format!("blk.{layer}.post_ffw_norm_1.weight"),
        ),
        "post_feedforward_layernorm_2.weight" => (
            DiffusionGemmaTextLayerRole::PostFeedForwardNorm2,
            format!("blk.{layer}.post_ffw_norm_2.weight"),
        ),
        "layer_scalar" => (
            DiffusionGemmaTextLayerRole::LayerOutputScale,
            format!("blk.{layer}.layer_output_scale.weight"),
        ),
        "self_attn.q_proj.weight" => (
            DiffusionGemmaTextLayerRole::Attention(DiffusionGemmaAttentionRole::Query),
            format!("blk.{layer}.attn_q.weight"),
        ),
        "self_attn.q_norm.weight" => (
            DiffusionGemmaTextLayerRole::Attention(DiffusionGemmaAttentionRole::QueryNorm),
            format!("blk.{layer}.attn_q_norm.weight"),
        ),
        "self_attn.k_proj.weight" => (
            DiffusionGemmaTextLayerRole::Attention(DiffusionGemmaAttentionRole::Key),
            format!("blk.{layer}.attn_k.weight"),
        ),
        "self_attn.k_norm.weight" => (
            DiffusionGemmaTextLayerRole::Attention(DiffusionGemmaAttentionRole::KeyNorm),
            format!("blk.{layer}.attn_k_norm.weight"),
        ),
        "self_attn.v_proj.weight" if !FULL_ATTENTION_LAYERS.contains(&layer) => (
            DiffusionGemmaTextLayerRole::Attention(DiffusionGemmaAttentionRole::Value),
            format!("blk.{layer}.attn_v.weight"),
        ),
        "self_attn.o_proj.weight" => (
            DiffusionGemmaTextLayerRole::Attention(DiffusionGemmaAttentionRole::Output),
            format!("blk.{layer}.attn_output.weight"),
        ),
        "mlp.gate_proj.weight" => (
            DiffusionGemmaTextLayerRole::DenseMlp(DiffusionGemmaProjection::Gate),
            format!("blk.{layer}.ffn_gate.weight"),
        ),
        "mlp.up_proj.weight" => (
            DiffusionGemmaTextLayerRole::DenseMlp(DiffusionGemmaProjection::Up),
            format!("blk.{layer}.ffn_up.weight"),
        ),
        "mlp.down_proj.weight" => (
            DiffusionGemmaTextLayerRole::DenseMlp(DiffusionGemmaProjection::Down),
            format!("blk.{layer}.ffn_down.weight"),
        ),
        "router.proj.weight" => (
            DiffusionGemmaTextLayerRole::Router,
            format!("blk.{layer}.ffn_gate_inp.weight"),
        ),
        "router.scale" => (
            DiffusionGemmaTextLayerRole::RouterScale,
            format!("blk.{layer}.ffn_gate_inp.scale"),
        ),
        "router.per_expert_scale" => (
            DiffusionGemmaTextLayerRole::RouterPerExpertScale,
            format!("blk.{layer}.ffn_down_exps.scale"),
        ),
        "experts.gate_up_proj" => (
            DiffusionGemmaTextLayerRole::PackedExperts(
                DiffusionGemmaPackedExpertProjection::GateUp,
            ),
            format!("blk.{layer}.ffn_gate_up_exps.weight"),
        ),
        "experts.down_proj" => (
            DiffusionGemmaTextLayerRole::PackedExperts(DiffusionGemmaPackedExpertProjection::Down),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ),
        _ => {
            return Err(invalid(format!(
                "unsupported decoder text tensor grammar: {source_name}"
            )));
        }
    };
    Ok(DiffusionGemmaSourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_family: DiffusionGemmaArtifactFamily::Text,
        tensor_role: DiffusionGemmaTensorRole::TextLayer { layer, role },
        foundation_output_name: output,
    })
}

fn vision_layer_mapping(
    source_name: &str,
    layer: u8,
    suffix: &str,
) -> Result<DiffusionGemmaSourceTensorMapping, DiffusionGemmaGgufFoundationError> {
    let (role, output) = match suffix {
        "input_layernorm.weight" => (
            DiffusionGemmaVisionLayerRole::AttentionNorm,
            format!("v.blk.{layer}.ln1.weight"),
        ),
        "post_attention_layernorm.weight" => (
            DiffusionGemmaVisionLayerRole::PostAttentionNorm,
            format!("v.blk.{layer}.attn_post_norm.weight"),
        ),
        "pre_feedforward_layernorm.weight" => (
            DiffusionGemmaVisionLayerRole::PreFeedForwardNorm,
            format!("v.blk.{layer}.ln2.weight"),
        ),
        "post_feedforward_layernorm.weight" => (
            DiffusionGemmaVisionLayerRole::PostFeedForwardNorm,
            format!("v.blk.{layer}.ffn_post_norm.weight"),
        ),
        "self_attn.q_proj.linear.weight" => (
            DiffusionGemmaVisionLayerRole::Attention(DiffusionGemmaAttentionRole::Query),
            format!("v.blk.{layer}.attn_q.weight"),
        ),
        "self_attn.q_norm.weight" => (
            DiffusionGemmaVisionLayerRole::Attention(DiffusionGemmaAttentionRole::QueryNorm),
            format!("v.blk.{layer}.attn_q_norm.weight"),
        ),
        "self_attn.k_proj.linear.weight" => (
            DiffusionGemmaVisionLayerRole::Attention(DiffusionGemmaAttentionRole::Key),
            format!("v.blk.{layer}.attn_k.weight"),
        ),
        "self_attn.k_norm.weight" => (
            DiffusionGemmaVisionLayerRole::Attention(DiffusionGemmaAttentionRole::KeyNorm),
            format!("v.blk.{layer}.attn_k_norm.weight"),
        ),
        "self_attn.v_proj.linear.weight" => (
            DiffusionGemmaVisionLayerRole::Attention(DiffusionGemmaAttentionRole::Value),
            format!("v.blk.{layer}.attn_v.weight"),
        ),
        "self_attn.o_proj.linear.weight" => (
            DiffusionGemmaVisionLayerRole::Attention(DiffusionGemmaAttentionRole::Output),
            format!("v.blk.{layer}.attn_out.weight"),
        ),
        "mlp.gate_proj.linear.weight" => (
            DiffusionGemmaVisionLayerRole::Mlp(DiffusionGemmaProjection::Gate),
            format!("v.blk.{layer}.ffn_gate.weight"),
        ),
        "mlp.up_proj.linear.weight" => (
            DiffusionGemmaVisionLayerRole::Mlp(DiffusionGemmaProjection::Up),
            format!("v.blk.{layer}.ffn_up.weight"),
        ),
        "mlp.down_proj.linear.weight" => (
            DiffusionGemmaVisionLayerRole::Mlp(DiffusionGemmaProjection::Down),
            format!("v.blk.{layer}.ffn_down.weight"),
        ),
        _ => {
            return Err(invalid(format!(
                "unsupported Gemma 4 vision tensor grammar: {source_name}"
            )));
        }
    };
    Ok(DiffusionGemmaSourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_family: DiffusionGemmaArtifactFamily::Vision,
        tensor_role: DiffusionGemmaTensorRole::VisionLayer { layer, role },
        foundation_output_name: output,
    })
}

/// Map one exact source name without inspecting its tensor header or payload.
pub fn map_diffusion_gemma_source_tensor(
    source_name: &str,
) -> Result<DiffusionGemmaSourceTensorMapping, DiffusionGemmaGgufFoundationError> {
    let root = match source_name {
        "model.decoder.embed_tokens.weight" => Some((
            DiffusionGemmaArtifactFamily::Text,
            DiffusionGemmaTensorRole::TokenEmbedding,
            "token_embd.weight",
        )),
        "model.decoder.norm.weight" => Some((
            DiffusionGemmaArtifactFamily::Text,
            DiffusionGemmaTensorRole::OutputNorm,
            "output_norm.weight",
        )),
        "model.encoder.vision_tower.patch_embedder.input_proj.weight" => Some((
            DiffusionGemmaArtifactFamily::Vision,
            DiffusionGemmaTensorRole::VisionPatchEmbedding,
            "v.patch_embd.weight",
        )),
        "model.encoder.vision_tower.patch_embedder.position_embedding_table" => Some((
            DiffusionGemmaArtifactFamily::Vision,
            DiffusionGemmaTensorRole::VisionPositionEmbedding,
            "v.position_embd",
        )),
        "model.encoder.vision_tower.std_bias" => Some((
            DiffusionGemmaArtifactFamily::Vision,
            DiffusionGemmaTensorRole::VisionStandardizationBias,
            "v.std_bias",
        )),
        "model.encoder.vision_tower.std_scale" => Some((
            DiffusionGemmaArtifactFamily::Vision,
            DiffusionGemmaTensorRole::VisionStandardizationScale,
            "v.std_scale",
        )),
        "model.encoder.embed_vision.embedding_projection.weight" => Some((
            DiffusionGemmaArtifactFamily::Projector,
            DiffusionGemmaTensorRole::ImageEmbeddingProjection,
            "mm.input_projection.weight",
        )),
        "model.decoder.self_conditioning.pre_norm.weight" => Some((
            DiffusionGemmaArtifactFamily::DiffusionSpecific,
            DiffusionGemmaTensorRole::SelfConditioning(DiffusionGemmaSelfConditioningRole::PreNorm),
            "diffusion.self_conditioning.pre_norm.weight",
        )),
        "model.decoder.self_conditioning.gate_proj.weight" => Some((
            DiffusionGemmaArtifactFamily::DiffusionSpecific,
            DiffusionGemmaTensorRole::SelfConditioning(DiffusionGemmaSelfConditioningRole::Gate),
            "diffusion.self_conditioning.gate.weight",
        )),
        "model.decoder.self_conditioning.up_proj.weight" => Some((
            DiffusionGemmaArtifactFamily::DiffusionSpecific,
            DiffusionGemmaTensorRole::SelfConditioning(DiffusionGemmaSelfConditioningRole::Up),
            "diffusion.self_conditioning.up.weight",
        )),
        "model.decoder.self_conditioning.down_proj.weight" => Some((
            DiffusionGemmaArtifactFamily::DiffusionSpecific,
            DiffusionGemmaTensorRole::SelfConditioning(DiffusionGemmaSelfConditioningRole::Down),
            "diffusion.self_conditioning.down.weight",
        )),
        _ => None,
    };
    if let Some((artifact_family, tensor_role, foundation_output_name)) = root {
        return Ok(DiffusionGemmaSourceTensorMapping {
            source_name: source_name.to_owned(),
            artifact_family,
            tensor_role,
            foundation_output_name: foundation_output_name.to_owned(),
        });
    }

    if let Some((layer, suffix)) = layer_and_suffix(
        source_name,
        "model.decoder.layers.",
        TEXT_LAYER_COUNT,
        "decoder layer",
    )? {
        return text_layer_mapping(source_name, layer, suffix);
    }
    if let Some((layer, suffix)) = layer_and_suffix(
        source_name,
        "model.encoder.vision_tower.encoder.layers.",
        VISION_LAYER_COUNT,
        "vision layer",
    )? {
        return vision_layer_mapping(source_name, layer, suffix);
    }
    if let Some((layer, suffix)) = layer_and_suffix(
        source_name,
        "model.encoder.language_model.layers.",
        TEXT_LAYER_COUNT,
        "encoder layer",
    )? {
        if suffix != "layer_scalar" {
            return Err(invalid(format!(
                "unsupported untied diffusion encoder tensor: {source_name}"
            )));
        }
        return Ok(DiffusionGemmaSourceTensorMapping {
            source_name: source_name.to_owned(),
            artifact_family: DiffusionGemmaArtifactFamily::DiffusionSpecific,
            tensor_role: DiffusionGemmaTensorRole::EncoderLayerOutputScale { layer },
            foundation_output_name: format!(
                "diffusion.encoder.blk.{layer}.layer_output_scale.weight"
            ),
        });
    }
    Err(invalid(format!("unknown source tensor: {source_name}")))
}

fn expected_source_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for name in [
        "model.decoder.embed_tokens.weight",
        "model.decoder.norm.weight",
        "model.decoder.self_conditioning.pre_norm.weight",
        "model.decoder.self_conditioning.gate_proj.weight",
        "model.decoder.self_conditioning.up_proj.weight",
        "model.decoder.self_conditioning.down_proj.weight",
        "model.encoder.embed_vision.embedding_projection.weight",
        "model.encoder.vision_tower.patch_embedder.input_proj.weight",
        "model.encoder.vision_tower.patch_embedder.position_embedding_table",
        "model.encoder.vision_tower.std_bias",
        "model.encoder.vision_tower.std_scale",
    ] {
        names.insert(name.to_owned());
    }
    let text_suffixes = [
        "experts.down_proj",
        "experts.gate_up_proj",
        "input_layernorm.weight",
        "layer_scalar",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
        "router.per_expert_scale",
        "router.proj.weight",
        "router.scale",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
    ];
    for layer in 0..TEXT_LAYER_COUNT {
        for suffix in text_suffixes {
            names.insert(format!("model.decoder.layers.{layer}.{suffix}"));
        }
        if !FULL_ATTENTION_LAYERS.contains(&layer) {
            names.insert(format!(
                "model.decoder.layers.{layer}.self_attn.v_proj.weight"
            ));
        }
        names.insert(format!(
            "model.encoder.language_model.layers.{layer}.layer_scalar"
        ));
    }
    let vision_suffixes = [
        "input_layernorm.weight",
        "mlp.down_proj.linear.weight",
        "mlp.gate_proj.linear.weight",
        "mlp.up_proj.linear.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.linear.weight",
        "self_attn.o_proj.linear.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.linear.weight",
        "self_attn.v_proj.linear.weight",
    ];
    for layer in 0..VISION_LAYER_COUNT {
        for suffix in vision_suffixes {
            names.insert(format!(
                "model.encoder.vision_tower.encoder.layers.{layer}.{suffix}"
            ));
        }
    }
    names
}

fn validate_shard_name(name: &str) -> Result<usize, DiffusionGemmaGgufFoundationError> {
    let middle = name
        .strip_prefix("model-")
        .and_then(|value| value.strip_suffix(".safetensors"))
        .ok_or_else(|| invalid(format!("invalid source shard name: {name}")))?;
    let (index, total) = middle
        .split_once("-of-")
        .ok_or_else(|| invalid(format!("invalid source shard name: {name}")))?;
    if index.len() != 5 || total != "00011" || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid(format!("invalid source shard name: {name}")));
    }
    let index = index
        .parse::<usize>()
        .map_err(|_| invalid(format!("invalid source shard index: {name}")))?;
    if !(1..=DIFFUSION_GEMMA_SHARD_COUNT).contains(&index) {
        return Err(invalid(format!(
            "source shard index is out of range: {name}"
        )));
    }
    Ok(index)
}

/// Validate the logical order for an already-packed expert axis.
pub fn validate_diffusion_gemma_expert_numeric_order(
    order: &[u16],
) -> Result<(), DiffusionGemmaGgufFoundationError> {
    if order.len() != usize::from(EXPERT_COUNT) {
        return Err(invalid("packed expert axis must contain exactly 128 slots"));
    }
    let mut seen = BTreeSet::new();
    for (position, expert) in order.iter().copied().enumerate() {
        if expert >= EXPERT_COUNT {
            return Err(invalid(format!("packed expert is out of range: {expert}")));
        }
        if !seen.insert(expert) {
            return Err(invalid(format!("duplicate packed expert: {expert}")));
        }
        if usize::from(expert) != position {
            return Err(invalid(format!(
                "packed expert order is not numeric at position {position}: {expert}"
            )));
        }
    }
    Ok(())
}

fn foundation_metadata() -> BTreeMap<String, GgufValue> {
    BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String(DIFFUSION_GEMMA_GGUF_ARCHITECTURE.to_owned()),
        ),
        ("general.alignment".to_owned(), GgufValue::U32(32)),
        (
            "general.type".to_owned(),
            GgufValue::String("model".to_owned()),
        ),
        (
            "general.name".to_owned(),
            GgufValue::String(format!(
                "{DIFFUSION_GEMMA_REPOSITORY}@{DIFFUSION_GEMMA_REVISION}"
            )),
        ),
        (
            "general.license".to_owned(),
            GgufValue::String(DIFFUSION_GEMMA_LICENSE.to_owned()),
        ),
        (
            "general.source.url".to_owned(),
            GgufValue::String(format!(
                "https://huggingface.co/{DIFFUSION_GEMMA_REPOSITORY}/tree/{DIFFUSION_GEMMA_REVISION}"
            )),
        ),
        (
            "diffusion-gemma.foundation.architecture_authority".to_owned(),
            GgufValue::String("sllm-write-disabled-foundation-not-upstream-approved".to_owned()),
        ),
        (
            "diffusion-gemma.foundation.llama_cpp_cross_check".to_owned(),
            GgufValue::String(DIFFUSION_GEMMA_LLAMA_CPP_DRAFT_PR.to_owned()),
        ),
        (
            "diffusion-gemma.foundation.llama_cpp_reference_commit".to_owned(),
            GgufValue::String(DIFFUSION_GEMMA_LLAMA_CPP_REFERENCE_COMMIT.to_owned()),
        ),
        (
            "diffusion-gemma.foundation.mapping_serialization".to_owned(),
            GgufValue::String(DIFFUSION_GEMMA_GGUF_MAPPING_SERIALIZATION.to_owned()),
        ),
        (
            "diffusion-gemma.foundation.transformers_reference_commit".to_owned(),
            GgufValue::String(DIFFUSION_GEMMA_TRANSFORMERS_REFERENCE_COMMIT.to_owned()),
        ),
        (
            "diffusion-gemma.source.revision".to_owned(),
            GgufValue::String(DIFFUSION_GEMMA_REVISION.to_owned()),
        ),
        (
            "diffusion-gemma.context_length".to_owned(),
            GgufValue::U32(262_144),
        ),
        (
            "diffusion-gemma.embedding_length".to_owned(),
            GgufValue::U32(2_816),
        ),
        (
            "diffusion-gemma.block_count".to_owned(),
            GgufValue::U32(u32::from(TEXT_LAYER_COUNT)),
        ),
        (
            "diffusion-gemma.feed_forward_length".to_owned(),
            GgufValue::U32(2_112),
        ),
        (
            "diffusion-gemma.expert_feed_forward_length".to_owned(),
            GgufValue::U32(704),
        ),
        (
            "diffusion-gemma.expert_count".to_owned(),
            GgufValue::U32(u32::from(EXPERT_COUNT)),
        ),
        (
            "diffusion-gemma.expert_used_count".to_owned(),
            GgufValue::U32(SELECTED_EXPERT_COUNT),
        ),
        (
            "diffusion-gemma.attention.head_count".to_owned(),
            GgufValue::U32(16),
        ),
        (
            "diffusion-gemma.attention.head_count_kv".to_owned(),
            GgufValue::U32(8),
        ),
        (
            "diffusion-gemma.attention.head_count_kv_full".to_owned(),
            GgufValue::U32(2),
        ),
        (
            "diffusion-gemma.attention.sliding_window_pattern".to_owned(),
            GgufValue::Array(GgufArray::Bool(
                (0..TEXT_LAYER_COUNT)
                    .map(|layer| !FULL_ATTENTION_LAYERS.contains(&layer))
                    .collect(),
            )),
        ),
        (
            "diffusion-gemma.canvas_length".to_owned(),
            GgufValue::U32(256),
        ),
        (
            "diffusion-gemma.vision.block_count".to_owned(),
            GgufValue::U32(u32::from(VISION_LAYER_COUNT)),
        ),
        (
            "diffusion-gemma.vision.embedding_length".to_owned(),
            GgufValue::U32(1_152),
        ),
        (
            "diffusion-gemma.vision.soft_token_count".to_owned(),
            GgufValue::U32(280),
        ),
        (
            "diffusion-gemma.manifest.tensor_count".to_owned(),
            GgufValue::U64(DIFFUSION_GEMMA_SOURCE_TENSOR_COUNT as u64),
        ),
        (
            "diffusion-gemma.manifest.index_advertised_payload_bytes".to_owned(),
            GgufValue::U64(DIFFUSION_GEMMA_INDEX_ADVERTISED_PAYLOAD_BYTES),
        ),
        (
            "diffusion-gemma.production_supported".to_owned(),
            GgufValue::Bool(false),
        ),
        (
            "diffusion-gemma.write_supported".to_owned(),
            GgufValue::Bool(false),
        ),
    ])
}

fn row_serialization(
    row: &DiffusionGemmaGgufCatalogRow,
) -> Result<String, DiffusionGemmaGgufFoundationError> {
    let role = row.tensor_role.canonical();
    let fields = [
        row.source_name.as_str(),
        row.source_shard.as_str(),
        row.artifact_family.canonical(),
        role.as_str(),
        row.foundation_output_name.as_str(),
    ];
    if fields
        .iter()
        .any(|field| field.contains(['\t', '\n', '\r']))
    {
        return Err(invalid(
            "mapping row contains a canonical serialization delimiter",
        ));
    }
    Ok(format!("{}\n", fields.join("\t")))
}

/// Build the pure mapping plan from source-name/shard entries.
///
/// The input may come from a future typed config/index adapter.  Duplicate
/// entries remain observable because a slice, rather than a map, is accepted.
pub fn build_diffusion_gemma_gguf_foundation_catalog_from_entries(
    entries: &[(String, String)],
) -> Result<DiffusionGemmaGgufFoundationCatalogPlan, DiffusionGemmaGgufFoundationError> {
    if entries.len() != DIFFUSION_GEMMA_SOURCE_TENSOR_COUNT {
        return Err(invalid(format!(
            "source tensor count differs: {}",
            entries.len()
        )));
    }
    let expected = expected_source_names();
    if expected.len() != DIFFUSION_GEMMA_SOURCE_TENSOR_COUNT {
        return Err(invalid("internal reviewed source catalog count differs"));
    }
    let mut by_source = BTreeMap::new();
    let mut shards = BTreeSet::new();
    for (source_name, shard) in entries {
        if source_name.is_empty() {
            return Err(invalid("source tensor name is empty"));
        }
        if by_source
            .insert(source_name.clone(), shard.clone())
            .is_some()
        {
            return Err(invalid(format!("duplicate source tensor: {source_name}")));
        }
        shards.insert(validate_shard_name(shard)?);
    }
    let observed = by_source.keys().cloned().collect::<BTreeSet<_>>();
    if let Some(unknown) = observed.difference(&expected).next() {
        return Err(invalid(format!("unknown source tensor: {unknown}")));
    }
    if let Some(missing) = expected.difference(&observed).next() {
        return Err(invalid(format!("missing source tensor: {missing}")));
    }
    if shards != (1..=DIFFUSION_GEMMA_SHARD_COUNT).collect() {
        return Err(invalid("source shard coverage differs from 1..=11"));
    }

    let mut rows = Vec::with_capacity(DIFFUSION_GEMMA_SOURCE_TENSOR_COUNT);
    let mut typed_keys = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let mut packed = BTreeMap::new();
    let mut family_counts = BTreeMap::<DiffusionGemmaArtifactFamily, usize>::new();
    let mut digest = Sha256::new();
    for (source_name, shard) in by_source {
        let mapping = map_diffusion_gemma_source_tensor(&source_name)?;
        let typed_key = format!(
            "{}\t{}",
            mapping.artifact_family.canonical(),
            mapping.tensor_role.canonical()
        );
        if !typed_keys.insert(typed_key) {
            return Err(invalid(format!("typed role collision: {source_name}")));
        }
        if !outputs.insert(mapping.foundation_output_name.clone()) {
            return Err(invalid(format!(
                "foundation output collision: {}",
                mapping.foundation_output_name
            )));
        }
        let count = family_counts.entry(mapping.artifact_family).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("artifact family count overflowed"))?;
        if let Some((layer, projection)) = mapping.tensor_role.packed_experts()
            && packed
                .insert(
                    (layer, projection),
                    (
                        mapping.source_name.clone(),
                        mapping.foundation_output_name.clone(),
                    ),
                )
                .is_some()
        {
            return Err(invalid(format!(
                "duplicate packed expert layer/projection: {source_name}"
            )));
        }
        let row = DiffusionGemmaGgufCatalogRow {
            source_name: mapping.source_name,
            source_shard: shard,
            artifact_family: mapping.artifact_family,
            tensor_role: mapping.tensor_role,
            foundation_output_name: mapping.foundation_output_name,
        };
        digest.update(row_serialization(&row)?.as_bytes());
        rows.push(row);
    }

    let numeric_expert_order = (0..EXPERT_COUNT).collect::<Vec<_>>();
    validate_diffusion_gemma_expert_numeric_order(&numeric_expert_order)?;
    let mut packed_expert_axes = Vec::with_capacity(
        usize::from(TEXT_LAYER_COUNT) * DiffusionGemmaPackedExpertProjection::ALL.len(),
    );
    for layer in 0..TEXT_LAYER_COUNT {
        for projection in DiffusionGemmaPackedExpertProjection::ALL {
            let (source_name, foundation_output_name) =
                packed.remove(&(layer, projection)).ok_or_else(|| {
                    invalid(format!(
                        "missing packed expert layer {layer} projection {}",
                        projection.canonical()
                    ))
                })?;
            packed_expert_axes.push(DiffusionGemmaPackedExpertAxisPlan {
                layer,
                projection,
                source_name,
                foundation_output_name,
                numeric_expert_order: numeric_expert_order.clone(),
            });
        }
    }
    if !packed.is_empty() {
        return Err(invalid("unknown packed expert layer/projection remained"));
    }

    let text = family_counts
        .get(&DiffusionGemmaArtifactFamily::Text)
        .copied()
        .unwrap_or(0);
    let vision = family_counts
        .get(&DiffusionGemmaArtifactFamily::Vision)
        .copied()
        .unwrap_or(0);
    let projector = family_counts
        .get(&DiffusionGemmaArtifactFamily::Projector)
        .copied()
        .unwrap_or(0);
    let diffusion = family_counts
        .get(&DiffusionGemmaArtifactFamily::DiffusionSpecific)
        .copied()
        .unwrap_or(0);
    if text != DIFFUSION_GEMMA_SOURCE_TEXT_TENSOR_COUNT
        || vision != DIFFUSION_GEMMA_SOURCE_VISION_TENSOR_COUNT
        || projector != DIFFUSION_GEMMA_SOURCE_PROJECTOR_TENSOR_COUNT
        || diffusion != DIFFUSION_GEMMA_SOURCE_DIFFUSION_TENSOR_COUNT
        || packed_expert_axes.len() != DIFFUSION_GEMMA_PACKED_EXPERT_SOURCE_TENSOR_COUNT
        || outputs.len() != DIFFUSION_GEMMA_PHYSICAL_CANDIDATE_COUNT
    {
        return Err(invalid(
            "catalog family or physical candidate accounting differs",
        ));
    }

    let target_metadata = foundation_metadata();
    if target_metadata.contains_key("general.file_type") {
        return Err(invalid("write-disabled foundation emitted a file type"));
    }
    Ok(DiffusionGemmaGgufFoundationCatalogPlan {
        architecture_authority:
            DiffusionGemmaGgufArchitectureAuthority::SllmWriteDisabledFoundationNotUpstreamApproved,
        target_metadata,
        source_rows: rows,
        packed_expert_axes,
        unresolved_payload_requirements: vec![
            DiffusionGemmaUnresolvedPayloadRequirement::PackedExpertAxisHeaderVerification,
            DiffusionGemmaUnresolvedPayloadRequirement::FullAttentionCombinedKeyValueHeaderReview,
            DiffusionGemmaUnresolvedPayloadRequirement::VisionQueryKeyRopeAxisPermutation,
        ],
        source_tensor_count: DIFFUSION_GEMMA_SOURCE_TENSOR_COUNT,
        source_text_tensor_count: text,
        source_vision_tensor_count: vision,
        source_projector_tensor_count: projector,
        source_diffusion_tensor_count: diffusion,
        packed_expert_source_tensor_count: DIFFUSION_GEMMA_PACKED_EXPERT_SOURCE_TENSOR_COUNT,
        physical_candidate_count: outputs.len(),
        mapping_sha256: format!("{:x}", digest.finalize()),
        upstream_gguf_architecture_approved: false,
        production_loadable: false,
        payload_headers_verified: false,
        payload_bytes_verified: false,
        payload_transforms_executed: false,
        dtype_conversion_executed: false,
        quantization_executed: false,
        writable_gguf_plan: false,
        output_path: None,
        output_file_type: None,
        output_payload_bytes: None,
        pass_scope: DIFFUSION_GEMMA_GGUF_PASS_SCOPE,
    })
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn exact_config(config_bytes: &[u8]) -> Result<(), DiffusionGemmaGgufFoundationError> {
    if sha256(config_bytes) != DIFFUSION_GEMMA_CONFIG_SHA256 {
        return Err(invalid("config SHA-256 differs from the reviewed revision"));
    }
    let config: Value = serde_json::from_slice(config_bytes)
        .map_err(|error| invalid(format!("config JSON is invalid: {error}")))?;
    if config.get("model_type").and_then(Value::as_str) != Some("diffusion_gemma")
        || config
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
            .and_then(Value::as_str)
            != Some("DiffusionGemmaForBlockDiffusion")
    {
        return Err(invalid("config model identity differs"));
    }
    Ok(())
}

fn exact_index_entries(
    index_bytes: &[u8],
) -> Result<Vec<(String, String)>, DiffusionGemmaGgufFoundationError> {
    if sha256(index_bytes) != DIFFUSION_GEMMA_INDEX_SHA256 {
        return Err(invalid("index SHA-256 differs from the reviewed revision"));
    }
    let index: Value = serde_json::from_slice(index_bytes)
        .map_err(|error| invalid(format!("index JSON is invalid: {error}")))?;
    let metadata = index
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("index metadata is missing"))?;
    if metadata.get("total_parameters").and_then(Value::as_u64)
        != Some(DIFFUSION_GEMMA_INDEX_ADVERTISED_PARAMETERS)
        || metadata.get("total_size").and_then(Value::as_u64)
            != Some(DIFFUSION_GEMMA_INDEX_ADVERTISED_PAYLOAD_BYTES)
    {
        return Err(invalid("index metadata totals differ"));
    }
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("index weight_map is missing"))?;
    let mut entries = Vec::with_capacity(weight_map.len());
    for (source_name, shard) in weight_map {
        let shard = shard
            .as_str()
            .ok_or_else(|| invalid(format!("source shard is not a string: {source_name}")))?;
        entries.push((source_name.clone(), shard.to_owned()));
    }
    Ok(entries)
}

/// Validate the exact official config/index bytes and return the write-disabled
/// catalog.  No shard, tensor header, or payload is opened by this API.
pub fn validate_diffusion_gemma_gguf_foundation_catalog(
    config_bytes: &[u8],
    index_bytes: &[u8],
) -> Result<DiffusionGemmaGgufFoundationCatalogPlan, DiffusionGemmaGgufFoundationError> {
    exact_config(config_bytes)?;
    let entries = exact_index_entries(index_bytes)?;
    let plan = build_diffusion_gemma_gguf_foundation_catalog_from_entries(&entries)?;
    if plan.mapping_sha256 != DIFFUSION_GEMMA_GGUF_MAPPING_SHA256 {
        return Err(invalid(format!(
            "mapping SHA-256 differs: {}",
            plan.mapping_sha256
        )));
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reviewed_entries() -> Vec<(String, String)> {
        expected_source_names()
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                (
                    name,
                    format!(
                        "model-{:05}-of-00011.safetensors",
                        index % DIFFUSION_GEMMA_SHARD_COUNT + 1
                    ),
                )
            })
            .collect()
    }

    #[test]
    fn sllm_write_disabled_architecture_key_is_not_upstream_approval() {
        let plan = build_diffusion_gemma_gguf_foundation_catalog_from_entries(&reviewed_entries())
            .expect("reviewed catalog");
        assert_eq!(
            plan.target_metadata.get("general.architecture"),
            Some(&GgufValue::String("diffusion-gemma".to_owned()))
        );
        assert_eq!(
            plan.architecture_authority,
            DiffusionGemmaGgufArchitectureAuthority::SllmWriteDisabledFoundationNotUpstreamApproved
        );
        assert!(!plan.upstream_gguf_architecture_approved);
        assert!(!plan.target_metadata.contains_key("general.file_type"));
        assert!(!plan.production_loadable);
        assert!(!plan.payload_headers_verified);
        assert!(!plan.payload_bytes_verified);
        assert!(!plan.writable_gguf_plan);
        assert_eq!(plan.output_path, None);
        assert_eq!(plan.output_file_type, None);
        assert_eq!(plan.output_payload_bytes, None);
    }

    #[test]
    fn family_counts_and_diffusion_specific_boundaries_are_exact() {
        let plan = build_diffusion_gemma_gguf_foundation_catalog_from_entries(&reviewed_entries())
            .expect("reviewed catalog");
        assert_eq!(plan.source_tensor_count, 1_047);
        assert_eq!(plan.source_text_tensor_count, 657);
        assert_eq!(plan.source_vision_tensor_count, 355);
        assert_eq!(plan.source_projector_tensor_count, 1);
        assert_eq!(plan.source_diffusion_tensor_count, 34);
        assert_eq!(plan.physical_candidate_count, 1_047);
        assert_eq!(
            map_diffusion_gemma_source_tensor("model.encoder.language_model.layers.0.layer_scalar")
                .expect("encoder layer zero")
                .artifact_family,
            DiffusionGemmaArtifactFamily::DiffusionSpecific
        );
        assert!(
            map_diffusion_gemma_source_tensor(
                "model.encoder.language_model.layers.29.layer_scalar"
            )
            .is_ok()
        );
        assert!(
            map_diffusion_gemma_source_tensor(
                "model.encoder.language_model.layers.30.layer_scalar"
            )
            .is_err()
        );
    }

    #[test]
    fn packed_expert_axes_use_strict_numeric_zero_through_127_order() {
        let plan = build_diffusion_gemma_gguf_foundation_catalog_from_entries(&reviewed_entries())
            .expect("reviewed catalog");
        assert_eq!(plan.packed_expert_axes.len(), 60);
        assert_eq!(plan.packed_expert_axes[0].layer, 0);
        assert_eq!(plan.packed_expert_axes[0].numeric_expert_order[0], 0);
        assert_eq!(plan.packed_expert_axes[0].numeric_expert_order[127], 127);
        assert_eq!(plan.packed_expert_axes[59].layer, 29);

        let mut duplicate = (0..EXPERT_COUNT).collect::<Vec<_>>();
        duplicate[127] = 0;
        assert!(validate_diffusion_gemma_expert_numeric_order(&duplicate).is_err());
        let mut out_of_range = (0..EXPERT_COUNT).collect::<Vec<_>>();
        out_of_range[127] = EXPERT_COUNT;
        assert!(validate_diffusion_gemma_expert_numeric_order(&out_of_range).is_err());
        let mut non_numeric = (0..EXPERT_COUNT).collect::<Vec<_>>();
        non_numeric.swap(9, 10);
        assert!(validate_diffusion_gemma_expert_numeric_order(&non_numeric).is_err());
    }

    #[test]
    fn missing_duplicate_unknown_and_layer_boundaries_fail_closed() {
        let entries = reviewed_entries();
        let mut missing = entries.clone();
        missing.pop();
        assert!(build_diffusion_gemma_gguf_foundation_catalog_from_entries(&missing).is_err());

        let mut duplicate = entries.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(build_diffusion_gemma_gguf_foundation_catalog_from_entries(&duplicate).is_err());

        let mut unknown = entries;
        unknown[0].0 = "model.decoder.layers.0.unknown.weight".to_owned();
        assert!(build_diffusion_gemma_gguf_foundation_catalog_from_entries(&unknown).is_err());
        assert!(
            map_diffusion_gemma_source_tensor("model.decoder.layers.05.self_attn.q_proj.weight")
                .is_err()
        );
        assert!(
            map_diffusion_gemma_source_tensor(
                "model.encoder.vision_tower.encoder.layers.27.self_attn.q_proj.linear.weight"
            )
            .is_err()
        );
        assert!(
            map_diffusion_gemma_source_tensor("model.decoder.layers.5.self_attn.v_proj.weight")
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires exact official Phase59 config/index fixture via SLLM_DIFFUSION_GEMMA_METADATA_DIR"]
    fn exact_official_catalog_counts_digest_and_packed_expert_order_are_fixed() {
        let root = std::path::PathBuf::from(
            std::env::var_os("SLLM_DIFFUSION_GEMMA_METADATA_DIR")
                .expect("set SLLM_DIFFUSION_GEMMA_METADATA_DIR"),
        );
        let config = std::fs::read(root.join("config.json")).expect("official config");
        let index =
            std::fs::read(root.join("model.safetensors.index.json")).expect("official index");
        let entries = exact_index_entries(&index).expect("exact index entries");
        let dry_run = build_diffusion_gemma_gguf_foundation_catalog_from_entries(&entries)
            .expect("exact dry-run mapping");
        eprintln!("DiffusionGemma mapping SHA-256: {}", dry_run.mapping_sha256);
        assert_eq!(sha256(&config), DIFFUSION_GEMMA_CONFIG_SHA256);
        assert_eq!(sha256(&index), DIFFUSION_GEMMA_INDEX_SHA256);
        assert_eq!(dry_run.source_rows.len(), 1_047);
        assert_eq!(dry_run.packed_expert_axes.len(), 60);
        assert_eq!(dry_run.mapping_sha256, DIFFUSION_GEMMA_GGUF_MAPPING_SHA256);
        validate_diffusion_gemma_gguf_foundation_catalog(&config, &index)
            .expect("exact official write-disabled catalog");
    }
}
