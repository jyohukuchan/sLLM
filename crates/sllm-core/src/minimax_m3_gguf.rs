//! Write-disabled GGUF catalog dry run for the reviewed MiniMax M3 artifact.
//!
//! This module maps only the exact validated config/index pair. It does not
//! inspect tensor headers or payload bytes, execute tensor transforms, choose
//! output dtypes or quantization, construct a `GgufWritePlan`, or write files.

use crate::gguf::GgufValue;
use crate::minimax_m3::{
    MINIMAX_M3_CATALOG_SHA256, MINIMAX_M3_INDEX_ADVERTISED_BYTES, MINIMAX_M3_LICENSE,
    MINIMAX_M3_REPOSITORY, MINIMAX_M3_REVISION, MINIMAX_M3_SHARD_FILE_BYTES,
    MINIMAX_M3_TENSOR_COUNT, MiniMaxM3Config, MiniMaxM3Index, MiniMaxM3ManifestState,
    classify_minimax_m3_tensor, validate_minimax_m3_config, validate_minimax_m3_index,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const MINIMAX_M3_GGUF_ARCHITECTURE: &str = "minimax-m3";
pub const MINIMAX_M3_GGUF_SOURCE_TEXT_TENSOR_COUNT: usize = 22_893;
pub const MINIMAX_M3_GGUF_SOURCE_VISION_PROJECTOR_TENSOR_COUNT: usize = 523;
pub const MINIMAX_M3_GGUF_ROUTED_EXPERT_SOURCE_TENSOR_COUNT: usize = 21_888;
pub const MINIMAX_M3_GGUF_DIRECT_SOURCE_TENSOR_COUNT: usize = 1_528;
pub const MINIMAX_M3_GGUF_STACKED_EXPERT_OUTPUT_COUNT: usize = 171;
pub const MINIMAX_M3_GGUF_COMBINED_PHYSICAL_CANDIDATE_COUNT: usize = 1_699;
pub const MINIMAX_M3_GGUF_PASS_SCOPE: &str =
    "exact-metadata-and-index-catalog-only-no-headers-no-payload-no-write";
pub const MINIMAX_M3_GGUF_MAPPING_SERIALIZATION: &str =
    "utf8-tsv-v1:source,shard,artifact-plane,typed-role,output-or-dash;lf-rows";

/// SHA-256 of [`MINIMAX_M3_GGUF_MAPPING_SERIALIZATION`] for the fixed
/// 23,416-row official index. This digest covers names and mapping decisions,
/// never uninspected tensor payload bytes.
pub const MINIMAX_M3_GGUF_MAPPING_SHA256: &str =
    "93ad9f5467bb9a7ba3b77c96db5aa0641e5d9e9801f99dc49bf46a8a4a18dd3f";

const REVIEWED_TEXT_ROOT_COUNT: usize = 3;
const REVIEWED_DENSE_TEXT_COUNT: usize = 33;
const REVIEWED_MOE_TEXT_COUNT: usize = 22_857;
const REVIEWED_VISION_COUNT: usize = 515;
const REVIEWED_MULTIMODAL_PROJECTOR_COUNT: usize = 4;
const REVIEWED_PATCH_MERGE_PROJECTOR_COUNT: usize = 4;
const REVIEWED_TEXT_LAYER_COUNT: u8 = 60;
const REVIEWED_DENSE_LAYER_COUNT: u8 = 3;
const REVIEWED_EXPERT_COUNT: u16 = 128;
const REVIEWED_PROJECTION_COUNT: usize = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MiniMaxM3GgufFoundationError {
    Invalid(String),
}

impl fmt::Display for MiniMaxM3GgufFoundationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(
                formatter,
                "invalid MiniMax M3 GGUF foundation catalog: {message}"
            ),
        }
    }
}

impl std::error::Error for MiniMaxM3GgufFoundationError {}

fn invalid(message: impl Into<String>) -> MiniMaxM3GgufFoundationError {
    MiniMaxM3GgufFoundationError::Invalid(message.into())
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3ArtifactPlane {
    Text,
    VisionProjector,
}

impl MiniMaxM3ArtifactPlane {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::VisionProjector => "vision-projector",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3Parameter {
    Weight,
    Bias,
}

impl MiniMaxM3Parameter {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Weight => "weight",
            Self::Bias => "bias",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3ExpertProjection {
    Gate,
    Down,
    Up,
}

impl MiniMaxM3ExpertProjection {
    const ALL: [Self; REVIEWED_PROJECTION_COUNT] = [Self::Gate, Self::Down, Self::Up];

    const fn canonical(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Down => "down",
            Self::Up => "up",
        }
    }

    const fn stacked_base(self) -> &'static str {
        match self {
            Self::Gate => "ffn_gate_exps",
            Self::Down => "ffn_down_exps",
            Self::Up => "ffn_up_exps",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3RootTensorRole {
    TokenEmbedding,
    OutputNorm,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3AttentionTensorRole {
    Norm,
    Query,
    QueryNorm,
    Key,
    KeyNorm,
    Value,
    Output,
    IndexQuery,
    IndexQueryNorm,
    IndexKey,
    IndexKeyNorm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3FeedForwardTensorRole {
    Norm,
    Dense {
        projection: MiniMaxM3ExpertProjection,
    },
    Router,
    RouterSelectionBias,
    SharedExpert {
        projection: MiniMaxM3ExpertProjection,
    },
    RoutedExpert {
        expert: u16,
        projection: MiniMaxM3ExpertProjection,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3VisionRootTensorRole {
    TemporalPatchEmbedding,
    PreLayerNorm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3VisionLayerTensorRole {
    AttentionNorm,
    FeedForwardNorm,
    Query,
    Key,
    Value,
    AttentionOutput,
    FeedForwardUp,
    FeedForwardDown,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3ProjectorKind {
    Multimodal,
    PatchMerge,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MiniMaxM3TensorRole {
    TextRoot(MiniMaxM3RootTensorRole),
    TextAttention {
        layer: u8,
        role: MiniMaxM3AttentionTensorRole,
    },
    TextFeedForward {
        layer: u8,
        role: MiniMaxM3FeedForwardTensorRole,
    },
    VisionRoot {
        role: MiniMaxM3VisionRootTensorRole,
        parameter: MiniMaxM3Parameter,
    },
    VisionLayer {
        layer: u8,
        role: MiniMaxM3VisionLayerTensorRole,
        parameter: MiniMaxM3Parameter,
    },
    Projector {
        kind: MiniMaxM3ProjectorKind,
        linear: u8,
        parameter: MiniMaxM3Parameter,
    },
}

impl MiniMaxM3TensorRole {
    fn canonical(self) -> String {
        match self {
            Self::TextRoot(role) => format!(
                "text-root:{}",
                match role {
                    MiniMaxM3RootTensorRole::TokenEmbedding => "token-embedding",
                    MiniMaxM3RootTensorRole::OutputNorm => "output-norm",
                    MiniMaxM3RootTensorRole::Output => "output",
                }
            ),
            Self::TextAttention { layer, role } => format!(
                "text-layer:{layer}:attention:{}",
                match role {
                    MiniMaxM3AttentionTensorRole::Norm => "norm",
                    MiniMaxM3AttentionTensorRole::Query => "query",
                    MiniMaxM3AttentionTensorRole::QueryNorm => "query-norm",
                    MiniMaxM3AttentionTensorRole::Key => "key",
                    MiniMaxM3AttentionTensorRole::KeyNorm => "key-norm",
                    MiniMaxM3AttentionTensorRole::Value => "value",
                    MiniMaxM3AttentionTensorRole::Output => "output",
                    MiniMaxM3AttentionTensorRole::IndexQuery => "index-query",
                    MiniMaxM3AttentionTensorRole::IndexQueryNorm => "index-query-norm",
                    MiniMaxM3AttentionTensorRole::IndexKey => "index-key",
                    MiniMaxM3AttentionTensorRole::IndexKeyNorm => "index-key-norm",
                }
            ),
            Self::TextFeedForward { layer, role } => match role {
                MiniMaxM3FeedForwardTensorRole::Norm => {
                    format!("text-layer:{layer}:feed-forward:norm")
                }
                MiniMaxM3FeedForwardTensorRole::Dense { projection } => format!(
                    "text-layer:{layer}:feed-forward:dense:{}",
                    projection.canonical()
                ),
                MiniMaxM3FeedForwardTensorRole::Router => {
                    format!("text-layer:{layer}:feed-forward:router")
                }
                MiniMaxM3FeedForwardTensorRole::RouterSelectionBias => {
                    format!("text-layer:{layer}:feed-forward:router-selection-bias")
                }
                MiniMaxM3FeedForwardTensorRole::SharedExpert { projection } => format!(
                    "text-layer:{layer}:feed-forward:shared-expert:{}",
                    projection.canonical()
                ),
                MiniMaxM3FeedForwardTensorRole::RoutedExpert { expert, projection } => format!(
                    "text-layer:{layer}:feed-forward:routed-expert:{expert}:{}",
                    projection.canonical()
                ),
            },
            Self::VisionRoot { role, parameter } => format!(
                "vision-root:{}:{}",
                match role {
                    MiniMaxM3VisionRootTensorRole::TemporalPatchEmbedding => {
                        "temporal-patch-embedding"
                    }
                    MiniMaxM3VisionRootTensorRole::PreLayerNorm => "pre-layer-norm",
                },
                parameter.canonical()
            ),
            Self::VisionLayer {
                layer,
                role,
                parameter,
            } => format!(
                "vision-layer:{layer}:{}:{}",
                match role {
                    MiniMaxM3VisionLayerTensorRole::AttentionNorm => "attention-norm",
                    MiniMaxM3VisionLayerTensorRole::FeedForwardNorm => "feed-forward-norm",
                    MiniMaxM3VisionLayerTensorRole::Query => "query",
                    MiniMaxM3VisionLayerTensorRole::Key => "key",
                    MiniMaxM3VisionLayerTensorRole::Value => "value",
                    MiniMaxM3VisionLayerTensorRole::AttentionOutput => "attention-output",
                    MiniMaxM3VisionLayerTensorRole::FeedForwardUp => "feed-forward-up",
                    MiniMaxM3VisionLayerTensorRole::FeedForwardDown => "feed-forward-down",
                },
                parameter.canonical()
            ),
            Self::Projector {
                kind,
                linear,
                parameter,
            } => format!(
                "projector:{}:linear-{linear}:{}",
                match kind {
                    MiniMaxM3ProjectorKind::Multimodal => "multimodal",
                    MiniMaxM3ProjectorKind::PatchMerge => "patch-merge",
                },
                parameter.canonical()
            ),
        }
    }

    const fn routed(self) -> Option<(u8, u16, MiniMaxM3ExpertProjection)> {
        match self {
            Self::TextFeedForward {
                layer,
                role: MiniMaxM3FeedForwardTensorRole::RoutedExpert { expert, projection },
            } => Some((layer, expert, projection)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3SourceTensorMapping {
    pub source_name: String,
    pub artifact_plane: MiniMaxM3ArtifactPlane,
    pub tensor_role: MiniMaxM3TensorRole,
    pub output_name: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3GgufCatalogRow {
    pub source_name: String,
    pub source_shard: String,
    pub artifact_plane: MiniMaxM3ArtifactPlane,
    pub tensor_role: MiniMaxM3TensorRole,
    pub output_name: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3RoutedExpertSource {
    pub expert: u16,
    pub source_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxM3ExpertStackPlan {
    pub layer: u8,
    pub projection: MiniMaxM3ExpertProjection,
    pub output_name: String,
    /// Strict numeric order, exactly `0..=127`.
    pub experts: Vec<MiniMaxM3RoutedExpertSource>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MiniMaxM3RequiredPayloadTransform {
    TextGemmaNormAddOne,
    RoutedExpertNumericAxisStack,
    VisionTemporalPatchSplit { parts: u32 },
    VisionQueryKeyRopeAxisPermutation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3GgufFoundationCatalogPlan {
    pub target_metadata: BTreeMap<String, GgufValue>,
    pub source_rows: Vec<MiniMaxM3GgufCatalogRow>,
    pub expert_stacks: Vec<MiniMaxM3ExpertStackPlan>,
    pub required_payload_transforms: Vec<MiniMaxM3RequiredPayloadTransform>,
    pub source_tensor_count: usize,
    pub source_text_tensor_count: usize,
    pub source_vision_projector_tensor_count: usize,
    pub direct_source_tensor_count: usize,
    pub routed_expert_source_tensor_count: usize,
    pub stacked_expert_output_count: usize,
    pub combined_physical_candidate_count: usize,
    pub mapping_sha256: String,
    pub production_loadable: bool,
    pub payload_headers_verified: bool,
    pub payload_bytes_verified: bool,
    pub payload_transforms_executed: bool,
    pub dtype_conversion_executed: bool,
    pub quantization_executed: bool,
    pub writable_gguf_plan: bool,
    pub output_payload_bytes: Option<u64>,
    pub pass_scope: &'static str,
}

fn parse_index(
    value: &str,
    label: &str,
    upper_exclusive: u16,
) -> Result<u16, MiniMaxM3GgufFoundationError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid(format!(
            "{label} is not canonical decimal: {value}"
        )));
    }
    let parsed = value
        .parse::<u16>()
        .map_err(|_| invalid(format!("{label} is invalid: {value}")))?;
    if parsed >= upper_exclusive {
        return Err(invalid(format!("{label} is out of range: {parsed}")));
    }
    Ok(parsed)
}

fn parameter(value: &str) -> Option<MiniMaxM3Parameter> {
    match value {
        "weight" => Some(MiniMaxM3Parameter::Weight),
        "bias" => Some(MiniMaxM3Parameter::Bias),
        _ => None,
    }
}

fn projection(value: &str) -> Option<MiniMaxM3ExpertProjection> {
    match value {
        "w1" | "gate_proj" => Some(MiniMaxM3ExpertProjection::Gate),
        "w2" | "down_proj" => Some(MiniMaxM3ExpertProjection::Down),
        "w3" | "up_proj" => Some(MiniMaxM3ExpertProjection::Up),
        _ => None,
    }
}

fn output(base: impl fmt::Display, parameter: MiniMaxM3Parameter) -> String {
    format!("{base}.{}", parameter.canonical())
}

fn root_mapping(source_name: &str) -> Option<MiniMaxM3SourceTensorMapping> {
    let (role, target) = match source_name {
        "language_model.model.embed_tokens.weight" => {
            (MiniMaxM3RootTensorRole::TokenEmbedding, "token_embd.weight")
        }
        "language_model.model.norm.weight" => {
            (MiniMaxM3RootTensorRole::OutputNorm, "output_norm.weight")
        }
        "language_model.lm_head.weight" => (MiniMaxM3RootTensorRole::Output, "output.weight"),
        _ => return None,
    };
    Some(MiniMaxM3SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_plane: MiniMaxM3ArtifactPlane::Text,
        tensor_role: MiniMaxM3TensorRole::TextRoot(role),
        output_name: Some(target.to_owned()),
    })
}

fn text_layer_mapping(
    source_name: &str,
    layer: u8,
    suffix: &[&str],
) -> Result<MiniMaxM3SourceTensorMapping, MiniMaxM3GgufFoundationError> {
    let (tensor_role, output_name) = match suffix {
        ["input_layernorm", "weight"] => (
            MiniMaxM3TensorRole::TextAttention {
                layer,
                role: MiniMaxM3AttentionTensorRole::Norm,
            },
            format!("blk.{layer}.attn_norm.weight"),
        ),
        ["post_attention_layernorm", "weight"] => (
            MiniMaxM3TensorRole::TextFeedForward {
                layer,
                role: MiniMaxM3FeedForwardTensorRole::Norm,
            },
            format!("blk.{layer}.ffn_norm.weight"),
        ),
        ["self_attn", component, "weight"] => {
            let (role, target) = match *component {
                "q_proj" => (MiniMaxM3AttentionTensorRole::Query, "attn_q"),
                "q_norm" => (MiniMaxM3AttentionTensorRole::QueryNorm, "attn_q_norm"),
                "k_proj" => (MiniMaxM3AttentionTensorRole::Key, "attn_k"),
                "k_norm" => (MiniMaxM3AttentionTensorRole::KeyNorm, "attn_k_norm"),
                "v_proj" => (MiniMaxM3AttentionTensorRole::Value, "attn_v"),
                "o_proj" => (MiniMaxM3AttentionTensorRole::Output, "attn_output"),
                "index_q_proj" if layer >= REVIEWED_DENSE_LAYER_COUNT => {
                    (MiniMaxM3AttentionTensorRole::IndexQuery, "indexer.q_proj")
                }
                "index_q_norm" if layer >= REVIEWED_DENSE_LAYER_COUNT => (
                    MiniMaxM3AttentionTensorRole::IndexQueryNorm,
                    "indexer.q_norm",
                ),
                "index_k_proj" if layer >= REVIEWED_DENSE_LAYER_COUNT => {
                    (MiniMaxM3AttentionTensorRole::IndexKey, "indexer.k_proj")
                }
                "index_k_norm" if layer >= REVIEWED_DENSE_LAYER_COUNT => {
                    (MiniMaxM3AttentionTensorRole::IndexKeyNorm, "indexer.k_norm")
                }
                _ => {
                    return Err(invalid(format!(
                        "unsupported attention tensor grammar: {source_name}"
                    )));
                }
            };
            (
                MiniMaxM3TensorRole::TextAttention { layer, role },
                format!("blk.{layer}.{target}.weight"),
            )
        }
        ["mlp", projection_name, "weight"] if layer < REVIEWED_DENSE_LAYER_COUNT => {
            let projection = projection(projection_name)
                .ok_or_else(|| invalid(format!("unknown dense projection: {source_name}")))?;
            (
                MiniMaxM3TensorRole::TextFeedForward {
                    layer,
                    role: MiniMaxM3FeedForwardTensorRole::Dense { projection },
                },
                format!("blk.{layer}.ffn_{}.weight", projection.canonical()),
            )
        }
        ["block_sparse_moe", "gate", "weight"] if layer >= REVIEWED_DENSE_LAYER_COUNT => (
            MiniMaxM3TensorRole::TextFeedForward {
                layer,
                role: MiniMaxM3FeedForwardTensorRole::Router,
            },
            format!("blk.{layer}.ffn_gate_inp.weight"),
        ),
        ["block_sparse_moe", "e_score_correction_bias"] if layer >= REVIEWED_DENSE_LAYER_COUNT => (
            MiniMaxM3TensorRole::TextFeedForward {
                layer,
                role: MiniMaxM3FeedForwardTensorRole::RouterSelectionBias,
            },
            format!("blk.{layer}.exp_probs_b.bias"),
        ),
        [
            "block_sparse_moe",
            "shared_experts",
            projection_name,
            "weight",
        ] if layer >= REVIEWED_DENSE_LAYER_COUNT => {
            let projection = projection(projection_name)
                .ok_or_else(|| invalid(format!("unknown shared projection: {source_name}")))?;
            (
                MiniMaxM3TensorRole::TextFeedForward {
                    layer,
                    role: MiniMaxM3FeedForwardTensorRole::SharedExpert { projection },
                },
                format!("blk.{layer}.ffn_{}_shexp.weight", projection.canonical()),
            )
        }
        [
            "block_sparse_moe",
            "experts",
            expert,
            projection_name,
            "weight",
        ] if layer >= REVIEWED_DENSE_LAYER_COUNT => {
            let expert = parse_index(expert, "expert", REVIEWED_EXPERT_COUNT)?;
            let projection = projection(projection_name)
                .ok_or_else(|| invalid(format!("unknown routed projection: {source_name}")))?;
            (
                MiniMaxM3TensorRole::TextFeedForward {
                    layer,
                    role: MiniMaxM3FeedForwardTensorRole::RoutedExpert { expert, projection },
                },
                format!("blk.{layer}.{}.weight", projection.stacked_base()),
            )
        }
        _ => {
            return Err(invalid(format!(
                "unsupported text layer tensor grammar: {source_name}"
            )));
        }
    };
    Ok(MiniMaxM3SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_plane: MiniMaxM3ArtifactPlane::Text,
        tensor_role,
        output_name: Some(output_name),
    })
}

fn vision_layer_mapping(
    source_name: &str,
    layer: u8,
    suffix: &[&str],
) -> Result<MiniMaxM3SourceTensorMapping, MiniMaxM3GgufFoundationError> {
    let (role, base, parameter) = match suffix {
        ["layer_norm1", tail] => (
            MiniMaxM3VisionLayerTensorRole::AttentionNorm,
            "ln1",
            parameter(tail),
        ),
        ["layer_norm2", tail] => (
            MiniMaxM3VisionLayerTensorRole::FeedForwardNorm,
            "ln2",
            parameter(tail),
        ),
        ["self_attn", "q_proj", tail] => (
            MiniMaxM3VisionLayerTensorRole::Query,
            "attn_q",
            parameter(tail),
        ),
        ["self_attn", "k_proj", tail] => (
            MiniMaxM3VisionLayerTensorRole::Key,
            "attn_k",
            parameter(tail),
        ),
        ["self_attn", "v_proj", tail] => (
            MiniMaxM3VisionLayerTensorRole::Value,
            "attn_v",
            parameter(tail),
        ),
        ["self_attn", "out_proj", tail] => (
            MiniMaxM3VisionLayerTensorRole::AttentionOutput,
            "attn_out",
            parameter(tail),
        ),
        ["mlp", "fc1", tail] => (
            MiniMaxM3VisionLayerTensorRole::FeedForwardUp,
            "ffn_up",
            parameter(tail),
        ),
        ["mlp", "fc2", tail] => (
            MiniMaxM3VisionLayerTensorRole::FeedForwardDown,
            "ffn_down",
            parameter(tail),
        ),
        _ => {
            return Err(invalid(format!(
                "unsupported vision layer tensor grammar: {source_name}"
            )));
        }
    };
    let parameter = parameter
        .ok_or_else(|| invalid(format!("unknown vision parameter suffix: {source_name}")))?;
    Ok(MiniMaxM3SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_plane: MiniMaxM3ArtifactPlane::VisionProjector,
        tensor_role: MiniMaxM3TensorRole::VisionLayer {
            layer,
            role,
            parameter,
        },
        output_name: Some(output(format!("v.blk.{layer}.{base}"), parameter)),
    })
}

fn projector_mapping(
    source_name: &str,
    kind: MiniMaxM3ProjectorKind,
    rest: &str,
) -> Result<MiniMaxM3SourceTensorMapping, MiniMaxM3GgufFoundationError> {
    let parts = rest.split('.').collect::<Vec<_>>();
    let [linear, parameter_name] = parts.as_slice() else {
        return Err(invalid(format!(
            "malformed projector tensor: {source_name}"
        )));
    };
    let linear = linear
        .strip_prefix("linear_")
        .ok_or_else(|| invalid(format!("malformed projector linear: {source_name}")))?;
    let linear = parse_index(linear, "projector linear", 3)? as u8;
    if linear == 0 {
        return Err(invalid(format!(
            "projector linear is out of range: {source_name}"
        )));
    }
    let parameter = parameter(parameter_name)
        .ok_or_else(|| invalid(format!("unknown projector parameter: {source_name}")))?;
    let base = match kind {
        MiniMaxM3ProjectorKind::Multimodal => format!("mm.{linear}"),
        MiniMaxM3ProjectorKind::PatchMerge => {
            format!("mm.merger.{}", if linear == 1 { "fc1" } else { "fc2" })
        }
    };
    Ok(MiniMaxM3SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_plane: MiniMaxM3ArtifactPlane::VisionProjector,
        tensor_role: MiniMaxM3TensorRole::Projector {
            kind,
            linear,
            parameter,
        },
        output_name: Some(output(base, parameter)),
    })
}

/// Parse one exact official source name into a typed, write-disabled mapping.
pub fn map_minimax_m3_source_tensor(
    source_name: &str,
) -> Result<MiniMaxM3SourceTensorMapping, MiniMaxM3GgufFoundationError> {
    if let Some(mapping) = root_mapping(source_name) {
        return Ok(mapping);
    }
    if let Some(rest) = source_name.strip_prefix("language_model.model.layers.") {
        let parts = rest.split('.').collect::<Vec<_>>();
        let (layer, suffix) = parts
            .split_first()
            .ok_or_else(|| invalid(format!("malformed text layer tensor: {source_name}")))?;
        let layer = parse_index(layer, "text layer", u16::from(REVIEWED_TEXT_LAYER_COUNT))? as u8;
        return text_layer_mapping(source_name, layer, suffix);
    }
    if source_name == "vision_tower.vision_model.embeddings.patch_embedding.weight" {
        return Ok(MiniMaxM3SourceTensorMapping {
            source_name: source_name.to_owned(),
            artifact_plane: MiniMaxM3ArtifactPlane::VisionProjector,
            tensor_role: MiniMaxM3TensorRole::VisionRoot {
                role: MiniMaxM3VisionRootTensorRole::TemporalPatchEmbedding,
                parameter: MiniMaxM3Parameter::Weight,
            },
            // One source must be split into two temporal outputs from payload
            // shape. No output name is frozen before that transform executes.
            output_name: None,
        });
    }
    if let Some(parameter_name) =
        source_name.strip_prefix("vision_tower.vision_model.pre_layrnorm.")
    {
        let parameter = parameter(parameter_name)
            .ok_or_else(|| invalid(format!("unknown vision pre-norm parameter: {source_name}")))?;
        return Ok(MiniMaxM3SourceTensorMapping {
            source_name: source_name.to_owned(),
            artifact_plane: MiniMaxM3ArtifactPlane::VisionProjector,
            tensor_role: MiniMaxM3TensorRole::VisionRoot {
                role: MiniMaxM3VisionRootTensorRole::PreLayerNorm,
                parameter,
            },
            output_name: Some(output("v.pre_ln", parameter)),
        });
    }
    if let Some(rest) = source_name.strip_prefix("vision_tower.vision_model.encoder.layers.") {
        let parts = rest.split('.').collect::<Vec<_>>();
        let (layer, suffix) = parts
            .split_first()
            .ok_or_else(|| invalid(format!("malformed vision layer tensor: {source_name}")))?;
        let layer = parse_index(layer, "vision layer", 32)? as u8;
        return vision_layer_mapping(source_name, layer, suffix);
    }
    if let Some(rest) = source_name.strip_prefix("multi_modal_projector.") {
        return projector_mapping(source_name, MiniMaxM3ProjectorKind::Multimodal, rest);
    }
    if let Some(rest) = source_name.strip_prefix("patch_merge_mlp.") {
        return projector_mapping(source_name, MiniMaxM3ProjectorKind::PatchMerge, rest);
    }
    Err(invalid(format!(
        "unsupported tensor grammar: {source_name}"
    )))
}

fn validate_reviewed_config(config: &MiniMaxM3Config) -> Result<(), MiniMaxM3GgufFoundationError> {
    let text = &config.text;
    let sparse = &text.sparse_attention;
    let vision = &config.vision;
    let multimodal = &config.multimodal;
    let schedule = |values: &[bool]| {
        values.len() == usize::from(REVIEWED_TEXT_LAYER_COUNT)
            && values[..usize::from(REVIEWED_DENSE_LAYER_COUNT)]
                .iter()
                .all(|value| !value)
            && values[usize::from(REVIEWED_DENSE_LAYER_COUNT)..]
                .iter()
                .all(|value| *value)
    };
    if text.hidden_size != 6_144
        || text.intermediate_size != 3_072
        || text.layer_count != 60
        || text.attention_heads != 64
        || text.kv_heads != 4
        || text.head_dim != 128
        || text.vocab_size != 200_064
        || text.max_position_embeddings != 1_048_576
        || text.rms_norm_epsilon.to_bits() != 1.0e-6_f64.to_bits()
        || text.rotary_dimension != 64
        || text.rope_theta != 5_000_000
        || text.dense_intermediate_size != 12_288
        || text.shared_intermediate_size != 3_072
        || text.expert_count != 128
        || text.selected_expert_count != 4
        || text.shared_expert_count != 1
        || !schedule(&text.moe_layers)
        || text.mtp_module_count != 7
        || text.nextn_predict_layers != 1
        || text.swiglu_alpha.to_bits() != 1.702_f64.to_bits()
        || text.swiglu_limit.to_bits() != 7.0_f64.to_bits()
        || text.routed_scaling_factor.to_bits() != 2.0_f64.to_bits()
        || sparse.index_dimension != 128
        || sparse.index_heads != 4
        || sparse.top_k_blocks != 16
        || sparse.block_size != 128
        || sparse.init_blocks != 0
        || sparse.local_blocks != 1
        || sparse.score_type != "max"
        || !schedule(&sparse.enabled_layers)
        || !schedule(&sparse.index_value_disabled_layers)
        || sparse.enabled_layers != text.moe_layers
        || sparse.index_value_disabled_layers != sparse.enabled_layers
        || vision.hidden_size != 1_280
        || vision.attention_heads != 16
        || vision.layer_count != 32
        || vision.intermediate_size != 5_120
        || vision.patch_size != 14
        || vision.image_size != 2_016
        || vision.projection_dimension != 6_144
        || vision.rope_mode != "3d"
        || vision.rope_theta.to_bits() != 10_000.0_f64.to_bits()
        || vision.max_frames != 4
        || multimodal.image_grid_count != 36
        || multimodal.image_sequence_length != 576
        || multimodal.image_token_id != 200_025
        || multimodal.video_token_id != 200_026
        || multimodal.spatial_merge_size != 2
        || multimodal.temporal_patch_size != 2
        || multimodal.projector_hidden_size != 6_144
        || multimodal.production_execution_enabled
        || config.indexed_mtp_tensor_count != 0
        || config.mtp_production_execution_enabled
    {
        return Err(invalid(
            "typed config differs from the reviewed exact MiniMax M3 contract",
        ));
    }
    Ok(())
}

/// Build canonical metadata without declaring an output dtype or file type.
pub fn minimax_m3_gguf_foundation_metadata(
    config: &MiniMaxM3Config,
) -> Result<BTreeMap<String, GgufValue>, MiniMaxM3GgufFoundationError> {
    validate_reviewed_config(config)?;
    let revision_url =
        format!("https://huggingface.co/{MINIMAX_M3_REPOSITORY}/tree/{MINIMAX_M3_REVISION}");
    Ok(BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String(MINIMAX_M3_GGUF_ARCHITECTURE.to_owned()),
        ),
        ("general.alignment".to_owned(), GgufValue::U32(32)),
        (
            "general.type".to_owned(),
            GgufValue::String("model".to_owned()),
        ),
        (
            "general.name".to_owned(),
            GgufValue::String(format!("{MINIMAX_M3_REPOSITORY}@{MINIMAX_M3_REVISION}")),
        ),
        (
            "general.license".to_owned(),
            GgufValue::String(MINIMAX_M3_LICENSE.to_owned()),
        ),
        (
            "general.source.url".to_owned(),
            GgufValue::String(revision_url),
        ),
        (
            "general.source.huggingface.repository".to_owned(),
            GgufValue::String(MINIMAX_M3_REPOSITORY.to_owned()),
        ),
        (
            "minimax-m3.source.revision".to_owned(),
            GgufValue::String(MINIMAX_M3_REVISION.to_owned()),
        ),
        (
            "minimax-m3.vocab_size".to_owned(),
            GgufValue::U32(config.text.vocab_size),
        ),
        (
            "minimax-m3.context_length".to_owned(),
            GgufValue::U32(config.text.max_position_embeddings),
        ),
        (
            "minimax-m3.embedding_length".to_owned(),
            GgufValue::U32(config.text.hidden_size),
        ),
        (
            "minimax-m3.block_count".to_owned(),
            GgufValue::U32(config.text.layer_count),
        ),
        (
            "minimax-m3.leading_dense_block_count".to_owned(),
            GgufValue::U32(3),
        ),
        (
            "minimax-m3.feed_forward_length".to_owned(),
            GgufValue::U32(config.text.dense_intermediate_size),
        ),
        (
            "minimax-m3.expert_feed_forward_length".to_owned(),
            GgufValue::U32(config.text.intermediate_size),
        ),
        (
            "minimax-m3.attention.head_count".to_owned(),
            GgufValue::U32(config.text.attention_heads),
        ),
        (
            "minimax-m3.attention.head_count_kv".to_owned(),
            GgufValue::U32(config.text.kv_heads),
        ),
        (
            "minimax-m3.attention.key_length".to_owned(),
            GgufValue::U32(config.text.head_dim),
        ),
        (
            "minimax-m3.attention.value_length".to_owned(),
            GgufValue::U32(config.text.head_dim),
        ),
        (
            "minimax-m3.attention.layer_norm_rms_epsilon".to_owned(),
            GgufValue::F32(config.text.rms_norm_epsilon as f32),
        ),
        (
            "minimax-m3.rope.dimension_count".to_owned(),
            GgufValue::U32(config.text.rotary_dimension),
        ),
        (
            "minimax-m3.rope.freq_base".to_owned(),
            GgufValue::F32(config.text.rope_theta as f32),
        ),
        (
            "minimax-m3.attention.indexer.block_size".to_owned(),
            GgufValue::U32(config.text.sparse_attention.block_size),
        ),
        (
            "minimax-m3.attention.indexer.top_k".to_owned(),
            GgufValue::U32(config.text.sparse_attention.top_k_blocks),
        ),
        (
            "minimax-m3.attention.indexer.head_count".to_owned(),
            GgufValue::U32(config.text.sparse_attention.index_heads),
        ),
        (
            "minimax-m3.attention.indexer.key_length".to_owned(),
            GgufValue::U32(config.text.sparse_attention.index_dimension),
        ),
        (
            "minimax-m3.attention.indexer.local_blocks".to_owned(),
            GgufValue::U32(config.text.sparse_attention.local_blocks),
        ),
        (
            "minimax-m3.expert_count".to_owned(),
            GgufValue::U32(config.text.expert_count),
        ),
        (
            "minimax-m3.expert_used_count".to_owned(),
            GgufValue::U32(config.text.selected_expert_count),
        ),
        (
            "minimax-m3.expert_shared_count".to_owned(),
            GgufValue::U32(config.text.shared_expert_count),
        ),
        (
            "minimax-m3.expert_weights_scale".to_owned(),
            GgufValue::F32(config.text.routed_scaling_factor as f32),
        ),
        (
            "minimax-m3.expert_weights_norm".to_owned(),
            GgufValue::Bool(true),
        ),
        (
            "minimax-m3.expert_gating_func".to_owned(),
            GgufValue::U32(2),
        ),
        (
            "minimax-m3.vision.embedding_length".to_owned(),
            GgufValue::U32(config.vision.hidden_size),
        ),
        (
            "minimax-m3.vision.attention.head_count".to_owned(),
            GgufValue::U32(config.vision.attention_heads),
        ),
        (
            "minimax-m3.vision.block_count".to_owned(),
            GgufValue::U32(config.vision.layer_count),
        ),
        (
            "minimax-m3.vision.feed_forward_length".to_owned(),
            GgufValue::U32(config.vision.intermediate_size),
        ),
        (
            "minimax-m3.vision.patch_size".to_owned(),
            GgufValue::U32(config.vision.patch_size),
        ),
        (
            "minimax-m3.vision.image_size".to_owned(),
            GgufValue::U32(config.vision.image_size),
        ),
        (
            "minimax-m3.vision.projection_length".to_owned(),
            GgufValue::U32(config.vision.projection_dimension),
        ),
        (
            "minimax-m3.vision.rope.mode".to_owned(),
            GgufValue::String(config.vision.rope_mode.to_owned()),
        ),
        (
            "minimax-m3.vision.rope.freq_base".to_owned(),
            GgufValue::F32(config.vision.rope_theta as f32),
        ),
        (
            "minimax-m3.vision.max_frames".to_owned(),
            GgufValue::U32(config.vision.max_frames),
        ),
        (
            "minimax-m3.vision.spatial_merge_size".to_owned(),
            GgufValue::U32(config.multimodal.spatial_merge_size),
        ),
        (
            "minimax-m3.vision.temporal_patch_size".to_owned(),
            GgufValue::U32(config.multimodal.temporal_patch_size),
        ),
        (
            "minimax-m3.multimodal.image_grid_count".to_owned(),
            GgufValue::U32(config.multimodal.image_grid_count),
        ),
        (
            "minimax-m3.multimodal.image_sequence_length".to_owned(),
            GgufValue::U32(config.multimodal.image_sequence_length),
        ),
        (
            "minimax-m3.multimodal.image_token_id".to_owned(),
            GgufValue::U32(config.multimodal.image_token_id),
        ),
        (
            "minimax-m3.multimodal.video_token_id".to_owned(),
            GgufValue::U32(config.multimodal.video_token_id),
        ),
        (
            "minimax-m3.multimodal.production_supported".to_owned(),
            GgufValue::Bool(false),
        ),
        (
            "minimax-m3.manifest.index_metadata_bytes".to_owned(),
            GgufValue::U64(MINIMAX_M3_INDEX_ADVERTISED_BYTES),
        ),
        (
            "minimax-m3.manifest.shard_file_bytes".to_owned(),
            GgufValue::U64(MINIMAX_M3_SHARD_FILE_BYTES),
        ),
        (
            "minimax-m3.mtp.config_module_count".to_owned(),
            GgufValue::U32(config.text.mtp_module_count),
        ),
        (
            "minimax-m3.mtp.indexed_tensor_count".to_owned(),
            GgufValue::U64(config.indexed_mtp_tensor_count as u64),
        ),
        (
            "minimax-m3.mtp.production_supported".to_owned(),
            GgufValue::Bool(false),
        ),
    ]))
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StackKey {
    layer: u8,
    projection: MiniMaxM3ExpertProjection,
}

fn insert_expert_source(
    sources: &mut BTreeMap<StackKey, BTreeMap<u16, String>>,
    layer: u8,
    expert: u16,
    projection: MiniMaxM3ExpertProjection,
    source_name: &str,
) -> Result<(), MiniMaxM3GgufFoundationError> {
    let by_expert = sources.entry(StackKey { layer, projection }).or_default();
    if by_expert.insert(expert, source_name.to_owned()).is_some() {
        return Err(invalid(format!(
            "duplicate routed expert layer/projection/expert: {source_name}"
        )));
    }
    Ok(())
}

fn row_serialization(
    row: &MiniMaxM3GgufCatalogRow,
) -> Result<String, MiniMaxM3GgufFoundationError> {
    let role = row.tensor_role.canonical();
    let fields = [
        row.source_name.as_str(),
        row.source_shard.as_str(),
        row.artifact_plane.canonical(),
        role.as_str(),
        row.output_name.as_deref().unwrap_or("-"),
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

fn finalize_expert_stacks(
    sources: BTreeMap<StackKey, BTreeMap<u16, String>>,
    direct_outputs: &BTreeSet<String>,
) -> Result<Vec<MiniMaxM3ExpertStackPlan>, MiniMaxM3GgufFoundationError> {
    let mut stacks = Vec::new();
    stacks
        .try_reserve_exact(MINIMAX_M3_GGUF_STACKED_EXPERT_OUTPUT_COUNT)
        .map_err(|_| invalid("expert stack plan allocation failed"))?;
    let mut stack_outputs = BTreeSet::new();
    for layer in REVIEWED_DENSE_LAYER_COUNT..REVIEWED_TEXT_LAYER_COUNT {
        for projection in MiniMaxM3ExpertProjection::ALL {
            let key = StackKey { layer, projection };
            let by_expert = sources.get(&key).ok_or_else(|| {
                invalid(format!("missing routed expert layer/projection: {key:?}"))
            })?;
            if by_expert.len() != usize::from(REVIEWED_EXPERT_COUNT) {
                return Err(invalid(format!(
                    "routed stack does not contain 128 experts: {key:?}"
                )));
            }
            let output_name = format!("blk.{layer}.{}.weight", projection.stacked_base());
            if direct_outputs.contains(&output_name) || !stack_outputs.insert(output_name.clone()) {
                return Err(invalid(format!("stacked output collision: {output_name}")));
            }
            let mut experts = Vec::new();
            experts
                .try_reserve_exact(usize::from(REVIEWED_EXPERT_COUNT))
                .map_err(|_| invalid("expert source plan allocation failed"))?;
            for expert in 0..REVIEWED_EXPERT_COUNT {
                let source_name = by_expert.get(&expert).ok_or_else(|| {
                    invalid(format!(
                        "routed expert coverage is not contiguous at layer {layer}, projection {}, expert {expert}",
                        projection.canonical()
                    ))
                })?;
                experts.push(MiniMaxM3RoutedExpertSource {
                    expert,
                    source_name: source_name.clone(),
                });
            }
            stacks.push(MiniMaxM3ExpertStackPlan {
                layer,
                projection,
                output_name,
                experts,
            });
        }
    }
    if sources.len() != MINIMAX_M3_GGUF_STACKED_EXPERT_OUTPUT_COUNT
        || stacks.len() != MINIMAX_M3_GGUF_STACKED_EXPERT_OUTPUT_COUNT
    {
        return Err(invalid(
            "unknown routed expert layer/projection was present",
        ));
    }
    Ok(stacks)
}

/// Build the foundation catalog from already validated typed inputs.
pub fn build_minimax_m3_gguf_foundation_catalog(
    config: &MiniMaxM3Config,
    index: &MiniMaxM3Index,
) -> Result<MiniMaxM3GgufFoundationCatalogPlan, MiniMaxM3GgufFoundationError> {
    validate_reviewed_config(config)?;
    if index.index_metadata_bytes() != MINIMAX_M3_INDEX_ADVERTISED_BYTES
        || index.shard_file_bytes() != MINIMAX_M3_SHARD_FILE_BYTES
        || index.manifest_state()
            != (MiniMaxM3ManifestState::IndexMetadataExceedsShardFiles {
                index_metadata_bytes: MINIMAX_M3_INDEX_ADVERTISED_BYTES,
                shard_file_bytes: MINIMAX_M3_SHARD_FILE_BYTES,
                delta_bytes: MINIMAX_M3_INDEX_ADVERTISED_BYTES - MINIMAX_M3_SHARD_FILE_BYTES,
            })
        || index.tensor_count() != MINIMAX_M3_TENSOR_COUNT
        || index.catalog_sha256() != MINIMAX_M3_CATALOG_SHA256
    {
        return Err(invalid(
            "typed index identity or manifest differs from the reviewed exact catalog",
        ));
    }
    let summary = index.summary();
    if summary.text_root != REVIEWED_TEXT_ROOT_COUNT
        || summary.dense_text_layers != REVIEWED_DENSE_TEXT_COUNT
        || summary.moe_text_layers != REVIEWED_MOE_TEXT_COUNT
        || summary.vision != REVIEWED_VISION_COUNT
        || summary.multimodal_projector != REVIEWED_MULTIMODAL_PROJECTOR_COUNT
        || summary.patch_merge_projector != REVIEWED_PATCH_MERGE_PROJECTOR_COUNT
        || summary.mtp != 0
    {
        return Err(invalid("typed index classification differs"));
    }

    let target_metadata = minimax_m3_gguf_foundation_metadata(config)?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(index.tensor_count())
        .map_err(|_| invalid("source row allocation failed"))?;
    let mut digest = Sha256::new();
    let mut typed_keys = BTreeSet::new();
    let mut direct_outputs = BTreeSet::new();
    let mut stack_sources = BTreeMap::<StackKey, BTreeMap<u16, String>>::new();
    let mut source_text = 0_usize;
    let mut source_vision = 0_usize;
    let mut direct = 0_usize;
    let mut routed = 0_usize;

    for (source_name, shard) in index.tensors() {
        // Keep the independent typed mapping consistent with the strict index
        // grammar without relying on the coarse class as the mapping itself.
        classify_minimax_m3_tensor(source_name)
            .map_err(|error| invalid(format!("source classification failed: {error}")))?;
        let mapping = map_minimax_m3_source_tensor(source_name)?;
        let row = MiniMaxM3GgufCatalogRow {
            source_name: mapping.source_name,
            source_shard: shard.to_owned(),
            artifact_plane: mapping.artifact_plane,
            tensor_role: mapping.tensor_role,
            output_name: mapping.output_name,
        };
        let plane_count = match row.artifact_plane {
            MiniMaxM3ArtifactPlane::Text => &mut source_text,
            MiniMaxM3ArtifactPlane::VisionProjector => &mut source_vision,
        };
        *plane_count = plane_count
            .checked_add(1)
            .ok_or_else(|| invalid("artifact-plane count overflowed"))?;

        let typed_key = format!(
            "{}\t{}",
            row.artifact_plane.canonical(),
            row.tensor_role.canonical()
        );
        if !typed_keys.insert(typed_key) {
            return Err(invalid(format!(
                "typed mapping collision: {}",
                row.source_name
            )));
        }

        if let Some((layer, expert, projection)) = row.tensor_role.routed() {
            routed = routed
                .checked_add(1)
                .ok_or_else(|| invalid("routed source count overflowed"))?;
            insert_expert_source(
                &mut stack_sources,
                layer,
                expert,
                projection,
                &row.source_name,
            )?;
        } else {
            direct = direct
                .checked_add(1)
                .ok_or_else(|| invalid("direct source count overflowed"))?;
            if let Some(output_name) = row.output_name.as_deref() {
                if !direct_outputs.insert(output_name.to_owned()) {
                    return Err(invalid(format!("direct output collision: {output_name}")));
                }
            }
            if row.output_name.is_none()
                && !matches!(
                    row.tensor_role,
                    MiniMaxM3TensorRole::VisionRoot {
                        role: MiniMaxM3VisionRootTensorRole::TemporalPatchEmbedding,
                        parameter: MiniMaxM3Parameter::Weight,
                    }
                )
            {
                return Err(invalid(format!(
                    "unexpected direct output dash: {}",
                    row.source_name
                )));
            }
        }
        digest.update(row_serialization(&row)?.as_bytes());
        rows.push(row);
    }

    let expert_stacks = finalize_expert_stacks(stack_sources, &direct_outputs)?;
    let combined_physical_candidate_count = direct
        .checked_add(expert_stacks.len())
        .ok_or_else(|| invalid("physical candidate count overflowed"))?;
    if rows.len() != MINIMAX_M3_TENSOR_COUNT
        || source_text != MINIMAX_M3_GGUF_SOURCE_TEXT_TENSOR_COUNT
        || source_vision != MINIMAX_M3_GGUF_SOURCE_VISION_PROJECTOR_TENSOR_COUNT
        || direct != MINIMAX_M3_GGUF_DIRECT_SOURCE_TENSOR_COUNT
        || routed != MINIMAX_M3_GGUF_ROUTED_EXPERT_SOURCE_TENSOR_COUNT
        || routed
            != usize::from(REVIEWED_TEXT_LAYER_COUNT - REVIEWED_DENSE_LAYER_COUNT)
                * usize::from(REVIEWED_EXPERT_COUNT)
                * REVIEWED_PROJECTION_COUNT
        || expert_stacks.len() != MINIMAX_M3_GGUF_STACKED_EXPERT_OUTPUT_COUNT
        || combined_physical_candidate_count != MINIMAX_M3_GGUF_COMBINED_PHYSICAL_CANDIDATE_COUNT
    {
        return Err(invalid("catalog or physical candidate accounting differs"));
    }

    Ok(MiniMaxM3GgufFoundationCatalogPlan {
        target_metadata,
        source_rows: rows,
        expert_stacks,
        required_payload_transforms: vec![
            MiniMaxM3RequiredPayloadTransform::TextGemmaNormAddOne,
            MiniMaxM3RequiredPayloadTransform::RoutedExpertNumericAxisStack,
            MiniMaxM3RequiredPayloadTransform::VisionTemporalPatchSplit { parts: 2 },
            MiniMaxM3RequiredPayloadTransform::VisionQueryKeyRopeAxisPermutation,
        ],
        source_tensor_count: MINIMAX_M3_TENSOR_COUNT,
        source_text_tensor_count: source_text,
        source_vision_projector_tensor_count: source_vision,
        direct_source_tensor_count: direct,
        routed_expert_source_tensor_count: routed,
        stacked_expert_output_count: MINIMAX_M3_GGUF_STACKED_EXPERT_OUTPUT_COUNT,
        combined_physical_candidate_count,
        mapping_sha256: format!("{:x}", digest.finalize()),
        production_loadable: false,
        payload_headers_verified: false,
        payload_bytes_verified: false,
        payload_transforms_executed: false,
        dtype_conversion_executed: false,
        quantization_executed: false,
        writable_gguf_plan: false,
        output_payload_bytes: None,
        pass_scope: MINIMAX_M3_GGUF_PASS_SCOPE,
    })
}

/// Validate the exact official metadata bytes and return the write-disabled
/// dry-run catalog. No shard is opened by this API.
pub fn validate_minimax_m3_gguf_foundation_catalog(
    config_bytes: &[u8],
    index_bytes: &[u8],
) -> Result<MiniMaxM3GgufFoundationCatalogPlan, MiniMaxM3GgufFoundationError> {
    let config = validate_minimax_m3_config(config_bytes)
        .map_err(|error| invalid(format!("config validation failed: {error}")))?;
    let index = validate_minimax_m3_index(index_bytes)
        .map_err(|error| invalid(format!("index validation failed: {error}")))?;
    build_minimax_m3_gguf_foundation_catalog(&config, &index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_m3::{
        MiniMaxM3MultimodalConfig, MiniMaxM3SparseAttentionConfig, MiniMaxM3TextConfig,
        MiniMaxM3VisionConfig,
    };

    fn fixture_config() -> MiniMaxM3Config {
        let mut schedule = vec![true; 60];
        schedule[..3].fill(false);
        MiniMaxM3Config {
            text: MiniMaxM3TextConfig {
                hidden_size: 6_144,
                intermediate_size: 3_072,
                layer_count: 60,
                attention_heads: 64,
                kv_heads: 4,
                head_dim: 128,
                vocab_size: 200_064,
                max_position_embeddings: 1_048_576,
                rms_norm_epsilon: 1.0e-6,
                rotary_dimension: 64,
                rope_theta: 5_000_000,
                dense_intermediate_size: 12_288,
                shared_intermediate_size: 3_072,
                expert_count: 128,
                selected_expert_count: 4,
                shared_expert_count: 1,
                moe_layers: schedule.clone(),
                mtp_module_count: 7,
                nextn_predict_layers: 1,
                swiglu_alpha: 1.702,
                swiglu_limit: 7.0,
                routed_scaling_factor: 2.0,
                sparse_attention: MiniMaxM3SparseAttentionConfig {
                    index_dimension: 128,
                    index_heads: 4,
                    top_k_blocks: 16,
                    block_size: 128,
                    init_blocks: 0,
                    local_blocks: 1,
                    score_type: "max",
                    enabled_layers: schedule.clone(),
                    index_value_disabled_layers: schedule,
                },
            },
            vision: MiniMaxM3VisionConfig {
                hidden_size: 1_280,
                attention_heads: 16,
                layer_count: 32,
                intermediate_size: 5_120,
                patch_size: 14,
                image_size: 2_016,
                projection_dimension: 6_144,
                rope_mode: "3d",
                rope_theta: 10_000.0,
                max_frames: 4,
            },
            multimodal: MiniMaxM3MultimodalConfig {
                image_grid_count: 36,
                image_sequence_length: 576,
                image_token_id: 200_025,
                video_token_id: 200_026,
                spatial_merge_size: 2,
                temporal_patch_size: 2,
                projector_hidden_size: 6_144,
                production_execution_enabled: false,
            },
            indexed_mtp_tensor_count: 0,
            mtp_production_execution_enabled: false,
        }
    }

    #[test]
    fn canonical_architecture_metadata_and_no_write_flags_are_fixed() {
        let metadata = minimax_m3_gguf_foundation_metadata(&fixture_config()).unwrap();
        assert_eq!(
            metadata["general.architecture"],
            GgufValue::String("minimax-m3".to_owned())
        );
        assert_eq!(
            metadata["minimax-m3.attention.indexer.block_size"],
            GgufValue::U32(128)
        );
        assert_eq!(
            metadata["minimax-m3.attention.indexer.local_blocks"],
            GgufValue::U32(1)
        );
        assert_eq!(metadata["minimax-m3.expert_gating_func"], GgufValue::U32(2));
        assert_eq!(
            metadata["minimax-m3.manifest.index_metadata_bytes"],
            GgufValue::U64(869_157_697_024)
        );
        assert_eq!(
            metadata["minimax-m3.manifest.shard_file_bytes"],
            GgufValue::U64(854_176_398_808)
        );
        assert!(!metadata.contains_key("general.file_type"));
        assert!(!metadata.contains_key("general.quantization_version"));
    }

    #[test]
    fn text_mapping_boundaries_and_numeric_experts_are_typed() {
        let root =
            map_minimax_m3_source_tensor("language_model.model.embed_tokens.weight").unwrap();
        assert_eq!(root.output_name.as_deref(), Some("token_embd.weight"));
        let dense =
            map_minimax_m3_source_tensor("language_model.model.layers.2.mlp.down_proj.weight")
                .unwrap();
        assert_eq!(dense.output_name.as_deref(), Some("blk.2.ffn_down.weight"));
        let sparse = map_minimax_m3_source_tensor(
            "language_model.model.layers.3.self_attn.index_q_proj.weight",
        )
        .unwrap();
        assert_eq!(
            sparse.output_name.as_deref(),
            Some("blk.3.indexer.q_proj.weight")
        );
        for expert in [2_u16, 9, 10, 127] {
            let mapping = map_minimax_m3_source_tensor(&format!(
                "language_model.model.layers.59.block_sparse_moe.experts.{expert}.w3.weight"
            ))
            .unwrap();
            assert_eq!(
                mapping.tensor_role,
                MiniMaxM3TensorRole::TextFeedForward {
                    layer: 59,
                    role: MiniMaxM3FeedForwardTensorRole::RoutedExpert {
                        expert,
                        projection: MiniMaxM3ExpertProjection::Up,
                    },
                }
            );
            assert_eq!(
                mapping.output_name.as_deref(),
                Some("blk.59.ffn_up_exps.weight")
            );
        }
    }

    #[test]
    fn unknown_missing_projection_layer_and_expert_are_rejected() {
        for source_name in [
            "language_model.model.layers.02.mlp.down_proj.weight",
            "language_model.model.layers.3.mlp.down_proj.weight",
            "language_model.model.layers.2.self_attn.index_q_proj.weight",
            "language_model.model.layers.60.input_layernorm.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.128.w1.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.01.w1.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.0.w4.weight",
            "vision_tower.vision_model.encoder.layers.32.layer_norm1.weight",
            "patch_merge_mlp.linear_3.weight",
            "unknown.weight",
        ] {
            assert!(
                map_minimax_m3_source_tensor(source_name).is_err(),
                "accepted {source_name}"
            );
        }
    }

    #[test]
    fn vision_patch_split_is_explicitly_unresolved_while_direct_names_are_typed() {
        let patch = map_minimax_m3_source_tensor(
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
        )
        .unwrap();
        assert_eq!(
            patch.artifact_plane,
            MiniMaxM3ArtifactPlane::VisionProjector
        );
        assert_eq!(patch.output_name, None);
        let query = map_minimax_m3_source_tensor(
            "vision_tower.vision_model.encoder.layers.31.self_attn.q_proj.bias",
        )
        .unwrap();
        assert_eq!(query.output_name.as_deref(), Some("v.blk.31.attn_q.bias"));
        let merger = map_minimax_m3_source_tensor("patch_merge_mlp.linear_2.weight").unwrap();
        assert_eq!(merger.output_name.as_deref(), Some("mm.merger.fc2.weight"));
    }

    #[test]
    fn config_drift_is_rejected_before_catalog_work() {
        let mut config = fixture_config();
        config.text.sparse_attention.top_k_blocks = 15;
        assert!(minimax_m3_gguf_foundation_metadata(&config).is_err());
        let mut config = fixture_config();
        config.multimodal.temporal_patch_size = 1;
        assert!(minimax_m3_gguf_foundation_metadata(&config).is_err());
        let mut config = fixture_config();
        config.text.moe_layers[59] = false;
        assert!(minimax_m3_gguf_foundation_metadata(&config).is_err());
        let mut config = fixture_config();
        config.indexed_mtp_tensor_count = 1;
        assert!(minimax_m3_gguf_foundation_metadata(&config).is_err());
    }

    #[test]
    fn stack_finalizer_rejects_duplicate_missing_and_non_contiguous_coverage() {
        let key = StackKey {
            layer: 3,
            projection: MiniMaxM3ExpertProjection::Gate,
        };
        let mut duplicate = BTreeMap::new();
        insert_expert_source(
            &mut duplicate,
            3,
            0,
            MiniMaxM3ExpertProjection::Gate,
            "expert.0.first",
        )
        .unwrap();
        let error = insert_expert_source(
            &mut duplicate,
            3,
            0,
            MiniMaxM3ExpertProjection::Gate,
            "expert.0.second",
        )
        .unwrap_err();
        assert!(error.to_string().contains("duplicate routed expert"));

        let mut incomplete = BTreeMap::new();
        incomplete.insert(
            key,
            (1..REVIEWED_EXPERT_COUNT)
                .map(|expert| (expert, format!("expert.{expert}")))
                .collect(),
        );
        let error = finalize_expert_stacks(incomplete, &BTreeSet::new()).unwrap_err();
        assert!(error.to_string().contains("does not contain 128 experts"));

        let mut non_contiguous = BTreeMap::new();
        non_contiguous.insert(
            key,
            (1..=REVIEWED_EXPERT_COUNT)
                .map(|expert| (expert, format!("expert.{expert}")))
                .collect(),
        );
        let error = finalize_expert_stacks(non_contiguous, &BTreeSet::new()).unwrap_err();
        assert!(error.to_string().contains("coverage is not contiguous"));

        let mut unknown = BTreeMap::new();
        unknown.insert(
            StackKey {
                layer: 2,
                projection: MiniMaxM3ExpertProjection::Gate,
            },
            (0..REVIEWED_EXPERT_COUNT)
                .map(|expert| (expert, format!("expert.{expert}")))
                .collect(),
        );
        assert!(finalize_expert_stacks(unknown, &BTreeSet::new()).is_err());
    }

    #[test]
    #[ignore = "requires fixed official metadata under /tmp/sllm-phase58.0TzSVe or SLLM_MINIMAX_M3_METADATA_DIR"]
    fn exact_official_catalog_counts_mapping_digest_and_stack_order_are_fixed() {
        let root = std::env::var_os("SLLM_MINIMAX_M3_METADATA_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp/sllm-phase58.0TzSVe"));
        let config = std::fs::read(root.join("config.json")).expect("read official config.json");
        let index = std::fs::read(root.join("index.json")).expect("read official index.json");
        let plan = validate_minimax_m3_gguf_foundation_catalog(&config, &index).unwrap();
        assert_eq!(plan.source_tensor_count, 23_416);
        assert_eq!(plan.source_text_tensor_count, 22_893);
        assert_eq!(plan.source_vision_projector_tensor_count, 523);
        assert_eq!(plan.direct_source_tensor_count, 1_528);
        assert_eq!(plan.routed_expert_source_tensor_count, 21_888);
        assert_eq!(plan.stacked_expert_output_count, 171);
        assert_eq!(plan.combined_physical_candidate_count, 1_699);
        assert_eq!(plan.mapping_sha256, MINIMAX_M3_GGUF_MAPPING_SHA256);
        assert!(plan.expert_stacks.iter().all(|stack| {
            stack.experts.len() == 128
                && stack
                    .experts
                    .iter()
                    .enumerate()
                    .all(|(index, source)| usize::from(source.expert) == index)
        }));
        assert!(!plan.production_loadable);
        assert!(!plan.payload_headers_verified);
        assert!(!plan.payload_bytes_verified);
        assert!(!plan.payload_transforms_executed);
        assert!(!plan.dtype_conversion_executed);
        assert!(!plan.quantization_executed);
        assert!(!plan.writable_gguf_plan);
        assert_eq!(plan.output_payload_bytes, None);
    }
}
