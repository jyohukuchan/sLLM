//! Metadata and tensor-catalog dry run for the reviewed DeepSeek V4 checkpoint.
//!
//! This is deliberately not a converter.  It cannot create a `GgufWritePlan`,
//! accepts no output path or write callback, and does not claim that any tensor
//! header or payload byte was inspected.  Its only job is to turn the exact
//! reviewed config/index pair into a fail-closed, typed catalog for later GGUF
//! conversion work.

use crate::deepseek_v4::{
    DEEPSEEK_V4_CATALOG_SHA256, DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT, DEEPSEEK_V4_LICENSE,
    DEEPSEEK_V4_MAIN_LAYER_COUNT, DEEPSEEK_V4_REPOSITORY, DEEPSEEK_V4_REVISION,
    DEEPSEEK_V4_TENSOR_COUNT, DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES, DeepSeekV4Compression,
    DeepSeekV4Config, DeepSeekV4Index, validate_deepseek_v4_config, validate_deepseek_v4_index,
};
use crate::gguf::{GgufArray, GgufValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DEEPSEEK_V4_GGUF_SOURCE_TARGET_TENSOR_COUNT: usize = 67_612;
pub const DEEPSEEK_V4_GGUF_SOURCE_DSPARK_TENSOR_COUNT: usize = 4_705;
pub const DEEPSEEK_V4_GGUF_DIRECT_TENSOR_COUNT: usize = 1_661;
pub const DEEPSEEK_V4_GGUF_ROUTED_EXPERT_SOURCE_TENSOR_COUNT: usize = 70_656;
pub const DEEPSEEK_V4_GGUF_STACKED_EXPERT_OUTPUT_COUNT: usize = 138;
pub const DEEPSEEK_V4_GGUF_MAIN_PHYSICAL_TENSOR_COUNT: usize = 1_693;
pub const DEEPSEEK_V4_GGUF_DSPARK_PHYSICAL_TENSOR_COUNT: usize = 106;
pub const DEEPSEEK_V4_GGUF_COMBINED_PHYSICAL_TENSOR_COUNT: usize = 1_799;
pub const DEEPSEEK_V4_GGUF_PASS_SCOPE: &str =
    "exact-metadata-and-index-catalog-only-no-payload-no-write";
pub const DEEPSEEK_V4_GGUF_MAPPING_SERIALIZATION: &str =
    "utf8-tsv-v1:source,shard,artifact-role,typed-role,output-or-dash,plane;lf-rows";

/// SHA-256 of the canonical serialization described by
/// [`DEEPSEEK_V4_GGUF_MAPPING_SERIALIZATION`] for the reviewed 72,317-row index.
/// Filled from the ignored exact-official-metadata test, never from payload data.
pub const DEEPSEEK_V4_GGUF_MAPPING_SHA256: &str =
    "69302fb84672fbafa9e5280e752ba1370a178853cc775f436cac33739d47db91";

const REVIEWED_SOURCE_ROOT_COUNT: usize = 6;
const REVIEWED_SOURCE_MAIN_COUNT: usize = 67_606;
const REVIEWED_MAIN_ROUTED_SOURCE_COUNT: usize = 66_048;
const REVIEWED_DSPARK_ROUTED_SOURCE_COUNT: usize = 4_608;
const REVIEWED_MAIN_STACK_COUNT: usize = 129;
const REVIEWED_DSPARK_STACK_COUNT: usize = 9;
const REVIEWED_EXPERT_COUNT: u16 = 256;
const REVIEWED_PROJECTION_COUNT: usize = 3;
const REVIEWED_COMPRESSION_RATIOS: [u32; 46] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4GgufFoundationError {
    Invalid(String),
}

impl fmt::Display for DeepSeekV4GgufFoundationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(
                    formatter,
                    "invalid DeepSeek V4 GGUF foundation catalog: {message}"
                )
            }
        }
    }
}

impl std::error::Error for DeepSeekV4GgufFoundationError {}

fn invalid(message: impl Into<String>) -> DeepSeekV4GgufFoundationError {
    DeepSeekV4GgufFoundationError::Invalid(message.into())
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4ArtifactRole {
    TargetRoot,
    TargetMain { layer: u8 },
    Dspark { stage: u8 },
}

impl DeepSeekV4ArtifactRole {
    fn canonical(self) -> String {
        match self {
            Self::TargetRoot => "target-root".to_owned(),
            Self::TargetMain { layer } => format!("target-main:{layer}"),
            Self::Dspark { stage } => format!("dspark:{stage}"),
        }
    }

    const fn is_dspark(self) -> bool {
        matches!(self, Self::Dspark { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4RootTensorRole {
    TokenEmbedding,
    OutputNorm,
    Output,
    OutputHyperConnectionFn,
    OutputHyperConnectionBase,
    OutputHyperConnectionScale,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4AttentionTensorRole {
    Sink,
    QueryA,
    QueryB,
    QueryANorm,
    KeyValue,
    KeyValueANorm,
    OutputA,
    OutputB,
    CompressorKeyValue,
    CompressorGate,
    CompressorApe,
    CompressorNorm,
    IndexerProjection,
    IndexerQueryB,
    IndexerCompressorKeyValue,
    IndexerCompressorGate,
    IndexerCompressorApe,
    IndexerCompressorNorm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4HyperConnectionSite {
    Attention,
    FeedForward,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4HyperConnectionParameter {
    Fn,
    Base,
    Scale,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4ExpertProjection {
    Gate,
    Down,
    Up,
}

impl DeepSeekV4ExpertProjection {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Down => "down",
            Self::Up => "up",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4FeedForwardTensorRole {
    Norm,
    Router,
    RouterSelectionBias,
    HashTable,
    SharedExpert {
        projection: DeepSeekV4ExpertProjection,
    },
    RoutedExpert {
        expert: u16,
        projection: DeepSeekV4ExpertProjection,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4DsparkSpecialTensorRole {
    MainNorm,
    MainProjection,
    ConfidenceProjection,
    OutputHyperConnectionFn,
    OutputHyperConnectionBase,
    OutputHyperConnectionScale,
    MarkovProjection1,
    MarkovProjection2,
    Norm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4TensorRole {
    Root(DeepSeekV4RootTensorRole),
    Attention(DeepSeekV4AttentionTensorRole),
    AttentionNorm,
    FeedForward(DeepSeekV4FeedForwardTensorRole),
    HyperConnection {
        site: DeepSeekV4HyperConnectionSite,
        parameter: DeepSeekV4HyperConnectionParameter,
    },
    DsparkSpecial(DeepSeekV4DsparkSpecialTensorRole),
}

impl DeepSeekV4TensorRole {
    fn canonical(self) -> String {
        match self {
            Self::Root(role) => format!(
                "root:{}",
                match role {
                    DeepSeekV4RootTensorRole::TokenEmbedding => "token-embedding",
                    DeepSeekV4RootTensorRole::OutputNorm => "output-norm",
                    DeepSeekV4RootTensorRole::Output => "output",
                    DeepSeekV4RootTensorRole::OutputHyperConnectionFn => "output-hc-fn",
                    DeepSeekV4RootTensorRole::OutputHyperConnectionBase => "output-hc-base",
                    DeepSeekV4RootTensorRole::OutputHyperConnectionScale => "output-hc-scale",
                }
            ),
            Self::Attention(role) => format!(
                "attention:{}",
                match role {
                    DeepSeekV4AttentionTensorRole::Sink => "sink",
                    DeepSeekV4AttentionTensorRole::QueryA => "query-a",
                    DeepSeekV4AttentionTensorRole::QueryB => "query-b",
                    DeepSeekV4AttentionTensorRole::QueryANorm => "query-a-norm",
                    DeepSeekV4AttentionTensorRole::KeyValue => "key-value",
                    DeepSeekV4AttentionTensorRole::KeyValueANorm => "key-value-a-norm",
                    DeepSeekV4AttentionTensorRole::OutputA => "output-a",
                    DeepSeekV4AttentionTensorRole::OutputB => "output-b",
                    DeepSeekV4AttentionTensorRole::CompressorKeyValue => {
                        "compressor-key-value"
                    }
                    DeepSeekV4AttentionTensorRole::CompressorGate => "compressor-gate",
                    DeepSeekV4AttentionTensorRole::CompressorApe => "compressor-ape",
                    DeepSeekV4AttentionTensorRole::CompressorNorm => "compressor-norm",
                    DeepSeekV4AttentionTensorRole::IndexerProjection => "indexer-projection",
                    DeepSeekV4AttentionTensorRole::IndexerQueryB => "indexer-query-b",
                    DeepSeekV4AttentionTensorRole::IndexerCompressorKeyValue => {
                        "indexer-compressor-key-value"
                    }
                    DeepSeekV4AttentionTensorRole::IndexerCompressorGate => {
                        "indexer-compressor-gate"
                    }
                    DeepSeekV4AttentionTensorRole::IndexerCompressorApe => {
                        "indexer-compressor-ape"
                    }
                    DeepSeekV4AttentionTensorRole::IndexerCompressorNorm => {
                        "indexer-compressor-norm"
                    }
                }
            ),
            Self::AttentionNorm => "attention-norm".to_owned(),
            Self::FeedForward(role) => match role {
                DeepSeekV4FeedForwardTensorRole::Norm => "feed-forward:norm".to_owned(),
                DeepSeekV4FeedForwardTensorRole::Router => "feed-forward:router".to_owned(),
                DeepSeekV4FeedForwardTensorRole::RouterSelectionBias => {
                    "feed-forward:router-selection-bias".to_owned()
                }
                DeepSeekV4FeedForwardTensorRole::HashTable => "feed-forward:hash-table".to_owned(),
                DeepSeekV4FeedForwardTensorRole::SharedExpert { projection } => {
                    format!("feed-forward:shared-expert:{}", projection.canonical())
                }
                DeepSeekV4FeedForwardTensorRole::RoutedExpert { expert, projection } => format!(
                    "feed-forward:routed-expert:{expert}:{}",
                    projection.canonical()
                ),
            },
            Self::HyperConnection { site, parameter } => format!(
                "hyper-connection:{}:{}",
                match site {
                    DeepSeekV4HyperConnectionSite::Attention => "attention",
                    DeepSeekV4HyperConnectionSite::FeedForward => "feed-forward",
                },
                match parameter {
                    DeepSeekV4HyperConnectionParameter::Fn => "fn",
                    DeepSeekV4HyperConnectionParameter::Base => "base",
                    DeepSeekV4HyperConnectionParameter::Scale => "scale",
                }
            ),
            Self::DsparkSpecial(role) => format!(
                "dspark-special:{}",
                match role {
                    DeepSeekV4DsparkSpecialTensorRole::MainNorm => "main-norm",
                    DeepSeekV4DsparkSpecialTensorRole::MainProjection => "main-projection",
                    DeepSeekV4DsparkSpecialTensorRole::ConfidenceProjection => {
                        "confidence-projection"
                    }
                    DeepSeekV4DsparkSpecialTensorRole::OutputHyperConnectionFn => "output-hc-fn",
                    DeepSeekV4DsparkSpecialTensorRole::OutputHyperConnectionBase => {
                        "output-hc-base"
                    }
                    DeepSeekV4DsparkSpecialTensorRole::OutputHyperConnectionScale => {
                        "output-hc-scale"
                    }
                    DeepSeekV4DsparkSpecialTensorRole::MarkovProjection1 => "markov-projection-1",
                    DeepSeekV4DsparkSpecialTensorRole::MarkovProjection2 => "markov-projection-2",
                    DeepSeekV4DsparkSpecialTensorRole::Norm => "norm",
                }
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeepSeekV4TensorPlane {
    Direct,
    Fp8E4m3Value,
    Ue8m0BlockScale,
    RoutedMxfp4Value,
    RoutedMxfp4Scale,
}

impl DeepSeekV4TensorPlane {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Fp8E4m3Value => "fp8-e4m3-value-i8-carrier",
            Self::Ue8m0BlockScale => "ue8m0-block-scale",
            Self::RoutedMxfp4Value => "routed-mxfp4-value",
            Self::RoutedMxfp4Scale => "routed-mxfp4-scale",
        }
    }

    const fn is_routed(self) -> bool {
        matches!(self, Self::RoutedMxfp4Value | Self::RoutedMxfp4Scale)
    }

    const fn is_direct_fp8(self) -> bool {
        matches!(self, Self::Fp8E4m3Value | Self::Ue8m0BlockScale)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4SourceTensorMapping {
    pub source_name: String,
    pub artifact_role: DeepSeekV4ArtifactRole,
    pub tensor_role: DeepSeekV4TensorRole,
    /// Canonical target name for main-model tensors.  DSpark names are
    /// intentionally left unfrozen at this foundation stage.
    pub output_name: Option<String>,
    pub plane: DeepSeekV4TensorPlane,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4GgufCatalogRow {
    pub source_name: String,
    pub source_shard: String,
    pub artifact_role: DeepSeekV4ArtifactRole,
    pub tensor_role: DeepSeekV4TensorRole,
    pub output_name: Option<String>,
    pub plane: DeepSeekV4TensorPlane,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4RoutedExpertPair {
    pub expert: u16,
    pub value_source: String,
    pub scale_source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4ExpertStackPlan {
    pub artifact_role: DeepSeekV4ArtifactRole,
    pub projection: DeepSeekV4ExpertProjection,
    /// Present only for the main target. DSpark output naming is not frozen.
    pub output_name: Option<String>,
    /// Strict numeric expert order, `0..=255`.
    pub experts: Vec<DeepSeekV4RoutedExpertPair>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4GgufFoundationCatalogPlan {
    pub target_metadata: BTreeMap<String, GgufValue>,
    pub source_rows: Vec<DeepSeekV4GgufCatalogRow>,
    pub expert_stacks: Vec<DeepSeekV4ExpertStackPlan>,
    pub source_tensor_count: usize,
    pub source_target_tensor_count: usize,
    pub source_dspark_tensor_count: usize,
    pub direct_tensor_count: usize,
    pub routed_expert_source_tensor_count: usize,
    pub stacked_expert_output_count: usize,
    pub main_physical_tensor_count: usize,
    pub dspark_physical_tensor_count: usize,
    pub combined_physical_tensor_count: usize,
    pub config_nextn_predict_layers: u32,
    pub checkpoint_dspark_stages: u32,
    pub mapping_sha256: String,
    pub production_loadable: bool,
    pub payload_headers_verified: bool,
    pub payload_bytes_verified: bool,
    pub writable_gguf_plan: bool,
    pub output_payload_bytes: Option<u64>,
    pub pass_scope: &'static str,
}

fn parse_index(
    value: &str,
    label: &str,
    upper_exclusive: u16,
) -> Result<u16, DeepSeekV4GgufFoundationError> {
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

fn root_mapping(source_name: &str) -> Option<(DeepSeekV4RootTensorRole, &'static str)> {
    Some(match source_name {
        "embed.weight" => (
            DeepSeekV4RootTensorRole::TokenEmbedding,
            "token_embd.weight",
        ),
        "norm.weight" => (DeepSeekV4RootTensorRole::OutputNorm, "output_norm.weight"),
        "head.weight" => (DeepSeekV4RootTensorRole::Output, "output.weight"),
        "hc_head_fn" => (
            DeepSeekV4RootTensorRole::OutputHyperConnectionFn,
            "output_hc_fn.weight",
        ),
        "hc_head_base" => (
            DeepSeekV4RootTensorRole::OutputHyperConnectionBase,
            "output_hc_base.weight",
        ),
        "hc_head_scale" => (
            DeepSeekV4RootTensorRole::OutputHyperConnectionScale,
            "output_hc_scale.weight",
        ),
        _ => return None,
    })
}

fn projection(value: &str) -> Option<DeepSeekV4ExpertProjection> {
    match value {
        "w1" => Some(DeepSeekV4ExpertProjection::Gate),
        "w2" => Some(DeepSeekV4ExpertProjection::Down),
        "w3" => Some(DeepSeekV4ExpertProjection::Up),
        _ => None,
    }
}

fn output_with_plane(base: &str, plane: DeepSeekV4TensorPlane) -> String {
    let suffix = match plane {
        DeepSeekV4TensorPlane::Ue8m0BlockScale => ".scale",
        DeepSeekV4TensorPlane::Direct
        | DeepSeekV4TensorPlane::Fp8E4m3Value
        | DeepSeekV4TensorPlane::RoutedMxfp4Value
        | DeepSeekV4TensorPlane::RoutedMxfp4Scale => ".weight",
    };
    format!("{base}{suffix}")
}

fn direct_or_fp8_plane(last: &str) -> Option<DeepSeekV4TensorPlane> {
    match last {
        "weight" => Some(DeepSeekV4TensorPlane::Fp8E4m3Value),
        "scale" => Some(DeepSeekV4TensorPlane::Ue8m0BlockScale),
        _ => None,
    }
}

fn map_layer_suffix(
    source_name: &str,
    artifact_role: DeepSeekV4ArtifactRole,
    suffix: &[&str],
) -> Result<DeepSeekV4SourceTensorMapping, DeepSeekV4GgufFoundationError> {
    let output_prefix = match artifact_role {
        DeepSeekV4ArtifactRole::TargetMain { layer } => Some(format!("blk.{layer}")),
        DeepSeekV4ArtifactRole::Dspark { .. } => None,
        DeepSeekV4ArtifactRole::TargetRoot => return Err(invalid("root used as a layer")),
    };
    let output = |base: &str, plane| {
        output_prefix
            .as_ref()
            .map(|prefix| output_with_plane(&format!("{prefix}.{base}"), plane))
    };
    let direct = DeepSeekV4TensorPlane::Direct;

    let (tensor_role, plane, output_name) = match suffix {
        ["attn", "attn_sink"] => (
            DeepSeekV4TensorRole::Attention(DeepSeekV4AttentionTensorRole::Sink),
            direct,
            output("attn_sinks", direct),
        ),
        [
            "attn",
            projection_name @ ("wq_a" | "wq_b" | "wkv" | "wo_a" | "wo_b"),
            tail,
        ] => {
            let plane = direct_or_fp8_plane(tail)
                .ok_or_else(|| invalid(format!("invalid FP8 projection plane: {source_name}")))?;
            let (role, base) = match *projection_name {
                "wq_a" => (DeepSeekV4AttentionTensorRole::QueryA, "attn_q_a"),
                "wq_b" => (DeepSeekV4AttentionTensorRole::QueryB, "attn_q_b"),
                "wkv" => (DeepSeekV4AttentionTensorRole::KeyValue, "attn_kv"),
                "wo_a" => (DeepSeekV4AttentionTensorRole::OutputA, "attn_output_a"),
                "wo_b" => (DeepSeekV4AttentionTensorRole::OutputB, "attn_output_b"),
                _ => unreachable!(),
            };
            (
                DeepSeekV4TensorRole::Attention(role),
                plane,
                output(base, plane),
            )
        }
        ["attn", norm @ ("q_norm" | "kv_norm"), "weight"] => {
            let (role, base) = if *norm == "q_norm" {
                (DeepSeekV4AttentionTensorRole::QueryANorm, "attn_q_a_norm")
            } else {
                (
                    DeepSeekV4AttentionTensorRole::KeyValueANorm,
                    "attn_kv_a_norm",
                )
            };
            (
                DeepSeekV4TensorRole::Attention(role),
                direct,
                output(base, direct),
            )
        }
        ["attn", "compressor", component] => {
            let (role, base) = match *component {
                "ape" => (
                    DeepSeekV4AttentionTensorRole::CompressorApe,
                    "attn_compressor_ape",
                ),
                _ => return Err(invalid(format!("invalid compressor tensor: {source_name}"))),
            };
            (
                DeepSeekV4TensorRole::Attention(role),
                direct,
                output(base, direct),
            )
        }
        ["attn", "compressor", component, "weight"] => {
            let (role, base) = match *component {
                "wkv" => (
                    DeepSeekV4AttentionTensorRole::CompressorKeyValue,
                    "attn_compressor_kv",
                ),
                "wgate" => (
                    DeepSeekV4AttentionTensorRole::CompressorGate,
                    "attn_compressor_gate",
                ),
                "norm" => (
                    DeepSeekV4AttentionTensorRole::CompressorNorm,
                    "attn_compressor_norm",
                ),
                _ => return Err(invalid(format!("invalid compressor tensor: {source_name}"))),
            };
            (
                DeepSeekV4TensorRole::Attention(role),
                direct,
                output(base, direct),
            )
        }
        ["attn", "indexer", "wq_b", tail] => {
            let plane = direct_or_fp8_plane(tail)
                .ok_or_else(|| invalid(format!("invalid indexer FP8 plane: {source_name}")))?;
            (
                DeepSeekV4TensorRole::Attention(DeepSeekV4AttentionTensorRole::IndexerQueryB),
                plane,
                output("indexer.attn_q_b", plane),
            )
        }
        ["attn", "indexer", "weights_proj", "weight"] => (
            DeepSeekV4TensorRole::Attention(DeepSeekV4AttentionTensorRole::IndexerProjection),
            direct,
            output("indexer.proj", direct),
        ),
        ["attn", "indexer", "compressor", component] => {
            let (role, base) = match *component {
                "ape" => (
                    DeepSeekV4AttentionTensorRole::IndexerCompressorApe,
                    "indexer_compressor_ape",
                ),
                _ => {
                    return Err(invalid(format!(
                        "invalid indexer compressor tensor: {source_name}"
                    )));
                }
            };
            (
                DeepSeekV4TensorRole::Attention(role),
                direct,
                output(base, direct),
            )
        }
        ["attn", "indexer", "compressor", component, "weight"] => {
            let (role, base) = match *component {
                "wkv" => (
                    DeepSeekV4AttentionTensorRole::IndexerCompressorKeyValue,
                    "indexer_compressor_kv",
                ),
                "wgate" => (
                    DeepSeekV4AttentionTensorRole::IndexerCompressorGate,
                    "indexer_compressor_gate",
                ),
                "norm" => (
                    DeepSeekV4AttentionTensorRole::IndexerCompressorNorm,
                    "indexer_compressor_norm",
                ),
                _ => {
                    return Err(invalid(format!(
                        "invalid indexer compressor tensor: {source_name}"
                    )));
                }
            };
            (
                DeepSeekV4TensorRole::Attention(role),
                direct,
                output(base, direct),
            )
        }
        ["attn_norm", "weight"] => (
            DeepSeekV4TensorRole::AttentionNorm,
            direct,
            output("attn_norm", direct),
        ),
        ["ffn_norm", "weight"] => (
            DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::Norm),
            direct,
            output("ffn_norm", direct),
        ),
        ["ffn", "gate", "weight"] => (
            DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::Router),
            direct,
            output("ffn_gate_inp", direct),
        ),
        ["ffn", "gate", "bias"] => (
            DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::RouterSelectionBias),
            direct,
            output_prefix
                .as_ref()
                .map(|prefix| format!("{prefix}.exp_probs_b.bias")),
        ),
        ["ffn", "gate", "tid2eid"] => (
            DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::HashTable),
            direct,
            output("ffn_gate_tid2eid", direct),
        ),
        ["ffn", "shared_experts", projection_name, tail] => {
            let projection = projection(projection_name).ok_or_else(|| {
                invalid(format!("invalid shared expert projection: {source_name}"))
            })?;
            let plane = direct_or_fp8_plane(tail)
                .ok_or_else(|| invalid(format!("invalid shared expert plane: {source_name}")))?;
            let base = match projection {
                DeepSeekV4ExpertProjection::Gate => "ffn_gate_shexp",
                DeepSeekV4ExpertProjection::Down => "ffn_down_shexp",
                DeepSeekV4ExpertProjection::Up => "ffn_up_shexp",
            };
            (
                DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::SharedExpert {
                    projection,
                }),
                plane,
                output(base, plane),
            )
        }
        [
            "ffn",
            "experts",
            expert,
            projection_name,
            tail @ ("weight" | "scale"),
        ] => {
            let expert = parse_index(expert, "expert", REVIEWED_EXPERT_COUNT)?;
            let projection = projection(projection_name).ok_or_else(|| {
                invalid(format!("invalid routed expert projection: {source_name}"))
            })?;
            let plane = if *tail == "weight" {
                DeepSeekV4TensorPlane::RoutedMxfp4Value
            } else {
                DeepSeekV4TensorPlane::RoutedMxfp4Scale
            };
            let base = format!("ffn_{}_exps", projection.canonical());
            (
                DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::RoutedExpert {
                    expert,
                    projection,
                }),
                plane,
                output(&base, plane),
            )
        }
        [
            hc @ ("hc_attn_fn" | "hc_attn_base" | "hc_attn_scale" | "hc_ffn_fn" | "hc_ffn_base"
            | "hc_ffn_scale"),
        ] => {
            let site = if hc.starts_with("hc_attn") {
                DeepSeekV4HyperConnectionSite::Attention
            } else {
                DeepSeekV4HyperConnectionSite::FeedForward
            };
            let parameter = if hc.ends_with("_fn") {
                DeepSeekV4HyperConnectionParameter::Fn
            } else if hc.ends_with("_base") {
                DeepSeekV4HyperConnectionParameter::Base
            } else {
                DeepSeekV4HyperConnectionParameter::Scale
            };
            let base = if site == DeepSeekV4HyperConnectionSite::Attention {
                match parameter {
                    DeepSeekV4HyperConnectionParameter::Fn => "hc_attn_fn",
                    DeepSeekV4HyperConnectionParameter::Base => "hc_attn_base",
                    DeepSeekV4HyperConnectionParameter::Scale => "hc_attn_scale",
                }
            } else {
                match parameter {
                    DeepSeekV4HyperConnectionParameter::Fn => "hc_ffn_fn",
                    DeepSeekV4HyperConnectionParameter::Base => "hc_ffn_base",
                    DeepSeekV4HyperConnectionParameter::Scale => "hc_ffn_scale",
                }
            };
            (
                DeepSeekV4TensorRole::HyperConnection { site, parameter },
                direct,
                output(base, direct),
            )
        }
        _ => {
            return Err(invalid(format!(
                "unsupported tensor grammar: {source_name}"
            )));
        }
    };

    Ok(DeepSeekV4SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_role,
        tensor_role,
        output_name,
        plane,
    })
}

fn validate_layer_semantics(
    source_name: &str,
    layer: u8,
    mapping: &DeepSeekV4SourceTensorMapping,
    compression: DeepSeekV4Compression,
) -> Result<(), DeepSeekV4GgufFoundationError> {
    let role = mapping.tensor_role;
    let is_compressor = matches!(
        role,
        DeepSeekV4TensorRole::Attention(
            DeepSeekV4AttentionTensorRole::CompressorKeyValue
                | DeepSeekV4AttentionTensorRole::CompressorGate
                | DeepSeekV4AttentionTensorRole::CompressorApe
                | DeepSeekV4AttentionTensorRole::CompressorNorm
        )
    );
    let is_indexer = matches!(
        role,
        DeepSeekV4TensorRole::Attention(
            DeepSeekV4AttentionTensorRole::IndexerProjection
                | DeepSeekV4AttentionTensorRole::IndexerQueryB
                | DeepSeekV4AttentionTensorRole::IndexerCompressorKeyValue
                | DeepSeekV4AttentionTensorRole::IndexerCompressorGate
                | DeepSeekV4AttentionTensorRole::IndexerCompressorApe
                | DeepSeekV4AttentionTensorRole::IndexerCompressorNorm
        )
    );
    if (is_compressor && compression == DeepSeekV4Compression::Uncompressed)
        || (is_indexer && compression != DeepSeekV4Compression::Csa4To1)
    {
        return Err(invalid(format!(
            "attention tensor disagrees with layer {layer} compression: {source_name}"
        )));
    }
    match role {
        DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::HashTable)
            if layer >= 3 =>
        {
            return Err(invalid(format!(
                "hash table appears after layer 2: {source_name}"
            )));
        }
        DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::RouterSelectionBias)
            if layer < 3 =>
        {
            return Err(invalid(format!(
                "selection bias appears in hash layer: {source_name}"
            )));
        }
        _ => {}
    }
    Ok(())
}

fn map_dspark_special(
    source_name: &str,
    stage: u8,
    suffix: &[&str],
) -> Result<Option<DeepSeekV4SourceTensorMapping>, DeepSeekV4GgufFoundationError> {
    let (role, plane) = match suffix {
        ["main_norm", "weight"] if stage == 0 => (
            DeepSeekV4DsparkSpecialTensorRole::MainNorm,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["main_proj", tail @ ("weight" | "scale")] if stage == 0 => (
            DeepSeekV4DsparkSpecialTensorRole::MainProjection,
            direct_or_fp8_plane(tail).expect("matched a direct FP8 plane"),
        ),
        ["confidence_head", "proj", "weight"] if stage == 2 => (
            DeepSeekV4DsparkSpecialTensorRole::ConfidenceProjection,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["hc_head_fn"] if stage == 2 => (
            DeepSeekV4DsparkSpecialTensorRole::OutputHyperConnectionFn,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["hc_head_base"] if stage == 2 => (
            DeepSeekV4DsparkSpecialTensorRole::OutputHyperConnectionBase,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["hc_head_scale"] if stage == 2 => (
            DeepSeekV4DsparkSpecialTensorRole::OutputHyperConnectionScale,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["markov_head", "markov_w1", "weight"] if stage == 2 => (
            DeepSeekV4DsparkSpecialTensorRole::MarkovProjection1,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["markov_head", "markov_w2", "weight"] if stage == 2 => (
            DeepSeekV4DsparkSpecialTensorRole::MarkovProjection2,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["norm", "weight"] if stage == 2 => (
            DeepSeekV4DsparkSpecialTensorRole::Norm,
            DeepSeekV4TensorPlane::Direct,
        ),
        ["main_norm", ..]
        | ["main_proj", ..]
        | ["confidence_head", ..]
        | ["hc_head_fn", ..]
        | ["hc_head_base", ..]
        | ["hc_head_scale", ..]
        | ["markov_head", ..]
        | ["norm", ..] => {
            return Err(invalid(format!(
                "DSpark special is in the wrong stage: {source_name}"
            )));
        }
        _ => return Ok(None),
    };
    Ok(Some(DeepSeekV4SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_role: DeepSeekV4ArtifactRole::Dspark { stage },
        tensor_role: DeepSeekV4TensorRole::DsparkSpecial(role),
        output_name: None,
        plane,
    }))
}

/// Parse one official source tensor name into a typed catalog mapping.
///
/// This function is grammar-based and range-checks every coordinate; it does
/// not perform regex replacement. Main layers are `0..=42`, routed experts are
/// `0..=255`, and DSpark stages are `0..=2`.
pub fn map_deepseek_v4_source_tensor(
    source_name: &str,
) -> Result<DeepSeekV4SourceTensorMapping, DeepSeekV4GgufFoundationError> {
    if let Some((role, output)) = root_mapping(source_name) {
        return Ok(DeepSeekV4SourceTensorMapping {
            source_name: source_name.to_owned(),
            artifact_role: DeepSeekV4ArtifactRole::TargetRoot,
            tensor_role: DeepSeekV4TensorRole::Root(role),
            output_name: Some(output.to_owned()),
            plane: DeepSeekV4TensorPlane::Direct,
        });
    }

    let parts = source_name.split('.').collect::<Vec<_>>();
    if parts.len() < 3 {
        return Err(invalid(format!(
            "unsupported tensor grammar: {source_name}"
        )));
    }
    match parts[0] {
        "layers" => {
            let layer =
                parse_index(parts[1], "main layer", DEEPSEEK_V4_MAIN_LAYER_COUNT as u16)? as u8;
            let mapping = map_layer_suffix(
                source_name,
                DeepSeekV4ArtifactRole::TargetMain { layer },
                &parts[2..],
            )?;
            let compression = REVIEWED_COMPRESSION_RATIOS[usize::from(layer)];
            let compression = match compression {
                0 => DeepSeekV4Compression::Uncompressed,
                4 => DeepSeekV4Compression::Csa4To1,
                128 => DeepSeekV4Compression::Hca128To1,
                _ => unreachable!(),
            };
            validate_layer_semantics(source_name, layer, &mapping, compression)?;
            Ok(mapping)
        }
        "mtp" => {
            let stage = parse_index(
                parts[1],
                "DSpark stage",
                DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT as u16,
            )? as u8;
            if let Some(mapping) = map_dspark_special(source_name, stage, &parts[2..])? {
                return Ok(mapping);
            }
            let mapping = map_layer_suffix(
                source_name,
                DeepSeekV4ArtifactRole::Dspark { stage },
                &parts[2..],
            )?;
            if matches!(
                mapping.tensor_role,
                DeepSeekV4TensorRole::Attention(
                    DeepSeekV4AttentionTensorRole::CompressorKeyValue
                        | DeepSeekV4AttentionTensorRole::CompressorGate
                        | DeepSeekV4AttentionTensorRole::CompressorApe
                        | DeepSeekV4AttentionTensorRole::CompressorNorm
                        | DeepSeekV4AttentionTensorRole::IndexerProjection
                        | DeepSeekV4AttentionTensorRole::IndexerQueryB
                        | DeepSeekV4AttentionTensorRole::IndexerCompressorKeyValue
                        | DeepSeekV4AttentionTensorRole::IndexerCompressorGate
                        | DeepSeekV4AttentionTensorRole::IndexerCompressorApe
                        | DeepSeekV4AttentionTensorRole::IndexerCompressorNorm
                ) | DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::HashTable)
            ) {
                return Err(invalid(format!("unsupported DSpark role: {source_name}")));
            }
            Ok(mapping)
        }
        _ => Err(invalid(format!(
            "unsupported tensor grammar: {source_name}"
        ))),
    }
}

fn validate_reviewed_config(
    config: &DeepSeekV4Config,
) -> Result<(), DeepSeekV4GgufFoundationError> {
    let ratios = config
        .compression
        .iter()
        .map(|compression| match compression {
            DeepSeekV4Compression::Uncompressed => 0,
            DeepSeekV4Compression::Csa4To1 => 4,
            DeepSeekV4Compression::Hca128To1 => 128,
        })
        .collect::<Vec<_>>();
    if config.hidden_size != 4_096
        || config.layer_count != 43
        || config.hash_layer_count != 3
        || config.attention_heads != 64
        || config.kv_heads != 1
        || config.head_dim != 512
        || config.index_heads != 64
        || config.index_head_dim != 128
        || config.index_top_k != 512
        || config.expert_count != 256
        || config.selected_expert_count != 6
        || config.shared_expert_count != 1
        || config.expert_intermediate_size != 2_048
        || config.max_position_embeddings != 1_048_576
        || config.vocab_size != 129_280
        || config.sliding_window != 128
        || config.hc_multiplier != 4
        || config.q_lora_rank != 1_024
        || config.o_lora_rank != 1_024
        || config.o_groups != 8
        || ratios.as_slice() != REVIEWED_COMPRESSION_RATIOS
        || config.dspark_block_size != 5
        || config.dspark_noise_token_id != 128_799
        || config.dspark_target_layer_ids != [40, 41, 42]
        || config.dspark_markov_rank != 256
        || config.next_token_prediction_layers != 1
        || config.rope.factor != 16
        || config.rope.original_max_position_embeddings != 65_536
        || config.rope.beta_fast != 32
        || config.rope.beta_slow != 1
        || config.rope.theta != 10_000
        || config.rope.compressed_theta != 160_000
        || !config.quantization.activation_dynamic
        || config.quantization.value_format != "E4M3"
        || config.quantization.scale_format != "UE8M0"
        || config.quantization.block_shape != [128, 128]
    {
        return Err(invalid(
            "typed config differs from the reviewed exact contract",
        ));
    }
    Ok(())
}

/// Build the target-only metadata portion of the foundation catalog.
///
/// The main 43-row compression schedule is emitted. The three DSpark rows are
/// intentionally represented by separate plan fields rather than being
/// appended to target metadata.
pub fn deepseek_v4_gguf_foundation_metadata(
    config: &DeepSeekV4Config,
) -> Result<BTreeMap<String, GgufValue>, DeepSeekV4GgufFoundationError> {
    validate_reviewed_config(config)?;
    let main_ratios = config.compression[..DEEPSEEK_V4_MAIN_LAYER_COUNT as usize]
        .iter()
        .map(|compression| match compression {
            DeepSeekV4Compression::Uncompressed => 0,
            DeepSeekV4Compression::Csa4To1 => 4,
            DeepSeekV4Compression::Hca128To1 => 128,
        })
        .collect::<Vec<_>>();
    let revision_url =
        format!("https://huggingface.co/{DEEPSEEK_V4_REPOSITORY}/tree/{DEEPSEEK_V4_REVISION}");
    Ok(BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String("deepseek4".to_owned()),
        ),
        ("general.alignment".to_owned(), GgufValue::U32(32)),
        (
            "general.type".to_owned(),
            GgufValue::String("model".to_owned()),
        ),
        (
            "general.name".to_owned(),
            GgufValue::String(format!("{DEEPSEEK_V4_REPOSITORY}@{DEEPSEEK_V4_REVISION}")),
        ),
        (
            "general.license".to_owned(),
            GgufValue::String(DEEPSEEK_V4_LICENSE.to_owned()),
        ),
        (
            "general.source.url".to_owned(),
            GgufValue::String(revision_url),
        ),
        (
            "general.source.huggingface.repository".to_owned(),
            GgufValue::String(DEEPSEEK_V4_REPOSITORY.to_owned()),
        ),
        ("deepseek4.vocab_size".to_owned(), GgufValue::U32(129_280)),
        (
            "deepseek4.context_length".to_owned(),
            GgufValue::U32(1_048_576),
        ),
        (
            "deepseek4.embedding_length".to_owned(),
            GgufValue::U32(4_096),
        ),
        (
            "deepseek4.embedding_length_out".to_owned(),
            GgufValue::U32(16_384),
        ),
        ("deepseek4.block_count".to_owned(), GgufValue::U32(43)),
        (
            "deepseek4.leading_dense_block_count".to_owned(),
            GgufValue::U32(0),
        ),
        (
            "deepseek4.hidden_activation".to_owned(),
            GgufValue::String("silu".to_owned()),
        ),
        (
            "deepseek4.attention.head_count".to_owned(),
            GgufValue::U32(64),
        ),
        (
            "deepseek4.attention.head_count_kv".to_owned(),
            GgufValue::U32(1),
        ),
        (
            "deepseek4.attention.key_length".to_owned(),
            GgufValue::U32(512),
        ),
        (
            "deepseek4.attention.value_length".to_owned(),
            GgufValue::U32(512),
        ),
        (
            "deepseek4.attention.layer_norm_rms_epsilon".to_owned(),
            GgufValue::F32(1.0e-6),
        ),
        (
            "deepseek4.attention.q_lora_rank".to_owned(),
            GgufValue::U32(1_024),
        ),
        (
            "deepseek4.attention.sliding_window".to_owned(),
            GgufValue::U32(128),
        ),
        (
            "deepseek4.attention.output_group_count".to_owned(),
            GgufValue::U32(8),
        ),
        (
            "deepseek4.attention.output_lora_rank".to_owned(),
            GgufValue::U32(1_024),
        ),
        (
            "deepseek4.attention.compress_rope_freq_base".to_owned(),
            GgufValue::F32(160_000.0),
        ),
        (
            "deepseek4.attention.indexer.head_count".to_owned(),
            GgufValue::U32(64),
        ),
        (
            "deepseek4.attention.indexer.key_length".to_owned(),
            GgufValue::U32(128),
        ),
        (
            "deepseek4.attention.indexer.top_k".to_owned(),
            GgufValue::U32(512),
        ),
        (
            "deepseek4.attention.compress_ratios".to_owned(),
            GgufValue::Array(GgufArray::U32(main_ratios)),
        ),
        (
            "deepseek4.expert_feed_forward_length".to_owned(),
            GgufValue::U32(2_048),
        ),
        ("deepseek4.expert_count".to_owned(), GgufValue::U32(256)),
        ("deepseek4.expert_used_count".to_owned(), GgufValue::U32(6)),
        (
            "deepseek4.expert_shared_count".to_owned(),
            GgufValue::U32(1),
        ),
        (
            "deepseek4.expert_weights_scale".to_owned(),
            GgufValue::F32(1.5),
        ),
        (
            "deepseek4.expert_weights_norm".to_owned(),
            GgufValue::Bool(true),
        ),
        ("deepseek4.expert_gating_func".to_owned(), GgufValue::U32(4)),
        (
            "deepseek4.swiglu_clamp_exp".to_owned(),
            GgufValue::Array(GgufArray::F32(vec![10.0; 43])),
        ),
        (
            "deepseek4.swiglu_clamp_shexp".to_owned(),
            GgufValue::Array(GgufArray::F32(vec![10.0; 43])),
        ),
        ("deepseek4.hash_layer_count".to_owned(), GgufValue::U32(3)),
        (
            "deepseek4.hyper_connection.count".to_owned(),
            GgufValue::U32(4),
        ),
        (
            "deepseek4.hyper_connection.sinkhorn_iterations".to_owned(),
            GgufValue::U32(20),
        ),
        (
            "deepseek4.hyper_connection.epsilon".to_owned(),
            GgufValue::F32(1.0e-6),
        ),
        (
            "deepseek4.rope.dimension_count".to_owned(),
            GgufValue::U32(64),
        ),
        (
            "deepseek4.rope.freq_base".to_owned(),
            GgufValue::F32(10_000.0),
        ),
        (
            "deepseek4.rope.scaling.type".to_owned(),
            GgufValue::String("yarn".to_owned()),
        ),
        (
            "deepseek4.rope.scaling.factor".to_owned(),
            GgufValue::F32(16.0),
        ),
        (
            "deepseek4.rope.scaling.original_context_length".to_owned(),
            GgufValue::U32(65_536),
        ),
        (
            "deepseek4.rope.scaling.yarn_beta_fast".to_owned(),
            GgufValue::F32(32.0),
        ),
        (
            "deepseek4.rope.scaling.yarn_beta_slow".to_owned(),
            GgufValue::F32(1.0),
        ),
    ]))
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StackKey {
    artifact_role: DeepSeekV4ArtifactRole,
    projection: DeepSeekV4ExpertProjection,
}

#[derive(Clone, Debug, Default)]
struct ExpertPairBuilder {
    value_source: Option<String>,
    scale_source: Option<String>,
}

fn row_serialization(
    row: &DeepSeekV4GgufCatalogRow,
) -> Result<String, DeepSeekV4GgufFoundationError> {
    let artifact_role = row.artifact_role.canonical();
    let tensor_role = row.tensor_role.canonical();
    let fields = [
        row.source_name.as_str(),
        row.source_shard.as_str(),
        artifact_role.as_str(),
        tensor_role.as_str(),
        row.output_name.as_deref().unwrap_or("-"),
        row.plane.canonical(),
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

fn direct_pair_key(row: &DeepSeekV4GgufCatalogRow) -> String {
    format!(
        "{}\t{}\t{}",
        row.artifact_role.canonical(),
        row.tensor_role.canonical(),
        match row.plane {
            DeepSeekV4TensorPlane::Fp8E4m3Value => "pair",
            DeepSeekV4TensorPlane::Ue8m0BlockScale => "pair",
            _ => "not-pair",
        }
    )
}

/// Build the write-disabled foundation catalog from already validated typed
/// config and index values.
pub fn build_deepseek_v4_gguf_foundation_catalog(
    config: &DeepSeekV4Config,
    index: &DeepSeekV4Index,
) -> Result<DeepSeekV4GgufFoundationCatalogPlan, DeepSeekV4GgufFoundationError> {
    validate_reviewed_config(config)?;
    if index.total_size() != DEEPSEEK_V4_TENSOR_PAYLOAD_BYTES
        || index.tensor_count() != DEEPSEEK_V4_TENSOR_COUNT
        || index.catalog_sha256() != DEEPSEEK_V4_CATALOG_SHA256
    {
        return Err(invalid(
            "typed index identity differs from the reviewed exact catalog",
        ));
    }
    let summary = index.summary();
    if summary.next_token_root != REVIEWED_SOURCE_ROOT_COUNT
        || summary.hash_routed_layers != 4_706
        || summary.next_token_layers != 58_179
        || summary.dspark_target_layers != 4_721
        || summary.dspark_stages != DEEPSEEK_V4_GGUF_SOURCE_DSPARK_TENSOR_COUNT
    {
        return Err(invalid("typed index classification differs"));
    }

    let metadata = deepseek_v4_gguf_foundation_metadata(config)?;
    let mut rows = Vec::with_capacity(index.tensor_count());
    let mut digest = Sha256::new();
    let mut typed_keys = BTreeSet::new();
    let mut direct_outputs = BTreeSet::new();
    let mut direct_fp8_pairs = BTreeMap::<String, u8>::new();
    let mut stack_pairs = BTreeMap::<StackKey, BTreeMap<u16, ExpertPairBuilder>>::new();
    let mut source_target = 0_usize;
    let mut source_dspark = 0_usize;
    let mut direct = 0_usize;
    let mut routed = 0_usize;

    for (source_name, shard) in index.tensors() {
        let mapping = map_deepseek_v4_source_tensor(source_name)?;
        let row = DeepSeekV4GgufCatalogRow {
            source_name: mapping.source_name,
            source_shard: shard.to_owned(),
            artifact_role: mapping.artifact_role,
            tensor_role: mapping.tensor_role,
            output_name: mapping.output_name,
            plane: mapping.plane,
        };
        if row.artifact_role.is_dspark() {
            source_dspark += 1;
            if row.output_name.is_some() {
                return Err(invalid(format!(
                    "DSpark output name was frozen unexpectedly: {}",
                    row.source_name
                )));
            }
        } else {
            source_target += 1;
        }

        let typed_key = format!(
            "{}\t{}\t{}",
            row.artifact_role.canonical(),
            row.tensor_role.canonical(),
            row.plane.canonical()
        );
        if !typed_keys.insert(typed_key) {
            return Err(invalid(format!(
                "typed mapping collision: {}",
                row.source_name
            )));
        }

        if row.plane.is_routed() {
            routed += 1;
            let (expert, projection) = match row.tensor_role {
                DeepSeekV4TensorRole::FeedForward(
                    DeepSeekV4FeedForwardTensorRole::RoutedExpert { expert, projection },
                ) => (expert, projection),
                _ => return Err(invalid("routed plane does not have a routed expert role")),
            };
            let pair = stack_pairs
                .entry(StackKey {
                    artifact_role: row.artifact_role,
                    projection,
                })
                .or_default()
                .entry(expert)
                .or_default();
            let slot = if row.plane == DeepSeekV4TensorPlane::RoutedMxfp4Value {
                &mut pair.value_source
            } else {
                &mut pair.scale_source
            };
            if slot.replace(row.source_name.clone()).is_some() {
                return Err(invalid(format!(
                    "duplicate routed expert plane: {}",
                    row.source_name
                )));
            }
        } else {
            direct += 1;
            if let Some(output) = row.output_name.as_deref()
                && !direct_outputs.insert(output.to_owned())
            {
                return Err(invalid(format!("direct target output collision: {output}")));
            }
            if row.plane.is_direct_fp8() {
                let mask = if row.plane == DeepSeekV4TensorPlane::Fp8E4m3Value {
                    1
                } else {
                    2
                };
                let entry = direct_fp8_pairs.entry(direct_pair_key(&row)).or_default();
                if *entry & mask != 0 {
                    return Err(invalid(format!(
                        "duplicate direct FP8 plane: {}",
                        row.source_name
                    )));
                }
                *entry |= mask;
            }
        }
        digest.update(row_serialization(&row)?.as_bytes());
        rows.push(row);
    }

    if direct_fp8_pairs.values().any(|mask| *mask != 3) {
        return Err(invalid("a direct FP8 value/UE8M0 scale pair is incomplete"));
    }

    let mut stacks = Vec::with_capacity(stack_pairs.len());
    let mut stack_outputs = BTreeSet::new();
    let mut main_stacks = 0_usize;
    let mut dspark_stacks = 0_usize;
    for (key, by_expert) in stack_pairs {
        if by_expert.len() != usize::from(REVIEWED_EXPERT_COUNT) {
            return Err(invalid(format!(
                "routed stack does not contain 256 experts: {:?}",
                key
            )));
        }
        let mut experts = Vec::with_capacity(usize::from(REVIEWED_EXPERT_COUNT));
        for expert in 0..REVIEWED_EXPERT_COUNT {
            let pair = by_expert
                .get(&expert)
                .ok_or_else(|| invalid(format!("routed stack is missing expert {expert}")))?;
            experts.push(DeepSeekV4RoutedExpertPair {
                expert,
                value_source: pair
                    .value_source
                    .clone()
                    .ok_or_else(|| invalid(format!("routed expert {expert} is missing value")))?,
                scale_source: pair
                    .scale_source
                    .clone()
                    .ok_or_else(|| invalid(format!("routed expert {expert} is missing scale")))?,
            });
        }
        let output_name = match key.artifact_role {
            DeepSeekV4ArtifactRole::TargetMain { layer } => {
                main_stacks += 1;
                let output = format!("blk.{layer}.ffn_{}_exps.weight", key.projection.canonical());
                if direct_outputs.contains(&output) || !stack_outputs.insert(output.clone()) {
                    return Err(invalid(format!(
                        "stacked target output collision: {output}"
                    )));
                }
                Some(output)
            }
            DeepSeekV4ArtifactRole::Dspark { .. } => {
                dspark_stacks += 1;
                None
            }
            DeepSeekV4ArtifactRole::TargetRoot => {
                return Err(invalid("root routed expert stack is impossible"));
            }
        };
        stacks.push(DeepSeekV4ExpertStackPlan {
            artifact_role: key.artifact_role,
            projection: key.projection,
            output_name,
            experts,
        });
    }

    let mapping_sha256 = format!("{:x}", digest.finalize());
    if rows.len() != DEEPSEEK_V4_TENSOR_COUNT
        || source_target != DEEPSEEK_V4_GGUF_SOURCE_TARGET_TENSOR_COUNT
        || source_dspark != DEEPSEEK_V4_GGUF_SOURCE_DSPARK_TENSOR_COUNT
        || direct != DEEPSEEK_V4_GGUF_DIRECT_TENSOR_COUNT
        || routed != DEEPSEEK_V4_GGUF_ROUTED_EXPERT_SOURCE_TENSOR_COUNT
        || main_stacks != REVIEWED_MAIN_STACK_COUNT
        || dspark_stacks != REVIEWED_DSPARK_STACK_COUNT
        || stacks.len() != DEEPSEEK_V4_GGUF_STACKED_EXPERT_OUTPUT_COUNT
        || routed
            != (DEEPSEEK_V4_MAIN_LAYER_COUNT as usize
                + DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT as usize)
                * usize::from(REVIEWED_EXPERT_COUNT)
                * REVIEWED_PROJECTION_COUNT
                * 2
        || source_target - REVIEWED_SOURCE_ROOT_COUNT != REVIEWED_SOURCE_MAIN_COUNT
        || source_target - REVIEWED_MAIN_ROUTED_SOURCE_COUNT + main_stacks
            != DEEPSEEK_V4_GGUF_MAIN_PHYSICAL_TENSOR_COUNT
        || source_dspark - REVIEWED_DSPARK_ROUTED_SOURCE_COUNT + dspark_stacks
            != DEEPSEEK_V4_GGUF_DSPARK_PHYSICAL_TENSOR_COUNT
        || direct + stacks.len() != DEEPSEEK_V4_GGUF_COMBINED_PHYSICAL_TENSOR_COUNT
    {
        return Err(invalid("catalog or physical tensor accounting differs"));
    }

    Ok(DeepSeekV4GgufFoundationCatalogPlan {
        target_metadata: metadata,
        source_rows: rows,
        expert_stacks: stacks,
        source_tensor_count: DEEPSEEK_V4_TENSOR_COUNT,
        source_target_tensor_count: source_target,
        source_dspark_tensor_count: source_dspark,
        direct_tensor_count: direct,
        routed_expert_source_tensor_count: routed,
        stacked_expert_output_count: DEEPSEEK_V4_GGUF_STACKED_EXPERT_OUTPUT_COUNT,
        main_physical_tensor_count: DEEPSEEK_V4_GGUF_MAIN_PHYSICAL_TENSOR_COUNT,
        dspark_physical_tensor_count: DEEPSEEK_V4_GGUF_DSPARK_PHYSICAL_TENSOR_COUNT,
        combined_physical_tensor_count: DEEPSEEK_V4_GGUF_COMBINED_PHYSICAL_TENSOR_COUNT,
        config_nextn_predict_layers: config.next_token_prediction_layers,
        checkpoint_dspark_stages: DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT,
        mapping_sha256,
        production_loadable: false,
        payload_headers_verified: false,
        payload_bytes_verified: false,
        writable_gguf_plan: false,
        output_payload_bytes: None,
        pass_scope: DEEPSEEK_V4_GGUF_PASS_SCOPE,
    })
}

/// Validate the exact official metadata bytes and return the write-disabled
/// foundation catalog. No safetensors shard is opened by this API.
pub fn validate_deepseek_v4_gguf_foundation_catalog(
    config_bytes: &[u8],
    index_bytes: &[u8],
) -> Result<DeepSeekV4GgufFoundationCatalogPlan, DeepSeekV4GgufFoundationError> {
    let config = validate_deepseek_v4_config(config_bytes)
        .map_err(|error| invalid(format!("config validation failed: {error}")))?;
    let index = validate_deepseek_v4_index(index_bytes)
        .map_err(|error| invalid(format!("index validation failed: {error}")))?;
    build_deepseek_v4_gguf_foundation_catalog(&config, &index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_config() -> DeepSeekV4Config {
        DeepSeekV4Config {
            hidden_size: 4_096,
            layer_count: 43,
            hash_layer_count: 3,
            attention_heads: 64,
            kv_heads: 1,
            head_dim: 512,
            index_heads: 64,
            index_head_dim: 128,
            index_top_k: 512,
            expert_count: 256,
            selected_expert_count: 6,
            shared_expert_count: 1,
            expert_intermediate_size: 2_048,
            max_position_embeddings: 1_048_576,
            vocab_size: 129_280,
            sliding_window: 128,
            hc_multiplier: 4,
            q_lora_rank: 1_024,
            o_lora_rank: 1_024,
            o_groups: 8,
            compression: REVIEWED_COMPRESSION_RATIOS
                .iter()
                .map(|ratio| match ratio {
                    0 => DeepSeekV4Compression::Uncompressed,
                    4 => DeepSeekV4Compression::Csa4To1,
                    128 => DeepSeekV4Compression::Hca128To1,
                    _ => unreachable!(),
                })
                .collect(),
            dspark_block_size: 5,
            dspark_noise_token_id: 128_799,
            dspark_target_layer_ids: [40, 41, 42],
            dspark_markov_rank: 256,
            next_token_prediction_layers: 1,
            rope: crate::deepseek_v4::DeepSeekV4YarnRope {
                factor: 16,
                original_max_position_embeddings: 65_536,
                beta_fast: 32,
                beta_slow: 1,
                theta: 10_000,
                compressed_theta: 160_000,
            },
            quantization: crate::deepseek_v4::DeepSeekV4Quantization {
                activation_dynamic: true,
                value_format: "E4M3",
                scale_format: "UE8M0",
                block_shape: [128, 128],
            },
        }
    }

    #[test]
    fn root_and_main_boundary_mappings_are_typed() {
        let root = map_deepseek_v4_source_tensor("embed.weight").unwrap();
        assert_eq!(root.artifact_role, DeepSeekV4ArtifactRole::TargetRoot);
        assert_eq!(root.output_name.as_deref(), Some("token_embd.weight"));

        let csa = map_deepseek_v4_source_tensor("layers.2.attn.indexer.wq_b.scale").unwrap();
        assert_eq!(
            csa.artifact_role,
            DeepSeekV4ArtifactRole::TargetMain { layer: 2 }
        );
        assert_eq!(csa.plane, DeepSeekV4TensorPlane::Ue8m0BlockScale);
        assert_eq!(
            csa.output_name.as_deref(),
            Some("blk.2.indexer.attn_q_b.scale")
        );

        let hca = map_deepseek_v4_source_tensor("layers.3.attn.compressor.wkv.weight").unwrap();
        assert_eq!(
            hca.tensor_role,
            DeepSeekV4TensorRole::Attention(DeepSeekV4AttentionTensorRole::CompressorKeyValue)
        );
        assert!(
            map_deepseek_v4_source_tensor("layers.3.attn.indexer.weights_proj.weight").is_err()
        );
        assert!(map_deepseek_v4_source_tensor("layers.43.attn.wkv.weight").is_err());
    }

    #[test]
    fn hash_bias_and_expert_numeric_boundaries_are_exact() {
        assert!(map_deepseek_v4_source_tensor("layers.2.ffn.gate.tid2eid").is_ok());
        assert!(map_deepseek_v4_source_tensor("layers.3.ffn.gate.tid2eid").is_err());
        assert!(map_deepseek_v4_source_tensor("layers.2.ffn.gate.bias").is_err());
        assert!(map_deepseek_v4_source_tensor("layers.3.ffn.gate.bias").is_ok());
        for expert in [2_u16, 9, 10, 255] {
            let mapping =
                map_deepseek_v4_source_tensor(&format!("layers.42.ffn.experts.{expert}.w3.scale"))
                    .unwrap();
            assert_eq!(
                mapping.tensor_role,
                DeepSeekV4TensorRole::FeedForward(DeepSeekV4FeedForwardTensorRole::RoutedExpert {
                    expert,
                    projection: DeepSeekV4ExpertProjection::Up,
                })
            );
        }
        assert!(map_deepseek_v4_source_tensor("layers.0.ffn.experts.0255.w1.weight").is_err());
        assert!(map_deepseek_v4_source_tensor("layers.0.ffn.experts.256.w1.weight").is_err());
    }

    #[test]
    fn dspark_stage_boundaries_remain_typed_without_output_names() {
        for source in [
            "mtp.0.main_proj.weight",
            "mtp.1.attn.wq_a.scale",
            "mtp.2.confidence_head.proj.weight",
        ] {
            let mapping = map_deepseek_v4_source_tensor(source).unwrap();
            assert!(mapping.artifact_role.is_dspark());
            assert_eq!(mapping.output_name, None);
        }
        assert!(map_deepseek_v4_source_tensor("mtp.1.main_proj.weight").is_err());
        assert!(map_deepseek_v4_source_tensor("mtp.3.attn.wq_a.weight").is_err());
    }

    #[test]
    fn target_metadata_types_and_foundation_flags_are_fixed() {
        let metadata = deepseek_v4_gguf_foundation_metadata(&fixture_config()).unwrap();
        assert_eq!(
            metadata["general.architecture"],
            GgufValue::String("deepseek4".to_owned())
        );
        assert_eq!(metadata["general.alignment"], GgufValue::U32(32));
        assert_eq!(metadata["deepseek4.expert_gating_func"], GgufValue::U32(4));
        assert_eq!(
            metadata["deepseek4.hyper_connection.epsilon"],
            GgufValue::F32(1.0e-6)
        );
        assert!(!metadata.contains_key("general.file_type"));
        assert!(!metadata.contains_key("general.quantization_version"));
        let ratios = match &metadata["deepseek4.attention.compress_ratios"] {
            GgufValue::Array(GgufArray::U32(values)) => values,
            _ => panic!("compression metadata type differs"),
        };
        assert_eq!(ratios.len(), 43);
        assert_eq!(&ratios[..4], &[0, 0, 4, 128]);

        let flags = (
            false,
            false,
            false,
            false,
            None::<u64>,
            DEEPSEEK_V4_GGUF_PASS_SCOPE,
        );
        assert_eq!(
            flags,
            (
                false,
                false,
                false,
                false,
                None,
                "exact-metadata-and-index-catalog-only-no-payload-no-write"
            )
        );
        assert_eq!(fixture_config().next_token_prediction_layers, 1);
        assert_eq!(DEEPSEEK_V4_DSPARK_CHECKPOINT_STAGE_COUNT, 3);
    }

    #[test]
    #[ignore = "requires exact official config.json and model.safetensors.index.json"]
    fn official_metadata_catalog_counts_and_mapping_digest_are_exact() {
        let root = std::env::var_os("SLLM_DEEPSEEK_V4_METADATA_DIR")
            .map(std::path::PathBuf::from)
            .expect("set SLLM_DEEPSEEK_V4_METADATA_DIR");
        let config = std::fs::read(root.join("config.json")).expect("read official config.json");
        let index =
            std::fs::read(root.join("model.safetensors.index.json")).expect("read official index");
        assert_eq!(
            format!("{:x}", Sha256::digest(&index)),
            crate::deepseek_v4::DEEPSEEK_V4_INDEX_SHA256
        );
        let plan = validate_deepseek_v4_gguf_foundation_catalog(&config, &index).unwrap();
        assert_eq!(plan.source_tensor_count, 72_317);
        assert_eq!(plan.source_target_tensor_count, 67_612);
        assert_eq!(plan.source_dspark_tensor_count, 4_705);
        assert_eq!(plan.direct_tensor_count, 1_661);
        assert_eq!(plan.routed_expert_source_tensor_count, 70_656);
        assert_eq!(plan.stacked_expert_output_count, 138);
        assert_eq!(plan.main_physical_tensor_count, 1_693);
        assert_eq!(plan.dspark_physical_tensor_count, 106);
        assert_eq!(plan.combined_physical_tensor_count, 1_799);
        assert_eq!(plan.config_nextn_predict_layers, 1);
        assert_eq!(plan.checkpoint_dspark_stages, 3);
        assert_eq!(plan.mapping_sha256, DEEPSEEK_V4_GGUF_MAPPING_SHA256);
        assert!(plan.expert_stacks.iter().all(|stack| {
            stack.experts.len() == 256
                && stack
                    .experts
                    .iter()
                    .enumerate()
                    .all(|(index, pair)| usize::from(pair.expert) == index)
        }));
        assert!(!plan.production_loadable);
        assert!(!plan.payload_headers_verified);
        assert!(!plan.payload_bytes_verified);
        assert!(!plan.writable_gguf_plan);
        assert_eq!(plan.output_payload_bytes, None);
    }
}
