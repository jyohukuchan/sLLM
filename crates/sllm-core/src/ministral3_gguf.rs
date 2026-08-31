//! Canonical Ministral 3 text-only GGUF catalog and official-artifact verifier.
//!
//! The dry run binds the exact reviewed safetensors config/index pair to the
//! fixed llama.cpp `b10453` text mapping. It describes, but does not execute,
//! the Q/K head permutations or BF16-to-F32 norm conversions. Consequently it
//! cannot construct a writable GGUF plan or claim an output-file hash.
//!
//! The separate official-GGUF verifier accepts an already parsed
//! [`VerifiedGguf`]. Its full-file LFS SHA-256 remains the responsibility of
//! the outer model lock; this module rechecks the exact header metadata and
//! 236-entry text tensor catalog and never describes the official file as an
//! sLLM-derived conversion output.

use crate::gguf::{GgufTensorType, GgufValue, VerifiedGguf};
use crate::ministral3::{
    MINISTRAL3_CONFIG_BYTES, MINISTRAL3_CONFIG_SHA256, MINISTRAL3_CONTEXT_LENGTH,
    MINISTRAL3_INDEX_TOTAL_PARAMETERS, MINISTRAL3_INDEX_TOTAL_SIZE, MINISTRAL3_LICENSE,
    MINISTRAL3_PHYSICAL_PARAMETERS, MINISTRAL3_REPOSITORY, MINISTRAL3_REVISION,
    MINISTRAL3_TENSOR_COUNT, MINISTRAL3_TEXT_FFN_SIZE, MINISTRAL3_TEXT_HIDDEN_SIZE,
    MINISTRAL3_TEXT_LAYER_COUNT, MINISTRAL3_VISION_LAYER_COUNT, MINISTRAL3_VOCAB_SIZE,
    Ministral3Config, validate_ministral3_config,
};
use crate::ministral3_headers::{Ministral3Index, validate_ministral3_index};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

pub const MINISTRAL3_GGUF_ARCHITECTURE: &str = "mistral3";
pub const MINISTRAL3_TEXT_MODEL_TYPE: &str = "ministral3";
pub const MINISTRAL3_GGUF_SOURCE_TEXT_TENSOR_COUNT: usize = 236;
pub const MINISTRAL3_GGUF_SOURCE_VISION_TENSOR_COUNT: usize = 218;
pub const MINISTRAL3_GGUF_SOURCE_PROJECTOR_TENSOR_COUNT: usize = 4;
pub const MINISTRAL3_GGUF_KNOWN_UNCONSUMED_TENSOR_COUNT: usize = 222;
pub const MINISTRAL3_GGUF_OUTPUT_CANDIDATE_COUNT: usize = 236;
pub const MINISTRAL3_GGUF_BF16_TENSOR_COUNT: usize = 183;
pub const MINISTRAL3_GGUF_F32_NORM_TENSOR_COUNT: usize = 53;
pub const MINISTRAL3_GGUF_QUERY_PERMUTATION_COUNT: usize = 26;
pub const MINISTRAL3_GGUF_KEY_PERMUTATION_COUNT: usize = 26;
pub const MINISTRAL3_GGUF_PASS_SCOPE: &str =
    "exact-config-index-text-mapping-only-no-source-payload-no-write";
pub const MINISTRAL3_GGUF_MAPPING_SERIALIZATION: &str = "utf8-tsv-v1:source,shard,artifact-plane,typed-role,output-or-dash,target-type-or-dash,required-transform;lf-rows";

/// SHA-256 over [`MINISTRAL3_GGUF_MAPPING_SERIALIZATION`] for the exact
/// 458-row source index. It binds catalog decisions, not tensor payload bytes.
pub const MINISTRAL3_GGUF_MAPPING_SHA256: &str =
    "b4c4061c4f9932c51fef2a8b01d1ae96a99b4c701ae1ece4869b852c46333da9";

pub const MINISTRAL3_OFFICIAL_GGUF_REPOSITORY: &str = "mistralai/Ministral-3-3B-Instruct-2512-GGUF";
pub const MINISTRAL3_OFFICIAL_GGUF_REVISION: &str = "eb599d408350ea2bb60452cb86be7c7b2fc28227";
pub const MINISTRAL3_OFFICIAL_GGUF_FILE_NAME: &str = "Ministral-3-3B-Instruct-2512-BF16.gguf";
pub const MINISTRAL3_OFFICIAL_GGUF_FILE_BYTES: u64 = 6_866_745_504;
pub const MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256: &str =
    "17ef932bea952e007f9dad63151da5699132ec513d1033d618df7382e24aa3ee";
/// Digest of the raw metadata byte range parsed by [`VerifiedGguf`].
pub const MINISTRAL3_OFFICIAL_GGUF_METADATA_SHA256: &str =
    "sha256:7e16085724a92d35c80e29982ff663860fc95b6a054fcbf57b0f28f881cd5f0e";
/// Digest of the raw tensor-catalog byte range parsed by [`VerifiedGguf`].
pub const MINISTRAL3_OFFICIAL_GGUF_TENSOR_CATALOG_SHA256: &str =
    "sha256:f40ed89f4535224c30c8a0c03a7a167435adcb06e909af07f33fd66f25dee95a";

const TEXT_ATTENTION_HEADS: u32 = 32;
const TEXT_KV_HEADS: u32 = 8;
const TEXT_HEAD_DIM: u32 = 128;
const YARN_ORIGINAL_CONTEXT: u32 = 16_384;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3GgufError {
    Invalid(String),
}

impl fmt::Display for Ministral3GgufError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(formatter, "invalid Ministral 3 GGUF catalog: {message}")
            }
        }
    }
}

impl std::error::Error for Ministral3GgufError {}

fn invalid(message: impl Into<String>) -> Ministral3GgufError {
    Ministral3GgufError::Invalid(message.into())
}

fn ensure(condition: bool, message: impl Into<String>) -> Result<(), Ministral3GgufError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn same_f64(actual: f64, expected: f64) -> bool {
    actual.to_bits() == expected.to_bits()
}

fn checked_add(counter: &mut usize, value: usize, label: &str) -> Result<(), Ministral3GgufError> {
    *counter = counter
        .checked_add(value)
        .ok_or_else(|| invalid(format!("{label} count overflows")))?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3ArtifactPlane {
    Text,
    Vision,
    Projector,
}

impl Ministral3ArtifactPlane {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision-known-unconsumed",
            Self::Projector => "projector-known-unconsumed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3TextRootRole {
    TokenEmbedding,
    OutputNorm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3TextLayerRole {
    AttentionNorm,
    Query,
    Key,
    Value,
    AttentionOutput,
    FeedForwardNorm,
    FeedForwardGate,
    FeedForwardDown,
    FeedForwardUp,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3VisionRootRole {
    PatchConvolution,
    PreNorm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3VisionLayerRole {
    AttentionNorm,
    Query,
    Key,
    Value,
    AttentionOutput,
    FeedForwardNorm,
    FeedForwardGate,
    FeedForwardDown,
    FeedForwardUp,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3ProjectorRole {
    LinearOne,
    LinearTwo,
    Norm,
    PatchMerge,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3TensorRole {
    TextRoot(Ministral3TextRootRole),
    TextLayer {
        layer: u8,
        role: Ministral3TextLayerRole,
    },
    VisionRoot(Ministral3VisionRootRole),
    VisionLayer {
        layer: u8,
        role: Ministral3VisionLayerRole,
    },
    Projector(Ministral3ProjectorRole),
}

impl Ministral3TensorRole {
    fn canonical(self) -> String {
        match self {
            Self::TextRoot(role) => format!(
                "text-root:{}",
                match role {
                    Ministral3TextRootRole::TokenEmbedding => "token-embedding",
                    Ministral3TextRootRole::OutputNorm => "output-norm",
                }
            ),
            Self::TextLayer { layer, role } => format!(
                "text-layer:{layer}:{}",
                match role {
                    Ministral3TextLayerRole::AttentionNorm => "attention-norm",
                    Ministral3TextLayerRole::Query => "attention-query",
                    Ministral3TextLayerRole::Key => "attention-key",
                    Ministral3TextLayerRole::Value => "attention-value",
                    Ministral3TextLayerRole::AttentionOutput => "attention-output",
                    Ministral3TextLayerRole::FeedForwardNorm => "feed-forward-norm",
                    Ministral3TextLayerRole::FeedForwardGate => "feed-forward-gate",
                    Ministral3TextLayerRole::FeedForwardDown => "feed-forward-down",
                    Ministral3TextLayerRole::FeedForwardUp => "feed-forward-up",
                }
            ),
            Self::VisionRoot(role) => format!(
                "vision-root:{}",
                match role {
                    Ministral3VisionRootRole::PatchConvolution => "patch-convolution",
                    Ministral3VisionRootRole::PreNorm => "pre-norm",
                }
            ),
            Self::VisionLayer { layer, role } => format!(
                "vision-layer:{layer}:{}",
                match role {
                    Ministral3VisionLayerRole::AttentionNorm => "attention-norm",
                    Ministral3VisionLayerRole::Query => "attention-query",
                    Ministral3VisionLayerRole::Key => "attention-key",
                    Ministral3VisionLayerRole::Value => "attention-value",
                    Ministral3VisionLayerRole::AttentionOutput => "attention-output",
                    Ministral3VisionLayerRole::FeedForwardNorm => "feed-forward-norm",
                    Ministral3VisionLayerRole::FeedForwardGate => "feed-forward-gate",
                    Ministral3VisionLayerRole::FeedForwardDown => "feed-forward-down",
                    Ministral3VisionLayerRole::FeedForwardUp => "feed-forward-up",
                }
            ),
            Self::Projector(role) => format!(
                "projector:{}",
                match role {
                    Ministral3ProjectorRole::LinearOne => "linear-1",
                    Ministral3ProjectorRole::LinearTwo => "linear-2",
                    Ministral3ProjectorRole::Norm => "norm",
                    Ministral3ProjectorRole::PatchMerge => "patch-merge",
                }
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ministral3PayloadTransform {
    DirectBf16,
    QueryHeadPermutation { heads: u32, head_dim: u32 },
    KeyHeadPermutation { heads: u32, head_dim: u32 },
    Bf16ToF32,
    KnownUnconsumed,
}

impl Ministral3PayloadTransform {
    fn canonical(self) -> String {
        match self {
            Self::DirectBf16 => "direct-bf16".to_owned(),
            Self::QueryHeadPermutation { heads, head_dim } => {
                format!("query-head-permutation:{heads}x{head_dim}")
            }
            Self::KeyHeadPermutation { heads, head_dim } => {
                format!("key-head-permutation:{heads}x{head_dim}")
            }
            Self::Bf16ToF32 => "bf16-to-f32".to_owned(),
            Self::KnownUnconsumed => "known-unconsumed".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3SourceTensorMapping {
    pub source_name: String,
    pub artifact_plane: Ministral3ArtifactPlane,
    pub tensor_role: Ministral3TensorRole,
    pub output_name: Option<String>,
    /// GGML/GGUF dimension order, or `None` for intentionally unconsumed
    /// vision/projector tensors.
    pub output_dimensions: Option<Vec<u64>>,
    pub output_tensor_type: Option<GgufTensorType>,
    pub required_transform: Ministral3PayloadTransform,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3GgufCatalogRow {
    pub source_name: String,
    pub source_shard: String,
    pub artifact_plane: Ministral3ArtifactPlane,
    pub tensor_role: Ministral3TensorRole,
    pub output_name: Option<String>,
    pub output_dimensions: Option<Vec<u64>>,
    pub output_tensor_type: Option<GgufTensorType>,
    pub required_transform: Ministral3PayloadTransform,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ministral3GgufDryRunPlan {
    pub target_metadata: BTreeMap<String, GgufValue>,
    pub source_rows: Vec<Ministral3GgufCatalogRow>,
    pub source_tensor_count: usize,
    pub source_text_tensor_count: usize,
    pub source_vision_tensor_count: usize,
    pub source_projector_tensor_count: usize,
    pub known_unconsumed_tensor_count: usize,
    pub output_candidate_count: usize,
    pub output_bf16_tensor_count: usize,
    pub output_f32_tensor_count: usize,
    pub query_permutation_count: usize,
    pub key_permutation_count: usize,
    pub mapping_sha256: String,
    pub tied_output_omitted: bool,
    pub source_payload_bytes_verified: bool,
    pub payload_transforms_executed: bool,
    pub dtype_conversions_executed: bool,
    pub quantization_executed: bool,
    pub writable_gguf_plan: bool,
    pub output_payload_bytes: Option<u64>,
    pub output_file_sha256: Option<String>,
    pub pass_scope: &'static str,
}

/// Parse only the exact official source index fixed by the artifact contract.
pub fn validate_ministral3_gguf_source_index(
    bytes: &[u8],
) -> Result<Ministral3Index, Ministral3GgufError> {
    validate_ministral3_index(bytes).map_err(|error| invalid(error.to_string()))
}

fn canonical_index(
    value: &str,
    upper_exclusive: u32,
    label: &str,
) -> Result<u8, Ministral3GgufError> {
    ensure(
        !value.is_empty()
            && !(value.len() > 1 && value.starts_with('0'))
            && value.bytes().all(|byte| byte.is_ascii_digit()),
        format!("{label} is not canonical decimal: {value}"),
    )?;
    let parsed = value
        .parse::<u32>()
        .map_err(|_| invalid(format!("{label} is invalid: {value}")))?;
    ensure(
        parsed < upper_exclusive,
        format!("{label} is out of range: {parsed}"),
    )?;
    u8::try_from(parsed).map_err(|_| invalid(format!("{label} exceeds u8")))
}

fn text_mapping(
    source_name: &str,
    role: Ministral3TensorRole,
    output_name: String,
    dimensions: Vec<u64>,
    tensor_type: GgufTensorType,
    transform: Ministral3PayloadTransform,
) -> Ministral3SourceTensorMapping {
    Ministral3SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_plane: Ministral3ArtifactPlane::Text,
        tensor_role: role,
        output_name: Some(output_name),
        output_dimensions: Some(dimensions),
        output_tensor_type: Some(tensor_type),
        required_transform: transform,
    }
}

fn known_unconsumed_mapping(
    source_name: &str,
    plane: Ministral3ArtifactPlane,
    role: Ministral3TensorRole,
) -> Ministral3SourceTensorMapping {
    Ministral3SourceTensorMapping {
        source_name: source_name.to_owned(),
        artifact_plane: plane,
        tensor_role: role,
        output_name: None,
        output_dimensions: None,
        output_tensor_type: None,
        required_transform: Ministral3PayloadTransform::KnownUnconsumed,
    }
}

fn map_text_layer(source_name: &str) -> Result<Ministral3SourceTensorMapping, Ministral3GgufError> {
    const PREFIX: &str = "language_model.model.layers.";
    let rest = source_name
        .strip_prefix(PREFIX)
        .ok_or_else(|| invalid(format!("unknown text tensor: {source_name}")))?;
    let (layer_text, suffix) = rest
        .split_once('.')
        .ok_or_else(|| invalid(format!("text layer tensor has no suffix: {source_name}")))?;
    let layer = canonical_index(layer_text, MINISTRAL3_TEXT_LAYER_COUNT, "text layer")?;
    let hidden = u64::from(MINISTRAL3_TEXT_HIDDEN_SIZE);
    let ffn = u64::from(MINISTRAL3_TEXT_FFN_SIZE);
    let q_width = u64::from(TEXT_ATTENTION_HEADS)
        .checked_mul(u64::from(TEXT_HEAD_DIM))
        .ok_or_else(|| invalid("query width overflows"))?;
    let kv_width = u64::from(TEXT_KV_HEADS)
        .checked_mul(u64::from(TEXT_HEAD_DIM))
        .ok_or_else(|| invalid("KV width overflows"))?;
    let (role, output_suffix, dimensions, tensor_type, transform) = match suffix {
        "input_layernorm.weight" => (
            Ministral3TextLayerRole::AttentionNorm,
            "attn_norm",
            vec![hidden],
            GgufTensorType::F32,
            Ministral3PayloadTransform::Bf16ToF32,
        ),
        "self_attn.q_proj.weight" => (
            Ministral3TextLayerRole::Query,
            "attn_q",
            vec![hidden, q_width],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::QueryHeadPermutation {
                heads: TEXT_ATTENTION_HEADS,
                head_dim: TEXT_HEAD_DIM,
            },
        ),
        "self_attn.k_proj.weight" => (
            Ministral3TextLayerRole::Key,
            "attn_k",
            vec![hidden, kv_width],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::KeyHeadPermutation {
                heads: TEXT_KV_HEADS,
                head_dim: TEXT_HEAD_DIM,
            },
        ),
        "self_attn.v_proj.weight" => (
            Ministral3TextLayerRole::Value,
            "attn_v",
            vec![hidden, kv_width],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::DirectBf16,
        ),
        "self_attn.o_proj.weight" => (
            Ministral3TextLayerRole::AttentionOutput,
            "attn_output",
            vec![q_width, hidden],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::DirectBf16,
        ),
        "post_attention_layernorm.weight" => (
            Ministral3TextLayerRole::FeedForwardNorm,
            "ffn_norm",
            vec![hidden],
            GgufTensorType::F32,
            Ministral3PayloadTransform::Bf16ToF32,
        ),
        "mlp.gate_proj.weight" => (
            Ministral3TextLayerRole::FeedForwardGate,
            "ffn_gate",
            vec![hidden, ffn],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::DirectBf16,
        ),
        "mlp.down_proj.weight" => (
            Ministral3TextLayerRole::FeedForwardDown,
            "ffn_down",
            vec![ffn, hidden],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::DirectBf16,
        ),
        "mlp.up_proj.weight" => (
            Ministral3TextLayerRole::FeedForwardUp,
            "ffn_up",
            vec![hidden, ffn],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::DirectBf16,
        ),
        _ => return Err(invalid(format!("unknown text layer tensor: {source_name}"))),
    };
    Ok(text_mapping(
        source_name,
        Ministral3TensorRole::TextLayer { layer, role },
        format!("blk.{layer}.{output_suffix}.weight"),
        dimensions,
        tensor_type,
        transform,
    ))
}

fn map_vision_layer(
    source_name: &str,
) -> Result<Ministral3SourceTensorMapping, Ministral3GgufError> {
    const PREFIX: &str = "vision_tower.transformer.layers.";
    let rest = source_name
        .strip_prefix(PREFIX)
        .ok_or_else(|| invalid(format!("unknown vision tensor: {source_name}")))?;
    let (layer_text, suffix) = rest
        .split_once('.')
        .ok_or_else(|| invalid(format!("vision layer tensor has no suffix: {source_name}")))?;
    let layer = canonical_index(layer_text, MINISTRAL3_VISION_LAYER_COUNT, "vision layer")?;
    let role = match suffix {
        "attention_norm.weight" => Ministral3VisionLayerRole::AttentionNorm,
        "attention.q_proj.weight" => Ministral3VisionLayerRole::Query,
        "attention.k_proj.weight" => Ministral3VisionLayerRole::Key,
        "attention.v_proj.weight" => Ministral3VisionLayerRole::Value,
        "attention.o_proj.weight" => Ministral3VisionLayerRole::AttentionOutput,
        "ffn_norm.weight" => Ministral3VisionLayerRole::FeedForwardNorm,
        "feed_forward.gate_proj.weight" => Ministral3VisionLayerRole::FeedForwardGate,
        "feed_forward.down_proj.weight" => Ministral3VisionLayerRole::FeedForwardDown,
        "feed_forward.up_proj.weight" => Ministral3VisionLayerRole::FeedForwardUp,
        _ => {
            return Err(invalid(format!(
                "unknown vision layer tensor: {source_name}"
            )));
        }
    };
    Ok(known_unconsumed_mapping(
        source_name,
        Ministral3ArtifactPlane::Vision,
        Ministral3TensorRole::VisionLayer { layer, role },
    ))
}

/// Parse one official source tensor name into the text conversion catalog or
/// an explicitly known-unconsumed vision/projector role.
pub fn map_ministral3_source_tensor(
    source_name: &str,
) -> Result<Ministral3SourceTensorMapping, Ministral3GgufError> {
    let hidden = u64::from(MINISTRAL3_TEXT_HIDDEN_SIZE);
    match source_name {
        "language_model.model.embed_tokens.weight" => Ok(text_mapping(
            source_name,
            Ministral3TensorRole::TextRoot(Ministral3TextRootRole::TokenEmbedding),
            "token_embd.weight".to_owned(),
            vec![hidden, u64::from(MINISTRAL3_VOCAB_SIZE)],
            GgufTensorType::Bf16,
            Ministral3PayloadTransform::DirectBf16,
        )),
        "language_model.model.norm.weight" => Ok(text_mapping(
            source_name,
            Ministral3TensorRole::TextRoot(Ministral3TextRootRole::OutputNorm),
            "output_norm.weight".to_owned(),
            vec![hidden],
            GgufTensorType::F32,
            Ministral3PayloadTransform::Bf16ToF32,
        )),
        "vision_tower.patch_conv.weight" => Ok(known_unconsumed_mapping(
            source_name,
            Ministral3ArtifactPlane::Vision,
            Ministral3TensorRole::VisionRoot(Ministral3VisionRootRole::PatchConvolution),
        )),
        "vision_tower.ln_pre.weight" => Ok(known_unconsumed_mapping(
            source_name,
            Ministral3ArtifactPlane::Vision,
            Ministral3TensorRole::VisionRoot(Ministral3VisionRootRole::PreNorm),
        )),
        "multi_modal_projector.linear_1.weight" => Ok(known_unconsumed_mapping(
            source_name,
            Ministral3ArtifactPlane::Projector,
            Ministral3TensorRole::Projector(Ministral3ProjectorRole::LinearOne),
        )),
        "multi_modal_projector.linear_2.weight" => Ok(known_unconsumed_mapping(
            source_name,
            Ministral3ArtifactPlane::Projector,
            Ministral3TensorRole::Projector(Ministral3ProjectorRole::LinearTwo),
        )),
        "multi_modal_projector.norm.weight" => Ok(known_unconsumed_mapping(
            source_name,
            Ministral3ArtifactPlane::Projector,
            Ministral3TensorRole::Projector(Ministral3ProjectorRole::Norm),
        )),
        "multi_modal_projector.patch_merger.merging_layer.weight" => Ok(known_unconsumed_mapping(
            source_name,
            Ministral3ArtifactPlane::Projector,
            Ministral3TensorRole::Projector(Ministral3ProjectorRole::PatchMerge),
        )),
        _ if source_name.starts_with("language_model.model.layers.") => map_text_layer(source_name),
        _ if source_name.starts_with("vision_tower.transformer.layers.") => {
            map_vision_layer(source_name)
        }
        _ => Err(invalid(format!("unknown source tensor: {source_name}"))),
    }
}

/// Require all three architecture spellings to stay distinct and exact.
pub fn validate_ministral3_architecture_spelling(
    outer_model_type: &str,
    text_model_type: &str,
    gguf_architecture: &str,
) -> Result<(), Ministral3GgufError> {
    ensure(
        outer_model_type == "mistral3",
        "outer model_type must be mistral3",
    )?;
    ensure(
        text_model_type == MINISTRAL3_TEXT_MODEL_TYPE,
        "text model_type must be ministral3",
    )?;
    ensure(
        gguf_architecture == MINISTRAL3_GGUF_ARCHITECTURE,
        "GGUF architecture must be mistral3",
    )
}

fn validate_reviewed_config(config: &Ministral3Config) -> Result<(), Ministral3GgufError> {
    let text = config.text;
    let rope = text.rope;
    let vision = config.vision;
    ensure(
        text.hidden_size == MINISTRAL3_TEXT_HIDDEN_SIZE
            && text.intermediate_size == MINISTRAL3_TEXT_FFN_SIZE
            && text.num_hidden_layers == MINISTRAL3_TEXT_LAYER_COUNT
            && text.num_attention_heads == TEXT_ATTENTION_HEADS
            && text.num_key_value_heads == TEXT_KV_HEADS
            && text.head_dim == TEXT_HEAD_DIM
            && text.vocab_size == MINISTRAL3_VOCAB_SIZE
            && text.max_position_embeddings == MINISTRAL3_CONTEXT_LENGTH
            && same_f64(text.rms_norm_eps, 1.0e-5)
            && text.tie_word_embeddings
            && text.use_cache,
        "text config differs from reviewed contract",
    )?;
    ensure(
        same_f64(rope.beta_fast, 32.0)
            && same_f64(rope.beta_slow, 1.0)
            && same_f64(rope.factor, 16.0)
            && same_f64(rope.llama_4_scaling_beta, 0.1)
            && same_f64(rope.mscale, 1.0)
            && same_f64(rope.mscale_all_dim, 1.0)
            && rope.original_max_position_embeddings == YARN_ORIGINAL_CONTEXT
            && same_f64(rope.rope_theta, 1_000_000.0)
            && rope.rope_type == "yarn",
        "YaRN config differs from reviewed contract",
    )?;
    ensure(
        vision.hidden_size == 1_024
            && vision.num_attention_heads == 16
            && vision.num_hidden_layers == MINISTRAL3_VISION_LAYER_COUNT
            && vision.intermediate_size == 4_096
            && vision.patch_size == 14
            && vision.image_size == 1_540
            && vision.num_channels == 3
            && vision.head_dim == 64
            && same_f64(vision.rope_theta, 10_000.0)
            && vision.rope_type == "default"
            && config.image_token_index == 10
            && config.spatial_merge_size == 2
            && config.vision_feature_layer == -1
            && !config.multimodal_projector_bias,
        "vision/projector config differs from reviewed contract",
    )
}

/// Construct the canonical text target metadata. This metadata is a dry-run
/// target description; it is not proof that payload transforms ran.
pub fn ministral3_gguf_metadata(
    config: &Ministral3Config,
) -> Result<BTreeMap<String, GgufValue>, Ministral3GgufError> {
    validate_reviewed_config(config)?;
    validate_ministral3_architecture_spelling("mistral3", "ministral3", "mistral3")?;
    let revision_url =
        format!("https://huggingface.co/{MINISTRAL3_REPOSITORY}/tree/{MINISTRAL3_REVISION}");
    Ok(BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String(MINISTRAL3_GGUF_ARCHITECTURE.to_owned()),
        ),
        ("general.alignment".to_owned(), GgufValue::U32(32)),
        (
            "general.type".to_owned(),
            GgufValue::String("model".to_owned()),
        ),
        (
            "general.name".to_owned(),
            GgufValue::String(format!("{MINISTRAL3_REPOSITORY}@{MINISTRAL3_REVISION}")),
        ),
        (
            "general.license".to_owned(),
            GgufValue::String(MINISTRAL3_LICENSE.to_owned()),
        ),
        (
            "general.source.url".to_owned(),
            GgufValue::String(revision_url),
        ),
        (
            "general.source.huggingface.repository".to_owned(),
            GgufValue::String(MINISTRAL3_REPOSITORY.to_owned()),
        ),
        ("general.file_type".to_owned(), GgufValue::U32(32)),
        ("general.quantization_version".to_owned(), GgufValue::U32(2)),
        (
            "mistral3.source.revision".to_owned(),
            GgufValue::String(MINISTRAL3_REVISION.to_owned()),
        ),
        (
            "mistral3.vocab_size".to_owned(),
            GgufValue::U32(config.text.vocab_size),
        ),
        (
            "mistral3.context_length".to_owned(),
            GgufValue::U32(config.text.max_position_embeddings),
        ),
        (
            "mistral3.embedding_length".to_owned(),
            GgufValue::U32(config.text.hidden_size),
        ),
        (
            "mistral3.block_count".to_owned(),
            GgufValue::U32(config.text.num_hidden_layers),
        ),
        (
            "mistral3.feed_forward_length".to_owned(),
            GgufValue::U32(config.text.intermediate_size),
        ),
        (
            "mistral3.attention.head_count".to_owned(),
            GgufValue::U32(config.text.num_attention_heads),
        ),
        (
            "mistral3.attention.head_count_kv".to_owned(),
            GgufValue::U32(config.text.num_key_value_heads),
        ),
        (
            "mistral3.attention.key_length".to_owned(),
            GgufValue::U32(config.text.head_dim),
        ),
        (
            "mistral3.attention.value_length".to_owned(),
            GgufValue::U32(config.text.head_dim),
        ),
        (
            "mistral3.attention.layer_norm_rms_epsilon".to_owned(),
            GgufValue::F32(config.text.rms_norm_eps as f32),
        ),
        (
            "mistral3.rope.dimension_count".to_owned(),
            GgufValue::U32(config.text.head_dim),
        ),
        (
            "mistral3.rope.freq_base".to_owned(),
            GgufValue::F32(config.text.rope.rope_theta as f32),
        ),
        (
            "mistral3.rope.scaling.type".to_owned(),
            GgufValue::String("yarn".to_owned()),
        ),
        (
            "mistral3.rope.scaling.factor".to_owned(),
            GgufValue::F32(config.text.rope.factor as f32),
        ),
        (
            "mistral3.rope.scaling.original_context_length".to_owned(),
            GgufValue::U32(config.text.rope.original_max_position_embeddings),
        ),
        (
            "mistral3.rope.scaling.yarn_beta_fast".to_owned(),
            GgufValue::F32(config.text.rope.beta_fast as f32),
        ),
        (
            "mistral3.rope.scaling.yarn_beta_slow".to_owned(),
            GgufValue::F32(config.text.rope.beta_slow as f32),
        ),
        (
            "mistral3.rope.scaling.yarn_log_multiplier".to_owned(),
            GgufValue::F32(config.text.rope.mscale_all_dim as f32),
        ),
        (
            "mistral3.attention.temperature_scale".to_owned(),
            GgufValue::F32(config.text.rope.llama_4_scaling_beta as f32),
        ),
        (
            "mistral3.source.index_total_parameters".to_owned(),
            GgufValue::U64(MINISTRAL3_INDEX_TOTAL_PARAMETERS),
        ),
        (
            "mistral3.source.physical_parameters".to_owned(),
            GgufValue::U64(MINISTRAL3_PHYSICAL_PARAMETERS),
        ),
        (
            "mistral3.source.payload_bytes".to_owned(),
            GgufValue::U64(MINISTRAL3_INDEX_TOTAL_SIZE),
        ),
        (
            "mistral3.source.text_tensor_count".to_owned(),
            GgufValue::U64(MINISTRAL3_GGUF_SOURCE_TEXT_TENSOR_COUNT as u64),
        ),
        (
            "mistral3.source.vision_tensor_count".to_owned(),
            GgufValue::U64(MINISTRAL3_GGUF_SOURCE_VISION_TENSOR_COUNT as u64),
        ),
        (
            "mistral3.source.projector_tensor_count".to_owned(),
            GgufValue::U64(MINISTRAL3_GGUF_SOURCE_PROJECTOR_TENSOR_COUNT as u64),
        ),
        (
            "mistral3.output.tied_embedding".to_owned(),
            GgufValue::Bool(true),
        ),
        (
            "mistral3.output.explicit_output_tensor".to_owned(),
            GgufValue::Bool(false),
        ),
        (
            "mistral3.vision.production_supported".to_owned(),
            GgufValue::Bool(false),
        ),
    ]))
}

fn tensor_type_canonical(tensor_type: GgufTensorType) -> &'static str {
    match tensor_type {
        GgufTensorType::F32 => "f32",
        GgufTensorType::Bf16 => "bf16",
        GgufTensorType::F16 => "f16",
        GgufTensorType::I8Carrier => "i8-carrier",
        GgufTensorType::Mxfp4 => "mxfp4",
        GgufTensorType::Nvfp4 => "nvfp4",
    }
}

fn row_serialization(row: &Ministral3GgufCatalogRow) -> Result<String, Ministral3GgufError> {
    let role = row.tensor_role.canonical();
    let transform = row.required_transform.canonical();
    let tensor_type = row
        .output_tensor_type
        .map(tensor_type_canonical)
        .unwrap_or("-");
    let fields = [
        row.source_name.as_str(),
        row.source_shard.as_str(),
        row.artifact_plane.canonical(),
        role.as_str(),
        row.output_name.as_deref().unwrap_or("-"),
        tensor_type,
        transform.as_str(),
    ];
    ensure(
        !fields
            .iter()
            .any(|field| field.contains(['\t', '\n', '\r'])),
        "mapping row contains a canonical serialization delimiter",
    )?;
    Ok(format!("{}\n", fields.join("\t")))
}

fn build_catalog(
    config: &Ministral3Config,
    index: &Ministral3Index,
    enforce_digest: bool,
) -> Result<Ministral3GgufDryRunPlan, Ministral3GgufError> {
    validate_reviewed_config(config)?;
    ensure(
        index.total_parameters() == MINISTRAL3_INDEX_TOTAL_PARAMETERS,
        "typed index parameter count differs",
    )?;
    ensure(
        index.total_size() == MINISTRAL3_INDEX_TOTAL_SIZE,
        "typed index payload size differs",
    )?;
    ensure(
        index.tensor_count() == MINISTRAL3_TENSOR_COUNT,
        "typed index tensor count differs",
    )?;

    let mut rows = Vec::new();
    rows.try_reserve_exact(index.tensor_count())
        .map_err(|_| invalid("catalog row allocation failed"))?;
    let mut outputs = BTreeSet::new();
    let mut text_count = 0usize;
    let mut vision_count = 0usize;
    let mut projector_count = 0usize;
    let mut bf16_count = 0usize;
    let mut f32_count = 0usize;
    let mut query_count = 0usize;
    let mut key_count = 0usize;
    let mut serialization = String::new();
    serialization
        .try_reserve(
            index
                .tensor_count()
                .checked_mul(160)
                .ok_or_else(|| invalid("mapping serialization capacity overflows"))?,
        )
        .map_err(|_| invalid("mapping serialization allocation failed"))?;

    for (source_name, source_shard) in index.tensors() {
        let mapping = map_ministral3_source_tensor(source_name)?;
        match mapping.artifact_plane {
            Ministral3ArtifactPlane::Text => checked_add(&mut text_count, 1, "text")?,
            Ministral3ArtifactPlane::Vision => checked_add(&mut vision_count, 1, "vision")?,
            Ministral3ArtifactPlane::Projector => {
                checked_add(&mut projector_count, 1, "projector")?
            }
        }
        if let Some(output_name) = &mapping.output_name {
            ensure(
                outputs.insert(output_name.clone()),
                format!("duplicate output tensor: {output_name}"),
            )?;
        }
        match mapping.output_tensor_type {
            Some(GgufTensorType::Bf16) => checked_add(&mut bf16_count, 1, "BF16 output")?,
            Some(GgufTensorType::F32) => checked_add(&mut f32_count, 1, "F32 output")?,
            Some(other) => {
                return Err(invalid(format!("unexpected target tensor type: {other:?}")));
            }
            None => {}
        }
        match mapping.required_transform {
            Ministral3PayloadTransform::QueryHeadPermutation { .. } => {
                checked_add(&mut query_count, 1, "query permutation")?
            }
            Ministral3PayloadTransform::KeyHeadPermutation { .. } => {
                checked_add(&mut key_count, 1, "key permutation")?
            }
            _ => {}
        }
        let row = Ministral3GgufCatalogRow {
            source_name: mapping.source_name,
            source_shard: source_shard.to_owned(),
            artifact_plane: mapping.artifact_plane,
            tensor_role: mapping.tensor_role,
            output_name: mapping.output_name,
            output_dimensions: mapping.output_dimensions,
            output_tensor_type: mapping.output_tensor_type,
            required_transform: mapping.required_transform,
        };
        serialization.push_str(&row_serialization(&row)?);
        rows.push(row);
    }
    let known_unconsumed_count = vision_count
        .checked_add(projector_count)
        .ok_or_else(|| invalid("known-unconsumed count overflows"))?;
    ensure(
        text_count == MINISTRAL3_GGUF_SOURCE_TEXT_TENSOR_COUNT,
        "text tensor count differs",
    )?;
    ensure(
        vision_count == MINISTRAL3_GGUF_SOURCE_VISION_TENSOR_COUNT,
        "vision tensor count differs",
    )?;
    ensure(
        projector_count == MINISTRAL3_GGUF_SOURCE_PROJECTOR_TENSOR_COUNT,
        "projector tensor count differs",
    )?;
    ensure(
        known_unconsumed_count == MINISTRAL3_GGUF_KNOWN_UNCONSUMED_TENSOR_COUNT,
        "known-unconsumed count differs",
    )?;
    ensure(
        outputs.len() == MINISTRAL3_GGUF_OUTPUT_CANDIDATE_COUNT,
        "output candidate count differs",
    )?;
    ensure(
        bf16_count == MINISTRAL3_GGUF_BF16_TENSOR_COUNT,
        "BF16 output count differs",
    )?;
    ensure(
        f32_count == MINISTRAL3_GGUF_F32_NORM_TENSOR_COUNT,
        "F32 norm count differs",
    )?;
    ensure(
        query_count == MINISTRAL3_GGUF_QUERY_PERMUTATION_COUNT,
        "query permutation count differs",
    )?;
    ensure(
        key_count == MINISTRAL3_GGUF_KEY_PERMUTATION_COUNT,
        "key permutation count differs",
    )?;
    ensure(
        !outputs.contains("output.weight"),
        "tied output must remain omitted",
    )?;
    let mapping_sha256 = format!("{:x}", Sha256::digest(serialization.as_bytes()));
    if enforce_digest {
        ensure(
            mapping_sha256 == MINISTRAL3_GGUF_MAPPING_SHA256,
            "source mapping digest differs",
        )?;
    }
    Ok(Ministral3GgufDryRunPlan {
        target_metadata: ministral3_gguf_metadata(config)?,
        source_rows: rows,
        source_tensor_count: index.tensor_count(),
        source_text_tensor_count: text_count,
        source_vision_tensor_count: vision_count,
        source_projector_tensor_count: projector_count,
        known_unconsumed_tensor_count: known_unconsumed_count,
        output_candidate_count: outputs.len(),
        output_bf16_tensor_count: bf16_count,
        output_f32_tensor_count: f32_count,
        query_permutation_count: query_count,
        key_permutation_count: key_count,
        mapping_sha256,
        tied_output_omitted: true,
        source_payload_bytes_verified: false,
        payload_transforms_executed: false,
        dtype_conversions_executed: false,
        quantization_executed: false,
        writable_gguf_plan: false,
        output_payload_bytes: None,
        output_file_sha256: None,
        pass_scope: MINISTRAL3_GGUF_PASS_SCOPE,
    })
}

pub fn build_ministral3_gguf_dry_run(
    config: &Ministral3Config,
    index: &Ministral3Index,
) -> Result<Ministral3GgufDryRunPlan, Ministral3GgufError> {
    build_catalog(config, index, true)
}

/// Validate the exact official config/index pair and return a write-disabled
/// canonical text mapping.
pub fn validate_ministral3_gguf_dry_run(
    config_bytes: &[u8],
    index_bytes: &[u8],
) -> Result<Ministral3GgufDryRunPlan, Ministral3GgufError> {
    ensure(
        config_bytes.len() == MINISTRAL3_CONFIG_BYTES,
        "config byte length differs",
    )?;
    ensure(
        format!("{:x}", Sha256::digest(config_bytes)) == MINISTRAL3_CONFIG_SHA256,
        "config SHA-256 differs",
    )?;
    let config =
        validate_ministral3_config(config_bytes).map_err(|error| invalid(error.to_string()))?;
    let index = validate_ministral3_gguf_source_index(index_bytes)?;
    build_ministral3_gguf_dry_run(&config, &index)
}

#[derive(Clone)]
pub struct VerifiedOfficialMinistral3Gguf {
    gguf: VerifiedGguf,
}

impl fmt::Debug for VerifiedOfficialMinistral3Gguf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedOfficialMinistral3Gguf")
            .field("path", &self.gguf.path())
            .field("file_size", &self.gguf.file_size())
            .field("tensor_count", &self.gguf.tensors().len())
            .finish_non_exhaustive()
    }
}

impl VerifiedOfficialMinistral3Gguf {
    /// Verify the official GGUF header/catalog. The caller must separately
    /// bind the full file to [`MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256`].
    pub fn verify(gguf: VerifiedGguf) -> Result<Self, Ministral3GgufError> {
        verify_official_gguf(&gguf, true)?;
        Ok(Self { gguf })
    }

    pub fn gguf(&self) -> &VerifiedGguf {
        &self.gguf
    }

    pub fn repository(&self) -> &'static str {
        MINISTRAL3_OFFICIAL_GGUF_REPOSITORY
    }

    pub fn revision(&self) -> &'static str {
        MINISTRAL3_OFFICIAL_GGUF_REVISION
    }

    pub fn expected_lfs_sha256(&self) -> &'static str {
        MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256
    }
}

pub fn verify_official_ministral3_gguf(
    gguf: VerifiedGguf,
) -> Result<VerifiedOfficialMinistral3Gguf, Ministral3GgufError> {
    VerifiedOfficialMinistral3Gguf::verify(gguf)
}

/// Open and fully authenticate the fixed official production GGUF.
///
/// `VerifiedGguf::open` opens a regular file with `O_NOFOLLOW` and retains
/// that descriptor. The full SHA-256 is then streamed from that same retained
/// descriptor, so replacing the pathname cannot switch the bytes between
/// header parsing and hashing. The returned wrapper continues to own the same
/// descriptor. This is the production admission entry point; the dry-run
/// converter catalog above never substitutes for it.
pub fn open_and_verify_official_ministral3_gguf(
    path: impl AsRef<Path>,
) -> Result<VerifiedOfficialMinistral3Gguf, Ministral3GgufError> {
    let gguf = VerifiedGguf::open(path.as_ref())
        .map_err(|error| invalid(format!("official GGUF open failed: {error}")))?;
    ensure(
        gguf.file_size() == MINISTRAL3_OFFICIAL_GGUF_FILE_BYTES,
        "official GGUF file size differs before full hash",
    )?;
    let full_sha256 = gguf
        .file_sha256()
        .map_err(|error| invalid(format!("official GGUF full hash failed: {error}")))?;
    ensure(
        full_sha256 == format!("sha256:{MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256}"),
        "official GGUF full-file SHA-256 differs",
    )?;
    verify_official_ministral3_gguf(gguf)
}

fn metadata_value<'a>(
    gguf: &'a VerifiedGguf,
    key: &str,
) -> Result<&'a GgufValue, Ministral3GgufError> {
    gguf.metadata_value(key)
        .ok_or_else(|| invalid(format!("official GGUF metadata is missing {key}")))
}

fn expect_metadata(
    gguf: &VerifiedGguf,
    key: &str,
    expected: GgufValue,
) -> Result<(), Ministral3GgufError> {
    ensure(
        metadata_value(gguf, key)? == &expected,
        format!("official GGUF metadata differs at {key}"),
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OfficialTensorExpectation {
    dimensions: Vec<u64>,
    tensor_type: GgufTensorType,
}

fn expected_official_tensor_catalog()
-> Result<BTreeMap<String, OfficialTensorExpectation>, Ministral3GgufError> {
    let mut expected = BTreeMap::new();
    let root_names = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
    ];
    for source_name in root_names {
        let mapping = map_ministral3_source_tensor(source_name)?;
        insert_official_expectation(&mut expected, mapping)?;
    }
    const SUFFIXES: [&str; 9] = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        "mlp.up_proj.weight",
    ];
    for layer in 0..MINISTRAL3_TEXT_LAYER_COUNT {
        for suffix in SUFFIXES {
            let source_name = format!("language_model.model.layers.{layer}.{suffix}");
            insert_official_expectation(
                &mut expected,
                map_ministral3_source_tensor(&source_name)?,
            )?;
        }
    }
    ensure(
        expected.len() == MINISTRAL3_GGUF_OUTPUT_CANDIDATE_COUNT,
        "official tensor expectation count differs",
    )?;
    Ok(expected)
}

fn insert_official_expectation(
    expected: &mut BTreeMap<String, OfficialTensorExpectation>,
    mapping: Ministral3SourceTensorMapping,
) -> Result<(), Ministral3GgufError> {
    let name = mapping
        .output_name
        .ok_or_else(|| invalid("text mapping unexpectedly has no output"))?;
    let dimensions = mapping
        .output_dimensions
        .ok_or_else(|| invalid("text mapping unexpectedly has no dimensions"))?;
    let tensor_type = mapping
        .output_tensor_type
        .ok_or_else(|| invalid("text mapping unexpectedly has no type"))?;
    ensure(
        expected
            .insert(
                name.clone(),
                OfficialTensorExpectation {
                    dimensions,
                    tensor_type,
                },
            )
            .is_none(),
        format!("duplicate expected official tensor: {name}"),
    )
}

fn verify_official_gguf(
    gguf: &VerifiedGguf,
    enforce_digests: bool,
) -> Result<(), Ministral3GgufError> {
    ensure(
        gguf.file_size() == MINISTRAL3_OFFICIAL_GGUF_FILE_BYTES,
        "official GGUF file size differs",
    )?;
    ensure(gguf.alignment() == 32, "official GGUF alignment differs")?;
    ensure(
        gguf.architecture() == MINISTRAL3_GGUF_ARCHITECTURE,
        "official GGUF architecture differs",
    )?;
    ensure(
        gguf.extension().is_none(),
        "official GGUF unexpectedly has an sLLM extension",
    )?;
    if enforce_digests {
        ensure(
            gguf.metadata_sha256() == MINISTRAL3_OFFICIAL_GGUF_METADATA_SHA256,
            "official GGUF metadata digest differs",
        )?;
        ensure(
            gguf.tensor_catalog_sha256() == MINISTRAL3_OFFICIAL_GGUF_TENSOR_CATALOG_SHA256,
            "official GGUF tensor catalog digest differs",
        )?;
    }
    expect_metadata(
        gguf,
        "general.architecture",
        GgufValue::String(MINISTRAL3_GGUF_ARCHITECTURE.to_owned()),
    )?;
    expect_metadata(gguf, "general.type", GgufValue::String("model".to_owned()))?;
    expect_metadata(gguf, "general.file_type", GgufValue::U32(32))?;
    expect_metadata(gguf, "general.quantization_version", GgufValue::U32(2))?;
    expect_metadata(gguf, "mistral3.block_count", GgufValue::U32(26))?;
    expect_metadata(gguf, "mistral3.context_length", GgufValue::U32(262_144))?;
    expect_metadata(gguf, "mistral3.embedding_length", GgufValue::U32(3_072))?;
    expect_metadata(gguf, "mistral3.feed_forward_length", GgufValue::U32(9_216))?;
    expect_metadata(gguf, "mistral3.attention.head_count", GgufValue::U32(32))?;
    expect_metadata(gguf, "mistral3.attention.head_count_kv", GgufValue::U32(8))?;
    expect_metadata(gguf, "mistral3.attention.key_length", GgufValue::U32(128))?;
    expect_metadata(gguf, "mistral3.attention.value_length", GgufValue::U32(128))?;
    expect_metadata(
        gguf,
        "mistral3.attention.layer_norm_rms_epsilon",
        GgufValue::F32(1.0e-5),
    )?;
    expect_metadata(gguf, "mistral3.rope.dimension_count", GgufValue::U32(128))?;
    expect_metadata(gguf, "mistral3.rope.freq_base", GgufValue::F32(1_000_000.0))?;
    expect_metadata(
        gguf,
        "mistral3.rope.scaling.type",
        GgufValue::String("yarn".to_owned()),
    )?;
    expect_metadata(gguf, "mistral3.rope.scaling.factor", GgufValue::F32(16.0))?;
    expect_metadata(
        gguf,
        "mistral3.rope.scaling.original_context_length",
        GgufValue::U32(16_384),
    )?;
    expect_metadata(
        gguf,
        "mistral3.rope.scaling.yarn_beta_fast",
        GgufValue::F32(32.0),
    )?;
    expect_metadata(
        gguf,
        "mistral3.rope.scaling.yarn_beta_slow",
        GgufValue::F32(1.0),
    )?;
    expect_metadata(
        gguf,
        "mistral3.rope.scaling.yarn_log_multiplier",
        GgufValue::F32(1.0),
    )?;
    expect_metadata(
        gguf,
        "mistral3.attention.temperature_scale",
        GgufValue::F32(0.1),
    )?;

    let expected = expected_official_tensor_catalog()?;
    ensure(
        gguf.tensors().len() == MINISTRAL3_GGUF_OUTPUT_CANDIDATE_COUNT,
        "official GGUF tensor count differs",
    )?;
    let mut bf16_count = 0usize;
    let mut f32_count = 0usize;
    for tensor in gguf.tensors() {
        let specification = expected
            .get(&tensor.name)
            .ok_or_else(|| invalid(format!("unknown official GGUF tensor: {}", tensor.name)))?;
        ensure(
            tensor.dimensions == specification.dimensions,
            format!("official GGUF tensor dimensions differ: {}", tensor.name),
        )?;
        ensure(
            tensor.tensor_type == specification.tensor_type,
            format!("official GGUF tensor type differs: {}", tensor.name),
        )?;
        match tensor.tensor_type {
            GgufTensorType::Bf16 => checked_add(&mut bf16_count, 1, "official BF16")?,
            GgufTensorType::F32 => checked_add(&mut f32_count, 1, "official F32")?,
            other => {
                return Err(invalid(format!(
                    "unsupported official tensor type: {other:?}"
                )));
            }
        }
    }
    ensure(
        bf16_count == MINISTRAL3_GGUF_BF16_TENSOR_COUNT,
        "official BF16 tensor count differs",
    )?;
    ensure(
        f32_count == MINISTRAL3_GGUF_F32_NORM_TENSOR_COUNT,
        "official F32 norm count differs",
    )?;
    ensure(
        gguf.tensor("output.weight").is_none(),
        "official GGUF must omit tied output.weight",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ministral3::{
        MINISTRAL3_INDEX_BYTES, Ministral3RopeParameters, Ministral3TextConfig,
        Ministral3VisionConfig,
    };
    use std::path::PathBuf;

    fn fixture_config() -> Ministral3Config {
        Ministral3Config {
            text: Ministral3TextConfig {
                hidden_size: 3_072,
                intermediate_size: 9_216,
                num_hidden_layers: 26,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                head_dim: 128,
                vocab_size: 131_072,
                max_position_embeddings: 262_144,
                rms_norm_eps: 1.0e-5,
                rope: Ministral3RopeParameters {
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    factor: 16.0,
                    llama_4_scaling_beta: 0.1,
                    mscale: 1.0,
                    mscale_all_dim: 1.0,
                    original_max_position_embeddings: 16_384,
                    rope_theta: 1_000_000.0,
                    rope_type: "yarn",
                },
                tie_word_embeddings: true,
                use_cache: true,
            },
            vision: Ministral3VisionConfig {
                hidden_size: 1_024,
                num_attention_heads: 16,
                num_hidden_layers: 24,
                intermediate_size: 4_096,
                patch_size: 14,
                image_size: 1_540,
                num_channels: 3,
                head_dim: 64,
                rope_theta: 10_000.0,
                rope_type: "default",
            },
            image_token_index: 10,
            spatial_merge_size: 2,
            vision_feature_layer: -1,
            multimodal_projector_bias: false,
        }
    }

    #[test]
    fn metadata_fixes_yarn_temperature_and_no_write_claim() {
        let metadata = ministral3_gguf_metadata(&fixture_config()).unwrap();
        assert_eq!(
            metadata["general.architecture"],
            GgufValue::String("mistral3".to_owned())
        );
        assert_eq!(metadata["general.file_type"], GgufValue::U32(32));
        assert_eq!(
            metadata["mistral3.rope.scaling.type"],
            GgufValue::String("yarn".to_owned())
        );
        assert_eq!(
            metadata["mistral3.rope.scaling.yarn_log_multiplier"],
            GgufValue::F32(1.0)
        );
        assert_eq!(
            metadata["mistral3.attention.temperature_scale"],
            GgufValue::F32(0.1)
        );
        assert_eq!(
            metadata["mistral3.output.explicit_output_tensor"],
            GgufValue::Bool(false)
        );
        assert_eq!(
            MINISTRAL3_GGUF_PASS_SCOPE,
            "exact-config-index-text-mapping-only-no-source-payload-no-write"
        );
    }

    #[test]
    fn root_layer_boundaries_and_required_transforms_are_typed() {
        let token =
            map_ministral3_source_tensor("language_model.model.embed_tokens.weight").unwrap();
        assert_eq!(token.output_name.as_deref(), Some("token_embd.weight"));
        assert_eq!(token.output_dimensions, Some(vec![3_072, 131_072]));
        assert_eq!(token.output_tensor_type, Some(GgufTensorType::Bf16));

        let norm = map_ministral3_source_tensor("language_model.model.norm.weight").unwrap();
        assert_eq!(norm.output_name.as_deref(), Some("output_norm.weight"));
        assert_eq!(norm.output_tensor_type, Some(GgufTensorType::F32));
        assert_eq!(
            norm.required_transform,
            Ministral3PayloadTransform::Bf16ToF32
        );

        let query =
            map_ministral3_source_tensor("language_model.model.layers.0.self_attn.q_proj.weight")
                .unwrap();
        assert_eq!(query.output_name.as_deref(), Some("blk.0.attn_q.weight"));
        assert_eq!(query.output_dimensions, Some(vec![3_072, 4_096]));
        assert_eq!(
            query.required_transform,
            Ministral3PayloadTransform::QueryHeadPermutation {
                heads: 32,
                head_dim: 128
            }
        );

        let key =
            map_ministral3_source_tensor("language_model.model.layers.25.self_attn.k_proj.weight")
                .unwrap();
        assert_eq!(key.output_name.as_deref(), Some("blk.25.attn_k.weight"));
        assert_eq!(key.output_dimensions, Some(vec![3_072, 1_024]));
        assert_eq!(
            key.required_transform,
            Ministral3PayloadTransform::KeyHeadPermutation {
                heads: 8,
                head_dim: 128
            }
        );

        assert!(
            map_ministral3_source_tensor("language_model.model.layers.26.self_attn.q_proj.weight")
                .is_err()
        );
        assert!(
            map_ministral3_source_tensor("language_model.model.layers.01.self_attn.q_proj.weight")
                .is_err()
        );
        assert!(map_ministral3_source_tensor("language_model.lm_head.weight").is_err());
        assert!(
            map_ministral3_source_tensor("language_model.model.layers.0.self_attn.q_proj.bias")
                .is_err()
        );
    }

    #[test]
    fn vision_projector_are_exact_known_unconsumed_families() {
        let vision = map_ministral3_source_tensor(
            "vision_tower.transformer.layers.23.feed_forward.down_proj.weight",
        )
        .unwrap();
        assert_eq!(vision.artifact_plane, Ministral3ArtifactPlane::Vision);
        assert_eq!(vision.output_name, None);
        assert_eq!(
            vision.required_transform,
            Ministral3PayloadTransform::KnownUnconsumed
        );
        let projector =
            map_ministral3_source_tensor("multi_modal_projector.patch_merger.merging_layer.weight")
                .unwrap();
        assert_eq!(projector.artifact_plane, Ministral3ArtifactPlane::Projector);
        assert!(
            map_ministral3_source_tensor(
                "vision_tower.transformer.layers.24.attention.q_proj.weight"
            )
            .is_err()
        );
        assert!(
            map_ministral3_source_tensor("vision_tower.transformer.layers.1.attention.bias")
                .is_err()
        );
        assert!(map_ministral3_source_tensor("multi_modal_projector.linear_3.weight").is_err());
    }

    #[test]
    fn spelling_and_config_drift_fail_closed() {
        assert!(
            validate_ministral3_architecture_spelling("mistral3", "ministral3", "mistral3").is_ok()
        );
        assert!(
            validate_ministral3_architecture_spelling("ministral3", "ministral3", "mistral3")
                .is_err()
        );
        assert!(
            validate_ministral3_architecture_spelling("mistral3", "mistral3", "mistral3").is_err()
        );
        assert!(
            validate_ministral3_architecture_spelling("mistral3", "ministral3", "ministral3")
                .is_err()
        );
        let mut config = fixture_config();
        config.text.rope.factor = 15.999;
        assert!(ministral3_gguf_metadata(&config).is_err());
        let mut config = fixture_config();
        config.text.num_hidden_layers = 25;
        assert!(ministral3_gguf_metadata(&config).is_err());
        let mut config = fixture_config();
        config.text.num_hidden_layers = 27;
        assert!(ministral3_gguf_metadata(&config).is_err());
    }

    #[test]
    fn expected_official_catalog_has_exact_types_shapes_and_tied_omission() {
        let catalog = expected_official_tensor_catalog().unwrap();
        assert_eq!(catalog.len(), 236);
        assert!(!catalog.contains_key("output.weight"));
        assert_eq!(
            catalog["blk.17.attn_output.weight"].dimensions,
            vec![4_096, 3_072]
        );
        assert_eq!(
            catalog["blk.17.ffn_down.weight"].dimensions,
            vec![9_216, 3_072]
        );
        let bf16 = catalog
            .values()
            .filter(|entry| entry.tensor_type == GgufTensorType::Bf16)
            .count();
        let f32 = catalog
            .values()
            .filter(|entry| entry.tensor_type == GgufTensorType::F32)
            .count();
        assert_eq!((bf16, f32), (183, 53));
    }

    #[test]
    fn locked_index_rejects_length_and_hash_boundaries_before_allocation() {
        assert!(validate_ministral3_gguf_source_index(&[]).is_err());
        assert!(
            validate_ministral3_gguf_source_index(&vec![0; MINISTRAL3_INDEX_BYTES - 1]).is_err()
        );
        assert!(validate_ministral3_gguf_source_index(&vec![0; MINISTRAL3_INDEX_BYTES]).is_err());
        assert!(
            validate_ministral3_gguf_source_index(&vec![0; MINISTRAL3_INDEX_BYTES + 1]).is_err()
        );
    }

    fn official_fixture_root() -> PathBuf {
        std::env::var_os("SLLM_MINISTRAL3_METADATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/sllm-phase60.2pDfxs"))
    }

    #[test]
    #[ignore = "requires exact official config/index under /tmp/sllm-phase60.2pDfxs or SLLM_MINISTRAL3_METADATA_DIR"]
    fn exact_official_source_mapping_digest_and_counts_are_fixed() {
        let root = official_fixture_root();
        let config_bytes = std::fs::read(root.join("config.json")).expect("read exact config");
        let index_bytes =
            std::fs::read(root.join("model.safetensors.index.json")).expect("read exact index");
        let config = validate_ministral3_config(&config_bytes).expect("exact config validates");
        let index =
            validate_ministral3_gguf_source_index(&index_bytes).expect("exact index validates");
        let plan = build_catalog(&config, &index, false).expect("exact source catalog builds");
        eprintln!("MINISTRAL3_GGUF_MAPPING_SHA256={}", plan.mapping_sha256);
        assert_eq!(plan.mapping_sha256, MINISTRAL3_GGUF_MAPPING_SHA256);
        assert_eq!(plan.source_tensor_count, 458);
        assert_eq!(
            (
                plan.source_text_tensor_count,
                plan.source_vision_tensor_count,
                plan.source_projector_tensor_count
            ),
            (236, 218, 4)
        );
        assert_eq!(
            (plan.output_bf16_tensor_count, plan.output_f32_tensor_count),
            (183, 53)
        );
        assert_eq!(
            (plan.query_permutation_count, plan.key_permutation_count),
            (26, 26)
        );
        assert!(plan.tied_output_omitted);
        assert!(!plan.source_payload_bytes_verified);
        assert!(!plan.payload_transforms_executed);
        assert!(!plan.dtype_conversions_executed);
        assert!(!plan.quantization_executed);
        assert!(!plan.writable_gguf_plan);
        assert_eq!(plan.output_payload_bytes, None);
        assert_eq!(plan.output_file_sha256, None);
    }

    #[test]
    #[ignore = "requires the fixed 6,866,745,504-byte official BF16 GGUF under /tmp/sllm-phase60.2pDfxs or SLLM_MINISTRAL3_METADATA_DIR"]
    fn exact_official_gguf_metadata_catalog_and_types_are_fixed() {
        let path = official_fixture_root().join(MINISTRAL3_OFFICIAL_GGUF_FILE_NAME);
        let gguf = VerifiedGguf::open(&path).expect("official GGUF parses");
        eprintln!(
            "MINISTRAL3_OFFICIAL_GGUF_METADATA_SHA256={}",
            gguf.metadata_sha256()
        );
        eprintln!(
            "MINISTRAL3_OFFICIAL_GGUF_TENSOR_CATALOG_SHA256={}",
            gguf.tensor_catalog_sha256()
        );
        verify_official_gguf(&gguf, false).expect("official semantic catalog validates");
        let header_verified =
            verify_official_ministral3_gguf(gguf).expect("official exact header validates");
        assert_eq!(
            header_verified.gguf().file_size(),
            MINISTRAL3_OFFICIAL_GGUF_FILE_BYTES
        );
        assert_eq!(header_verified.gguf().tensors().len(), 236);
        drop(header_verified);
        let verified = open_and_verify_official_ministral3_gguf(&path)
            .expect("official full-file identity validates");
        assert_eq!(
            verified.expected_lfs_sha256(),
            MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256
        );
    }
}
