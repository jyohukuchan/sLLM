//! Strict identity and typed configuration contract for the reviewed
//! Ministral 3 3B BF16 artifact.
//!
//! This module validates bounded metadata only.  It does not load weights,
//! infer support from an architecture string, or treat the physical tensor
//! count as the index's tied-embedding parameter count.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt;

pub const MINISTRAL3_REPOSITORY: &str = "mistralai/Ministral-3-3B-Instruct-2512-BF16";
pub const MINISTRAL3_REVISION: &str = "b6d637bef2393152b3da2b2fde72eecdee30557e";
pub const MINISTRAL3_LICENSE: &str = "Apache-2.0";
pub const MINISTRAL3_CONFIG_BYTES: usize = 1_579;
pub const MINISTRAL3_CONFIG_SHA256: &str =
    "c89d1a0b4f237d2892ce911d1fe03e9e5a4834579f7149ebc715a4c3fa564214";
pub const MINISTRAL3_INDEX_BYTES: usize = 45_577;
pub const MINISTRAL3_INDEX_SHA256: &str =
    "7829dcf0040e34f1172b401563fcbb27cc3c5a0244ef01e6af18b7a64d63a81e";
pub const MINISTRAL3_SHARD_COUNT: usize = 2;
pub const MINISTRAL3_TENSOR_COUNT: usize = 458;
/// The index counts the tied embedding twice.
pub const MINISTRAL3_INDEX_TOTAL_PARAMETERS: u64 = 4_251_743_232;
/// Physical BF16 tensor numel from the 458 header tensor rows.
pub const MINISTRAL3_PHYSICAL_PARAMETERS: u64 = 3_849_090_048;
pub const MINISTRAL3_INDEX_TOTAL_SIZE: u64 = 7_698_180_096;
pub const MINISTRAL3_SHARD_FILE_BYTES: u64 = 7_698_241_056;
pub const MINISTRAL3_HEADER_BYTES: u64 = 60_960;
pub const MINISTRAL3_TENSOR_PAYLOAD_BYTES: u64 = MINISTRAL3_INDEX_TOTAL_SIZE;
/// Admission includes the complete safetensors files, not just tensor data.
pub const MINISTRAL3_CAPACITY_ADMISSION_BYTES: u64 = MINISTRAL3_SHARD_FILE_BYTES;
pub const MINISTRAL3_CONTEXT_LENGTH: u32 = 262_144;
pub const MINISTRAL3_TEXT_LAYER_COUNT: u32 = 26;
pub const MINISTRAL3_VISION_LAYER_COUNT: u32 = 24;
pub const MINISTRAL3_TEXT_HIDDEN_SIZE: u32 = 3_072;
pub const MINISTRAL3_TEXT_FFN_SIZE: u32 = 9_216;
pub const MINISTRAL3_VOCAB_SIZE: u32 = 131_072;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ministral3ArtifactEvidence {
    pub hub_shard_lfs_identity_fixed: bool,
    pub local_full_shard_payload_sha256_verified: bool,
}

pub const MINISTRAL3_ARTIFACT_EVIDENCE: Ministral3ArtifactEvidence = Ministral3ArtifactEvidence {
    hub_shard_lfs_identity_fixed: true,
    local_full_shard_payload_sha256_verified: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ministral3ShardIdentity {
    pub file_name: &'static str,
    pub file_size: u64,
    /// Full-file LFS SHA-256 reported by the fixed-revision Hub metadata.
    /// This is not computed from the bounded header bytes.
    pub lfs_sha256: &'static str,
    pub header_length: u64,
    /// SHA-256 over the eight-byte length field plus JSON header only.
    pub header_sha256: &'static str,
    pub indexed_tensor_count: usize,
}

pub const MINISTRAL3_SHARDS: [Ministral3ShardIdentity; MINISTRAL3_SHARD_COUNT] = [
    Ministral3ShardIdentity {
        file_name: "model-00001-of-00002.safetensors",
        file_size: 4_967_581_832,
        lfs_sha256: "b3821ebc30884f66e3d26d339e161641f34b91ed916627011e7b08e5f1edd884",
        header_length: 47_232,
        header_sha256: "0a9c9a62103a14b6d5a9f04958e1df0137f9b296cd66eaaf79959b1d549839c9",
        indexed_tensor_count: 353,
    },
    Ministral3ShardIdentity {
        file_name: "model-00002-of-00002.safetensors",
        file_size: 2_730_659_224,
        lfs_sha256: "718d087fa591fd4356b7241f293c24219399d86f092d46cf36f765051498033a",
        header_length: 13_712,
        header_sha256: "a1e93e8c71240094ee375c1641fd7dedb8ba250dd88ff74c8373dcfe0571756f",
        indexed_tensor_count: 105,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ministral3SupportFileRole {
    RepositoryAttributes,
    ChatTemplate,
    ModelConfig,
    GenerationConfig,
    SafetensorsIndex,
    Params,
    ProcessorConfig,
    SpecialTokensMap,
    TokenizerModel,
    TokenizerConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ministral3SupportFileIdentity {
    pub file_name: &'static str,
    pub role: Ministral3SupportFileRole,
    pub size: usize,
    pub sha256: &'static str,
}

pub const MINISTRAL3_SUPPORT_FILES: [Ministral3SupportFileIdentity; 11] = [
    Ministral3SupportFileIdentity {
        file_name: ".gitattributes",
        role: Ministral3SupportFileRole::RepositoryAttributes,
        size: 1_618,
        sha256: "1e11e8e6b9eb852a71eadd8fe191d4b9e329eb81308602dec19ab16da7e172b2",
    },
    Ministral3SupportFileIdentity {
        file_name: "chat_template.jinja",
        role: Ministral3SupportFileRole::ChatTemplate,
        size: 11_912,
        sha256: "0701cfbdc2b7d44fdbad104dff604faee4b0543e8247624568777fe465746f9b",
    },
    Ministral3SupportFileIdentity {
        file_name: "config.json",
        role: Ministral3SupportFileRole::ModelConfig,
        size: MINISTRAL3_CONFIG_BYTES,
        sha256: MINISTRAL3_CONFIG_SHA256,
    },
    Ministral3SupportFileIdentity {
        file_name: "generation_config.json",
        role: Ministral3SupportFileRole::GenerationConfig,
        size: 131,
        sha256: "e0923390059f84a9180b00e5501778acc45ea9856cd7f2fd68208b360927c677",
    },
    Ministral3SupportFileIdentity {
        file_name: "model.safetensors.index.json",
        role: Ministral3SupportFileRole::SafetensorsIndex,
        size: MINISTRAL3_INDEX_BYTES,
        sha256: MINISTRAL3_INDEX_SHA256,
    },
    Ministral3SupportFileIdentity {
        file_name: "params.json",
        role: Ministral3SupportFileRole::Params,
        size: 1_096,
        sha256: "d15587c57574d2a92c66ff2552c6db5173d4c440ad47f7454dc51bad0ecaeb59",
    },
    Ministral3SupportFileIdentity {
        file_name: "processor_config.json",
        role: Ministral3SupportFileRole::ProcessorConfig,
        size: 976,
        sha256: "ece2373c2ae391bce18785a1810543bc6173a1f28b6767cffab2c35dbea5f002",
    },
    Ministral3SupportFileIdentity {
        file_name: "special_tokens_map.json",
        role: Ministral3SupportFileRole::SpecialTokensMap,
        size: 147_094,
        sha256: "0a5c981e8c5c6f8886ee007a6d4543a0be6b221cb9ca32a8709384a4c6fc8cbb",
    },
    Ministral3SupportFileIdentity {
        file_name: "tekken.json",
        role: Ministral3SupportFileRole::TokenizerModel,
        size: 16_753_784,
        sha256: "600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843",
    },
    Ministral3SupportFileIdentity {
        file_name: "tokenizer.json",
        role: Ministral3SupportFileRole::TokenizerModel,
        size: 17_078_128,
        sha256: "d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135",
    },
    Ministral3SupportFileIdentity {
        file_name: "tokenizer_config.json",
        role: Ministral3SupportFileRole::TokenizerConfig,
        size: 198_094,
        sha256: "f59f7294e4f26383d0ea93840fe21cf197784be0842a8301a0343e8c34ed0d6d",
    },
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ministral3RopeParameters {
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub factor: f64,
    pub llama_4_scaling_beta: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    pub original_max_position_embeddings: u32,
    pub rope_theta: f64,
    pub rope_type: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ministral3TextConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f64,
    pub rope: Ministral3RopeParameters,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ministral3VisionConfig {
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    pub patch_size: u32,
    pub image_size: u32,
    pub num_channels: u32,
    pub head_dim: u32,
    pub rope_theta: f64,
    pub rope_type: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ministral3Config {
    pub text: Ministral3TextConfig,
    pub vision: Ministral3VisionConfig,
    pub image_token_index: u32,
    pub spatial_merge_size: u32,
    pub vision_feature_layer: i32,
    pub multimodal_projector_bias: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3Error {
    Invalid(String),
}

impl fmt::Display for Ministral3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid Ministral 3 artifact: {message}"),
        }
    }
}

impl std::error::Error for Ministral3Error {}

fn invalid(message: impl Into<String>) -> Ministral3Error {
    Ministral3Error::Invalid(message.into())
}

fn ensure(condition: bool, message: impl Into<String>) -> Result<(), Ministral3Error> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn same_f64(actual: f64, expected: f64) -> bool {
    actual.to_bits() == expected.to_bits()
}

fn locked_bytes(
    bytes: &[u8],
    expected_size: usize,
    expected_sha256: &str,
    label: &str,
) -> Result<(), Ministral3Error> {
    ensure(
        bytes.len() == expected_size,
        format!("{label} byte length differs"),
    )?;
    let digest = format!("{:x}", Sha256::digest(bytes));
    ensure(
        digest == expected_sha256,
        format!("{label} SHA-256 differs"),
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRopeParameters {
    beta_fast: f64,
    beta_slow: f64,
    factor: f64,
    llama_4_scaling_beta: f64,
    mscale: f64,
    mscale_all_dim: f64,
    original_max_position_embeddings: u32,
    rope_theta: f64,
    rope_type: String,
    #[serde(rename = "type")]
    rope_type_alias: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTextConfig {
    attention_dropout: f64,
    head_dim: u32,
    hidden_act: String,
    hidden_size: u32,
    initializer_range: f64,
    intermediate_size: u32,
    max_position_embeddings: u32,
    model_type: String,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    rms_norm_eps: f64,
    rope_parameters: RawRopeParameters,
    sliding_window: Option<u32>,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVisionRopeParameters {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVisionConfig {
    attention_dropout: f64,
    head_dim: u32,
    hidden_act: String,
    hidden_size: u32,
    image_size: u32,
    initializer_range: f64,
    intermediate_size: u32,
    model_type: String,
    num_attention_heads: u32,
    num_channels: u32,
    num_hidden_layers: u32,
    patch_size: u32,
    rope_parameters: RawVisionRopeParameters,
    rope_theta: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    architectures: Vec<String>,
    dtype: String,
    image_token_index: u32,
    model_type: String,
    multimodal_projector_bias: bool,
    projector_hidden_act: String,
    spatial_merge_size: u32,
    text_config: RawTextConfig,
    transformers_version: String,
    vision_config: RawVisionConfig,
    vision_feature_layer: i32,
}

fn validate_config_document(raw: RawConfig) -> Result<Ministral3Config, Ministral3Error> {
    ensure(
        raw.architectures == ["Mistral3ForConditionalGeneration"],
        "root architecture changed",
    )?;
    ensure(raw.dtype == "bfloat16", "root dtype changed")?;
    ensure(raw.model_type == "mistral3", "root model type changed")?;
    ensure(raw.image_token_index == 10, "image token index changed")?;
    ensure(
        !raw.multimodal_projector_bias,
        "multimodal projector bias changed",
    )?;
    ensure(
        raw.projector_hidden_act == "gelu",
        "projector activation changed",
    )?;
    ensure(raw.spatial_merge_size == 2, "spatial merge size changed")?;
    ensure(
        raw.transformers_version == "5.0.0.dev0",
        "Transformers version changed",
    )?;
    ensure(
        raw.vision_feature_layer == -1,
        "vision feature layer changed",
    )?;

    let text = raw.text_config;
    ensure(
        text.attention_dropout.to_bits() == 0,
        "text attention dropout changed",
    )?;
    ensure(text.head_dim == 128, "text head dimension changed")?;
    ensure(text.hidden_act == "silu", "text activation changed")?;
    ensure(
        text.hidden_size == MINISTRAL3_TEXT_HIDDEN_SIZE,
        "text hidden size changed",
    )?;
    ensure(
        same_f64(text.initializer_range, 0.02),
        "text initializer range changed",
    )?;
    ensure(
        text.intermediate_size == MINISTRAL3_TEXT_FFN_SIZE,
        "text FFN size changed",
    )?;
    ensure(
        text.max_position_embeddings == MINISTRAL3_CONTEXT_LENGTH,
        "text context length changed",
    )?;
    ensure(text.model_type == "ministral3", "text model type changed")?;
    ensure(
        text.num_attention_heads == 32,
        "text attention head count changed",
    )?;
    ensure(
        text.num_hidden_layers == MINISTRAL3_TEXT_LAYER_COUNT,
        "text layer count changed",
    )?;
    ensure(text.num_key_value_heads == 8, "text KV head count changed")?;
    ensure(
        same_f64(text.rms_norm_eps, 1e-5),
        "text RMS epsilon changed",
    )?;
    ensure(text.sliding_window.is_none(), "text sliding window changed")?;
    ensure(text.tie_word_embeddings, "text embeddings must remain tied")?;
    ensure(text.use_cache, "text cache setting changed")?;
    ensure(
        text.vocab_size == MINISTRAL3_VOCAB_SIZE,
        "text vocabulary size changed",
    )?;
    let rope = text.rope_parameters;
    ensure(same_f64(rope.beta_fast, 32.0), "YaRN beta_fast changed")?;
    ensure(same_f64(rope.beta_slow, 1.0), "YaRN beta_slow changed")?;
    ensure(same_f64(rope.factor, 16.0), "YaRN factor changed")?;
    ensure(
        same_f64(rope.llama_4_scaling_beta, 0.1),
        "Llama 4 scaling beta changed",
    )?;
    ensure(same_f64(rope.mscale, 1.0), "YaRN mscale changed")?;
    ensure(
        same_f64(rope.mscale_all_dim, 1.0),
        "YaRN mscale_all_dim changed",
    )?;
    ensure(
        rope.original_max_position_embeddings == 16_384,
        "YaRN original context changed",
    )?;
    ensure(
        same_f64(rope.rope_theta, 1_000_000.0),
        "text RoPE theta changed",
    )?;
    ensure(rope.rope_type == "yarn", "text RoPE type changed")?;
    ensure(
        rope.rope_type_alias == "yarn",
        "text RoPE type alias changed",
    )?;

    let vision = raw.vision_config;
    ensure(
        vision.attention_dropout.to_bits() == 0,
        "vision attention dropout changed",
    )?;
    ensure(vision.head_dim == 64, "vision head dimension changed")?;
    ensure(vision.hidden_act == "silu", "vision activation changed")?;
    ensure(vision.hidden_size == 1_024, "vision hidden size changed")?;
    ensure(vision.image_size == 1_540, "vision image size changed")?;
    ensure(
        same_f64(vision.initializer_range, 0.02),
        "vision initializer range changed",
    )?;
    ensure(vision.intermediate_size == 4_096, "vision FFN size changed")?;
    ensure(vision.model_type == "pixtral", "vision model type changed")?;
    ensure(
        vision.num_attention_heads == 16,
        "vision head count changed",
    )?;
    ensure(vision.num_channels == 3, "vision channel count changed")?;
    ensure(
        vision.num_hidden_layers == MINISTRAL3_VISION_LAYER_COUNT,
        "vision layer count changed",
    )?;
    ensure(vision.patch_size == 14, "vision patch size changed")?;
    ensure(
        same_f64(vision.rope_theta, 10_000.0),
        "vision RoPE theta changed",
    )?;
    ensure(
        vision.rope_parameters.rope_type == "default",
        "vision RoPE type changed",
    )?;
    ensure(
        same_f64(vision.rope_parameters.rope_theta, vision.rope_theta),
        "vision RoPE theta fields diverged",
    )?;

    Ok(Ministral3Config {
        text: Ministral3TextConfig {
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            vocab_size: text.vocab_size,
            max_position_embeddings: text.max_position_embeddings,
            rms_norm_eps: text.rms_norm_eps,
            rope: Ministral3RopeParameters {
                beta_fast: rope.beta_fast,
                beta_slow: rope.beta_slow,
                factor: rope.factor,
                llama_4_scaling_beta: rope.llama_4_scaling_beta,
                mscale: rope.mscale,
                mscale_all_dim: rope.mscale_all_dim,
                original_max_position_embeddings: rope.original_max_position_embeddings,
                rope_theta: rope.rope_theta,
                rope_type: "yarn",
            },
            tie_word_embeddings: text.tie_word_embeddings,
            use_cache: text.use_cache,
        },
        vision: Ministral3VisionConfig {
            hidden_size: vision.hidden_size,
            num_attention_heads: vision.num_attention_heads,
            num_hidden_layers: vision.num_hidden_layers,
            intermediate_size: vision.intermediate_size,
            patch_size: vision.patch_size,
            image_size: vision.image_size,
            num_channels: vision.num_channels,
            head_dim: vision.head_dim,
            rope_theta: vision.rope_theta,
            rope_type: "default",
        },
        image_token_index: raw.image_token_index,
        spatial_merge_size: raw.spatial_merge_size,
        vision_feature_layer: raw.vision_feature_layer,
        multimodal_projector_bias: raw.multimodal_projector_bias,
    })
}

/// Validate the exact official config bytes and return the typed contract.
pub fn validate_ministral3_config(bytes: &[u8]) -> Result<Ministral3Config, Ministral3Error> {
    locked_bytes(
        bytes,
        MINISTRAL3_CONFIG_BYTES,
        MINISTRAL3_CONFIG_SHA256,
        "config",
    )?;
    let raw: RawConfig = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("config JSON is not strict-valid: {error}")))?;
    validate_config_document(raw)
}

pub fn ministral3_shard(file_name: &str) -> Option<Ministral3ShardIdentity> {
    if file_name.is_empty()
        || file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
    {
        return None;
    }
    MINISTRAL3_SHARDS
        .iter()
        .copied()
        .find(|shard| shard.file_name == file_name)
}

/// Validate Hub-reported full-shard metadata without pretending that the
/// payload was downloaded locally.
pub fn validate_ministral3_shard_lfs_identity(
    file_name: &str,
    file_size: u64,
    lfs_sha256: &str,
) -> Result<Ministral3ShardIdentity, Ministral3Error> {
    let identity = ministral3_shard(file_name)
        .ok_or_else(|| invalid(format!("unknown or unsafe shard: {file_name}")))?;
    ensure(
        file_size == identity.file_size,
        format!("shard file size differs: {file_name}"),
    )?;
    ensure(
        lfs_sha256 == identity.lfs_sha256,
        format!("shard LFS SHA-256 differs: {file_name}"),
    )?;
    Ok(identity)
}

pub fn ministral3_support_file(file_name: &str) -> Option<Ministral3SupportFileIdentity> {
    if file_name.is_empty()
        || file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
    {
        return None;
    }
    MINISTRAL3_SUPPORT_FILES
        .iter()
        .copied()
        .find(|file| file.file_name == file_name)
}

pub fn validate_ministral3_support_file(
    file_name: &str,
    bytes: &[u8],
) -> Result<Ministral3SupportFileIdentity, Ministral3Error> {
    let identity = ministral3_support_file(file_name)
        .ok_or_else(|| invalid(format!("unknown or unsafe support file: {file_name}")))?;
    locked_bytes(bytes, identity.size, identity.sha256, file_name)?;
    Ok(identity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_identity_constants_are_consistent() {
        assert_eq!(MINISTRAL3_SHARDS.len(), 2);
        assert_eq!(
            MINISTRAL3_SHARDS.iter().map(|s| s.file_size).sum::<u64>(),
            MINISTRAL3_SHARD_FILE_BYTES
        );
        assert_eq!(
            MINISTRAL3_SHARDS
                .iter()
                .map(|s| s.header_length + 8)
                .sum::<u64>(),
            MINISTRAL3_HEADER_BYTES
        );
        assert_eq!(
            MINISTRAL3_SHARDS
                .iter()
                .map(|s| s.indexed_tensor_count)
                .sum::<usize>(),
            MINISTRAL3_TENSOR_COUNT
        );
        assert_eq!(MINISTRAL3_LICENSE, "Apache-2.0");
    }

    #[test]
    fn path_traversal_and_unknown_support_are_rejected() {
        assert!(ministral3_shard("../model-00001-of-00002.safetensors").is_none());
        assert!(ministral3_shard("model-00003-of-00002.safetensors").is_none());
        assert!(ministral3_support_file("../config.json").is_none());
        assert!(validate_ministral3_support_file("missing.json", &[]).is_err());
        let identity = MINISTRAL3_SHARDS[0];
        assert!(
            validate_ministral3_shard_lfs_identity(
                identity.file_name,
                identity.file_size,
                identity.lfs_sha256,
            )
            .is_ok()
        );
        assert!(
            validate_ministral3_shard_lfs_identity(
                identity.file_name,
                identity.file_size - 1,
                identity.lfs_sha256,
            )
            .is_err()
        );
    }

    #[test]
    fn strict_config_shape_and_unknown_key_contract_is_fail_closed() {
        // The fixed-size/hash gate rejects synthetic or truncated config data.
        assert!(validate_ministral3_config(b"{}").is_err());
        let mut unknown = vec![b' '; MINISTRAL3_CONFIG_BYTES];
        unknown[0] = b'{';
        assert!(validate_ministral3_config(&unknown).is_err());
    }
}
