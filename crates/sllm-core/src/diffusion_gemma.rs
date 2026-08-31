//! Reviewed host-only foundation for the fixed DiffusionGemma artifact.
//!
//! The shard identities are Git LFS identities reported by the fixed-revision
//! Hub API. They are not evidence that the 51.6 GB payload was downloaded and
//! hashed locally. Small support files can be validated independently through
//! [`validate_diffusion_gemma_support_file`].

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DIFFUSION_GEMMA_REPOSITORY: &str = "google/diffusiongemma-26B-A4B-it";
pub const DIFFUSION_GEMMA_REVISION: &str = "f7f5b7f5fa82ffc52addd066915886d497f5517b";
pub const DIFFUSION_GEMMA_LICENSE: &str = "Apache-2.0";
pub const DIFFUSION_GEMMA_CONFIG_BYTES: usize = 3_469;
pub const DIFFUSION_GEMMA_CONFIG_SHA256: &str =
    "13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506";
pub const DIFFUSION_GEMMA_INDEX_BYTES: usize = 104_650;
pub const DIFFUSION_GEMMA_INDEX_SHA256: &str =
    "6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a";
pub const DIFFUSION_GEMMA_CATALOG_SHA256: &str =
    "1f3a74edcf1578781417eb810fd4aefcb874f97b39a3f510b8a73d2af664b08a";
pub const DIFFUSION_GEMMA_SHARD_COUNT: usize = 11;
pub const DIFFUSION_GEMMA_TENSOR_COUNT: usize = 1_047;
pub const DIFFUSION_GEMMA_TOTAL_PARAMETERS: u64 = 25_823_778_864;
pub const DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES: u64 = 51_647_562_456;
pub const DIFFUSION_GEMMA_SHARD_FILE_BYTES: u64 = 51_647_701_024;
pub const DIFFUSION_GEMMA_MANIFEST_DELTA_BYTES: u64 = 138_568;
pub const DIFFUSION_GEMMA_CAPACITY_ADMISSION_BYTES: u64 =
    if DIFFUSION_GEMMA_SHARD_FILE_BYTES > DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES {
        DIFFUSION_GEMMA_SHARD_FILE_BYTES
    } else {
        DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES
    };
pub const DIFFUSION_GEMMA_TEXT_LAYER_COUNT: u32 = 30;
pub const DIFFUSION_GEMMA_VISION_LAYER_COUNT: u32 = 27;
pub const DIFFUSION_GEMMA_EXPERT_COUNT: u32 = 128;
pub const DIFFUSION_GEMMA_SELECTED_EXPERT_COUNT: u32 = 8;
pub const DIFFUSION_GEMMA_CANVAS_LENGTH: u32 = 256;

const DECODER_EMBEDDING_TENSOR_COUNT: usize = 1;
const DECODER_NORM_TENSOR_COUNT: usize = 1;
const SELF_CONDITIONING_TENSOR_COUNT: usize = 4;
const DECODER_LAYER_TENSOR_COUNT: usize = 655;
const ENCODER_PROJECTION_TENSOR_COUNT: usize = 1;
const ENCODER_LAYER_SCALAR_TENSOR_COUNT: usize = 30;
const VISION_ROOT_TENSOR_COUNT: usize = 4;
const VISION_LAYER_TENSOR_COUNT: usize = 351;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaShardIdentity {
    pub file_name: &'static str,
    pub size: u64,
    /// Git LFS SHA-256 OID reported by the fixed-revision Hub API.
    pub lfs_sha256: &'static str,
    pub indexed_tensor_count: usize,
}

pub const DIFFUSION_GEMMA_SHARDS: [DiffusionGemmaShardIdentity; DIFFUSION_GEMMA_SHARD_COUNT] = [
    DiffusionGemmaShardIdentity {
        file_name: "model-00001-of-00011.safetensors",
        size: 4_732_780_476,
        lfs_sha256: "3efe137998af7d2bde4e3ab04ab3524823699a4ac3130adace5003ef40cceeb6",
        indexed_tensor_count: 45,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00002-of-00011.safetensors",
        size: 4_884_578_006,
        lfs_sha256: "4a39d68c756fb26bbd2a54f2b8d550047ea98f3152f87ec75db825c8e17934a7",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00003-of-00011.safetensors",
        size: 4_913_414_742,
        lfs_sha256: "ac6083e3489215ca032501714b78832e5cc4c945a8dbbb905b5292a4e95bc75e",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00004-of-00011.safetensors",
        size: 4_884_578_030,
        lfs_sha256: "865b66393de5a9c752e67beae1bd2c860c12786b72ad86ca9345c00b7b586e60",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00005-of-00011.safetensors",
        size: 4_913_414_814,
        lfs_sha256: "a87e01bed77ad9d2234851267d99af583ff79c7a86d7494fe08ae0f9de9cd318",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00006-of-00011.safetensors",
        size: 4_884_578_070,
        lfs_sha256: "077e841b3b138fbc38df2c36665abdaafa96b8c96f968e2263a7452afaa912ab",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00007-of-00011.safetensors",
        size: 4_913_414_814,
        lfs_sha256: "13a18b3c04a7f19a16385dd8a0d7fd0b9cc89ef131e4294a063c31c7d90c58ef",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00008-of-00011.safetensors",
        size: 4_884_578_070,
        lfs_sha256: "aca5d3bdfc84700bd55b475781cb46e5608104c4832d2abe111a119ddcb23ff7",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00009-of-00011.safetensors",
        size: 4_913_414_814,
        lfs_sha256: "8e2418867354e5cb356c0af8ccfdcc60bc363500674d6cf395991e7d6219eb29",
        indexed_tensor_count: 65,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00010-of-00011.safetensors",
        size: 4_884_578_070,
        lfs_sha256: "93d564b7dd686464a5c068ff9665cd5d3bca399c2ce320aecd41bd011e3787d5",
        indexed_tensor_count: 66,
    },
    DiffusionGemmaShardIdentity {
        file_name: "model-00011-of-00011.safetensors",
        size: 2_838_371_118,
        lfs_sha256: "afec047176bb2a05f078566576aec6bdb71ad4d041275d0d0c89473fda6d6d87",
        indexed_tensor_count: 412,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaSupportFileRole {
    ModelCard,
    ChatTemplate,
    ModelConfig,
    GenerationConfig,
    SafetensorsIndex,
    PipelineIndex,
    ProcessorConfig,
    SchedulerConfig,
    TokenizerModel,
    TokenizerConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaSupportFileIdentity {
    pub file_name: &'static str,
    pub role: DiffusionGemmaSupportFileRole,
    pub size: usize,
    pub sha256: &'static str,
}

pub const DIFFUSION_GEMMA_SUPPORT_FILES: [DiffusionGemmaSupportFileIdentity; 10] = [
    DiffusionGemmaSupportFileIdentity {
        file_name: "README.md",
        role: DiffusionGemmaSupportFileRole::ModelCard,
        size: 19_722,
        sha256: "3acbbf7e4c01297291113b1297732b6bcecf0d7769a2571fefeca4f0f5909747",
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "chat_template.jinja",
        role: DiffusionGemmaSupportFileRole::ChatTemplate,
        size: 18_575,
        sha256: "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf",
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "config.json",
        role: DiffusionGemmaSupportFileRole::ModelConfig,
        size: DIFFUSION_GEMMA_CONFIG_BYTES,
        sha256: DIFFUSION_GEMMA_CONFIG_SHA256,
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "generation_config.json",
        role: DiffusionGemmaSupportFileRole::GenerationConfig,
        size: 357,
        sha256: "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200",
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "model.safetensors.index.json",
        role: DiffusionGemmaSupportFileRole::SafetensorsIndex,
        size: DIFFUSION_GEMMA_INDEX_BYTES,
        sha256: DIFFUSION_GEMMA_INDEX_SHA256,
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "model_index.json",
        role: DiffusionGemmaSupportFileRole::PipelineIndex,
        size: 295,
        sha256: "989c453462f9b84b90f02fe82b89f23c743b513a767a1789292da76f056f5117",
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "processor_config.json",
        role: DiffusionGemmaSupportFileRole::ProcessorConfig,
        size: 1_689,
        sha256: "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "scheduler/scheduler_config.json",
        role: DiffusionGemmaSupportFileRole::SchedulerConfig,
        size: 209,
        sha256: "5e5536ea036c284bcc09c7b08045868d741a41a3090069f6c035697bba31c6c7",
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "tokenizer.json",
        role: DiffusionGemmaSupportFileRole::TokenizerModel,
        size: 32_169_626,
        sha256: "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
    DiffusionGemmaSupportFileIdentity {
        file_name: "tokenizer_config.json",
        role: DiffusionGemmaSupportFileRole::TokenizerConfig,
        size: 2_741,
        sha256: "a284d1243b62be31faa9c13e1c28cece940c4abaa7bd9ad87b94f61b40687200",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaArtifactEvidence {
    pub hub_shard_lfs_identity_fixed: bool,
    pub local_full_shard_payload_sha256_verified: bool,
}

pub const DIFFUSION_GEMMA_ARTIFACT_EVIDENCE: DiffusionGemmaArtifactEvidence =
    DiffusionGemmaArtifactEvidence {
        hub_shard_lfs_identity_fixed: true,
        local_full_shard_payload_sha256_verified: false,
    };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaTextLayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffusionGemmaRopeConfig {
    pub sliding_theta: f64,
    pub full_theta: f64,
    pub full_partial_rotary_factor: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaTextConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub moe_intermediate_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub sliding_kv_heads: u32,
    pub global_kv_heads: u32,
    pub head_dim: u32,
    pub global_head_dim: u32,
    pub expert_count: u32,
    pub selected_expert_count: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub sliding_window: u32,
    pub rms_norm_epsilon: f64,
    pub layer_types: Vec<DiffusionGemmaTextLayerType>,
    pub rope: DiffusionGemmaRopeConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaVisionConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub max_position_embeddings: u32,
    pub patch_size: u32,
    pub pooling_kernel_size: u32,
    pub position_embedding_size: u32,
    pub soft_tokens_per_image: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaConfig {
    pub canvas_length: u32,
    pub beginning_of_image_token_id: u32,
    pub end_of_image_token_id: u32,
    pub image_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    pub text: DiffusionGemmaTextConfig,
    pub vision: DiffusionGemmaVisionConfig,
    pub production_loader_enabled: bool,
    pub autoregressive_fallback_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DiffusionGemmaTensorSummary {
    pub decoder_embedding: usize,
    pub decoder_norm: usize,
    pub self_conditioning: usize,
    pub decoder_layers: usize,
    pub encoder_projection: usize,
    pub encoder_layer_scalars: usize,
    pub vision_root: usize,
    pub vision_layers: usize,
}

impl DiffusionGemmaTensorSummary {
    pub fn checked_total(self) -> Result<usize, DiffusionGemmaError> {
        self.decoder_embedding
            .checked_add(self.decoder_norm)
            .and_then(|value| value.checked_add(self.self_conditioning))
            .and_then(|value| value.checked_add(self.decoder_layers))
            .and_then(|value| value.checked_add(self.encoder_projection))
            .and_then(|value| value.checked_add(self.encoder_layer_scalars))
            .and_then(|value| value.checked_add(self.vision_root))
            .and_then(|value| value.checked_add(self.vision_layers))
            .ok_or_else(|| invalid("tensor family count overflowed"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaTensorClass {
    DecoderEmbedding,
    DecoderNorm,
    SelfConditioning,
    DecoderLayer,
    EncoderProjection,
    EncoderLayerScalar,
    VisionRoot,
    VisionLayer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaManifestState {
    Consistent {
        bytes: u64,
    },
    IndexAdvertisedExceedsShardFiles {
        index_advertised_bytes: u64,
        shard_file_bytes: u64,
        delta_bytes: u64,
    },
    ShardFilesExceedIndexAdvertised {
        index_advertised_bytes: u64,
        shard_file_bytes: u64,
        delta_bytes: u64,
    },
}

impl DiffusionGemmaManifestState {
    pub const fn admission_base_bytes(self) -> u64 {
        match self {
            Self::Consistent { bytes } => bytes,
            Self::IndexAdvertisedExceedsShardFiles {
                index_advertised_bytes,
                shard_file_bytes,
                delta_bytes: _,
            }
            | Self::ShardFilesExceedIndexAdvertised {
                index_advertised_bytes,
                shard_file_bytes,
                delta_bytes: _,
            } => {
                if index_advertised_bytes > shard_file_bytes {
                    index_advertised_bytes
                } else {
                    shard_file_bytes
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaIndex {
    total_parameters: u64,
    index_advertised_bytes: u64,
    shard_file_bytes: u64,
    manifest_state: DiffusionGemmaManifestState,
    catalog_sha256: String,
    summary: DiffusionGemmaTensorSummary,
    weight_map: BTreeMap<String, String>,
}

impl DiffusionGemmaIndex {
    pub const fn total_parameters(&self) -> u64 {
        self.total_parameters
    }

    pub const fn index_advertised_bytes(&self) -> u64 {
        self.index_advertised_bytes
    }

    pub const fn shard_file_bytes(&self) -> u64 {
        self.shard_file_bytes
    }

    pub const fn manifest_state(&self) -> DiffusionGemmaManifestState {
        self.manifest_state
    }

    pub const fn summary(&self) -> DiffusionGemmaTensorSummary {
        self.summary
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn source_file(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    pub fn checked_admission_bytes(
        &self,
        resident_copy_count: u64,
        additional_bytes: u64,
    ) -> Result<u64, DiffusionGemmaError> {
        if resident_copy_count == 0 {
            return Err(invalid("resident copy count must be nonzero"));
        }
        self.manifest_state
            .admission_base_bytes()
            .checked_mul(resident_copy_count)
            .and_then(|value| value.checked_add(additional_bytes))
            .ok_or_else(|| invalid("capacity admission byte count overflowed"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaCapacityDecision {
    pub required_bytes: u64,
    pub available_bytes: u64,
    pub fits: bool,
    pub shortfall_bytes: u64,
    pub manifest_state: DiffusionGemmaManifestState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaError {
    Invalid(String),
}

impl fmt::Display for DiffusionGemmaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(formatter, "invalid DiffusionGemma artifact: {message}")
            }
        }
    }
}

impl std::error::Error for DiffusionGemmaError {}

fn invalid(message: impl Into<String>) -> DiffusionGemmaError {
    DiffusionGemmaError::Invalid(message.into())
}

fn ensure(condition: bool, message: impl Into<String>) -> Result<(), DiffusionGemmaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn same_f64(actual: f64, expected: f64) -> bool {
    actual.to_bits() == expected.to_bits()
}

fn validate_locked_document(
    bytes: &[u8],
    expected_size: usize,
    expected_sha256: &str,
    label: &str,
) -> Result<(), DiffusionGemmaError> {
    ensure(
        bytes.len() == expected_size,
        format!(
            "{label} byte length {} does not match reviewed {expected_size}",
            bytes.len()
        ),
    )?;
    let actual = sha256_hex(bytes);
    ensure(
        actual == expected_sha256,
        format!("{label} SHA-256 {actual} does not match reviewed {expected_sha256}"),
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTextFullRope {
    partial_rotary_factor: f64,
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTextSlidingRope {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTextRopeParameters {
    full_attention: RawTextFullRope,
    sliding_attention: RawTextSlidingRope,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTextConfig {
    attention_bias: bool,
    attention_dropout: f64,
    bos_token_id: u32,
    dtype: String,
    eos_token_id: u32,
    final_logit_softcapping: f64,
    global_head_dim: u32,
    head_dim: u32,
    hidden_activation: String,
    hidden_size: u32,
    initializer_range: f64,
    intermediate_size: u32,
    layer_types: Vec<String>,
    max_position_embeddings: u32,
    model_type: String,
    moe_intermediate_size: u32,
    num_attention_heads: u32,
    num_experts: u32,
    num_global_key_value_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    pad_token_id: u32,
    rms_norm_eps: f64,
    rope_parameters: RawTextRopeParameters,
    sliding_window: u32,
    tie_word_embeddings: bool,
    top_k_experts: u32,
    use_bidirectional_attention: String,
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
    #[serde(rename = "_name_or_path")]
    name_or_path: String,
    architectures: serde_json::Value,
    attention_bias: bool,
    attention_dropout: f64,
    chunk_size_feed_forward: u32,
    default_output_length: u32,
    dtype: String,
    global_head_dim: u32,
    head_dim: u32,
    hidden_activation: String,
    hidden_size: u32,
    id2label: BTreeMap<String, String>,
    initializer_range: f64,
    intermediate_size: u32,
    is_encoder_decoder: bool,
    label2id: BTreeMap<String, u32>,
    max_position_embeddings: u32,
    model_type: String,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    output_attentions: bool,
    output_hidden_states: bool,
    patch_size: u32,
    pooling_kernel_size: u32,
    position_embedding_size: u32,
    problem_type: serde_json::Value,
    return_dict: bool,
    rms_norm_eps: f64,
    rope_parameters: RawVisionRopeParameters,
    standardize: bool,
    use_clipped_linears: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    architectures: Vec<String>,
    boi_token_id: u32,
    canvas_length: u32,
    dtype: String,
    eoi_token_id: u32,
    eos_token_id: Vec<u32>,
    image_token_id: u32,
    initializer_range: f64,
    model_type: String,
    text_config: RawTextConfig,
    tie_word_embeddings: bool,
    transformers_version: String,
    vision_config: RawVisionConfig,
    vision_soft_tokens_per_image: u32,
}

fn reviewed_layer_type(layer: u32) -> DiffusionGemmaTextLayerType {
    if layer % 6 == 5 {
        DiffusionGemmaTextLayerType::FullAttention
    } else {
        DiffusionGemmaTextLayerType::SlidingAttention
    }
}

fn checked_product(values: &[u64], label: &str) -> Result<u64, DiffusionGemmaError> {
    ensure(!values.is_empty(), format!("{label} shape is empty"))?;
    values.iter().try_fold(1_u64, |product, &value| {
        ensure(
            value != 0,
            format!("{label} shape contains a zero dimension"),
        )?;
        product
            .checked_mul(value)
            .ok_or_else(|| invalid(format!("{label} shape element count overflowed")))
    })
}

fn validate_config_document(raw: RawConfig) -> Result<DiffusionGemmaConfig, DiffusionGemmaError> {
    ensure(
        raw.architectures == ["DiffusionGemmaForBlockDiffusion"],
        "root architecture changed",
    )?;
    ensure(
        raw.model_type == "diffusion_gemma",
        "root model type changed",
    )?;
    ensure(raw.dtype == "bfloat16", "root dtype changed")?;
    ensure(
        same_f64(raw.initializer_range, 0.02),
        "root initializer range changed",
    )?;
    ensure(
        raw.canvas_length == DIFFUSION_GEMMA_CANVAS_LENGTH,
        "canvas length changed",
    )?;
    ensure(
        raw.boi_token_id == 255_999,
        "beginning-of-image token changed",
    )?;
    ensure(raw.eoi_token_id == 258_882, "end-of-image token changed")?;
    ensure(raw.image_token_id == 258_880, "image token changed")?;
    ensure(raw.eos_token_id == [1, 106], "root EOS token set changed")?;
    ensure(raw.tie_word_embeddings, "root embeddings must remain tied")?;
    ensure(
        raw.transformers_version == "5.8.0.dev0",
        "Transformers version identity changed",
    )?;
    ensure(
        raw.vision_soft_tokens_per_image == 280,
        "vision soft-token count changed",
    )?;

    let text = raw.text_config;
    ensure(
        !text.attention_bias,
        "text attention bias must remain disabled",
    )?;
    ensure(
        same_f64(text.attention_dropout, 0.0),
        "text attention dropout changed",
    )?;
    ensure(text.bos_token_id == 2, "text BOS token changed")?;
    ensure(text.eos_token_id == 1, "text EOS token changed")?;
    ensure(text.pad_token_id == 0, "text padding token changed")?;
    ensure(text.dtype == "bfloat16", "text dtype changed")?;
    ensure(
        same_f64(text.final_logit_softcapping, 30.0),
        "final logit soft cap changed",
    )?;
    ensure(text.global_head_dim == 512, "global head dimension changed")?;
    ensure(text.head_dim == 256, "sliding head dimension changed")?;
    ensure(
        text.hidden_activation == "gelu_pytorch_tanh",
        "text activation changed",
    )?;
    ensure(text.hidden_size == 2_816, "text hidden size changed")?;
    ensure(
        same_f64(text.initializer_range, 0.02),
        "text initializer range changed",
    )?;
    ensure(
        text.intermediate_size == 2_112,
        "text intermediate size changed",
    )?;
    ensure(
        text.max_position_embeddings == 262_144,
        "text context length changed",
    )?;
    ensure(
        text.model_type == "diffusion_gemma_text",
        "text model type changed",
    )?;
    ensure(
        text.moe_intermediate_size == 704,
        "MoE intermediate size changed",
    )?;
    ensure(
        text.num_attention_heads == 16,
        "text attention head count changed",
    )?;
    ensure(
        text.num_experts == DIFFUSION_GEMMA_EXPERT_COUNT,
        "expert count changed",
    )?;
    ensure(
        text.num_global_key_value_heads == 2,
        "global KV head count changed",
    )?;
    ensure(
        text.num_hidden_layers == DIFFUSION_GEMMA_TEXT_LAYER_COUNT,
        "text layer count changed",
    )?;
    ensure(
        text.num_key_value_heads == 8,
        "sliding KV head count changed",
    )?;
    ensure(
        same_f64(text.rms_norm_eps, 1e-6),
        "text RMS epsilon changed",
    )?;
    ensure(text.sliding_window == 1_024, "sliding window changed")?;
    ensure(text.tie_word_embeddings, "text embeddings must remain tied")?;
    ensure(
        text.top_k_experts == DIFFUSION_GEMMA_SELECTED_EXPERT_COUNT,
        "selected expert count changed",
    )?;
    ensure(
        text.use_bidirectional_attention == "vision",
        "bidirectional attention mode changed",
    )?;
    ensure(text.vocab_size == 262_144, "text vocabulary size changed")?;
    ensure(
        text.layer_types.len() == DIFFUSION_GEMMA_TEXT_LAYER_COUNT as usize,
        "text layer schedule length changed",
    )?;
    let mut layer_types = Vec::with_capacity(text.layer_types.len());
    for (layer, actual) in text.layer_types.iter().enumerate() {
        let expected = reviewed_layer_type(layer as u32);
        let expected_name = match expected {
            DiffusionGemmaTextLayerType::SlidingAttention => "sliding_attention",
            DiffusionGemmaTextLayerType::FullAttention => "full_attention",
        };
        ensure(
            actual == expected_name,
            format!("text layer {layer} type {actual} does not match {expected_name}"),
        )?;
        layer_types.push(expected);
    }
    let rope = text.rope_parameters;
    ensure(
        same_f64(rope.full_attention.partial_rotary_factor, 0.25),
        "full-attention partial rotary factor changed",
    )?;
    ensure(
        same_f64(rope.full_attention.rope_theta, 1_000_000.0),
        "full-attention RoPE theta changed",
    )?;
    ensure(
        rope.full_attention.rope_type == "proportional",
        "full-attention RoPE type changed",
    )?;
    ensure(
        same_f64(rope.sliding_attention.rope_theta, 10_000.0),
        "sliding-attention RoPE theta changed",
    )?;
    ensure(
        rope.sliding_attention.rope_type == "default",
        "sliding-attention RoPE type changed",
    )?;
    ensure(
        text.num_attention_heads % text.num_key_value_heads == 0,
        "text heads are not divisible by sliding KV heads",
    )?;
    ensure(
        text.num_attention_heads % text.num_global_key_value_heads == 0,
        "text heads are not divisible by global KV heads",
    )?;
    ensure(
        text.top_k_experts <= text.num_experts,
        "selected expert count exceeds expert count",
    )?;
    ensure(
        raw.canvas_length <= text.max_position_embeddings,
        "canvas length exceeds text context length",
    )?;
    ensure(
        [raw.boi_token_id, raw.eoi_token_id, raw.image_token_id]
            .iter()
            .all(|&token| token < text.vocab_size),
        "special image token exceeds vocabulary",
    )?;
    checked_product(
        &[
            u64::from(text.num_experts),
            u64::from(text.moe_intermediate_size),
            u64::from(text.hidden_size),
        ],
        "MoE projection",
    )?;
    checked_product(
        &[
            u64::from(text.num_attention_heads),
            u64::from(text.head_dim),
        ],
        "sliding query projection",
    )?;

    let vision = raw.vision_config;
    ensure(
        vision.name_or_path.is_empty(),
        "vision name-or-path changed",
    )?;
    ensure(
        vision.architectures.is_null(),
        "vision architectures must remain null",
    )?;
    ensure(
        !vision.attention_bias,
        "vision attention bias must remain disabled",
    )?;
    ensure(
        same_f64(vision.attention_dropout, 0.0),
        "vision attention dropout changed",
    )?;
    ensure(
        vision.chunk_size_feed_forward == 0,
        "vision feed-forward chunk changed",
    )?;
    ensure(
        vision.default_output_length == 280,
        "vision default output length changed",
    )?;
    ensure(vision.dtype == "bfloat16", "vision dtype changed")?;
    ensure(
        vision.global_head_dim == 72,
        "vision global head dimension changed",
    )?;
    ensure(vision.head_dim == 72, "vision head dimension changed")?;
    ensure(
        vision.hidden_activation == "gelu_pytorch_tanh",
        "vision activation changed",
    )?;
    ensure(vision.hidden_size == 1_152, "vision hidden size changed")?;
    ensure(
        vision.id2label
            == BTreeMap::from([
                ("0".to_owned(), "LABEL_0".to_owned()),
                ("1".to_owned(), "LABEL_1".to_owned()),
            ]),
        "vision id-to-label mapping changed",
    )?;
    ensure(
        same_f64(vision.initializer_range, 0.02),
        "vision initializer range changed",
    )?;
    ensure(
        vision.intermediate_size == 4_304,
        "vision intermediate size changed",
    )?;
    ensure(
        !vision.is_encoder_decoder,
        "vision encoder-decoder flag changed",
    )?;
    ensure(
        vision.label2id == BTreeMap::from([("LABEL_0".to_owned(), 0), ("LABEL_1".to_owned(), 1)]),
        "vision label-to-id mapping changed",
    )?;
    ensure(
        vision.max_position_embeddings == 131_072,
        "vision context length changed",
    )?;
    ensure(
        vision.model_type == "gemma4_vision",
        "vision model type changed",
    )?;
    ensure(
        vision.num_attention_heads == 16,
        "vision attention head count changed",
    )?;
    ensure(
        vision.num_hidden_layers == DIFFUSION_GEMMA_VISION_LAYER_COUNT,
        "vision layer count changed",
    )?;
    ensure(
        vision.num_key_value_heads == 16,
        "vision KV head count changed",
    )?;
    ensure(
        !vision.output_attentions,
        "vision output-attention flag changed",
    )?;
    ensure(
        !vision.output_hidden_states,
        "vision output-hidden-state flag changed",
    )?;
    ensure(vision.patch_size == 16, "vision patch size changed")?;
    ensure(
        vision.pooling_kernel_size == 3,
        "vision pooling kernel changed",
    )?;
    ensure(
        vision.position_embedding_size == 10_240,
        "vision position embedding size changed",
    )?;
    ensure(
        vision.problem_type.is_null(),
        "vision problem type must remain null",
    )?;
    ensure(
        vision.return_dict,
        "vision return-dict flag must remain enabled",
    )?;
    ensure(
        same_f64(vision.rms_norm_eps, 1e-6),
        "vision RMS epsilon changed",
    )?;
    ensure(
        same_f64(vision.rope_parameters.rope_theta, 100.0),
        "vision RoPE theta changed",
    )?;
    ensure(
        vision.rope_parameters.rope_type == "default",
        "vision RoPE type changed",
    )?;
    ensure(
        vision.standardize,
        "vision standardization must remain enabled",
    )?;
    ensure(
        !vision.use_clipped_linears,
        "vision clipped linears must remain disabled",
    )?;
    let vision_width = vision
        .num_attention_heads
        .checked_mul(vision.head_dim)
        .ok_or_else(|| invalid("vision attention width overflowed"))?;
    ensure(
        vision_width == vision.hidden_size,
        "vision attention shape does not match hidden size",
    )?;

    Ok(DiffusionGemmaConfig {
        canvas_length: raw.canvas_length,
        beginning_of_image_token_id: raw.boi_token_id,
        end_of_image_token_id: raw.eoi_token_id,
        image_token_id: raw.image_token_id,
        eos_token_ids: raw.eos_token_id,
        text: DiffusionGemmaTextConfig {
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            moe_intermediate_size: text.moe_intermediate_size,
            layer_count: text.num_hidden_layers,
            attention_heads: text.num_attention_heads,
            sliding_kv_heads: text.num_key_value_heads,
            global_kv_heads: text.num_global_key_value_heads,
            head_dim: text.head_dim,
            global_head_dim: text.global_head_dim,
            expert_count: text.num_experts,
            selected_expert_count: text.top_k_experts,
            vocab_size: text.vocab_size,
            context_length: text.max_position_embeddings,
            sliding_window: text.sliding_window,
            rms_norm_epsilon: text.rms_norm_eps,
            layer_types,
            rope: DiffusionGemmaRopeConfig {
                sliding_theta: rope.sliding_attention.rope_theta,
                full_theta: rope.full_attention.rope_theta,
                full_partial_rotary_factor: rope.full_attention.partial_rotary_factor,
            },
        },
        vision: DiffusionGemmaVisionConfig {
            hidden_size: vision.hidden_size,
            intermediate_size: vision.intermediate_size,
            layer_count: vision.num_hidden_layers,
            attention_heads: vision.num_attention_heads,
            kv_heads: vision.num_key_value_heads,
            head_dim: vision.head_dim,
            max_position_embeddings: vision.max_position_embeddings,
            patch_size: vision.patch_size,
            pooling_kernel_size: vision.pooling_kernel_size,
            position_embedding_size: vision.position_embedding_size,
            soft_tokens_per_image: raw.vision_soft_tokens_per_image,
        },
        production_loader_enabled: false,
        autoregressive_fallback_enabled: false,
    })
}

/// Validates the exact fixed-revision config and returns a typed foundation contract.
pub fn validate_diffusion_gemma_config(
    bytes: &[u8],
) -> Result<DiffusionGemmaConfig, DiffusionGemmaError> {
    validate_locked_document(
        bytes,
        DIFFUSION_GEMMA_CONFIG_BYTES,
        DIFFUSION_GEMMA_CONFIG_SHA256,
        "config",
    )?;
    let raw: RawConfig = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("config JSON is not strict-valid: {error}")))?;
    validate_config_document(raw)
}

pub fn diffusion_gemma_support_file(
    file_name: &str,
) -> Option<&'static DiffusionGemmaSupportFileIdentity> {
    DIFFUSION_GEMMA_SUPPORT_FILES
        .iter()
        .find(|identity| identity.file_name == file_name)
}

/// Validates one small support file without implying that shard payloads exist locally.
pub fn validate_diffusion_gemma_support_file(
    file_name: &str,
    bytes: &[u8],
) -> Result<&'static DiffusionGemmaSupportFileIdentity, DiffusionGemmaError> {
    ensure(
        is_canonical_repository_path(file_name),
        format!("support file name is not a canonical relative path: {file_name}"),
    )?;
    let identity = diffusion_gemma_support_file(file_name)
        .ok_or_else(|| invalid(format!("unknown support file: {file_name}")))?;
    validate_locked_document(bytes, identity.size, identity.sha256, file_name)?;
    Ok(identity)
}

fn is_canonical_repository_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path.split('/').all(|component| {
            !component.is_empty()
                && component != "."
                && component != ".."
                && component.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b'.'
                })
        })
}

pub fn diffusion_gemma_locked_shard(
    file_name: &str,
) -> Option<&'static DiffusionGemmaShardIdentity> {
    DIFFUSION_GEMMA_SHARDS
        .iter()
        .find(|identity| identity.file_name == file_name)
}

/// Validates size and Hub LFS identity without claiming a local payload hash.
pub fn validate_diffusion_gemma_shard_lfs_identity(
    file_name: &str,
    size: u64,
    lfs_sha256: &str,
) -> Result<&'static DiffusionGemmaShardIdentity, DiffusionGemmaError> {
    ensure(
        !file_name.contains('/') && !file_name.contains('\\') && !file_name.contains(".."),
        format!("shard name is not a canonical base name: {file_name}"),
    )?;
    let identity = diffusion_gemma_locked_shard(file_name)
        .ok_or_else(|| invalid(format!("unknown shard: {file_name}")))?;
    ensure(
        identity.size == size,
        format!(
            "shard {file_name} size {size} does not match reviewed {}",
            identity.size
        ),
    )?;
    ensure(
        lfs_sha256.len() == 64
            && lfs_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            && lfs_sha256.bytes().all(|byte| !byte.is_ascii_uppercase()),
        format!("shard {file_name} LFS OID is not lowercase SHA-256"),
    )?;
    ensure(
        identity.lfs_sha256 == lfs_sha256,
        format!("shard {file_name} LFS identity changed"),
    )?;
    Ok(identity)
}

pub fn checked_diffusion_gemma_shard_file_bytes() -> Result<u64, DiffusionGemmaError> {
    DIFFUSION_GEMMA_SHARDS
        .iter()
        .try_fold(0_u64, |total, shard| {
            total
                .checked_add(shard.size)
                .ok_or_else(|| invalid("shard byte total overflowed"))
        })
}

fn parse_canonical_index(value: &str, limit: u32, label: &str) -> Result<u32, DiffusionGemmaError> {
    ensure(!value.is_empty(), format!("missing {label}"))?;
    ensure(
        value == "0" || !value.starts_with('0'),
        format!("{label} is not canonical decimal: {value}"),
    )?;
    ensure(
        value.bytes().all(|byte| byte.is_ascii_digit()),
        format!("{label} is not decimal: {value}"),
    )?;
    let parsed = value
        .parse::<u32>()
        .map_err(|_| invalid(format!("{label} overflowed: {value}")))?;
    ensure(
        parsed < limit,
        format!("{label} {parsed} is outside 0..{limit}"),
    )?;
    Ok(parsed)
}

fn validate_tensor_name_characters(name: &str) -> Result<(), DiffusionGemmaError> {
    ensure(!name.is_empty(), "tensor name is empty")?;
    ensure(
        !name.starts_with('.') && !name.ends_with('.') && !name.contains(".."),
        format!("tensor name has an empty component: {name}"),
    )?;
    ensure(
        name.bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.'),
        format!("tensor name contains a forbidden character: {name}"),
    )
}

const DECODER_LAYER_SUFFIXES: [&str; 22] = [
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
    "self_attn.v_proj.weight",
];

const VISION_LAYER_SUFFIXES: [&str; 13] = [
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

/// Parses an official tensor name with exact family grammar and index ranges.
pub fn classify_diffusion_gemma_tensor(
    name: &str,
) -> Result<DiffusionGemmaTensorClass, DiffusionGemmaError> {
    validate_tensor_name_characters(name)?;
    match name {
        "model.decoder.embed_tokens.weight" => {
            return Ok(DiffusionGemmaTensorClass::DecoderEmbedding);
        }
        "model.decoder.norm.weight" => return Ok(DiffusionGemmaTensorClass::DecoderNorm),
        "model.decoder.self_conditioning.down_proj.weight"
        | "model.decoder.self_conditioning.gate_proj.weight"
        | "model.decoder.self_conditioning.pre_norm.weight"
        | "model.decoder.self_conditioning.up_proj.weight" => {
            return Ok(DiffusionGemmaTensorClass::SelfConditioning);
        }
        "model.encoder.embed_vision.embedding_projection.weight" => {
            return Ok(DiffusionGemmaTensorClass::EncoderProjection);
        }
        "model.encoder.vision_tower.patch_embedder.input_proj.weight"
        | "model.encoder.vision_tower.patch_embedder.position_embedding_table"
        | "model.encoder.vision_tower.std_bias"
        | "model.encoder.vision_tower.std_scale" => {
            return Ok(DiffusionGemmaTensorClass::VisionRoot);
        }
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("model.decoder.layers.") {
        let (layer, suffix) = rest
            .split_once('.')
            .ok_or_else(|| invalid(format!("malformed decoder layer tensor: {name}")))?;
        let layer =
            parse_canonical_index(layer, DIFFUSION_GEMMA_TEXT_LAYER_COUNT, "decoder layer")?;
        ensure(
            DECODER_LAYER_SUFFIXES.contains(&suffix),
            format!("unknown decoder layer tensor suffix: {suffix}"),
        )?;
        ensure(
            reviewed_layer_type(layer) != DiffusionGemmaTextLayerType::FullAttention
                || suffix != "self_attn.v_proj.weight",
            format!("full-attention decoder layer {layer} has no V projection tensor"),
        )?;
        return Ok(DiffusionGemmaTensorClass::DecoderLayer);
    }
    if let Some(rest) = name.strip_prefix("model.encoder.language_model.layers.") {
        let (layer, suffix) = rest
            .split_once('.')
            .ok_or_else(|| invalid(format!("malformed encoder layer tensor: {name}")))?;
        parse_canonical_index(layer, DIFFUSION_GEMMA_TEXT_LAYER_COUNT, "encoder layer")?;
        ensure(
            suffix == "layer_scalar",
            format!("unknown encoder layer tensor suffix: {suffix}"),
        )?;
        return Ok(DiffusionGemmaTensorClass::EncoderLayerScalar);
    }
    if let Some(rest) = name.strip_prefix("model.encoder.vision_tower.encoder.layers.") {
        let (layer, suffix) = rest
            .split_once('.')
            .ok_or_else(|| invalid(format!("malformed vision layer tensor: {name}")))?;
        parse_canonical_index(layer, DIFFUSION_GEMMA_VISION_LAYER_COUNT, "vision layer")?;
        ensure(
            VISION_LAYER_SUFFIXES.contains(&suffix),
            format!("unknown vision layer tensor suffix: {suffix}"),
        )?;
        return Ok(DiffusionGemmaTensorClass::VisionLayer);
    }
    Err(invalid(format!("unknown tensor family: {name}")))
}

fn expected_tensor_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for name in [
        "model.decoder.embed_tokens.weight",
        "model.decoder.norm.weight",
        "model.decoder.self_conditioning.down_proj.weight",
        "model.decoder.self_conditioning.gate_proj.weight",
        "model.decoder.self_conditioning.pre_norm.weight",
        "model.decoder.self_conditioning.up_proj.weight",
        "model.encoder.embed_vision.embedding_projection.weight",
        "model.encoder.vision_tower.patch_embedder.input_proj.weight",
        "model.encoder.vision_tower.patch_embedder.position_embedding_table",
        "model.encoder.vision_tower.std_bias",
        "model.encoder.vision_tower.std_scale",
    ] {
        names.insert(name.to_owned());
    }
    for layer in 0..DIFFUSION_GEMMA_TEXT_LAYER_COUNT {
        for suffix in DECODER_LAYER_SUFFIXES {
            if reviewed_layer_type(layer) == DiffusionGemmaTextLayerType::FullAttention
                && suffix == "self_attn.v_proj.weight"
            {
                continue;
            }
            names.insert(format!("model.decoder.layers.{layer}.{suffix}"));
        }
        names.insert(format!(
            "model.encoder.language_model.layers.{layer}.layer_scalar"
        ));
    }
    for layer in 0..DIFFUSION_GEMMA_VISION_LAYER_COUNT {
        for suffix in VISION_LAYER_SUFFIXES {
            names.insert(format!(
                "model.encoder.vision_tower.encoder.layers.{layer}.{suffix}"
            ));
        }
    }
    names
}

fn increment_summary(
    summary: &mut DiffusionGemmaTensorSummary,
    class: DiffusionGemmaTensorClass,
) -> Result<(), DiffusionGemmaError> {
    let count = match class {
        DiffusionGemmaTensorClass::DecoderEmbedding => &mut summary.decoder_embedding,
        DiffusionGemmaTensorClass::DecoderNorm => &mut summary.decoder_norm,
        DiffusionGemmaTensorClass::SelfConditioning => &mut summary.self_conditioning,
        DiffusionGemmaTensorClass::DecoderLayer => &mut summary.decoder_layers,
        DiffusionGemmaTensorClass::EncoderProjection => &mut summary.encoder_projection,
        DiffusionGemmaTensorClass::EncoderLayerScalar => &mut summary.encoder_layer_scalars,
        DiffusionGemmaTensorClass::VisionRoot => &mut summary.vision_root,
        DiffusionGemmaTensorClass::VisionLayer => &mut summary.vision_layers,
    };
    *count = count
        .checked_add(1)
        .ok_or_else(|| invalid("tensor family count overflowed"))?;
    Ok(())
}

fn validate_summary(summary: DiffusionGemmaTensorSummary) -> Result<(), DiffusionGemmaError> {
    ensure(
        summary.decoder_embedding == DECODER_EMBEDDING_TENSOR_COUNT,
        "decoder embedding coverage changed",
    )?;
    ensure(
        summary.decoder_norm == DECODER_NORM_TENSOR_COUNT,
        "decoder norm coverage changed",
    )?;
    ensure(
        summary.self_conditioning == SELF_CONDITIONING_TENSOR_COUNT,
        "self-conditioning coverage changed",
    )?;
    ensure(
        summary.decoder_layers == DECODER_LAYER_TENSOR_COUNT,
        "decoder layer coverage changed",
    )?;
    ensure(
        summary.encoder_projection == ENCODER_PROJECTION_TENSOR_COUNT,
        "encoder projection coverage changed",
    )?;
    ensure(
        summary.encoder_layer_scalars == ENCODER_LAYER_SCALAR_TENSOR_COUNT,
        "encoder layer-scalar coverage changed",
    )?;
    ensure(
        summary.vision_root == VISION_ROOT_TENSOR_COUNT,
        "vision root coverage changed",
    )?;
    ensure(
        summary.vision_layers == VISION_LAYER_TENSOR_COUNT,
        "vision layer coverage changed",
    )?;
    ensure(
        summary.checked_total()? == DIFFUSION_GEMMA_TENSOR_COUNT,
        "tensor total changed",
    )
}

fn validate_tensor_name_set(
    actual: &BTreeSet<String>,
) -> Result<DiffusionGemmaTensorSummary, DiffusionGemmaError> {
    let expected = expected_tensor_names();
    ensure(
        expected.len() == DIFFUSION_GEMMA_TENSOR_COUNT,
        "internal expected tensor catalog count changed",
    )?;
    if let Some(missing) = expected.difference(actual).next() {
        return Err(invalid(format!("index is missing tensor: {missing}")));
    }
    if let Some(extra) = actual.difference(&expected).next() {
        return Err(invalid(format!("index has extra tensor: {extra}")));
    }
    let mut summary = DiffusionGemmaTensorSummary::default();
    for name in actual {
        increment_summary(&mut summary, classify_diffusion_gemma_tensor(name)?)?;
    }
    validate_summary(summary)?;
    Ok(summary)
}

#[derive(Debug)]
struct UniqueWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct WeightMapVisitor;

        impl<'de> Visitor<'de> for WeightMapVisitor {
            type Value = UniqueWeightMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor-to-shard map with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((name, shard)) = map.next_entry::<String, String>()? {
                    if values.insert(name.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor name {name}"
                        )));
                    }
                }
                Ok(UniqueWeightMap(values))
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndexMetadata {
    total_parameters: u64,
    total_size: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndex {
    metadata: RawIndexMetadata,
    weight_map: UniqueWeightMap,
}

fn catalog_sha256(weight_map: &BTreeMap<String, String>) -> Result<String, DiffusionGemmaError> {
    let mut hasher = Sha256::new();
    for (name, shard) in weight_map {
        let row = serde_json::to_vec(&(name, shard))
            .map_err(|error| invalid(format!("catalog serialization failed: {error}")))?;
        hasher.update(row);
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn diffusion_gemma_manifest_state(
    index_advertised_bytes: u64,
    shard_file_bytes: u64,
) -> DiffusionGemmaManifestState {
    match index_advertised_bytes.cmp(&shard_file_bytes) {
        std::cmp::Ordering::Equal => DiffusionGemmaManifestState::Consistent {
            bytes: index_advertised_bytes,
        },
        std::cmp::Ordering::Greater => {
            DiffusionGemmaManifestState::IndexAdvertisedExceedsShardFiles {
                index_advertised_bytes,
                shard_file_bytes,
                delta_bytes: index_advertised_bytes - shard_file_bytes,
            }
        }
        std::cmp::Ordering::Less => DiffusionGemmaManifestState::ShardFilesExceedIndexAdvertised {
            index_advertised_bytes,
            shard_file_bytes,
            delta_bytes: shard_file_bytes - index_advertised_bytes,
        },
    }
}

fn validate_index_document(raw: RawIndex) -> Result<DiffusionGemmaIndex, DiffusionGemmaError> {
    ensure(
        raw.metadata.total_parameters == DIFFUSION_GEMMA_TOTAL_PARAMETERS,
        format!(
            "index parameter count {} does not match reviewed {}",
            raw.metadata.total_parameters, DIFFUSION_GEMMA_TOTAL_PARAMETERS
        ),
    )?;
    ensure(
        raw.metadata.total_size == DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES,
        format!(
            "index advertised bytes {} do not match reviewed {}",
            raw.metadata.total_size, DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES
        ),
    )?;
    ensure(
        raw.weight_map.0.len() == DIFFUSION_GEMMA_TENSOR_COUNT,
        format!(
            "index tensor count {} does not match reviewed {}",
            raw.weight_map.0.len(),
            DIFFUSION_GEMMA_TENSOR_COUNT
        ),
    )?;
    let names = raw.weight_map.0.keys().cloned().collect::<BTreeSet<_>>();
    let summary = validate_tensor_name_set(&names)?;

    let mut shard_counts = BTreeMap::<&str, usize>::new();
    for shard_name in raw.weight_map.0.values() {
        ensure(
            !shard_name.contains('/') && !shard_name.contains('\\') && !shard_name.contains(".."),
            format!("index shard path is not a canonical base name: {shard_name}"),
        )?;
        let shard = diffusion_gemma_locked_shard(shard_name)
            .ok_or_else(|| invalid(format!("index references unknown shard: {shard_name}")))?;
        let count = shard_counts.entry(shard.file_name).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("per-shard tensor count overflowed"))?;
    }
    ensure(
        shard_counts.len() == DIFFUSION_GEMMA_SHARD_COUNT,
        "index shard coverage changed",
    )?;
    for shard in &DIFFUSION_GEMMA_SHARDS {
        let actual = shard_counts.get(shard.file_name).copied().unwrap_or(0);
        ensure(
            actual == shard.indexed_tensor_count,
            format!(
                "shard {} indexes {actual} tensors, reviewed {}",
                shard.file_name, shard.indexed_tensor_count
            ),
        )?;
    }

    let shard_file_bytes = checked_diffusion_gemma_shard_file_bytes()?;
    ensure(
        shard_file_bytes == DIFFUSION_GEMMA_SHARD_FILE_BYTES,
        "locked shard byte total changed",
    )?;
    let manifest_state = diffusion_gemma_manifest_state(raw.metadata.total_size, shard_file_bytes);
    ensure(
        manifest_state
            == DiffusionGemmaManifestState::ShardFilesExceedIndexAdvertised {
                index_advertised_bytes: DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES,
                shard_file_bytes: DIFFUSION_GEMMA_SHARD_FILE_BYTES,
                delta_bytes: DIFFUSION_GEMMA_MANIFEST_DELTA_BYTES,
            },
        "index/shard manifest relation changed",
    )?;
    let digest = catalog_sha256(&raw.weight_map.0)?;
    ensure(
        digest == DIFFUSION_GEMMA_CATALOG_SHA256,
        format!(
            "catalog SHA-256 {digest} does not match reviewed {DIFFUSION_GEMMA_CATALOG_SHA256}"
        ),
    )?;

    Ok(DiffusionGemmaIndex {
        total_parameters: raw.metadata.total_parameters,
        index_advertised_bytes: raw.metadata.total_size,
        shard_file_bytes,
        manifest_state,
        catalog_sha256: digest,
        summary,
        weight_map: raw.weight_map.0,
    })
}

/// Validates exact index bytes, all 1,047 names, and all shard assignments.
pub fn validate_diffusion_gemma_index(
    bytes: &[u8],
) -> Result<DiffusionGemmaIndex, DiffusionGemmaError> {
    validate_locked_document(
        bytes,
        DIFFUSION_GEMMA_INDEX_BYTES,
        DIFFUSION_GEMMA_INDEX_SHA256,
        "index",
    )?;
    let raw: RawIndex = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("index JSON is not strict-valid: {error}")))?;
    validate_index_document(raw)
}

pub fn diffusion_gemma_capacity_decision(
    index: &DiffusionGemmaIndex,
    available_bytes: u64,
    resident_copy_count: u64,
    additional_bytes: u64,
) -> Result<DiffusionGemmaCapacityDecision, DiffusionGemmaError> {
    let required_bytes = index.checked_admission_bytes(resident_copy_count, additional_bytes)?;
    Ok(DiffusionGemmaCapacityDecision {
        required_bytes,
        available_bytes,
        fits: available_bytes >= required_bytes,
        shortfall_bytes: required_bytes.saturating_sub(available_bytes),
        manifest_state: index.manifest_state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn foundation_index() -> DiffusionGemmaIndex {
        let manifest_state = diffusion_gemma_manifest_state(
            DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES,
            DIFFUSION_GEMMA_SHARD_FILE_BYTES,
        );
        DiffusionGemmaIndex {
            total_parameters: DIFFUSION_GEMMA_TOTAL_PARAMETERS,
            index_advertised_bytes: DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES,
            shard_file_bytes: DIFFUSION_GEMMA_SHARD_FILE_BYTES,
            manifest_state,
            catalog_sha256: DIFFUSION_GEMMA_CATALOG_SHA256.to_owned(),
            summary: DiffusionGemmaTensorSummary {
                decoder_embedding: DECODER_EMBEDDING_TENSOR_COUNT,
                decoder_norm: DECODER_NORM_TENSOR_COUNT,
                self_conditioning: SELF_CONDITIONING_TENSOR_COUNT,
                decoder_layers: DECODER_LAYER_TENSOR_COUNT,
                encoder_projection: ENCODER_PROJECTION_TENSOR_COUNT,
                encoder_layer_scalars: ENCODER_LAYER_SCALAR_TENSOR_COUNT,
                vision_root: VISION_ROOT_TENSOR_COUNT,
                vision_layers: VISION_LAYER_TENSOR_COUNT,
            },
            weight_map: BTreeMap::new(),
        }
    }

    #[test]
    fn serde_rejects_duplicate_unknown_missing_and_overflow() {
        let duplicate = br#"{"architectures": [], "architectures": []}"#;
        let error = serde_json::from_slice::<RawConfig>(duplicate)
            .expect_err("duplicate config field must fail")
            .to_string();
        assert!(error.contains("duplicate field `architectures`"), "{error}");

        let error = serde_json::from_slice::<RawConfig>(br#"{"unknown": true}"#)
            .expect_err("unknown config field must fail")
            .to_string();
        assert!(error.contains("unknown field `unknown`"), "{error}");
        let error = serde_json::from_slice::<RawConfig>(br#"{}"#)
            .expect_err("missing config field must fail")
            .to_string();
        assert!(error.contains("missing field"), "{error}");

        let duplicate_tensor = br#"{
            "metadata": {"total_parameters": 1, "total_size": 1},
            "weight_map": {"a": "x", "a": "y"}
        }"#;
        let error = serde_json::from_slice::<RawIndex>(duplicate_tensor)
            .expect_err("duplicate tensor must fail")
            .to_string();
        assert!(error.contains("duplicate tensor name a"), "{error}");

        let overflow = br#"{
            "metadata": {"total_parameters": 1, "total_size": 18446744073709551616},
            "weight_map": {}
        }"#;
        assert!(serde_json::from_slice::<RawIndex>(overflow).is_err());
    }

    #[test]
    fn layer_schedule_and_tensor_grammar_cover_boundaries() {
        for layer in [0, 4, 6, 28] {
            assert_eq!(
                reviewed_layer_type(layer),
                DiffusionGemmaTextLayerType::SlidingAttention
            );
        }
        for layer in [5, 11, 17, 23, 29] {
            assert_eq!(
                reviewed_layer_type(layer),
                DiffusionGemmaTextLayerType::FullAttention
            );
        }

        let accepted = [
            (
                "model.decoder.embed_tokens.weight",
                DiffusionGemmaTensorClass::DecoderEmbedding,
            ),
            (
                "model.decoder.self_conditioning.pre_norm.weight",
                DiffusionGemmaTensorClass::SelfConditioning,
            ),
            (
                "model.decoder.layers.0.self_attn.v_proj.weight",
                DiffusionGemmaTensorClass::DecoderLayer,
            ),
            (
                "model.decoder.layers.5.self_attn.q_proj.weight",
                DiffusionGemmaTensorClass::DecoderLayer,
            ),
            (
                "model.decoder.layers.29.router.per_expert_scale",
                DiffusionGemmaTensorClass::DecoderLayer,
            ),
            (
                "model.encoder.language_model.layers.29.layer_scalar",
                DiffusionGemmaTensorClass::EncoderLayerScalar,
            ),
            (
                "model.encoder.vision_tower.encoder.layers.26.self_attn.v_proj.linear.weight",
                DiffusionGemmaTensorClass::VisionLayer,
            ),
        ];
        for (name, expected) in accepted {
            assert_eq!(
                classify_diffusion_gemma_tensor(name),
                Ok(expected),
                "{name}"
            );
        }

        for name in [
            "model.decoder.layers.05.self_attn.q_proj.weight",
            "model.decoder.layers.5.self_attn.v_proj.weight",
            "model.decoder.layers.30.self_attn.q_proj.weight",
            "model.decoder.layers.42949672960.self_attn.q_proj.weight",
            "model.encoder.language_model.layers.30.layer_scalar",
            "model.encoder.vision_tower.encoder.layers.27.self_attn.q_proj.linear.weight",
            "model.decoder.layers.0.unknown.weight",
            "../model.decoder.embed_tokens.weight",
            "model//decoder.weight",
        ] {
            assert!(
                classify_diffusion_gemma_tensor(name).is_err(),
                "accepted {name}"
            );
        }
    }

    #[test]
    fn exact_catalog_rejects_missing_and_extra_names() {
        let exact = expected_tensor_names();
        assert_eq!(exact.len(), DIFFUSION_GEMMA_TENSOR_COUNT);
        let summary = validate_tensor_name_set(&exact).expect("reviewed catalog");
        validate_summary(summary).expect("reviewed family counts");

        let mut missing = exact.clone();
        let removed = missing
            .iter()
            .next()
            .expect("nonempty expected catalog")
            .clone();
        assert!(missing.remove(&removed));
        let error = validate_tensor_name_set(&missing).expect_err("missing tensor must fail");
        assert!(error.to_string().contains("missing tensor"));

        let mut extra = exact;
        assert!(extra.insert("model.extra.weight".to_owned()));
        let error = validate_tensor_name_set(&extra).expect_err("extra tensor must fail");
        assert!(error.to_string().contains("extra tensor"));
    }

    #[test]
    fn shapes_reject_empty_zero_and_overflow() {
        assert_eq!(
            checked_product(&[16, 72], "vision attention").expect("valid vision shape"),
            1_152
        );
        assert_eq!(
            checked_product(&[128, 704, 2_816], "MoE projection")
                .expect("valid non-power-of-two shape"),
            253_755_392
        );
        assert!(checked_product(&[], "empty").is_err());
        assert!(checked_product(&[17, 0, 19], "zero").is_err());
        assert!(checked_product(&[u64::MAX, 2], "overflow").is_err());
    }

    #[test]
    fn shard_and_support_identities_reject_traversal_and_mismatch() {
        assert_eq!(DIFFUSION_GEMMA_SHARDS.len(), DIFFUSION_GEMMA_SHARD_COUNT);
        assert_eq!(
            checked_diffusion_gemma_shard_file_bytes().expect("sum shard sizes"),
            DIFFUSION_GEMMA_SHARD_FILE_BYTES
        );
        assert_eq!(
            DIFFUSION_GEMMA_SHARDS
                .iter()
                .map(|shard| shard.indexed_tensor_count)
                .sum::<usize>(),
            DIFFUSION_GEMMA_TENSOR_COUNT
        );
        let first = DIFFUSION_GEMMA_SHARDS[0];
        assert_eq!(
            validate_diffusion_gemma_shard_lfs_identity(
                first.file_name,
                first.size,
                first.lfs_sha256
            ),
            Ok(&first)
        );
        assert!(
            validate_diffusion_gemma_shard_lfs_identity(
                "../model-00001-of-00011.safetensors",
                first.size,
                first.lfs_sha256
            )
            .is_err()
        );
        assert!(
            validate_diffusion_gemma_shard_lfs_identity(
                first.file_name,
                first.size + 1,
                first.lfs_sha256
            )
            .is_err()
        );
        assert!(
            validate_diffusion_gemma_shard_lfs_identity(
                first.file_name,
                first.size,
                &"0".repeat(64)
            )
            .is_err()
        );
        assert!(validate_diffusion_gemma_support_file("../config.json", &[]).is_err());
        assert!(validate_diffusion_gemma_support_file("config.json", &[]).is_err());
        assert!(is_canonical_repository_path(
            "scheduler/scheduler_config.json"
        ));
        assert!(!is_canonical_repository_path(
            "scheduler/../scheduler_config.json"
        ));
        let tokenizer =
            diffusion_gemma_support_file("tokenizer.json").expect("fixed tokenizer model identity");
        assert_eq!(tokenizer.size, 32_169_626);
        assert_eq!(
            tokenizer.sha256,
            "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
        );
        assert!(diffusion_gemma_support_file("unknown.json").is_none());
    }

    #[test]
    fn manifest_and_capacity_use_larger_shard_total() {
        let state = diffusion_gemma_manifest_state(
            DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES,
            DIFFUSION_GEMMA_SHARD_FILE_BYTES,
        );
        assert_eq!(
            state,
            DiffusionGemmaManifestState::ShardFilesExceedIndexAdvertised {
                index_advertised_bytes: DIFFUSION_GEMMA_INDEX_ADVERTISED_BYTES,
                shard_file_bytes: DIFFUSION_GEMMA_SHARD_FILE_BYTES,
                delta_bytes: DIFFUSION_GEMMA_MANIFEST_DELTA_BYTES,
            }
        );
        assert_eq!(
            state.admission_base_bytes(),
            DIFFUSION_GEMMA_CAPACITY_ADMISSION_BYTES
        );
        assert_eq!(
            diffusion_gemma_manifest_state(19, 17).admission_base_bytes(),
            19
        );
        assert_eq!(
            diffusion_gemma_manifest_state(23, 23).admission_base_bytes(),
            23
        );

        let index = foundation_index();
        let required = DIFFUSION_GEMMA_CAPACITY_ADMISSION_BYTES + 13;
        let below = diffusion_gemma_capacity_decision(&index, required - 1, 1, 13)
            .expect("finite capacity decision");
        assert!(!below.fits);
        assert_eq!(below.shortfall_bytes, 1);
        let exact =
            diffusion_gemma_capacity_decision(&index, required, 1, 13).expect("capacity boundary");
        assert!(exact.fits);
        assert_eq!(exact.shortfall_bytes, 0);
        assert!(diffusion_gemma_capacity_decision(&index, u64::MAX, 0, 0).is_err());
        assert!(diffusion_gemma_capacity_decision(&index, u64::MAX, u64::MAX, 0).is_err());
        assert!(diffusion_gemma_capacity_decision(&index, u64::MAX, 1, u64::MAX).is_err());
    }

    fn metadata_file(directory: &Path, name: &str) -> PathBuf {
        let path = directory.join(name);
        assert!(
            path.is_file(),
            "missing official metadata file {}",
            path.display()
        );
        path
    }

    #[test]
    #[ignore = "requires fixed-revision metadata via SLLM_DIFFUSION_GEMMA_METADATA_DIR"]
    fn official_metadata_matches_locked_foundation() {
        let directory = PathBuf::from(
            std::env::var("SLLM_DIFFUSION_GEMMA_METADATA_DIR")
                .expect("set SLLM_DIFFUSION_GEMMA_METADATA_DIR"),
        );
        let config = validate_diffusion_gemma_config(
            &fs::read(metadata_file(&directory, "config.json")).expect("read config"),
        )
        .expect("validate config");
        let index = validate_diffusion_gemma_index(
            &fs::read(metadata_file(&directory, "model.safetensors.index.json"))
                .expect("read index"),
        )
        .expect("validate index");
        for identity in &DIFFUSION_GEMMA_SUPPORT_FILES {
            let fixture_name = if identity.file_name == "scheduler/scheduler_config.json" {
                "scheduler_scheduler_config.json"
            } else {
                identity.file_name
            };
            let fixture_path = directory.join(fixture_name);
            if identity.file_name == "tokenizer.json" && !fixture_path.is_file() {
                // The bounded metadata fixture intentionally omits the 32 MB LFS
                // tokenizer. Its Hub identity remains fixed, but it is not
                // represented as locally byte-verified by this test.
                continue;
            }
            validate_diffusion_gemma_support_file(
                identity.file_name,
                &fs::read(&fixture_path).expect("read support file"),
            )
            .expect("validate support file");
        }

        assert_eq!(config.canvas_length, DIFFUSION_GEMMA_CANVAS_LENGTH);
        assert_eq!(config.text.layer_count, DIFFUSION_GEMMA_TEXT_LAYER_COUNT);
        assert_eq!(
            config.vision.layer_count,
            DIFFUSION_GEMMA_VISION_LAYER_COUNT
        );
        assert!(!config.production_loader_enabled);
        assert!(!config.autoregressive_fallback_enabled);
        assert_eq!(index.tensor_count(), DIFFUSION_GEMMA_TENSOR_COUNT);
        assert_eq!(index.catalog_sha256(), DIFFUSION_GEMMA_CATALOG_SHA256);
        assert_eq!(
            index.manifest_state().admission_base_bytes(),
            DIFFUSION_GEMMA_CAPACITY_ADMISSION_BYTES
        );
    }

    #[test]
    fn summary_overflow_is_fail_closed() {
        assert!(
            DiffusionGemmaTensorSummary {
                decoder_embedding: usize::MAX,
                decoder_norm: 1,
                ..Default::default()
            }
            .checked_total()
            .is_err()
        );
    }
}
