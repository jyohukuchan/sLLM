//! Offline, fail-closed source contract for the reviewed Gemma 4 12B MTP
//! assistant.  The assistant is a draft model paired with one exact target;
//! it is never accepted as a standalone text model and model-provided code is
//! neither downloaded nor executed.

use crate::gemma4::{
    GEMMA4_12B_IT_ALIAS, GEMMA4_12B_IT_FINGERPRINT, GEMMA4_12B_IT_REPO_ID, GEMMA4_12B_IT_REVISION,
    Gemma4LayerType, Gemma4ModelLock,
};
use crate::model::{
    LockedFile, ModelError, TensorDType, TensorDescriptor, fingerprint_for_json,
    parse_model_source_json,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const GEMMA4_MTP_REPO_ID: &str = "google/gemma-4-12B-it-assistant";
pub const GEMMA4_MTP_REVISION: &str = "46d4c6f13f0ac0ad827b915669b8df9b81c64c51";
pub const GEMMA4_MTP_ALIAS: &str = "gemma4-12b-it-assistant-bf16";
pub const GEMMA4_MTP_FINGERPRINT: &str =
    "sha256:c2528168a0b31fab8dd9e447a2af14bf5016ac9698110230e9e67e3463636841";
pub const GEMMA4_MTP_MODEL_BYTES: u64 = 845_719_296;
pub const GEMMA4_MTP_MODEL_SHA256: &str =
    "3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df";
pub const GEMMA4_MTP_HEADER_BYTES: u64 = 5_360;
pub const GEMMA4_MTP_HEADER_SHA256: &str =
    "d0f1537ec1254122003a892254cefcf44c538f2cc42ba612b5791f4c6c5fdcb4";
pub const GEMMA4_MTP_CATALOG_SHA256: &str =
    "fd87240fd7fe1beac3b7f39ff3d4ae93e4c5a3fb4192fc556a8a2f28d892cc3d";
pub const GEMMA4_MTP_TENSOR_COUNT: u64 = 48;
pub const GEMMA4_MTP_VOCAB_SIZE: u64 = 262_144;
pub const GEMMA4_MTP_VOCAB_SEMANTIC_SHA256: &str =
    "fa92326adf8e68460cd13e22dd88df97e28263eb489ec91e6372c83e1cc6be4c";
pub const GEMMA4_MTP_MERGES_SEMANTIC_SHA256: &str =
    "9554731abedaef2a69332e3072ed3582a702203dc7dab4997057085653bdaa45";
pub const GEMMA4_MTP_MERGE_COUNT: u64 = 514_906;
pub const GEMMA4_MTP_DRAFT_TO_TARGET_KV_LAYERS: [u32; 4] = [46, 46, 46, 47];

const LOCK_SCHEMA: &str = "gemma4-mtp-model-lock-v1";
const DATA_BUFFER_START: u64 = GEMMA4_MTP_HEADER_BYTES + 8;
const PAYLOAD_BYTES: u64 = GEMMA4_MTP_MODEL_BYTES - DATA_BUFFER_START;
const MAX_SUPPORT_FILE_BYTES: u64 = 64 * 1024 * 1024;
const HASH_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpLicenseContract {
    pub id: String,
    pub statement: String,
    pub evidence_paths: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpArchitectureContract {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub custom_code: bool,
    pub model_provided_code: bool,
    pub dtype: TensorDType,
    pub hidden_size: u64,
    pub backbone_hidden_size: u64,
    pub pre_projection_input_size: u64,
    pub post_projection_output_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub num_global_key_value_heads: u64,
    pub head_dim: u64,
    pub global_head_dim: u64,
    pub sliding_window: u64,
    pub max_position_embeddings: u64,
    pub vocab_size: u64,
    pub num_kv_shared_layers: u64,
    pub own_kv_projection_tensor_count: u64,
    pub num_centroids: u64,
    pub centroid_intermediate_top_k: u64,
    pub use_ordered_embeddings: bool,
    pub layer_types: Vec<Gemma4LayerType>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpTargetCompatibility {
    pub repo_id: String,
    pub resolved_revision: String,
    pub model_fingerprint: String,
    pub alias: String,
    pub target_hidden_size: u64,
    pub target_vocab_size: u64,
    pub target_layer_count: u64,
    pub draft_to_target_kv_layers: [u32; 4],
    pub sliding_target_layer: u32,
    pub full_target_layer: u32,
    pub wire_tokenizer_source: String,
    pub assistant_named_video_token_present: bool,
    pub target_named_video_token_id: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpTensorContract {
    pub container: String,
    pub source_path: String,
    pub header_length_field_bytes: u64,
    pub header_length_bytes: u64,
    pub data_buffer_start: u64,
    pub header_sha256: String,
    pub catalog_sha256: String,
    pub tensor_count: u64,
    pub dtype: TensorDType,
    pub payload_bytes: u64,
    pub unknown_policy: String,
    pub duplicate_policy: String,
    pub catalog_policy: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpTokenizerContract {
    pub files: Vec<String>,
    pub tokenizer_class: String,
    pub wire_source: String,
    pub vocab_size: u64,
    pub vocab_semantic_sha256: String,
    pub merges_semantic_sha256: String,
    pub merge_count: u64,
    pub common_generation_token_ids: BTreeMap<String, u64>,
    pub assistant_named_video_token_present: bool,
    pub target_named_video_token_id: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpGenerationContract {
    pub bos_token_id: u64,
    pub eos_token_id: u64,
    pub pad_token_id: u64,
    pub do_sample: bool,
    pub temperature: String,
    pub top_k: u64,
    pub top_p: String,
    pub suppress_tokens: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpLockedModel {
    pub repo_id: String,
    pub repo_type: String,
    pub requested_revision: String,
    pub resolved_revision: String,
    pub license: Gemma4MtpLicenseContract,
    pub evidence_files: Vec<String>,
    pub files: Vec<LockedFile>,
    pub architecture: Gemma4MtpArchitectureContract,
    pub target_compatibility: Gemma4MtpTargetCompatibility,
    pub tensor_contract: Gemma4MtpTensorContract,
    pub tokenizer_contract: Gemma4MtpTokenizerContract,
    pub generation_contract: Gemma4MtpGenerationContract,
    pub derivation: Option<()>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4MtpModelLock {
    pub schema_version: String,
    pub model: Gemma4MtpLockedModel,
    pub fingerprint: String,
    pub aliases: Vec<String>,
    pub generated_at: String,
}

impl Gemma4MtpModelLock {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn target_fingerprint(&self) -> &str {
        &self.model.target_compatibility.model_fingerprint
    }

    /// The assistant tokenizer is integrity-checked, but the paired target
    /// tokenizer remains the wire/tokenization authority.
    pub fn requires_target_tokenizer(&self) -> bool {
        true
    }

    pub fn verify_cache(
        &self,
        cache_root: impl AsRef<Path>,
        target: &Gemma4ModelLock,
    ) -> Result<VerifiedGemma4Mtp, ModelError> {
        verify_gemma4_mtp_cache(self, cache_root, target)
    }
}

/// Closed semantic config returned only after the assistant `config.json`
/// passes its complete field/topology contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpConfig {
    pub hidden_size: u32,
    pub backbone_hidden_size: u32,
    pub intermediate_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub global_kv_heads: u32,
    pub head_dim: u32,
    pub global_head_dim: u32,
    pub sliding_window: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub layer_types: Vec<Gemma4LayerType>,
    pub draft_to_target_kv_layers: [u32; 4],
}

/// Read-only assistant weight source shared by the immutable safetensors
/// cache and its canonical lossless GGUF derivative.
pub trait Gemma4MtpWeightSource {
    fn lock_fingerprint(&self) -> &str;
    fn target_fingerprint(&self) -> &str;
    fn config(&self) -> &Gemma4MtpConfig;
    fn tensors(&self) -> &BTreeMap<String, TensorDescriptor>;
    fn read_tensor_range(
        &self,
        name: &str,
        tensor_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, ModelError>;
}

struct BoundFile {
    file: Arc<File>,
    size: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

/// Fully verified assistant cache. It owns the exact opened file descriptors
/// used for subsequent reads, preventing path replacement from changing the
/// admitted source after validation.
pub struct VerifiedGemma4Mtp {
    lock_fingerprint: String,
    target_fingerprint: String,
    cache_root: PathBuf,
    files: BTreeMap<String, BoundFile>,
    tensors: BTreeMap<String, TensorDescriptor>,
    config: Gemma4MtpConfig,
}

impl fmt::Debug for VerifiedGemma4Mtp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedGemma4Mtp")
            .field("lock_fingerprint", &self.lock_fingerprint)
            .field("target_fingerprint", &self.target_fingerprint)
            .field("file_count", &self.files.len())
            .field("tensor_count", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

impl VerifiedGemma4Mtp {
    pub fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    pub fn target_fingerprint(&self) -> &str {
        &self.target_fingerprint
    }

    pub fn cache_root(&self) -> &Path {
        &self.cache_root
    }

    pub fn config(&self) -> &Gemma4MtpConfig {
        &self.config
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorDescriptor> {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    pub fn read_tensor_range(
        &self,
        name: &str,
        tensor_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, ModelError> {
        let descriptor = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("unknown Gemma 4 MTP tensor: {name}")))?;
        let length_u64 = u64::try_from(length)
            .map_err(|_| invalid("MTP tensor read length does not fit u64"))?;
        let end = tensor_offset
            .checked_add(length_u64)
            .ok_or_else(|| invalid("MTP tensor read range overflowed"))?;
        if end > descriptor.byte_size {
            return Err(invalid("MTP tensor read exceeds the verified tensor range"));
        }
        let absolute = descriptor
            .absolute_start()
            .checked_add(tensor_offset)
            .ok_or_else(|| invalid("MTP tensor absolute read offset overflowed"))?;
        let source = self
            .files
            .get(&descriptor.source_file)
            .ok_or_else(|| invalid("verified MTP model source is absent"))?;
        read_bound_range(source, absolute, length)
    }
}

impl Gemma4MtpWeightSource for VerifiedGemma4Mtp {
    fn lock_fingerprint(&self) -> &str {
        self.lock_fingerprint()
    }

    fn target_fingerprint(&self) -> &str {
        self.target_fingerprint()
    }

    fn config(&self) -> &Gemma4MtpConfig {
        self.config()
    }

    fn tensors(&self) -> &BTreeMap<String, TensorDescriptor> {
        self.tensors()
    }

    fn read_tensor_range(
        &self,
        name: &str,
        tensor_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, ModelError> {
        self.read_tensor_range(name, tensor_offset, length)
    }
}

pub fn parse_gemma4_mtp_model_lock(bytes: &[u8]) -> Result<Gemma4MtpModelLock, ModelError> {
    let computed = fingerprint_for_json(bytes)?;
    let lock: Gemma4MtpModelLock =
        serde_json::from_slice(bytes).map_err(|error| ModelError::Schema(error.to_string()))?;
    if lock.fingerprint != computed {
        return Err(ModelError::FingerprintMismatch {
            expected: lock.fingerprint,
            actual: computed,
        });
    }
    validate_gemma4_mtp_lock(&lock)?;
    Ok(lock)
}

fn validate_gemma4_mtp_lock(lock: &Gemma4MtpModelLock) -> Result<(), ModelError> {
    if lock.schema_version != LOCK_SCHEMA
        || lock.model.repo_id != GEMMA4_MTP_REPO_ID
        || lock.model.repo_type != "model"
        || lock.model.requested_revision != GEMMA4_MTP_REVISION
        || lock.model.resolved_revision != GEMMA4_MTP_REVISION
        || lock.fingerprint != GEMMA4_MTP_FINGERPRINT
        || lock.aliases != [GEMMA4_MTP_ALIAS]
        || lock.model.derivation.is_some()
    {
        return Err(invalid("Gemma 4 MTP immutable identity differs"));
    }
    if lock.model.license.id != "Apache-2.0"
        || lock.model.license.statement != "Apache 2.0"
        || lock.model.license.evidence_paths != ["README.md"]
        || lock.model.evidence_files != ["README.md"]
    {
        return Err(invalid("Gemma 4 MTP license evidence differs"));
    }
    validate_locked_files(&lock.model.files)?;

    let architecture = &lock.model.architecture;
    if architecture.architectures != ["Gemma4UnifiedAssistantForCausalLM"]
        || architecture.model_type != "gemma4_unified_assistant"
        || architecture.custom_code
        || architecture.model_provided_code
        || architecture.dtype != TensorDType::Bf16
        || architecture.hidden_size != 1_024
        || architecture.backbone_hidden_size != 3_840
        || architecture.pre_projection_input_size != 7_680
        || architecture.post_projection_output_size != 3_840
        || architecture.intermediate_size != 8_192
        || architecture.num_hidden_layers != 4
        || architecture.num_attention_heads != 16
        || architecture.num_key_value_heads != 8
        || architecture.num_global_key_value_heads != 1
        || architecture.head_dim != 256
        || architecture.global_head_dim != 512
        || architecture.sliding_window != 1_024
        || architecture.max_position_embeddings != 262_144
        || architecture.vocab_size != GEMMA4_MTP_VOCAB_SIZE
        || architecture.num_kv_shared_layers != 4
        || architecture.own_kv_projection_tensor_count != 0
        || architecture.num_centroids != 2_048
        || architecture.centroid_intermediate_top_k != 32
        || architecture.use_ordered_embeddings
        || architecture.layer_types != reviewed_mtp_layer_schedule()
    {
        return Err(invalid("Gemma 4 MTP reviewed architecture differs"));
    }
    let target = &lock.model.target_compatibility;
    if target.repo_id != GEMMA4_12B_IT_REPO_ID
        || target.resolved_revision != GEMMA4_12B_IT_REVISION
        || target.model_fingerprint != GEMMA4_12B_IT_FINGERPRINT
        || target.alias != GEMMA4_12B_IT_ALIAS
        || target.target_hidden_size != 3_840
        || target.target_vocab_size != GEMMA4_MTP_VOCAB_SIZE
        || target.target_layer_count != 48
        || target.draft_to_target_kv_layers != GEMMA4_MTP_DRAFT_TO_TARGET_KV_LAYERS
        || target.sliding_target_layer != 46
        || target.full_target_layer != 47
        || target.wire_tokenizer_source != "target-model-lock"
        || target.assistant_named_video_token_present
        || target.target_named_video_token_id != 258_884
    {
        return Err(invalid("Gemma 4 MTP target compatibility differs"));
    }
    let tensor = &lock.model.tensor_contract;
    if tensor.container != "direct-safetensors"
        || tensor.source_path != "model.safetensors"
        || tensor.header_length_field_bytes != 8
        || tensor.header_length_bytes != GEMMA4_MTP_HEADER_BYTES
        || tensor.data_buffer_start != DATA_BUFFER_START
        || tensor.header_sha256 != GEMMA4_MTP_HEADER_SHA256
        || tensor.catalog_sha256 != GEMMA4_MTP_CATALOG_SHA256
        || tensor.tensor_count != GEMMA4_MTP_TENSOR_COUNT
        || tensor.dtype != TensorDType::Bf16
        || tensor.payload_bytes != PAYLOAD_BYTES
        || tensor.unknown_policy != "reject"
        || tensor.duplicate_policy != "reject"
        || tensor.catalog_policy != "exact-derived-name-shape-dtype-and-range"
    {
        return Err(invalid("Gemma 4 MTP safetensors contract differs"));
    }
    let tokenizer = &lock.model.tokenizer_contract;
    let common_ids = BTreeMap::from([
        ("bos".to_owned(), 2),
        ("eos".to_owned(), 1),
        ("pad".to_owned(), 0),
        ("tool_response_begin".to_owned(), 50),
        ("turn_end".to_owned(), 106),
    ]);
    if tokenizer.files != ["tokenizer.json", "tokenizer_config.json"]
        || tokenizer.tokenizer_class != "GemmaTokenizer"
        || tokenizer.wire_source != "target-model-lock"
        || tokenizer.vocab_size != GEMMA4_MTP_VOCAB_SIZE
        || tokenizer.vocab_semantic_sha256 != GEMMA4_MTP_VOCAB_SEMANTIC_SHA256
        || tokenizer.merges_semantic_sha256 != GEMMA4_MTP_MERGES_SEMANTIC_SHA256
        || tokenizer.merge_count != GEMMA4_MTP_MERGE_COUNT
        || tokenizer.common_generation_token_ids != common_ids
        || tokenizer.assistant_named_video_token_present
        || tokenizer.target_named_video_token_id != 258_884
    {
        return Err(invalid("Gemma 4 MTP tokenizer contract differs"));
    }
    let generation = &lock.model.generation_contract;
    if generation.bos_token_id != 2
        || generation.eos_token_id != 1
        || generation.pad_token_id != 0
        || !generation.do_sample
        || generation.temperature != "1"
        || generation.top_k != 64
        || generation.top_p != "0.95"
        || generation.suppress_tokens != [258_883, 258_882]
    {
        return Err(invalid("Gemma 4 MTP generation contract differs"));
    }
    Ok(())
}

fn validate_locked_files(files: &[LockedFile]) -> Result<(), ModelError> {
    let expected = BTreeMap::from([
        (
            ".gitattributes",
            (
                1_624,
                "484fac0cb8b057eefe1992c8b72ac6e7438c7d17bd60c0e278b401c2190f7e72",
                "602d20f6eefed4b62821062891a7a495e25435f9",
                None,
            ),
        ),
        (
            "README.md",
            (
                29_898,
                "8adf126758a3da9a545dcc3fee8bb91817107779f668d2c5574504750db93de9",
                "fb9dbb93675f28d62e5586e75afd94d03fb80939",
                None,
            ),
        ),
        (
            "config.json",
            (
                2_346,
                "b6f19209588fcefe41f65b193fad6148446253c470d36e29441ecc5158a54e6d",
                "9889070ccf194bd3cab6f61954218a4058696537",
                None,
            ),
        ),
        (
            "generation_config.json",
            (
                233,
                "02b56bd11e1cd1e363e701a85a2fd7fbaa2992ec3358c1cd7cc44ead7208f505",
                "6e4ef65e8cf563a9177fd933423f89eed2f74d74",
                None,
            ),
        ),
        (
            "model.safetensors",
            (
                GEMMA4_MTP_MODEL_BYTES,
                GEMMA4_MTP_MODEL_SHA256,
                "dcecf2bfdf2086d661e44134663920bdaacec1e7",
                Some("sha256:3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df"),
            ),
        ),
        (
            "tokenizer.json",
            (
                32_169_884,
                "c001d9ada50af662c94f5ab17ec7e09f6438d1bec8246c47fee6510693d8de35",
                "0645e1e88640880b270ab18c952d52b04e7e396d",
                Some("sha256:c001d9ada50af662c94f5ab17ec7e09f6438d1bec8246c47fee6510693d8de35"),
            ),
        ),
        (
            "tokenizer_config.json",
            (
                822,
                "089594a3924fcfd4cb1c596a7906fbf476193519e5198f780912eed02b177e42",
                "1a6bee041ca75778c514a071efbdb568b0f3d7b0",
                None,
            ),
        ),
    ]);
    if files.len() != expected.len() {
        return Err(invalid("Gemma 4 MTP locked file count differs"));
    }
    let mut seen = BTreeSet::new();
    for file in files {
        if !seen.insert(file.path.as_str()) {
            return Err(invalid("Gemma 4 MTP locked file path is duplicated"));
        }
        let Some((size, sha256, git_blob, lfs_oid)) = expected.get(file.path.as_str()) else {
            return Err(invalid("Gemma 4 MTP locked file set differs"));
        };
        let source = format!(
            "https://huggingface.co/{}/blob/{}/{}",
            GEMMA4_MTP_REPO_ID, GEMMA4_MTP_REVISION, file.path
        );
        let download = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            GEMMA4_MTP_REPO_ID, GEMMA4_MTP_REVISION, file.path
        );
        if file.size_bytes != *size
            || file.sha256 != *sha256
            || file.git_blob != *git_blob
            || file.lfs_oid.as_deref() != *lfs_oid
            || file.source_page_url != source
            || file.download_url != download
        {
            return Err(invalid(format!(
                "Gemma 4 MTP locked file identity differs: {}",
                file.path
            )));
        }
    }
    Ok(())
}

pub fn reviewed_mtp_layer_schedule() -> Vec<Gemma4LayerType> {
    vec![
        Gemma4LayerType::SlidingAttention,
        Gemma4LayerType::SlidingAttention,
        Gemma4LayerType::SlidingAttention,
        Gemma4LayerType::FullAttention,
    ]
}

pub fn validate_gemma4_mtp_config(bytes: &[u8]) -> Result<Gemma4MtpConfig, ModelError> {
    let value = parse_model_source_json(bytes, "Gemma 4 MTP config")?;
    let root = object(&value, "Gemma 4 MTP config root")?;
    require_keys(
        root,
        &[
            "architectures",
            "audio_token_id",
            "backbone_hidden_size",
            "boa_token_id",
            "boi_token_id",
            "centroid_intermediate_top_k",
            "dtype",
            "eoa_token_index",
            "eoi_token_id",
            "image_token_id",
            "model_type",
            "num_centroids",
            "text_config",
            "tie_word_embeddings",
            "transformers_version",
            "use_ordered_embeddings",
        ],
        "Gemma 4 MTP config root",
    )?;
    if string_array(root, "architectures")? != ["Gemma4UnifiedAssistantForCausalLM"]
        || string(root, "model_type")? != "gemma4_unified_assistant"
        || string(root, "dtype")? != "bfloat16"
        || u64_value(root, "backbone_hidden_size")? != 3_840
        || u64_value(root, "centroid_intermediate_top_k")? != 32
        || u64_value(root, "num_centroids")? != 2_048
        || !bool_value(root, "tie_word_embeddings")?
        || bool_value(root, "use_ordered_embeddings")?
        || string(root, "transformers_version")? != "5.10.0.dev0"
        || u64_value(root, "audio_token_id")? != 258_881
        || u64_value(root, "boa_token_id")? != 256_000
        || u64_value(root, "boi_token_id")? != 255_999
        || u64_value(root, "eoa_token_index")? != 258_883
        || u64_value(root, "eoi_token_id")? != 258_882
        || u64_value(root, "image_token_id")? != 258_880
    {
        return Err(invalid("Gemma 4 MTP top-level config differs"));
    }
    let text = object_field(root, "text_config")?;
    require_keys(
        text,
        &[
            "_name_or_path",
            "architectures",
            "attention_bias",
            "attention_dropout",
            "attention_k_eq_v",
            "bos_token_id",
            "chunk_size_feed_forward",
            "dtype",
            "enable_moe_block",
            "eos_token_id",
            "final_logit_softcapping",
            "global_head_dim",
            "head_dim",
            "hidden_activation",
            "hidden_size",
            "hidden_size_per_layer_input",
            "id2label",
            "initializer_range",
            "intermediate_size",
            "is_encoder_decoder",
            "label2id",
            "layer_types",
            "max_position_embeddings",
            "model_type",
            "moe_intermediate_size",
            "num_attention_heads",
            "num_experts",
            "num_global_key_value_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_kv_shared_layers",
            "output_attentions",
            "output_hidden_states",
            "pad_token_id",
            "problem_type",
            "return_dict",
            "rms_norm_eps",
            "rope_parameters",
            "sliding_window",
            "tie_word_embeddings",
            "top_k_experts",
            "use_bidirectional_attention",
            "use_cache",
            "use_double_wide_mlp",
            "vocab_size",
            "vocab_size_per_layer_input",
        ],
        "Gemma 4 MTP text config",
    )?;
    let expected_schedule = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ];
    if !string(text, "_name_or_path")?.is_empty()
        || !is_null(text, "architectures")
        || string(text, "model_type")? != "gemma4_unified_text"
        || string(text, "dtype")? != "bfloat16"
        || u64_value(text, "hidden_size")? != 1_024
        || u64_value(text, "intermediate_size")? != 8_192
        || u64_value(text, "num_hidden_layers")? != 4
        || u64_value(text, "num_attention_heads")? != 16
        || u64_value(text, "num_key_value_heads")? != 8
        || u64_value(text, "num_global_key_value_heads")? != 1
        || u64_value(text, "num_kv_shared_layers")? != 4
        || u64_value(text, "head_dim")? != 256
        || u64_value(text, "global_head_dim")? != 512
        || string(text, "hidden_activation")? != "gelu_pytorch_tanh"
        || u64_value(text, "max_position_embeddings")? != 262_144
        || u64_value(text, "sliding_window")? != 1_024
        || f64_value(text, "rms_norm_eps")?.to_bits() != 1.0e-6_f64.to_bits()
        || f64_value(text, "initializer_range")?.to_bits() != 0.02_f64.to_bits()
        || bool_value(text, "attention_bias")?
        || f64_value(text, "attention_dropout")?.to_bits() != 0.0_f64.to_bits()
        || !bool_value(text, "attention_k_eq_v")?
        || !is_null(text, "final_logit_softcapping")
        || !bool_value(text, "tie_word_embeddings")?
        || !bool_value(text, "use_cache")?
        || bool_value(text, "enable_moe_block")?
        || bool_value(text, "use_double_wide_mlp")?
        || string(text, "use_bidirectional_attention")? != "vision"
        || u64_value(text, "vocab_size")? != GEMMA4_MTP_VOCAB_SIZE
        || string_array(text, "layer_types")? != expected_schedule
        || u64_value(text, "bos_token_id")? != 2
        || u64_value(text, "eos_token_id")? != 1
        || u64_value(text, "pad_token_id")? != 0
        || u64_value(text, "chunk_size_feed_forward")? != 0
        || u64_value(text, "hidden_size_per_layer_input")? != 0
        || u64_value(text, "vocab_size_per_layer_input")? != 0
        || bool_value(text, "is_encoder_decoder")?
        || bool_value(text, "output_attentions")?
        || bool_value(text, "output_hidden_states")?
        || !bool_value(text, "return_dict")?
        || !is_null(text, "moe_intermediate_size")
        || !is_null(text, "num_experts")
        || !is_null(text, "problem_type")
        || !is_null(text, "top_k_experts")
    {
        return Err(invalid("Gemma 4 MTP text config differs"));
    }
    if object_field(text, "id2label")?
        != &Map::from_iter([
            ("0".to_owned(), Value::String("LABEL_0".to_owned())),
            ("1".to_owned(), Value::String("LABEL_1".to_owned())),
        ])
        || object_field(text, "label2id")?
            != &Map::from_iter([
                ("LABEL_0".to_owned(), Value::from(0)),
                ("LABEL_1".to_owned(), Value::from(1)),
            ])
    {
        return Err(invalid("Gemma 4 MTP label metadata differs"));
    }
    let rope = object_field(text, "rope_parameters")?;
    require_keys(
        rope,
        &["full_attention", "sliding_attention"],
        "Gemma 4 MTP RoPE",
    )?;
    let sliding = object_field(rope, "sliding_attention")?;
    let full = object_field(rope, "full_attention")?;
    require_keys(
        sliding,
        &["rope_theta", "rope_type"],
        "Gemma 4 MTP sliding RoPE",
    )?;
    require_keys(
        full,
        &["partial_rotary_factor", "rope_theta", "rope_type"],
        "Gemma 4 MTP full RoPE",
    )?;
    if string(sliding, "rope_type")? != "default"
        || f64_value(sliding, "rope_theta")?.to_bits() != 10_000.0_f64.to_bits()
        || string(full, "rope_type")? != "proportional"
        || f64_value(full, "rope_theta")?.to_bits() != 1_000_000.0_f64.to_bits()
        || f64_value(full, "partial_rotary_factor")?.to_bits() != 0.25_f64.to_bits()
    {
        return Err(invalid("Gemma 4 MTP dual-RoPE config differs"));
    }
    Ok(Gemma4MtpConfig {
        hidden_size: 1_024,
        backbone_hidden_size: 3_840,
        intermediate_size: 8_192,
        layer_count: 4,
        attention_heads: 16,
        kv_heads: 8,
        global_kv_heads: 1,
        head_dim: 256,
        global_head_dim: 512,
        sliding_window: 1_024,
        max_position_embeddings: 262_144,
        vocab_size: 262_144,
        layer_types: reviewed_mtp_layer_schedule(),
        draft_to_target_kv_layers: GEMMA4_MTP_DRAFT_TO_TARGET_KV_LAYERS,
    })
}

pub fn validate_gemma4_mtp_generation_config(bytes: &[u8]) -> Result<(), ModelError> {
    let value = parse_model_source_json(bytes, "Gemma 4 MTP generation config")?;
    let root = object(&value, "Gemma 4 MTP generation config root")?;
    require_keys(
        root,
        &[
            "bos_token_id",
            "do_sample",
            "eos_token_id",
            "pad_token_id",
            "suppress_tokens",
            "temperature",
            "top_k",
            "top_p",
            "transformers_version",
        ],
        "Gemma 4 MTP generation config",
    )?;
    if u64_value(root, "bos_token_id")? != 2
        || u64_value(root, "eos_token_id")? != 1
        || u64_value(root, "pad_token_id")? != 0
        || !bool_value(root, "do_sample")?
        || f64_value(root, "temperature")?.to_bits() != 1.0_f64.to_bits()
        || u64_value(root, "top_k")? != 64
        || f64_value(root, "top_p")?.to_bits() != 0.95_f64.to_bits()
        || u64_array(root, "suppress_tokens")? != [258_883, 258_882]
        || string(root, "transformers_version")? != "5.10.0.dev0"
    {
        return Err(invalid("Gemma 4 MTP generation config differs"));
    }
    Ok(())
}

pub fn validate_gemma4_mtp_tokenizer_config(bytes: &[u8]) -> Result<(), ModelError> {
    let value = parse_model_source_json(bytes, "Gemma 4 MTP tokenizer config")?;
    let root = object(&value, "Gemma 4 MTP tokenizer config root")?;
    require_keys(
        root,
        &[
            "audio_token",
            "backend",
            "boa_token",
            "boi_token",
            "bos_token",
            "eoa_token",
            "eoc_token",
            "eoi_token",
            "eos_token",
            "eot_token",
            "escape_token",
            "etc_token",
            "etd_token",
            "etr_token",
            "extra_special_tokens",
            "image_token",
            "mask_token",
            "model_max_length",
            "pad_token",
            "padding_side",
            "soc_token",
            "sot_token",
            "stc_token",
            "std_token",
            "str_token",
            "think_token",
            "tokenizer_class",
            "unk_token",
        ],
        "Gemma 4 MTP tokenizer config",
    )?;
    let expected = BTreeMap::from([
        ("audio_token", "<|audio|>"),
        ("backend", "tokenizers"),
        ("boa_token", "<|audio>"),
        ("boi_token", "<|image>"),
        ("bos_token", "<bos>"),
        ("eoa_token", "<audio|>"),
        ("eoc_token", "<channel|>"),
        ("eoi_token", "<image|>"),
        ("eos_token", "<eos>"),
        ("eot_token", "<turn|>"),
        ("escape_token", "<|\"|>"),
        ("etc_token", "<tool_call|>"),
        ("etd_token", "<tool|>"),
        ("etr_token", "<tool_response|>"),
        ("image_token", "<|image|>"),
        ("mask_token", "<mask>"),
        ("pad_token", "<pad>"),
        ("padding_side", "left"),
        ("soc_token", "<|channel>"),
        ("sot_token", "<|turn>"),
        ("stc_token", "<|tool_call>"),
        ("std_token", "<|tool>"),
        ("str_token", "<|tool_response>"),
        ("think_token", "<|think|>"),
        ("tokenizer_class", "GemmaTokenizer"),
        ("unk_token", "<unk>"),
    ]);
    if expected
        .iter()
        .any(|(field, expected)| string(root, field).ok() != Some(*expected))
        || root
            .get("extra_special_tokens")
            .and_then(Value::as_array)
            .is_none_or(|values| !values.is_empty())
        || f64_value(root, "model_max_length")?.to_bits() != 1.0e30_f64.to_bits()
    {
        return Err(invalid("Gemma 4 MTP tokenizer config differs"));
    }
    Ok(())
}

pub fn validate_gemma4_mtp_target(
    lock: &Gemma4MtpModelLock,
    target: &Gemma4ModelLock,
) -> Result<(), ModelError> {
    validate_gemma4_mtp_lock(lock)?;
    let text = &target.model.architecture.text;
    let tokenizer = &target.model.tokenizer_contract;
    if target.model.repo_id != GEMMA4_12B_IT_REPO_ID
        || target.model.requested_revision != GEMMA4_12B_IT_REVISION
        || target.model.resolved_revision != GEMMA4_12B_IT_REVISION
        || target.fingerprint != GEMMA4_12B_IT_FINGERPRINT
        || target.aliases != [GEMMA4_12B_IT_ALIAS]
        || text.hidden_size != lock.model.target_compatibility.target_hidden_size
        || text.vocab_size != lock.model.target_compatibility.target_vocab_size
        || text.num_hidden_layers != lock.model.target_compatibility.target_layer_count
        || text.layer_types.len() != 48
        || text.layer_types.get(46) != Some(&Gemma4LayerType::SlidingAttention)
        || text.layer_types.get(47) != Some(&Gemma4LayerType::FullAttention)
        || tokenizer.tokenizer_class != "GemmaTokenizer"
        || tokenizer.vocab_size != GEMMA4_MTP_VOCAB_SIZE
        || tokenizer.special_token_ids.get("bos") != Some(&2)
        || tokenizer.special_token_ids.get("eos") != Some(&1)
        || tokenizer.special_token_ids.get("pad") != Some(&0)
        || tokenizer.special_token_ids.get("tool_response_begin") != Some(&50)
        || tokenizer.special_token_ids.get("turn_end") != Some(&106)
        || tokenizer.special_token_ids.get("video") != Some(&258_884)
    {
        return Err(invalid(
            "Gemma 4 MTP target fingerprint, hidden size, vocabulary, tokenizer, or KV mapping differs",
        ));
    }
    for (draft_layer, target_layer) in lock
        .model
        .architecture
        .layer_types
        .iter()
        .zip(lock.model.target_compatibility.draft_to_target_kv_layers)
    {
        if text.layer_types.get(target_layer as usize) != Some(draft_layer) {
            return Err(invalid("Gemma 4 MTP target KV layer type differs"));
        }
    }
    Ok(())
}

pub fn expected_gemma4_mtp_tensor_catalog() -> Result<BTreeMap<String, TensorDescriptor>, ModelError>
{
    let mut shapes = BTreeMap::new();
    insert_shape(&mut shapes, "model.embed_tokens.weight", &[262_144, 1_024])?;
    for (layer, layer_type) in reviewed_mtp_layer_schedule().into_iter().enumerate() {
        let prefix = format!("model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "pre_feedforward_layernorm.weight",
        ] {
            insert_shape(&mut shapes, format!("{prefix}.{suffix}"), &[1_024])?;
        }
        insert_shape(&mut shapes, format!("{prefix}.layer_scalar"), &[1])?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.mlp.down_proj.weight"),
            &[1_024, 8_192],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.mlp.gate_proj.weight"),
            &[8_192, 1_024],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.mlp.up_proj.weight"),
            &[8_192, 1_024],
        )?;
        let head_dim = match layer_type {
            Gemma4LayerType::SlidingAttention => 256,
            Gemma4LayerType::FullAttention => 512,
        };
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.o_proj.weight"),
            &[1_024, 16 * head_dim],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.q_norm.weight"),
            &[head_dim],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.q_proj.weight"),
            &[16 * head_dim, 1_024],
        )?;
    }
    insert_shape(&mut shapes, "model.norm.weight", &[1_024])?;
    insert_shape(&mut shapes, "post_projection.weight", &[3_840, 1_024])?;
    insert_shape(&mut shapes, "pre_projection.weight", &[1_024, 7_680])?;
    if shapes.len() as u64 != GEMMA4_MTP_TENSOR_COUNT
        || shapes
            .keys()
            .any(|name| name.contains(".k_proj.") || name.contains(".v_proj."))
    {
        return Err(invalid("derived Gemma 4 MTP tensor topology differs"));
    }

    let mut cursor = 0_u64;
    let mut catalog = BTreeMap::new();
    for (name, shape) in shapes {
        let elements = shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| invalid(format!("MTP tensor element count overflowed: {name}")))?;
        let byte_size = elements
            .checked_mul(2)
            .ok_or_else(|| invalid(format!("MTP tensor byte size overflowed: {name}")))?;
        let end = cursor
            .checked_add(byte_size)
            .ok_or_else(|| invalid("MTP payload range overflowed"))?;
        let absolute_start = DATA_BUFFER_START
            .checked_add(cursor)
            .ok_or_else(|| invalid("MTP tensor absolute start overflowed"))?;
        let absolute_end = DATA_BUFFER_START
            .checked_add(end)
            .ok_or_else(|| invalid("MTP tensor absolute end overflowed"))?;
        catalog.insert(
            name.clone(),
            TensorDescriptor {
                tensor_name: name,
                source_file: "model.safetensors".to_owned(),
                dtype: TensorDType::Bf16,
                shape,
                header_length_field_bytes: 8,
                header_length_bytes: GEMMA4_MTP_HEADER_BYTES,
                data_buffer_start: DATA_BUFFER_START,
                data_offset_basis: "data-buffer-relative".to_owned(),
                data_offsets: [cursor, end],
                absolute_byte_range: [absolute_start, absolute_end],
                byte_size,
            },
        );
        cursor = end;
    }
    if cursor != PAYLOAD_BYTES || DATA_BUFFER_START + cursor != GEMMA4_MTP_MODEL_BYTES {
        return Err(invalid("derived Gemma 4 MTP payload size differs"));
    }
    if gemma4_mtp_catalog_sha256(&catalog) != GEMMA4_MTP_CATALOG_SHA256 {
        return Err(invalid("derived Gemma 4 MTP catalog digest differs"));
    }
    Ok(catalog)
}

pub fn gemma4_mtp_catalog_sha256(catalog: &BTreeMap<String, TensorDescriptor>) -> String {
    let mut hasher = Sha256::new();
    for descriptor in catalog.values() {
        hasher.update(descriptor.tensor_name.as_bytes());
        hasher.update(b"\t");
        let dtype = match descriptor.dtype {
            TensorDType::Bf16 => "BF16",
            TensorDType::F16 => "F16",
            TensorDType::F32 => "F32",
            TensorDType::I32 => "I32",
            TensorDType::I64 => "I64",
            TensorDType::U8 => "U8",
        };
        hasher.update(dtype.as_bytes());
        hasher.update(b"\t");
        for (index, dimension) in descriptor.shape.iter().enumerate() {
            if index != 0 {
                hasher.update(b"x");
            }
            hasher.update(dimension.to_string().as_bytes());
        }
        hasher.update(b"\t");
        hasher.update(descriptor.data_offsets[0].to_string().as_bytes());
        hasher.update(b"-");
        hasher.update(descriptor.data_offsets[1].to_string().as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn insert_shape(
    shapes: &mut BTreeMap<String, Vec<u64>>,
    name: impl Into<String>,
    shape: &[u64],
) -> Result<(), ModelError> {
    let name = name.into();
    if shape.is_empty()
        || shape.contains(&0)
        || shapes.insert(name.clone(), shape.to_vec()).is_some()
    {
        return Err(invalid(format!("invalid or duplicate MTP tensor: {name}")));
    }
    Ok(())
}

pub fn verify_gemma4_mtp_cache(
    lock: &Gemma4MtpModelLock,
    cache_root: impl AsRef<Path>,
    target: &Gemma4ModelLock,
) -> Result<VerifiedGemma4Mtp, ModelError> {
    validate_gemma4_mtp_lock(lock)?;
    validate_gemma4_mtp_target(lock, target)?;
    let cache_root = cache_root.as_ref();
    let root_metadata = std::fs::symlink_metadata(cache_root)
        .map_err(|error| invalid(format!("MTP cache root metadata failed: {error}")))?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(invalid("Gemma 4 MTP cache root is not a bound directory"));
    }
    validate_cache_entry_set(cache_root, &lock.model.files)?;

    // Every complete content hash is admitted before config parsing or tensor
    // planning. The opened descriptors are retained for all later reads.
    let mut files = BTreeMap::new();
    for locked in &lock.model.files {
        let bound = open_and_verify_file(cache_root, locked)?;
        files.insert(locked.path.clone(), bound);
    }

    let config_bytes = read_small_file(&files, "config.json", 64 * 1024)?;
    let config = validate_gemma4_mtp_config(&config_bytes)?;
    let generation = read_small_file(&files, "generation_config.json", 64 * 1024)?;
    validate_gemma4_mtp_generation_config(&generation)?;
    let tokenizer_config = read_small_file(&files, "tokenizer_config.json", 256 * 1024)?;
    validate_gemma4_mtp_tokenizer_config(&tokenizer_config)?;
    let tokenizer = read_small_file(
        &files,
        "tokenizer.json",
        usize::try_from(MAX_SUPPORT_FILE_BYTES)
            .map_err(|_| invalid("support file limit does not fit usize"))?,
    )?;
    validate_gemma4_mtp_tokenizer(&tokenizer, &lock.model.tokenizer_contract)?;
    let tensors = validate_gemma4_mtp_safetensors(&files, &lock.model.tensor_contract)?;

    for file in files.values() {
        assert_bound_file_stable(file)?;
    }
    Ok(VerifiedGemma4Mtp {
        lock_fingerprint: lock.fingerprint.clone(),
        target_fingerprint: target.fingerprint.clone(),
        cache_root: cache_root.to_path_buf(),
        files,
        tensors,
        config,
    })
}

fn validate_cache_entry_set(cache_root: &Path, locked: &[LockedFile]) -> Result<(), ModelError> {
    let expected = locked
        .iter()
        .map(|file| file.path.clone())
        .collect::<BTreeSet<_>>();
    let entries = std::fs::read_dir(cache_root)
        .map_err(|error| invalid(format!("MTP cache root listing failed: {error}")))?;
    let mut actual = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|error| invalid(format!("MTP cache entry failed: {error}")))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("MTP cache entry name is not UTF-8"))?;
        let metadata = std::fs::symlink_metadata(entry.path())
            .map_err(|error| invalid(format!("MTP cache entry metadata failed: {error}")))?;
        if name == ".cache" && metadata.is_dir() && !metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.file_type().is_symlink() || !metadata.is_file() || !actual.insert(name) {
            return Err(invalid(
                "MTP cache contains a non-regular or duplicate entry",
            ));
        }
    }
    if actual != expected {
        return Err(invalid(
            "MTP cache file set differs from the seven-file lock",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn open_and_verify_file(cache_root: &Path, locked: &LockedFile) -> Result<BoundFile, ModelError> {
    let path = cache_root.join(&locked.path);
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&path)
        .map_err(|error| {
            invalid(format!(
                "MTP locked file open failed: {}: {error}",
                locked.path
            ))
        })?;
    let before = file
        .metadata()
        .map_err(|error| invalid(format!("MTP locked file metadata failed: {error}")))?;
    if !before.is_file() || before.len() != locked.size_bytes {
        return Err(invalid(format!(
            "MTP locked file size/type differs: {}",
            locked.path
        )));
    }
    let mut hasher = Sha256::new();
    let mut offset = 0_u64;
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    while offset < before.len() {
        let remaining = before.len() - offset;
        let request = buffer
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let read = file
            .read_at(&mut buffer[..request], offset)
            .map_err(|error| invalid(format!("MTP locked file hash read failed: {error}")))?;
        if read == 0 {
            return Err(invalid("MTP locked file ended during hashing"));
        }
        hasher.update(&buffer[..read]);
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| invalid("MTP locked file hash offset overflowed"))?;
    }
    if format!("{:x}", hasher.finalize()) != locked.sha256 {
        return Err(invalid(format!(
            "MTP locked file SHA-256 differs: {}",
            locked.path
        )));
    }
    let after = file
        .metadata()
        .map_err(|error| invalid(format!("MTP locked file restat failed: {error}")))?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
    {
        return Err(invalid("MTP locked file changed during hashing"));
    }
    Ok(BoundFile {
        file: Arc::new(file),
        size: before.len(),
        device: before.dev(),
        inode: before.ino(),
        modified_seconds: before.mtime(),
        modified_nanoseconds: before.mtime_nsec(),
        changed_seconds: before.ctime(),
        changed_nanoseconds: before.ctime_nsec(),
    })
}

#[cfg(not(unix))]
fn open_and_verify_file(_cache_root: &Path, _locked: &LockedFile) -> Result<BoundFile, ModelError> {
    Err(invalid(
        "Gemma 4 MTP bound cache verification requires Unix",
    ))
}

#[cfg(unix)]
fn assert_bound_file_stable(file: &BoundFile) -> Result<(), ModelError> {
    let metadata = file
        .file
        .metadata()
        .map_err(|error| invalid(format!("MTP bound file restat failed: {error}")))?;
    if metadata.dev() != file.device
        || metadata.ino() != file.inode
        || metadata.len() != file.size
        || metadata.mtime() != file.modified_seconds
        || metadata.mtime_nsec() != file.modified_nanoseconds
        || metadata.ctime() != file.changed_seconds
        || metadata.ctime_nsec() != file.changed_nanoseconds
    {
        return Err(invalid(
            "MTP bound file identity changed after verification",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn assert_bound_file_stable(_file: &BoundFile) -> Result<(), ModelError> {
    Err(invalid(
        "Gemma 4 MTP bound cache verification requires Unix",
    ))
}

#[cfg(unix)]
fn read_bound_range(file: &BoundFile, offset: u64, length: usize) -> Result<Vec<u8>, ModelError> {
    let end = offset
        .checked_add(length as u64)
        .ok_or_else(|| invalid("MTP bound read range overflowed"))?;
    if end > file.size {
        return Err(invalid("MTP bound read exceeds file"));
    }
    let mut bytes = vec![0_u8; length];
    let mut read_total = 0_usize;
    while read_total < length {
        let read_offset = offset
            .checked_add(read_total as u64)
            .ok_or_else(|| invalid("MTP bound read offset overflowed"))?;
        let read = file
            .file
            .read_at(&mut bytes[read_total..], read_offset)
            .map_err(|error| invalid(format!("MTP bound read failed: {error}")))?;
        if read == 0 {
            return Err(invalid("MTP bound file ended during range read"));
        }
        read_total += read;
    }
    assert_bound_file_stable(file)?;
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_bound_range(
    _file: &BoundFile,
    _offset: u64,
    _length: usize,
) -> Result<Vec<u8>, ModelError> {
    Err(invalid(
        "Gemma 4 MTP bound cache verification requires Unix",
    ))
}

fn read_small_file(
    files: &BTreeMap<String, BoundFile>,
    path: &str,
    limit: usize,
) -> Result<Vec<u8>, ModelError> {
    let file = files
        .get(path)
        .ok_or_else(|| invalid(format!("verified MTP file is absent: {path}")))?;
    let length = usize::try_from(file.size)
        .map_err(|_| invalid(format!("MTP support file size does not fit usize: {path}")))?;
    if length > limit {
        return Err(invalid(format!(
            "MTP support file exceeds read limit: {path}"
        )));
    }
    read_bound_range(file, 0, length)
}

fn validate_gemma4_mtp_tokenizer(
    bytes: &[u8],
    contract: &Gemma4MtpTokenizerContract,
) -> Result<(), ModelError> {
    // Full file identity is already fixed before this parse. The semantic
    // digests intentionally cover the common target/assistant BPE core while
    // the exact one-token named-video difference is checked separately.
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| ModelError::Json(format!("MTP tokenizer JSON: {error}")))?;
    let root = object(&value, "Gemma 4 MTP tokenizer root")?;
    let model = object_field(root, "model")?;
    if string(model, "type")? != "BPE" {
        return Err(invalid("MTP tokenizer model is not BPE"));
    }
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("MTP tokenizer vocabulary is absent"))?;
    if vocab.len() as u64 != contract.vocab_size {
        return Err(invalid("MTP tokenizer vocabulary size differs"));
    }
    let mut vocab_entries = Vec::with_capacity(vocab.len());
    let mut seen_ids = BTreeSet::new();
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or_else(|| invalid("MTP tokenizer vocabulary ID is invalid"))?;
        if id >= contract.vocab_size || !seen_ids.insert(id) {
            return Err(invalid(
                "MTP tokenizer vocabulary IDs are not a permutation",
            ));
        }
        vocab_entries.push((id, token.as_bytes()));
    }
    vocab_entries.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(right.1)));
    let mut vocab_digest = Sha256::new();
    vocab_digest.update(b"sllm-tokenizer-vocab-v1");
    vocab_digest.update((vocab_entries.len() as u64).to_le_bytes());
    for (id, token) in vocab_entries {
        vocab_digest.update(id.to_le_bytes());
        vocab_digest.update((token.len() as u64).to_le_bytes());
        vocab_digest.update(token);
    }
    if format!("{:x}", vocab_digest.finalize()) != contract.vocab_semantic_sha256 {
        return Err(invalid("MTP tokenizer vocabulary semantic digest differs"));
    }

    let merges = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("MTP tokenizer merges are absent"))?;
    if merges.len() as u64 != contract.merge_count {
        return Err(invalid("MTP tokenizer merge count differs"));
    }
    let mut merge_digest = Sha256::new();
    merge_digest.update(b"sllm-tokenizer-merges-v1");
    merge_digest.update((merges.len() as u64).to_le_bytes());
    for merge in merges {
        let pair = merge
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| invalid("MTP tokenizer merge is not a two-token pair"))?;
        let left = pair[0]
            .as_str()
            .ok_or_else(|| invalid("MTP tokenizer merge left token is not a string"))?;
        let right = pair[1]
            .as_str()
            .ok_or_else(|| invalid("MTP tokenizer merge right token is not a string"))?;
        let joined_length = left
            .len()
            .checked_add(1)
            .and_then(|length| length.checked_add(right.len()))
            .ok_or_else(|| invalid("MTP tokenizer merge length overflowed"))?;
        merge_digest.update((joined_length as u64).to_le_bytes());
        merge_digest.update(left.as_bytes());
        merge_digest.update(b" ");
        merge_digest.update(right.as_bytes());
    }
    if format!("{:x}", merge_digest.finalize()) != contract.merges_semantic_sha256 {
        return Err(invalid("MTP tokenizer merges semantic digest differs"));
    }

    let added = root
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("MTP tokenizer added tokens are absent"))?;
    let expected_added = BTreeMap::from([
        (0, "<pad>"),
        (1, "<eos>"),
        (2, "<bos>"),
        (3, "<unk>"),
        (4, "<mask>"),
        (46, "<|tool>"),
        (47, "<tool|>"),
        (48, "<|tool_call>"),
        (49, "<tool_call|>"),
        (50, "<|tool_response>"),
        (51, "<tool_response|>"),
        (52, "<|\"|>"),
        (98, "<|think|>"),
        (100, "<|channel>"),
        (101, "<channel|>"),
        (105, "<|turn>"),
        (106, "<turn|>"),
        (255_999, "<|image>"),
        (256_000, "<|audio>"),
        (258_880, "<|image|>"),
        (258_881, "<|audio|>"),
        (258_882, "<image|>"),
        (258_883, "<audio|>"),
    ]);
    let mut actual_added = BTreeMap::new();
    for token in added {
        let token = object(token, "MTP added token")?;
        let id = u64_value(token, "id")?;
        let content = string(token, "content")?;
        if token.get("special").and_then(Value::as_bool) != Some(true)
            || actual_added.insert(id, content).is_some()
        {
            return Err(invalid("MTP added token identity differs"));
        }
    }
    if actual_added != expected_added
        || actual_added.contains_key(&contract.target_named_video_token_id)
        || contract.assistant_named_video_token_present
    {
        return Err(invalid(
            "MTP assistant added tokens differ from the reviewed target-pair exception",
        ));
    }
    Ok(())
}

fn validate_gemma4_mtp_safetensors(
    files: &BTreeMap<String, BoundFile>,
    contract: &Gemma4MtpTensorContract,
) -> Result<BTreeMap<String, TensorDescriptor>, ModelError> {
    let file = files
        .get(&contract.source_path)
        .ok_or_else(|| invalid("verified MTP safetensors source is absent"))?;
    let length_bytes = read_bound_range(file, 0, 8)?;
    let header_length = u64::from_le_bytes(
        length_bytes
            .try_into()
            .map_err(|_| invalid("MTP header length field is not eight bytes"))?,
    );
    if header_length != contract.header_length_bytes
        || contract.header_length_field_bytes != 8
        || contract.data_buffer_start != header_length + 8
    {
        return Err(invalid("MTP safetensors header geometry differs"));
    }
    let header_length_usize = usize::try_from(header_length)
        .map_err(|_| invalid("MTP safetensors header length does not fit usize"))?;
    let header_bytes = read_bound_range(file, 8, header_length_usize)?;
    if format!("{:x}", Sha256::digest(&header_bytes)) != contract.header_sha256 {
        return Err(invalid("MTP safetensors header SHA-256 differs"));
    }
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|error| ModelError::Json(format!("MTP safetensors header: {error}")))?;
    let header = object(&header, "MTP safetensors header")?;
    let metadata = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("MTP safetensors metadata is absent"))?;
    if metadata.len() != 1 || metadata.get("format").and_then(Value::as_str) != Some("pt") {
        return Err(invalid("MTP safetensors metadata differs"));
    }
    let mut tensors = BTreeMap::new();
    let mut spans = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let tensor = object(value, "MTP tensor metadata")?;
        require_keys(
            tensor,
            &["data_offsets", "dtype", "shape"],
            "MTP tensor metadata",
        )?;
        if string(tensor, "dtype")? != "BF16" {
            return Err(invalid(format!("MTP tensor dtype differs: {name}")));
        }
        let shape = u64_array(tensor, "shape")?;
        if shape.is_empty() || shape.contains(&0) {
            return Err(invalid(format!("MTP tensor shape is empty/zero: {name}")));
        }
        let offsets = u64_array(tensor, "data_offsets")?;
        if offsets.len() != 2 || offsets[0] >= offsets[1] {
            return Err(invalid(format!("MTP tensor offsets differ: {name}")));
        }
        let expected_bytes = shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| invalid(format!("MTP tensor size overflowed: {name}")))?;
        if offsets[1] - offsets[0] != expected_bytes {
            return Err(invalid(format!("MTP tensor shape/range differs: {name}")));
        }
        let absolute_start = DATA_BUFFER_START
            .checked_add(offsets[0])
            .ok_or_else(|| invalid("MTP tensor absolute start overflowed"))?;
        let absolute_end = DATA_BUFFER_START
            .checked_add(offsets[1])
            .ok_or_else(|| invalid("MTP tensor absolute end overflowed"))?;
        if absolute_end > file.size {
            return Err(invalid(format!("MTP tensor exceeds source: {name}")));
        }
        tensors.insert(
            name.clone(),
            TensorDescriptor {
                tensor_name: name.clone(),
                source_file: contract.source_path.clone(),
                dtype: TensorDType::Bf16,
                shape,
                header_length_field_bytes: 8,
                header_length_bytes: header_length,
                data_buffer_start: DATA_BUFFER_START,
                data_offset_basis: "data-buffer-relative".to_owned(),
                data_offsets: [offsets[0], offsets[1]],
                absolute_byte_range: [absolute_start, absolute_end],
                byte_size: expected_bytes,
            },
        );
        spans.push((offsets[0], offsets[1], name.as_str()));
    }
    if tensors.len() as u64 != contract.tensor_count {
        return Err(invalid("MTP safetensors tensor count differs"));
    }
    spans.sort_unstable();
    let mut cursor = 0_u64;
    for (start, end, name) in spans {
        if start != cursor {
            return Err(invalid(format!(
                "MTP safetensors payload has a gap/overlap: {name}"
            )));
        }
        cursor = end;
    }
    if cursor != contract.payload_bytes || DATA_BUFFER_START + cursor != file.size {
        return Err(invalid("MTP safetensors payload coverage differs"));
    }
    let expected = expected_gemma4_mtp_tensor_catalog()?;
    if tensors != expected || gemma4_mtp_catalog_sha256(&tensors) != contract.catalog_sha256 {
        return Err(invalid("MTP safetensors exact catalog differs"));
    }
    assert_bound_file_stable(file)?;
    Ok(tensors)
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

fn object<'a>(value: &'a Value, scope: &str) -> Result<&'a Map<String, Value>, ModelError> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{scope} must be an object")))
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Map<String, Value>, ModelError> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(format!("{field} must be an object")))
}

fn require_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    scope: &str,
) -> Result<(), ModelError> {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(format!("{scope} field set differs")));
    }
    Ok(())
}

fn string<'a>(object: &'a Map<String, Value>, field: &str) -> Result<&'a str, ModelError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{field} must be a string")))
}

fn string_array(object: &Map<String, Value>, field: &str) -> Result<Vec<String>, ModelError> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{field} must be an array")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid(format!("{field} entries must be strings")))
        })
        .collect()
}

fn u64_value(object: &Map<String, Value>, field: &str) -> Result<u64, ModelError> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("{field} must be a non-negative integer")))
}

fn u64_array(object: &Map<String, Value>, field: &str) -> Result<Vec<u64>, ModelError> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{field} must be an array")))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid(format!("{field} entries must be non-negative integers")))
        })
        .collect()
}

fn f64_value(object: &Map<String, Value>, field: &str) -> Result<f64, ModelError> {
    object
        .get(field)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid(format!("{field} must be finite numeric data")))
}

fn bool_value(object: &Map<String, Value>, field: &str) -> Result<bool, ModelError> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid(format!("{field} must be boolean")))
}

fn is_null(object: &Map<String, Value>, field: &str) -> bool {
    object.get(field).is_some_and(Value::is_null)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK_BYTES: &[u8] =
        include_bytes!("../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json");
    const TARGET_LOCK_BYTES: &[u8] =
        include_bytes!("../../../docs/models/locks/gemma4-12b-it-bf16.json");

    const CONFIG_FIXTURE: &[u8] = br#"{
      "architectures":["Gemma4UnifiedAssistantForCausalLM"],
      "audio_token_id":258881,"backbone_hidden_size":3840,"boa_token_id":256000,
      "boi_token_id":255999,"centroid_intermediate_top_k":32,"dtype":"bfloat16",
      "eoa_token_index":258883,"eoi_token_id":258882,"image_token_id":258880,
      "model_type":"gemma4_unified_assistant","num_centroids":2048,
      "text_config":{
        "_name_or_path":"","architectures":null,"attention_bias":false,
        "attention_dropout":0.0,"attention_k_eq_v":true,"bos_token_id":2,
        "chunk_size_feed_forward":0,"dtype":"bfloat16","enable_moe_block":false,
        "eos_token_id":1,"final_logit_softcapping":null,"global_head_dim":512,
        "head_dim":256,"hidden_activation":"gelu_pytorch_tanh","hidden_size":1024,
        "hidden_size_per_layer_input":0,"id2label":{"0":"LABEL_0","1":"LABEL_1"},
        "initializer_range":0.02,"intermediate_size":8192,"is_encoder_decoder":false,
        "label2id":{"LABEL_0":0,"LABEL_1":1},
        "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
        "max_position_embeddings":262144,"model_type":"gemma4_unified_text",
        "moe_intermediate_size":null,"num_attention_heads":16,"num_experts":null,
        "num_global_key_value_heads":1,"num_hidden_layers":4,"num_key_value_heads":8,
        "num_kv_shared_layers":4,"output_attentions":false,"output_hidden_states":false,
        "pad_token_id":0,"problem_type":null,"return_dict":true,"rms_norm_eps":1e-6,
        "rope_parameters":{
          "full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},
          "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}
        },
        "sliding_window":1024,"tie_word_embeddings":true,"top_k_experts":null,
        "use_bidirectional_attention":"vision","use_cache":true,"use_double_wide_mlp":false,
        "vocab_size":262144,"vocab_size_per_layer_input":0
      },
      "tie_word_embeddings":true,"transformers_version":"5.10.0.dev0",
      "use_ordered_embeddings":false
    }"#;

    fn generation_fixture() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "bos_token_id": 2,
            "do_sample": true,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "suppress_tokens": [258883, 258882],
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "transformers_version": "5.10.0.dev0"
        }))
        .expect("fixture serializes")
    }

    const TOKENIZER_CONFIG_FIXTURE: &[u8] = br#"{
      "audio_token":"<|audio|>","backend":"tokenizers","boa_token":"<|audio>",
      "boi_token":"<|image>","bos_token":"<bos>","eoa_token":"<audio|>",
      "eoc_token":"<channel|>","eoi_token":"<image|>","eos_token":"<eos>",
      "eot_token":"<turn|>","escape_token":"<|\"|>","etc_token":"<tool_call|>",
      "etd_token":"<tool|>","etr_token":"<tool_response|>","extra_special_tokens":[],
      "image_token":"<|image|>","mask_token":"<mask>",
      "model_max_length":1000000000000000019884624838656,"pad_token":"<pad>",
      "padding_side":"left","soc_token":"<|channel>","sot_token":"<|turn>",
      "stc_token":"<|tool_call>","std_token":"<|tool>",
      "str_token":"<|tool_response>","think_token":"<|think|>",
      "tokenizer_class":"GemmaTokenizer","unk_token":"<unk>"
    }"#;

    fn mutated_lock(mut mutate: impl FnMut(&mut Value)) -> Vec<u8> {
        let mut value: Value = serde_json::from_slice(LOCK_BYTES).expect("tracked lock JSON");
        mutate(&mut value);
        let provisional = serde_json::to_vec(&value).expect("mutated lock serializes");
        value["fingerprint"] = Value::String(
            fingerprint_for_json(&provisional).expect("mutated lock fingerprint computes"),
        );
        serde_json::to_vec(&value).expect("fingerprinted mutation serializes")
    }

    #[test]
    fn tracked_lock_and_exact_target_pair_are_accepted() {
        let lock = parse_gemma4_mtp_model_lock(LOCK_BYTES).expect("tracked MTP lock is valid");
        let target = crate::gemma4::parse_gemma4_model_lock(TARGET_LOCK_BYTES)
            .expect("tracked target lock is valid");
        validate_gemma4_mtp_target(&lock, &target).expect("reviewed target pair is valid");
        assert_eq!(lock.fingerprint(), GEMMA4_MTP_FINGERPRINT);
        assert_eq!(lock.target_fingerprint(), GEMMA4_12B_IT_FINGERPRINT);
        assert!(lock.requires_target_tokenizer());
        assert!(
            !lock
                .model
                .tokenizer_contract
                .assistant_named_video_token_present
        );
        assert_eq!(
            lock.model.tokenizer_contract.target_named_video_token_id,
            258_884
        );
    }

    #[test]
    fn lock_rejects_re_fingerprinted_identity_and_topology_substitution() {
        let mutations = [
            mutated_lock(|value| value["model"]["resolved_revision"] = "1".repeat(40).into()),
            mutated_lock(|value| value["model"]["architecture"]["head_dim"] = 257.into()),
            mutated_lock(|value| {
                value["model"]["target_compatibility"]["draft_to_target_kv_layers"][2] = 45.into();
            }),
            mutated_lock(|value| {
                value["model"]["tokenizer_contract"]["wire_source"] = "assistant".into();
            }),
        ];
        for bytes in mutations {
            assert!(parse_gemma4_mtp_model_lock(&bytes).is_err());
        }
    }

    #[test]
    fn config_generation_and_tokenizer_configs_are_closed() {
        let config_bytes = CONFIG_FIXTURE.to_vec();
        let config = validate_gemma4_mtp_config(&config_bytes).expect("reviewed config is valid");
        assert_eq!(config.layer_types, reviewed_mtp_layer_schedule());
        assert_eq!(
            config.draft_to_target_kv_layers,
            GEMMA4_MTP_DRAFT_TO_TARGET_KV_LAYERS
        );
        validate_gemma4_mtp_generation_config(&generation_fixture())
            .expect("reviewed generation config is valid");
        validate_gemma4_mtp_tokenizer_config(TOKENIZER_CONFIG_FIXTURE)
            .expect("reviewed tokenizer config is valid");

        let mut wrong_head: Value = serde_json::from_slice(&config_bytes).unwrap();
        wrong_head["text_config"]["head_dim"] = 255.into();
        assert!(validate_gemma4_mtp_config(&serde_json::to_vec(&wrong_head).unwrap()).is_err());
        let mut unknown: Value = serde_json::from_slice(&config_bytes).unwrap();
        unknown["text_config"]["future_attention"] = true.into();
        assert!(validate_gemma4_mtp_config(&serde_json::to_vec(&unknown).unwrap()).is_err());
        let mut wrong_generation: Value =
            serde_json::from_slice(&generation_fixture()).expect("generation fixture");
        wrong_generation["suppress_tokens"] = serde_json::json!([258884, 258882]);
        assert!(
            validate_gemma4_mtp_generation_config(&serde_json::to_vec(&wrong_generation).unwrap())
                .is_err()
        );
    }

    #[test]
    fn derived_catalog_is_the_exact_48_tensor_kv_shared_layout() {
        let catalog = expected_gemma4_mtp_tensor_catalog().expect("catalog derives exactly");
        assert_eq!(catalog.len(), GEMMA4_MTP_TENSOR_COUNT as usize);
        assert_eq!(
            gemma4_mtp_catalog_sha256(&catalog),
            GEMMA4_MTP_CATALOG_SHA256
        );
        assert!(
            catalog
                .keys()
                .all(|name| !name.contains(".k_proj.") && !name.contains(".v_proj."))
        );
        assert_eq!(
            catalog["model.layers.0.self_attn.q_proj.weight"].shape,
            [4096, 1024]
        );
        assert_eq!(
            catalog["model.layers.3.self_attn.q_proj.weight"].shape,
            [8192, 1024]
        );
        assert_eq!(
            catalog["pre_projection.weight"].absolute_end(),
            GEMMA4_MTP_MODEL_BYTES
        );
    }

    #[test]
    fn exact_target_pair_mutations_are_rejected() {
        let lock = parse_gemma4_mtp_model_lock(LOCK_BYTES).expect("tracked MTP lock is valid");
        let mut target = crate::gemma4::parse_gemma4_model_lock(TARGET_LOCK_BYTES)
            .expect("tracked target lock is valid");
        target.model.architecture.text.hidden_size = 3_839;
        assert!(validate_gemma4_mtp_target(&lock, &target).is_err());
    }

    #[test]
    #[ignore = "requires the reviewed external 845 MB MTP cache"]
    fn reviewed_external_cache_passes_full_identity_catalog_and_read_checks() {
        let lock = parse_gemma4_mtp_model_lock(LOCK_BYTES).expect("tracked MTP lock is valid");
        let target = crate::gemma4::parse_gemma4_model_lock(TARGET_LOCK_BYTES)
            .expect("tracked target lock is valid");
        let verified = verify_gemma4_mtp_cache(
            &lock,
            "/home/homelab1/.cache/sllm/models/google--gemma-4-12B-it-assistant",
            &target,
        )
        .expect("reviewed assistant cache verifies");
        assert_eq!(verified.tensors().len(), GEMMA4_MTP_TENSOR_COUNT as usize);
        assert_eq!(
            verified.config().draft_to_target_kv_layers,
            [46, 46, 46, 47]
        );
        assert_eq!(
            verified
                .read_tensor_range("model.layers.0.layer_scalar", 0, 2)
                .expect("bound tensor read")
                .len(),
            2
        );
        assert!(
            verified
                .read_tensor_range("model.layers.0.layer_scalar", 1, 2)
                .is_err()
        );
    }
}
