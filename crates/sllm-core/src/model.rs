//! Offline, fail-closed model-lock and safetensors host validation.
//!
//! This module deliberately has no network, custom-code, CPU numerical, mmap,
//! or unsafe execution path.  `parse_model_lock` accepts only the restricted
//! model-lock JSON domain: JSON strings, booleans, null, and integers in the
//! RFC 8785/I-JSON safe range.  In particular, floating-point JSON numbers are
//! rejected while computing a lock identity.  A source model config may use a
//! JSON number for `rms_norm_eps`; `validate_model_config` accepts that one
//! finite positive value and compares it with the lock's decimal text.  This
//! exception is intentionally outside the fingerprint domain.

use serde::de::{self, DeserializeOwned, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, Metadata, OpenOptions};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};

const JCS_SAFE_INTEGER_MIN: i128 = -9_007_199_254_740_991;
const JCS_SAFE_INTEGER_MAX: u128 = 9_007_199_254_740_991;
const MAX_LOCK_JSON_BYTES: usize = 1024 * 1024;
const MAX_CONFIG_JSON_BYTES: usize = 1024 * 1024;
const MAX_INDEX_JSON_BYTES: usize = 1024 * 1024;
const MAX_TOKENIZER_JSON_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOKENIZER_CONFIG_JSON_BYTES: usize = 256 * 1024;
const MAX_CHAT_TEMPLATE_JINJA_BYTES: usize = 64 * 1024;
const MAX_SAFE_TENSOR_HEADER: u64 = 4 * 1024 * 1024;
const MAX_JSON_DEPTH: usize = 64;
const MAX_JSON_COLLECTION_ITEMS: usize = 1_000_000;
const MAX_JSON_STRING_BYTES: usize = 1024 * 1024;
const MAX_RANGE_READ_BYTES: usize = 16 * 1024 * 1024;
const QWEN_REPO_ID: &str = "Qwen/Qwen3.5-4B";
const QWEN_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
const QWEN_FINGERPRINT: &str =
    "sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935";

// Linux is the supported host platform.  These are kept local instead of
// adding a new direct libc dependency solely for model-lock file opening.
#[cfg(unix)]
const O_NONBLOCK: i32 = 0o4000;
#[cfg(unix)]
const O_NOFOLLOW: i32 = 0o400000;
#[cfg(unix)]
const O_CLOEXEC: i32 = 0o2000000;

/// A backend-independent error from model-lock, config, or cache validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    Json(String),
    Schema(String),
    FingerprintMismatch { expected: String, actual: String },
    Invalid(String),
    Io { path: PathBuf, message: String },
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(message) => write!(formatter, "invalid model JSON: {message}"),
            Self::Schema(message) => write!(formatter, "model-lock schema error: {message}"),
            Self::FingerprintMismatch { expected, actual } => write!(
                formatter,
                "model-lock fingerprint mismatch: expected {expected}, computed {actual}"
            ),
            Self::Invalid(message) => write!(formatter, "invalid model contract: {message}"),
            Self::Io { path, message } => write!(
                formatter,
                "model I/O error at {}: {message}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ModelError {}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

fn io_error(path: &Path, error: impl fmt::Display) -> ModelError {
    ModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

/// Numeric types accepted by the lock's typed model and safetensors metadata.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub enum TensorDType {
    #[serde(rename = "BF16")]
    Bf16,
    #[serde(rename = "F16")]
    F16,
    #[serde(rename = "F32")]
    F32,
    #[serde(rename = "I32")]
    I32,
    #[serde(rename = "I64")]
    I64,
    #[serde(rename = "U8")]
    U8,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub enum AccumulationDType {
    #[serde(rename = "FP32")]
    Fp32,
}

impl TensorDType {
    fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub enum LayerType {
    #[serde(rename = "linear_attention")]
    LinearAttention,
    #[serde(rename = "full_attention")]
    FullAttention,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub enum ComponentStatus {
    #[serde(rename = "consumed")]
    Consumed,
    #[serde(rename = "known-unconsumed")]
    KnownUnconsumed,
    #[serde(rename = "absent")]
    Absent,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub enum ClassificationStatus {
    #[serde(rename = "consumed")]
    Consumed,
    #[serde(rename = "known-unconsumed")]
    KnownUnconsumed,
    #[serde(rename = "partially-consumed")]
    PartiallyConsumed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub enum NormalizationKind {
    #[serde(rename = "rmsnorm")]
    RmsNorm,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
pub enum ScaleMode {
    #[serde(rename = "offset-one")]
    OffsetOne,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub intermediate_size: u64,
    pub dtype: TensorDType,
    pub rms_norm_eps: String,
    pub full_attention_interval: u64,
    pub layer_types: Vec<LayerType>,
    pub tie_word_embeddings: bool,
    pub vocab_size: u64,
    pub mtp_num_hidden_layers: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayerSchedule {
    pub kind: String,
    pub num_hidden_layers: u64,
    pub full_attention_interval: u64,
    pub layer_types: Vec<LayerType>,
    pub allowed_types: Vec<LayerType>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComponentMetadata {
    pub present: bool,
    pub tensor_prefix: String,
    pub tensor_count: u64,
    pub phase3_status: ComponentStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelArchitecture {
    pub architectures: Vec<String>,
    pub top_level_architecture: String,
    pub model_type: String,
    pub text_model_type: String,
    pub phase_scope: String,
    pub custom_code: bool,
    pub converted: bool,
    pub moe: bool,
    pub vision: ComponentMetadata,
    pub mtp: ComponentMetadata,
    pub text_config: TextConfig,
    pub layer_schedule: LayerSchedule,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TensorClassification {
    pub id: String,
    pub prefix: String,
    pub tensor_count: u64,
    pub phase3_status: ClassificationStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TensorContract {
    pub index_path: String,
    pub indexed_tensor_count: u64,
    pub shards: Vec<String>,
    pub classifications: Vec<TensorClassification>,
    pub unknown_policy: String,
    pub duplicate_policy: String,
    pub index_policy: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalizationContract {
    pub kind: NormalizationKind,
    pub scale_mode: ScaleMode,
    pub effective_scale: String,
    pub epsilon: String,
    pub activation_dtype: TensorDType,
    pub weight_dtype: TensorDType,
    pub accumulation_dtype: AccumulationDType,
    pub output_dtype: TensorDType,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SliceContract {
    pub tensor_name: String,
    pub source_file: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub header_length_field_bytes: u64,
    pub header_length_bytes: u64,
    pub data_buffer_start: u64,
    pub data_offset_basis: String,
    pub data_offsets: [u64; 2],
    pub absolute_byte_range: [u64; 2],
    pub byte_size: u64,
    pub normalization: NormalizationContract,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BaseModel {
    pub repo_id: String,
    pub revision: Option<String>,
    pub evidence_path: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LicenseInfo {
    pub id: Option<String>,
    pub statement: String,
    pub evidence_paths: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LockedFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub git_blob: String,
    pub source_page_url: String,
    pub download_url: String,
    pub lfs_oid: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExcludedFile {
    pub path: String,
    pub git_blob: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StopIdentity {
    pub config_eos: ConfigEos,
    pub tokenizer_eos: TokenizerEos,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConfigEos {
    pub token: String,
    pub token_id: u64,
    pub source_file: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TokenizerEos {
    pub token: String,
    pub token_id: u64,
    pub source_files: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum StopEvaluation {
    #[serde(rename = "newly_generated_after_argmax")]
    NewlyGeneratedAfterArgmax,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PromptEvaluation {
    #[serde(rename = "never_stop")]
    NeverStop,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BudgetBoundary {
    #[serde(rename = "stop_token_wins")]
    StopTokenWins,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MaxNewTokensZero {
    #[serde(rename = "max_new_tokens_before_decode")]
    MaxNewTokensBeforeDecode,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StopTokenHandling {
    pub visible_output: bool,
    pub subsequent_decode_input: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationStopPolicyV1 {
    pub version: u8,
    pub stop_token_ids: Vec<u32>,
    pub evaluation: StopEvaluation,
    pub prompt_evaluation: PromptEvaluation,
    pub stop_token: StopTokenHandling,
    pub budget_boundary: BudgetBoundary,
    pub max_new_tokens_zero: MaxNewTokensZero,
    pub reason_version: u8,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TokenizerContract {
    pub files: Vec<String>,
    pub chat_template_path: String,
    pub vocab_size: u64,
    pub eos_token_id: u64,
    pub special_token_ids: BTreeMap<String, u64>,
    pub stop_identity: StopIdentity,
    pub generation_stop_policy: GenerationStopPolicyV1,
}

impl TokenizerContract {
    /// Return the immutable generation stop-policy contract from the model lock.
    pub fn generation_stop_policy(&self) -> &GenerationStopPolicyV1 {
        &self.generation_stop_policy
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    pub present: bool,
    pub path: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LockedModel {
    pub repo_id: String,
    pub repo_type: String,
    pub requested_revision: String,
    pub resolved_revision: String,
    pub license: LicenseInfo,
    pub base_models: Vec<BaseModel>,
    pub evidence_files: Vec<String>,
    pub files: Vec<LockedFile>,
    pub excluded_files: Vec<ExcludedFile>,
    pub architecture: ModelArchitecture,
    pub tensor_contract: TensorContract,
    pub slice_contract: SliceContract,
    pub tokenizer_contract: TokenizerContract,
    pub generation_config: GenerationConfig,
    pub derivation: Option<()>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelLock {
    pub schema_version: String,
    pub model: LockedModel,
    pub fingerprint: String,
    pub aliases: Vec<String>,
    pub generated_at: String,
}

impl ModelLock {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn model(&self) -> &LockedModel {
        &self.model
    }

    /// Return the immutable generation stop-policy contract from the model lock.
    pub fn generation_stop_policy(&self) -> &GenerationStopPolicyV1 {
        self.model.tokenizer_contract.generation_stop_policy()
    }

    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    /// Verifies a cache without downloading, executing, or copying tensor payloads.
    pub fn verify_cache(&self, cache_root: impl AsRef<Path>) -> Result<VerifiedCache, ModelError> {
        verify_model_cache(self, cache_root)
    }
}

/// Parse a model-lock-v1 document with duplicate-key and unknown-field rejection.
pub fn parse_model_lock(bytes: &[u8]) -> Result<ModelLock, ModelError> {
    let value = parse_json(bytes, true, MAX_LOCK_JSON_BYTES, "model lock")?;
    let document: ModelLock = from_value(value.clone())?;
    validate_lock(&document, &value)?;
    Ok(document)
}

/// Read and parse a model lock from a local file.  This function never accesses a URL.
pub fn read_model_lock(path: impl AsRef<Path>) -> Result<ModelLock, ModelError> {
    let path = path.as_ref();
    let bytes = read_bound_regular_file(path, MAX_LOCK_JSON_BYTES, "model lock")?;
    parse_model_lock(&bytes)
}

/// Read a small local JSON file through one descriptor and bind it back to its
/// path before and after the positional read.  In particular, never inspect a
/// path and then reopen it: the descriptor is opened first with no-follow and
/// close-on-exec, sized with `fstat` before allocation, and remains the sole
/// source of bytes.
fn read_bound_regular_file(
    path: &Path,
    max_bytes: usize,
    purpose: &str,
) -> Result<Vec<u8>, ModelError> {
    #[cfg(unix)]
    {
        let mut options = OpenOptions::new();
        options.read(true);
        options.custom_flags(O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
        let file = options.open(path).map_err(|error| io_error(path, error))?;
        let fd_before = file.metadata().map_err(|error| io_error(path, error))?;
        if !fd_before.is_file() {
            return Err(invalid(format!(
                "{purpose} must be a regular non-symlink file"
            )));
        }
        if fd_before.len() > max_bytes as u64 {
            return Err(invalid(format!("{purpose} exceeds the parser byte limit")));
        }
        let path_before = fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
        if path_before.file_type().is_symlink()
            || metadata_identity(&path_before) != metadata_identity(&fd_before)
        {
            return Err(invalid(format!("{purpose} path changed while opening")));
        }
        let length = usize::try_from(fd_before.len())
            .map_err(|_| invalid(format!("{purpose} size does not fit usize")))?;
        let mut bytes = vec![0u8; length];
        read_at_exact(&file, &mut bytes, 0, path)?;
        let fd_after = file.metadata().map_err(|error| io_error(path, error))?;
        let path_after = fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
        if !fd_after.is_file()
            || path_after.file_type().is_symlink()
            || metadata_identity(&fd_before) != metadata_identity(&fd_after)
            || metadata_identity(&fd_after) != metadata_identity(&path_after)
        {
            return Err(invalid(format!("{purpose} changed while reading")));
        }
        Ok(bytes)
    }
    #[cfg(not(unix))]
    {
        let _ = (path, max_bytes, purpose);
        Err(invalid("bound model-lock reads require Unix"))
    }
}

/// Recompute the restricted lock fingerprint for a parsed JSON document.
pub fn fingerprint_for_json(bytes: &[u8]) -> Result<String, ModelError> {
    let value = parse_json(bytes, true, MAX_LOCK_JSON_BYTES, "model lock")?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("model-lock root must be an object"))?;
    fingerprint_for_value(object)
}

fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, ModelError> {
    serde_json::from_value(value).map_err(|error| ModelError::Schema(error.to_string()))
}

fn validate_lock(document: &ModelLock, value: &Value) -> Result<(), ModelError> {
    validate_lock_json_bounds(value)?;
    if document.schema_version != "model-lock-v1" {
        return Err(invalid("unsupported schema_version"));
    }
    if !is_sha256_fingerprint(&document.fingerprint) {
        return Err(invalid(
            "fingerprint must be sha256: followed by 64 lowercase hex digits",
        ));
    }
    let object = value
        .as_object()
        .ok_or_else(|| invalid("model-lock root must be an object"))?;
    let computed = fingerprint_for_value(object)?;
    if document.fingerprint != computed {
        return Err(ModelError::FingerprintMismatch {
            expected: document.fingerprint.clone(),
            actual: computed,
        });
    }
    if document.aliases.is_empty() {
        return Err(invalid("aliases must not be empty"));
    }
    let mut aliases = HashSet::new();
    for alias in &document.aliases {
        if !is_alias(alias) || !aliases.insert(alias) {
            return Err(invalid(format!("invalid or duplicate alias: {alias:?}")));
        }
    }
    validate_datetime(&document.generated_at)?;
    validate_model(&document.model)?;

    if document.model.repo_id == QWEN_REPO_ID {
        if document.model.requested_revision != "main"
            || document.model.resolved_revision != QWEN_REVISION
            || document.aliases != ["qwen3.5-4b-bf16".to_owned()]
        {
            return Err(invalid(
                "Qwen lock is not bound to the reviewed immutable identity",
            ));
        }
        if document.fingerprint != QWEN_FINGERPRINT {
            return Err(invalid("reviewed Qwen lock fingerprint differs"));
        }
        validate_qwen_lock_contract(document)?;
    }
    Ok(())
}

fn validate_qwen_lock_contract(document: &ModelLock) -> Result<(), ModelError> {
    let model = &document.model;
    if model.license.id.as_deref() != Some("Apache-2.0")
        || model.license.statement != "Apache-2.0"
        || model.base_models
            != [BaseModel {
                repo_id: "Qwen/Qwen3.5-4B-Base".to_owned(),
                revision: None,
                evidence_path: "README.md".to_owned(),
            }]
    {
        return Err(invalid(
            "Qwen reviewed evidence or license contract differs",
        ));
    }
    if model.license.evidence_paths != ["LICENSE", "README.md"]
        || model.evidence_files != ["LICENSE", "README.md"]
    {
        return Err(invalid("Qwen reviewed evidence paths differ"));
    }
    let architecture = &model.architecture;
    let expected_layers = qwen_layer_schedule();
    if architecture.architectures != ["Qwen3_5ForConditionalGeneration"]
        || architecture.top_level_architecture != "Qwen3_5ForConditionalGeneration"
        || architecture.model_type != "qwen3_5"
        || architecture.text_model_type != "qwen3_5_text"
        || architecture.phase_scope != "text-only"
        || architecture.custom_code
        || architecture.converted
        || architecture.moe
        || architecture.vision
            != (ComponentMetadata {
                present: true,
                tensor_prefix: "model.visual.".to_owned(),
                tensor_count: 297,
                phase3_status: ComponentStatus::KnownUnconsumed,
            })
        || architecture.mtp
            != (ComponentMetadata {
                present: true,
                tensor_prefix: "mtp.".to_owned(),
                tensor_count: 15,
                phase3_status: ComponentStatus::KnownUnconsumed,
            })
        || architecture.text_config
            != (TextConfig {
                model_type: "qwen3_5_text".to_owned(),
                hidden_size: 2560,
                num_hidden_layers: 32,
                num_attention_heads: 16,
                num_key_value_heads: 4,
                head_dim: 256,
                intermediate_size: 9216,
                dtype: TensorDType::Bf16,
                rms_norm_eps: "1e-6".to_owned(),
                full_attention_interval: 4,
                layer_types: expected_layers.clone(),
                tie_word_embeddings: true,
                vocab_size: 248320,
                mtp_num_hidden_layers: 1,
            })
        || architecture.layer_schedule
            != (LayerSchedule {
                kind: "explicit".to_owned(),
                num_hidden_layers: 32,
                full_attention_interval: 4,
                layer_types: expected_layers,
                allowed_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
            })
    {
        return Err(invalid(
            "Qwen reviewed architecture/config contract differs",
        ));
    }
    if model.tensor_contract.index_path != "model.safetensors.index.json"
        || model.tensor_contract.indexed_tensor_count != 738
        || model.tensor_contract.shards
            != [
                "model.safetensors-00001-of-00002.safetensors".to_owned(),
                "model.safetensors-00002-of-00002.safetensors".to_owned(),
            ]
        || model.tensor_contract.classifications
            != [
                TensorClassification {
                    id: "text".to_owned(),
                    prefix: "model.language_model.".to_owned(),
                    tensor_count: 426,
                    phase3_status: ClassificationStatus::PartiallyConsumed,
                },
                TensorClassification {
                    id: "vision".to_owned(),
                    prefix: "model.visual.".to_owned(),
                    tensor_count: 297,
                    phase3_status: ClassificationStatus::KnownUnconsumed,
                },
                TensorClassification {
                    id: "mtp".to_owned(),
                    prefix: "mtp.".to_owned(),
                    tensor_count: 15,
                    phase3_status: ClassificationStatus::KnownUnconsumed,
                },
            ]
    {
        return Err(invalid("Qwen reviewed tensor contract differs"));
    }
    let tokenizer = &model.tokenizer_contract;
    if tokenizer.files
        != [
            "chat_template.jinja".to_owned(),
            "merges.txt".to_owned(),
            "tokenizer.json".to_owned(),
            "tokenizer_config.json".to_owned(),
            "vocab.json".to_owned(),
        ]
        || tokenizer.chat_template_path != "chat_template.jinja"
        || tokenizer.vocab_size != 248320
        || tokenizer.eos_token_id != 248044
        || tokenizer.special_token_ids
            != BTreeMap::from([
                ("vision_start".to_owned(), 248053),
                ("vision_end".to_owned(), 248054),
                ("vision_pad".to_owned(), 248055),
                ("image_pad".to_owned(), 248056),
                ("video_pad".to_owned(), 248057),
            ])
        || tokenizer.stop_identity
            != (StopIdentity {
                config_eos: ConfigEos {
                    token: "<|endoftext|>".to_owned(),
                    token_id: 248044,
                    source_file: "config.json".to_owned(),
                },
                tokenizer_eos: TokenizerEos {
                    token: "<|im_end|>".to_owned(),
                    token_id: 248046,
                    source_files: vec![
                        "tokenizer_config.json".to_owned(),
                        "tokenizer.json".to_owned(),
                    ],
                },
            })
        || tokenizer.generation_stop_policy
            != (GenerationStopPolicyV1 {
                version: 1,
                stop_token_ids: vec![248046, 248044],
                evaluation: StopEvaluation::NewlyGeneratedAfterArgmax,
                prompt_evaluation: PromptEvaluation::NeverStop,
                stop_token: StopTokenHandling {
                    visible_output: false,
                    subsequent_decode_input: false,
                },
                budget_boundary: BudgetBoundary::StopTokenWins,
                max_new_tokens_zero: MaxNewTokensZero::MaxNewTokensBeforeDecode,
                reason_version: 1,
            })
    {
        return Err(invalid(
            "Qwen reviewed tokenizer/stop identity contract differs",
        ));
    }
    if model.generation_config
        != (GenerationConfig {
            present: false,
            path: None,
        })
    {
        return Err(invalid("Qwen generation_config must be explicitly absent"));
    }
    Ok(())
}

fn qwen_layer_schedule() -> Vec<LayerType> {
    (0..32)
        .map(|layer| {
            if layer % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect()
}

/// Returns the reviewed Qwen3.5-4B checkpoint namespace as a one-to-one,
/// machine-readable tensor-name/class/dtype catalog.  This is intentionally a
/// catalog rather than a prefix/count test: a swapped layer, omitted tensor,
/// or unknown tensor must not become an accepted member of a broad prefix.
#[derive(Clone, Debug)]
struct QwenTextShapeInputs {
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    head_dim: u64,
    intermediate_size: u64,
    vocab_size: u64,
    full_attention_interval: u64,
    layer_types: Vec<LayerType>,
    linear_conv_kernel_dim: u64,
    linear_key_head_dim: u64,
    linear_num_key_heads: u64,
    linear_num_value_heads: u64,
    linear_value_head_dim: u64,
}

#[derive(Clone, Debug)]
struct QwenVisionShapeInputs {
    depth: u64,
    hidden_size: u64,
    in_channels: u64,
    temporal_patch_size: u64,
    patch_size: u64,
    spatial_merge_size: u64,
    intermediate_size: u64,
    out_hidden_size: u64,
    num_position_embeddings: u64,
}

#[derive(Clone, Debug)]
struct QwenShapeInputs {
    text: QwenTextShapeInputs,
    vision: QwenVisionShapeInputs,
    mtp_num_hidden_layers: u64,
    mtp_use_dedicated_embeddings: bool,
    tie_word_embeddings: bool,
}

type QwenTensorCatalog = BTreeMap<String, (&'static str, TensorDType, Vec<u64>)>;

fn require_positive_u64(
    object: &Map<String, Value>,
    field: &str,
    scope: &str,
) -> Result<u64, ModelError> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("{scope} field {field} must be an integer")))?;
    if value == 0 {
        return Err(invalid(format!("{scope} field {field} must be positive")));
    }
    Ok(value)
}

fn checked_shape_mul(left: u64, right: u64, field: &str) -> Result<u64, ModelError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(format!("Qwen shape arithmetic overflow: {field}")))
}

fn checked_shape_add(left: u64, right: u64, field: &str) -> Result<u64, ModelError> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("Qwen shape arithmetic overflow: {field}")))
}

fn validate_qwen_shape_inputs(
    root: &Map<String, Value>,
    architecture: &ModelArchitecture,
) -> Result<QwenShapeInputs, ModelError> {
    let text = root
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Qwen text_config is not an object"))?;
    let vision = root
        .get("vision_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Qwen vision_config is not an object"))?;

    let text_fields = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "full_attention_interval",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mtp_num_hidden_layers",
    ];
    for field in text_fields {
        require_positive_u64(text, field, "Qwen text_config")?;
    }
    let text_inputs = QwenTextShapeInputs {
        hidden_size: require_positive_u64(text, "hidden_size", "Qwen text_config")?,
        num_hidden_layers: require_positive_u64(text, "num_hidden_layers", "Qwen text_config")?,
        num_attention_heads: require_positive_u64(text, "num_attention_heads", "Qwen text_config")?,
        num_key_value_heads: require_positive_u64(text, "num_key_value_heads", "Qwen text_config")?,
        head_dim: require_positive_u64(text, "head_dim", "Qwen text_config")?,
        intermediate_size: require_positive_u64(text, "intermediate_size", "Qwen text_config")?,
        vocab_size: require_positive_u64(text, "vocab_size", "Qwen text_config")?,
        full_attention_interval: require_positive_u64(
            text,
            "full_attention_interval",
            "Qwen text_config",
        )?,
        layer_types: architecture.text_config.layer_types.clone(),
        linear_conv_kernel_dim: require_positive_u64(
            text,
            "linear_conv_kernel_dim",
            "Qwen text_config",
        )?,
        linear_key_head_dim: require_positive_u64(text, "linear_key_head_dim", "Qwen text_config")?,
        linear_num_key_heads: require_positive_u64(
            text,
            "linear_num_key_heads",
            "Qwen text_config",
        )?,
        linear_num_value_heads: require_positive_u64(
            text,
            "linear_num_value_heads",
            "Qwen text_config",
        )?,
        linear_value_head_dim: require_positive_u64(
            text,
            "linear_value_head_dim",
            "Qwen text_config",
        )?,
    };
    if text_inputs.num_hidden_layers != text_inputs.layer_types.len() as u64
        || text_inputs.num_hidden_layers != architecture.text_config.num_hidden_layers
        || text_inputs.full_attention_interval != 4
        || text_inputs.layer_types != qwen_layer_schedule()
        || text_inputs.num_key_value_heads > text_inputs.num_attention_heads
    {
        return Err(invalid(
            "Qwen text shape inputs do not match the explicit reviewed schedule",
        ));
    }

    let vision_fields = [
        "depth",
        "hidden_size",
        "in_channels",
        "temporal_patch_size",
        "patch_size",
        "spatial_merge_size",
        "intermediate_size",
        "num_heads",
        "num_position_embeddings",
        "out_hidden_size",
    ];
    for field in vision_fields {
        require_positive_u64(vision, field, "Qwen vision_config")?;
    }
    let deepstack_visual_indexes = vision
        .get("deepstack_visual_indexes")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Qwen vision deepstack_visual_indexes must be an array"))?;
    if !deepstack_visual_indexes.is_empty() {
        return Err(invalid(
            "Qwen vision deepstack_visual_indexes must be explicitly empty",
        ));
    }
    let vision_inputs = QwenVisionShapeInputs {
        depth: require_positive_u64(vision, "depth", "Qwen vision_config")?,
        hidden_size: require_positive_u64(vision, "hidden_size", "Qwen vision_config")?,
        in_channels: require_positive_u64(vision, "in_channels", "Qwen vision_config")?,
        temporal_patch_size: require_positive_u64(
            vision,
            "temporal_patch_size",
            "Qwen vision_config",
        )?,
        patch_size: require_positive_u64(vision, "patch_size", "Qwen vision_config")?,
        spatial_merge_size: require_positive_u64(
            vision,
            "spatial_merge_size",
            "Qwen vision_config",
        )?,
        intermediate_size: require_positive_u64(vision, "intermediate_size", "Qwen vision_config")?,
        out_hidden_size: require_positive_u64(vision, "out_hidden_size", "Qwen vision_config")?,
        num_position_embeddings: require_positive_u64(
            vision,
            "num_position_embeddings",
            "Qwen vision_config",
        )?,
    };
    if vision_inputs.depth != 24 {
        return Err(invalid(
            "Qwen vision depth must produce the reviewed 297 tensors",
        ));
    }

    let mtp_num_hidden_layers =
        require_positive_u64(text, "mtp_num_hidden_layers", "Qwen text_config")?;
    let mtp_use_dedicated_embeddings = text
        .get("mtp_use_dedicated_embeddings")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid("Qwen text_config MTP embedding condition is missing"))?;
    let tie_word_embeddings = root
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid("Qwen tie_word_embeddings condition is missing"))?;
    if mtp_num_hidden_layers != 1 || mtp_use_dedicated_embeddings || !tie_word_embeddings {
        return Err(invalid(
            "Qwen MTP requires one layer, tied embeddings, and no dedicated embeddings",
        ));
    }
    Ok(QwenShapeInputs {
        text: text_inputs,
        vision: vision_inputs,
        mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings,
        tie_word_embeddings,
    })
}

fn qwen_tensor_catalog(inputs: &QwenShapeInputs) -> Result<QwenTensorCatalog, ModelError> {
    let mut catalog = BTreeMap::new();
    let mut add = |name: String,
                   class: &'static str,
                   dtype: TensorDType,
                   shape: Vec<u64>|
     -> Result<(), ModelError> {
        if catalog
            .insert(name.clone(), (class, dtype, shape))
            .is_some()
        {
            return Err(invalid(format!(
                "duplicate Qwen tensor catalog entry: {name}"
            )));
        }
        Ok(())
    };
    let text = &inputs.text;
    let vision = &inputs.vision;
    let linear_projection_width = checked_shape_mul(
        text.linear_num_value_heads,
        text.linear_value_head_dim,
        "linear projection width",
    )?;
    let linear_qkv_width = checked_shape_add(
        checked_shape_mul(
            2,
            checked_shape_mul(
                text.linear_num_key_heads,
                text.linear_key_head_dim,
                "linear qkv query/key width",
            )?,
            "linear qkv query/key width",
        )?,
        checked_shape_mul(
            text.linear_num_value_heads,
            text.linear_value_head_dim,
            "linear qkv value width",
        )?,
        "linear qkv width",
    )?;
    let full_q_width = checked_shape_mul(
        2,
        checked_shape_mul(text.num_attention_heads, text.head_dim, "full query width")?,
        "full query/gate width",
    )?;
    let full_kv_width =
        checked_shape_mul(text.num_key_value_heads, text.head_dim, "full KV width")?;
    let full_output_width =
        checked_shape_mul(text.num_attention_heads, text.head_dim, "full output width")?;

    add(
        "model.language_model.embed_tokens.weight".to_owned(),
        "text",
        TensorDType::Bf16,
        vec![text.vocab_size, text.hidden_size],
    )?;
    add(
        "model.language_model.norm.weight".to_owned(),
        "text",
        TensorDType::Bf16,
        vec![text.hidden_size],
    )?;
    for (layer, layer_type) in text.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        add(
            format!("{prefix}.input_layernorm.weight"),
            "text",
            TensorDType::Bf16,
            vec![text.hidden_size],
        )?;
        add(
            format!("{prefix}.post_attention_layernorm.weight"),
            "text",
            TensorDType::Bf16,
            vec![text.hidden_size],
        )?;
        add(
            format!("{prefix}.mlp.gate_proj.weight"),
            "text",
            TensorDType::Bf16,
            vec![text.intermediate_size, text.hidden_size],
        )?;
        add(
            format!("{prefix}.mlp.up_proj.weight"),
            "text",
            TensorDType::Bf16,
            vec![text.intermediate_size, text.hidden_size],
        )?;
        add(
            format!("{prefix}.mlp.down_proj.weight"),
            "text",
            TensorDType::Bf16,
            vec![text.hidden_size, text.intermediate_size],
        )?;
        match layer_type {
            LayerType::LinearAttention => {
                add(
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    "text",
                    TensorDType::Bf16,
                    vec![linear_qkv_width, text.hidden_size],
                )?;
                add(
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    "text",
                    TensorDType::Bf16,
                    vec![linear_projection_width, text.hidden_size],
                )?;
                for suffix in ["in_proj_b.weight", "in_proj_a.weight"] {
                    add(
                        format!("{prefix}.linear_attn.{suffix}"),
                        "text",
                        TensorDType::Bf16,
                        vec![text.linear_num_value_heads, text.hidden_size],
                    )?;
                }
                add(
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    "text",
                    TensorDType::Bf16,
                    vec![linear_qkv_width, 1, text.linear_conv_kernel_dim],
                )?;
                add(
                    format!("{prefix}.linear_attn.A_log"),
                    "text",
                    TensorDType::F32,
                    vec![text.linear_num_value_heads],
                )?;
                add(
                    format!("{prefix}.linear_attn.dt_bias"),
                    "text",
                    TensorDType::Bf16,
                    vec![text.linear_num_value_heads],
                )?;
                add(
                    format!("{prefix}.linear_attn.norm.weight"),
                    "text",
                    TensorDType::F32,
                    vec![text.linear_value_head_dim],
                )?;
                add(
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    "text",
                    TensorDType::Bf16,
                    vec![text.hidden_size, linear_projection_width],
                )?;
            }
            LayerType::FullAttention => {
                add(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    "text",
                    TensorDType::Bf16,
                    vec![full_q_width, text.hidden_size],
                )?;
                for suffix in ["k_proj.weight", "v_proj.weight"] {
                    add(
                        format!("{prefix}.self_attn.{suffix}"),
                        "text",
                        TensorDType::Bf16,
                        vec![full_kv_width, text.hidden_size],
                    )?;
                }
                add(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    "text",
                    TensorDType::Bf16,
                    vec![text.hidden_size, full_output_width],
                )?;
                for suffix in ["q_norm.weight", "k_norm.weight"] {
                    add(
                        format!("{prefix}.self_attn.{suffix}"),
                        "text",
                        TensorDType::Bf16,
                        vec![text.head_dim],
                    )?;
                }
            }
        }
    }

    let spatial_merge_area = checked_shape_mul(
        vision.spatial_merge_size,
        vision.spatial_merge_size,
        "vision spatial merge area",
    )?;
    let merged_width = checked_shape_mul(
        vision.hidden_size,
        spatial_merge_area,
        "vision merged width",
    )?;
    let qkv_width = checked_shape_mul(3, vision.hidden_size, "vision qkv width")?;
    for block in 0..vision.depth {
        let prefix = format!("model.visual.blocks.{block}");
        for (suffix, shape) in [
            (
                "attn.proj.weight",
                vec![vision.hidden_size, vision.hidden_size],
            ),
            ("attn.proj.bias", vec![vision.hidden_size]),
            ("attn.qkv.weight", vec![qkv_width, vision.hidden_size]),
            ("attn.qkv.bias", vec![qkv_width]),
            (
                "mlp.linear_fc1.weight",
                vec![vision.intermediate_size, vision.hidden_size],
            ),
            ("mlp.linear_fc1.bias", vec![vision.intermediate_size]),
            (
                "mlp.linear_fc2.weight",
                vec![vision.hidden_size, vision.intermediate_size],
            ),
            ("mlp.linear_fc2.bias", vec![vision.hidden_size]),
            ("norm1.weight", vec![vision.hidden_size]),
            ("norm1.bias", vec![vision.hidden_size]),
            ("norm2.weight", vec![vision.hidden_size]),
            ("norm2.bias", vec![vision.hidden_size]),
        ] {
            add(
                format!("{prefix}.{suffix}"),
                "vision",
                TensorDType::Bf16,
                shape,
            )?;
        }
    }
    for (suffix, shape) in [
        ("merger.linear_fc1.weight", vec![merged_width, merged_width]),
        ("merger.linear_fc1.bias", vec![merged_width]),
        (
            "merger.linear_fc2.weight",
            vec![vision.out_hidden_size, merged_width],
        ),
        ("merger.linear_fc2.bias", vec![vision.out_hidden_size]),
        ("merger.norm.weight", vec![vision.hidden_size]),
        ("merger.norm.bias", vec![vision.hidden_size]),
        (
            "patch_embed.proj.weight",
            vec![
                vision.hidden_size,
                vision.in_channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ],
        ),
        ("patch_embed.proj.bias", vec![vision.hidden_size]),
        (
            "pos_embed.weight",
            vec![vision.num_position_embeddings, vision.hidden_size],
        ),
    ] {
        add(
            format!("model.visual.{suffix}"),
            "vision",
            TensorDType::Bf16,
            shape,
        )?;
    }

    if inputs.mtp_num_hidden_layers != 1
        || inputs.mtp_use_dedicated_embeddings
        || !inputs.tie_word_embeddings
    {
        return Err(invalid("Qwen MTP shape conditions are not satisfied"));
    }
    let mtp_q_width = checked_shape_mul(
        2,
        checked_shape_mul(text.num_attention_heads, text.head_dim, "MTP query width")?,
        "MTP query/gate width",
    )?;
    let mtp_kv_width = checked_shape_mul(text.num_key_value_heads, text.head_dim, "MTP KV width")?;
    let mtp_output_width =
        checked_shape_mul(text.num_attention_heads, text.head_dim, "MTP output width")?;
    for (suffix, shape) in [
        (
            "fc.weight",
            vec![
                text.hidden_size,
                checked_shape_mul(2, text.hidden_size, "MTP fc width")?,
            ],
        ),
        ("layers.0.input_layernorm.weight", vec![text.hidden_size]),
        (
            "layers.0.post_attention_layernorm.weight",
            vec![text.hidden_size],
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            vec![text.intermediate_size, text.hidden_size],
        ),
        (
            "layers.0.mlp.up_proj.weight",
            vec![text.intermediate_size, text.hidden_size],
        ),
        (
            "layers.0.mlp.down_proj.weight",
            vec![text.hidden_size, text.intermediate_size],
        ),
        (
            "layers.0.self_attn.q_proj.weight",
            vec![mtp_q_width, text.hidden_size],
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            vec![mtp_kv_width, text.hidden_size],
        ),
        (
            "layers.0.self_attn.v_proj.weight",
            vec![mtp_kv_width, text.hidden_size],
        ),
        (
            "layers.0.self_attn.o_proj.weight",
            vec![text.hidden_size, mtp_output_width],
        ),
        ("layers.0.self_attn.q_norm.weight", vec![text.head_dim]),
        ("layers.0.self_attn.k_norm.weight", vec![text.head_dim]),
        ("norm.weight", vec![text.hidden_size]),
        ("pre_fc_norm_embedding.weight", vec![text.hidden_size]),
        ("pre_fc_norm_hidden.weight", vec![text.hidden_size]),
    ] {
        add(format!("mtp.{suffix}"), "mtp", TensorDType::Bf16, shape)?;
    }
    if catalog.len() != 738 {
        return Err(invalid(format!(
            "Qwen tensor catalog cardinality differs from 738: {}",
            catalog.len()
        )));
    }
    Ok(catalog)
}

fn validate_qwen_header_catalog(
    actual: &BTreeMap<String, TensorDescriptor>,
    classifications: &[TensorClassification],
    expected: &QwenTensorCatalog,
) -> Result<(), ModelError> {
    if expected.len() != 738 {
        return Err(invalid(
            "Qwen tensor catalog cardinality differs from the reviewed 738 tensors",
        ));
    }
    let missing: Vec<&str> = expected
        .keys()
        .filter(|name| !actual.contains_key(*name))
        .map(String::as_str)
        .collect();
    let extra: Vec<&str> = actual
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() || !extra.is_empty() {
        let missing_count = missing.len();
        let extra_count = extra.len();
        let missing_preview = &missing[..missing.len().min(4)];
        let extra_preview = &extra[..extra.len().min(4)];
        return Err(invalid(format!(
            "Qwen tensor names do not match the reviewed exact catalog; missing_count={missing_count} missing_preview={missing_preview:?} extra_count={extra_count} extra_preview={extra_preview:?}"
        )));
    }
    for (name, descriptor) in actual {
        let (expected_class, expected_dtype, expected_shape) = expected
            .get(name)
            .expect("exact Qwen tensor name-set equality was verified before metadata lookup");
        let classified = classifications
            .iter()
            .find(|classification| name.starts_with(&classification.prefix))
            .map(|classification| classification.id.as_str());
        if classified != Some(*expected_class) {
            return Err(invalid(format!(
                "Qwen tensor class differs from the reviewed catalog: {name}"
            )));
        }
        if descriptor.dtype != *expected_dtype {
            return Err(invalid(format!(
                "Qwen tensor dtype differs from the reviewed catalog: {name}"
            )));
        }
        if descriptor.shape != *expected_shape {
            return Err(invalid(format!(
                "Qwen tensor shape differs from the reviewed catalog: {name}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod qwen_shape_tests {
    use super::*;

    type QwenActualHeaderCatalog = BTreeMap<String, TensorDescriptor>;
    type QwenHeaderCatalogMutation = fn(&mut QwenActualHeaderCatalog);

    fn inputs() -> QwenShapeInputs {
        QwenShapeInputs {
            text: QwenTextShapeInputs {
                hidden_size: 2560,
                num_hidden_layers: 32,
                num_attention_heads: 16,
                num_key_value_heads: 4,
                head_dim: 256,
                intermediate_size: 9216,
                vocab_size: 248320,
                full_attention_interval: 4,
                layer_types: qwen_layer_schedule(),
                linear_conv_kernel_dim: 4,
                linear_key_head_dim: 128,
                linear_num_key_heads: 16,
                linear_num_value_heads: 32,
                linear_value_head_dim: 128,
            },
            vision: QwenVisionShapeInputs {
                depth: 24,
                hidden_size: 1024,
                in_channels: 3,
                temporal_patch_size: 2,
                patch_size: 16,
                spatial_merge_size: 2,
                intermediate_size: 4096,
                out_hidden_size: 2560,
                num_position_embeddings: 2304,
            },
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
            tie_word_embeddings: true,
        }
    }

    fn qwen_actual_header_catalog_from_expected(
        expected: &QwenTensorCatalog,
    ) -> QwenActualHeaderCatalog {
        expected
            .iter()
            .map(|(name, (_, dtype, shape))| {
                (
                    name.clone(),
                    TensorDescriptor {
                        tensor_name: name.clone(),
                        source_file: "synthetic.safetensors".to_owned(),
                        dtype: *dtype,
                        shape: shape.clone(),
                        header_length_field_bytes: 8,
                        header_length_bytes: 0,
                        data_buffer_start: 0,
                        data_offset_basis: "data-buffer-relative".to_owned(),
                        data_offsets: [0, 1],
                        absolute_byte_range: [0, 1],
                        byte_size: 1,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn representative_shapes_cover_rank_width_dtype_and_counts() {
        let catalog = qwen_tensor_catalog(&inputs()).expect("reviewed shape inputs build");
        assert_eq!(catalog.len(), 738);
        assert_eq!(
            catalog["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"].2,
            [8192, 2560]
        );
        assert_eq!(
            catalog["model.language_model.layers.0.linear_attn.conv1d.weight"].2,
            [8192, 1, 4]
        );
        assert_eq!(
            catalog["model.language_model.layers.3.self_attn.q_proj.weight"].2,
            [8192, 2560]
        );
        assert_eq!(
            catalog["model.visual.patch_embed.proj.weight"].2,
            [1024, 3, 2, 16, 16]
        );
        assert_eq!(
            catalog["model.visual.merger.linear_fc1.weight"].2,
            [4096, 4096]
        );
        assert_eq!(
            catalog["mtp.layers.0.self_attn.o_proj.weight"].2,
            [2560, 4096]
        );
        assert_eq!(
            catalog["model.language_model.layers.0.linear_attn.A_log"].1,
            TensorDType::F32
        );
        assert_eq!(
            catalog["model.language_model.layers.0.linear_attn.dt_bias"].1,
            TensorDType::Bf16
        );
        assert_eq!(
            catalog["model.language_model.layers.0.linear_attn.norm.weight"].1,
            TensorDType::F32
        );
    }

    #[test]
    fn shape_arithmetic_rejects_overflow_and_accepts_non_aligned_boundaries() {
        for value in [1, 3, 17, u64::MAX] {
            assert_eq!(checked_shape_mul(value, 1, "test"), Ok(value));
            assert_eq!(checked_shape_add(value, 0, "test"), Ok(value));
        }
        assert!(checked_shape_mul(u64::MAX, 2, "overflow").is_err());
        assert!(checked_shape_add(u64::MAX, 1, "overflow").is_err());

        let mut object = Map::new();
        for value in [1, 3, 17, u64::MAX] {
            object.insert("field".to_owned(), Value::from(value));
            assert_eq!(require_positive_u64(&object, "field", "test"), Ok(value));
        }
        object.insert("field".to_owned(), Value::from(0u64));
        assert!(require_positive_u64(&object, "field", "test").is_err());
        object.insert("field".to_owned(), Value::from(true));
        assert!(require_positive_u64(&object, "field", "test").is_err());
    }

    #[test]
    fn exact_header_catalog_rejects_qwen_name_shape_rank_dimension_and_dtype_mutations() {
        let expected = qwen_tensor_catalog(&inputs()).expect("reviewed shape inputs build");
        let classifications = [
            TensorClassification {
                id: "text".to_owned(),
                prefix: "model.language_model.".to_owned(),
                tensor_count: 426,
                phase3_status: ClassificationStatus::PartiallyConsumed,
            },
            TensorClassification {
                id: "vision".to_owned(),
                prefix: "model.visual.".to_owned(),
                tensor_count: 297,
                phase3_status: ClassificationStatus::KnownUnconsumed,
            },
            TensorClassification {
                id: "mtp".to_owned(),
                prefix: "mtp.".to_owned(),
                tensor_count: 15,
                phase3_status: ClassificationStatus::KnownUnconsumed,
            },
        ];
        let valid = qwen_actual_header_catalog_from_expected(&expected);
        assert_eq!(valid.len(), 738);
        validate_qwen_header_catalog(&valid, &classifications, &expected)
            .expect("generated actual catalog is valid");

        let expected_elements = 8192_u64 * 2560;
        let rank_mutation_elements = [8192_u64, 2560, 1].into_iter().product::<u64>();
        assert_eq!(expected_elements, rank_mutation_elements);
        assert_eq!(8192_u64 * 2560, 10240 * 2048);
        assert_eq!(
            TensorDType::Bf16.byte_width(),
            TensorDType::F16.byte_width()
        );
        let mutations: [(&str, QwenHeaderCatalogMutation); 4] = [
            ("missing name", |actual| {
                actual.remove("model.language_model.layers.0.linear_attn.A_log");
            }),
            ("wrong rank", |actual| {
                actual
                    .get_mut("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
                    .unwrap()
                    .shape = vec![8192, 2560, 1];
            }),
            ("wrong dimension", |actual| {
                actual
                    .get_mut("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
                    .unwrap()
                    .shape = vec![10240, 2048];
            }),
            ("same-width wrong dtype", |actual| {
                actual
                    .get_mut("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
                    .unwrap()
                    .dtype = TensorDType::F16;
            }),
        ];
        for (label, mutate) in mutations {
            let mut actual = valid.clone();
            mutate(&mut actual);
            let error = validate_qwen_header_catalog(&actual, &classifications, &expected)
                .expect_err("Qwen exact catalog accepted mutation");
            if label == "missing name" {
                assert!(error.to_string().contains("missing_count=1"));
                assert!(error.to_string().contains(
                    "missing_preview=[\"model.language_model.layers.0.linear_attn.A_log\"]"
                ));
                assert!(error.to_string().contains("extra_count=0"));
                assert!(error.to_string().contains("extra_preview=[]"));
            }
        }
    }
}

fn validate_model(model: &LockedModel) -> Result<(), ModelError> {
    validate_repo_id(&model.repo_id, "model.repo_id")?;
    if model.repo_type != "model" {
        return Err(invalid("model.repo_type must be model"));
    }
    validate_clean_text(&model.requested_revision, "requested_revision", 256)?;
    validate_sha40(&model.resolved_revision, "resolved_revision")?;
    if model.license.statement.is_empty() || model.license.evidence_paths.is_empty() {
        return Err(invalid("license statement/evidence must be present"));
    }
    validate_clean_text(&model.license.statement, "license statement", 4096)?;
    if let Some(id) = &model.license.id {
        validate_clean_text(id, "license id", 256)?;
        if id.is_empty()
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b".+-".contains(&byte))
        {
            return Err(invalid("license id is outside the allowed range"));
        }
    }
    let mut files = BTreeMap::new();
    for file in &model.files {
        validate_file(file, &model.repo_id, &model.resolved_revision)?;
        if files.insert(file.path.clone(), file).is_some() {
            return Err(invalid(format!("duplicate locked file: {}", file.path)));
        }
    }
    if files.is_empty() {
        return Err(invalid("files must not be empty"));
    }
    let paths: BTreeSet<&str> = files.keys().map(String::as_str).collect();
    for path in &model.evidence_files {
        validate_safe_path(path, "evidence file")?;
        if !paths.contains(path.as_str()) {
            return Err(invalid(format!("evidence file is not locked: {path}")));
        }
    }
    if model.evidence_files.is_empty() || !unique_strings(&model.evidence_files) {
        return Err(invalid("evidence_files must be non-empty and unique"));
    }
    for path in &model.license.evidence_paths {
        validate_safe_path(path, "license evidence path")?;
        if !paths.contains(path.as_str()) {
            return Err(invalid(format!(
                "license evidence path is not locked: {path}"
            )));
        }
    }
    for base in &model.base_models {
        validate_repo_id(&base.repo_id, "base model repo_id")?;
        if let Some(revision) = &base.revision {
            validate_sha40(revision, "base model revision")?;
        }
        validate_safe_path(&base.evidence_path, "base model evidence path")?;
        if !paths.contains(base.evidence_path.as_str()) {
            return Err(invalid(format!(
                "base model evidence is not locked: {}",
                base.evidence_path
            )));
        }
    }
    let mut excluded = HashSet::new();
    for entry in &model.excluded_files {
        validate_safe_path(&entry.path, "excluded file path")?;
        validate_sha40(&entry.git_blob, "excluded file git_blob")?;
        validate_clean_text(&entry.reason, "excluded file reason", 4096)?;
        if entry.reason.is_empty()
            || !excluded.insert(&entry.path)
            || paths.contains(entry.path.as_str())
        {
            return Err(invalid(format!(
                "invalid or overlapping excluded file: {}",
                entry.path
            )));
        }
    }
    validate_architecture(&model.architecture)?;
    validate_tensor_contract(&model.tensor_contract, &paths)?;
    validate_slice_contract(
        &model.slice_contract,
        &model.architecture.text_config,
        &paths,
    )?;
    validate_tokenizer_contract(&model.tokenizer_contract, &paths)?;
    if model.generation_config.present != model.generation_config.path.is_some() {
        return Err(invalid("generation_config present/path disagree"));
    }
    if model.derivation.is_some() {
        return Err(invalid("model-lock-v1 accepts only derivation: null"));
    }
    Ok(())
}

fn validate_file(file: &LockedFile, repo_id: &str, revision: &str) -> Result<(), ModelError> {
    validate_safe_path(&file.path, "locked file path")?;
    validate_sha256(&file.sha256, "file sha256")?;
    validate_sha40(&file.git_blob, "file git_blob")?;
    if let Some(oid) = &file.lfs_oid {
        if oid != &format!("sha256:{}", file.sha256) {
            return Err(invalid(format!(
                "LFS OID does not match file SHA-256: {}",
                file.path
            )));
        }
    }
    let expected_source = format!(
        "https://huggingface.co/{repo_id}/blob/{revision}/{}",
        file.path
    );
    let expected_download = format!(
        "https://huggingface.co/{repo_id}/resolve/{revision}/{}",
        file.path
    );
    if file.source_page_url != expected_source || file.download_url != expected_download {
        return Err(invalid(format!(
            "file URLs are not immutable resolved-SHA URLs: {}",
            file.path
        )));
    }
    Ok(())
}

fn validate_architecture(architecture: &ModelArchitecture) -> Result<(), ModelError> {
    if architecture.architectures.is_empty()
        || architecture.architectures.iter().any(String::is_empty)
        || architecture.top_level_architecture.is_empty()
        || architecture.model_type.is_empty()
        || architecture.text_model_type.is_empty()
        || architecture.phase_scope != "text-only"
    {
        return Err(invalid("architecture has an empty or unsupported identity"));
    }
    for name in &architecture.architectures {
        validate_clean_text(name, "architecture name", 256)?;
    }
    validate_clean_text(
        &architecture.top_level_architecture,
        "top-level architecture",
        256,
    )?;
    validate_clean_text(&architecture.model_type, "model_type", 256)?;
    validate_clean_text(&architecture.text_model_type, "text_model_type", 256)?;
    validate_component(&architecture.vision, "vision")?;
    validate_component(&architecture.mtp, "mtp")?;
    validate_text_config(&architecture.text_config)?;
    let schedule = &architecture.layer_schedule;
    if schedule.kind != "explicit"
        || schedule.num_hidden_layers != architecture.text_config.num_hidden_layers
        || schedule.full_attention_interval != architecture.text_config.full_attention_interval
        || schedule.layer_types != architecture.text_config.layer_types
        || schedule.layer_types.is_empty()
        || schedule.allowed_types.is_empty()
        || !unique_copy(&schedule.allowed_types)
        || schedule
            .layer_types
            .iter()
            .any(|kind| !schedule.allowed_types.contains(kind))
    {
        return Err(invalid(
            "layer schedule is not explicit, complete, and allow-listed",
        ));
    }
    Ok(())
}

fn validate_component(component: &ComponentMetadata, name: &str) -> Result<(), ModelError> {
    validate_clean_text(&component.tensor_prefix, "component tensor prefix", 512)?;
    if component.tensor_prefix.is_empty() {
        return Err(invalid(format!("{name} tensor prefix is empty")));
    }
    if (!component.present) != (component.phase3_status == ComponentStatus::Absent)
        || (component.present && component.phase3_status == ComponentStatus::Absent)
    {
        return Err(invalid(format!("{name} present/status mismatch")));
    }
    Ok(())
}

fn validate_text_config(config: &TextConfig) -> Result<(), ModelError> {
    validate_clean_text(&config.model_type, "text model_type", 256)?;
    if config.model_type.is_empty()
        || config.dtype != TensorDType::Bf16
        || config.hidden_size == 0
        || config.num_hidden_layers == 0
        || config.num_attention_heads == 0
        || config.num_key_value_heads == 0
        || config.num_key_value_heads > config.num_attention_heads
        || config.head_dim == 0
        || config.intermediate_size == 0
        || config.vocab_size == 0
        || config.full_attention_interval == 0
        || config.layer_types.len() as u64 != config.num_hidden_layers
    {
        return Err(invalid(
            "text_config contains an out-of-range or inconsistent field",
        ));
    }
    validate_decimal_epsilon(&config.rms_norm_eps)?;
    Ok(())
}

fn validate_tensor_contract(
    contract: &TensorContract,
    paths: &BTreeSet<&str>,
) -> Result<(), ModelError> {
    if contract.index_path.is_empty()
        || contract.indexed_tensor_count == 0
        || contract.shards.is_empty()
        || !unique_strings(&contract.shards)
        || contract.classifications.is_empty()
        || contract.unknown_policy != "reject"
        || contract.duplicate_policy != "reject"
        || contract.index_policy != "exact-weight-map-and-shard-metadata"
    {
        return Err(invalid(
            "invalid tensor contract policy or empty collection",
        ));
    }
    validate_safe_path(&contract.index_path, "tensor index path")?;
    if !paths.contains(contract.index_path.as_str()) {
        return Err(invalid("tensor index is not locked"));
    }
    for shard in &contract.shards {
        validate_safe_path(shard, "tensor shard path")?;
        if !paths.contains(shard.as_str()) {
            return Err(invalid(format!("tensor shard is not locked: {shard}")));
        }
    }
    let mut ids = HashSet::new();
    for classification in &contract.classifications {
        validate_clean_text(&classification.id, "tensor classification id", 128)?;
        validate_clean_text(&classification.prefix, "tensor classification prefix", 512)?;
        if classification.id.is_empty()
            || classification.prefix.is_empty()
            || !ids.insert(&classification.id)
        {
            return Err(invalid(
                "tensor classifications have empty or duplicate IDs",
            ));
        }
    }
    Ok(())
}

fn validate_slice_contract(
    contract: &SliceContract,
    text_config: &TextConfig,
    paths: &BTreeSet<&str>,
) -> Result<(), ModelError> {
    validate_clean_text(&contract.tensor_name, "slice tensor name", 1024)?;
    let absolute_start = contract
        .data_buffer_start
        .checked_add(contract.data_offsets[0]);
    let absolute_end = contract
        .data_buffer_start
        .checked_add(contract.data_offsets[1]);
    if contract.tensor_name.is_empty()
        || contract.dtype != TensorDType::Bf16
        || contract.shape.is_empty()
        || contract.shape.contains(&0)
        || contract.header_length_field_bytes != 8
        || !(8..=MAX_SAFE_TENSOR_HEADER).contains(&contract.header_length_bytes)
        || contract.data_buffer_start != contract.header_length_bytes + 8
        || contract.data_offset_basis != "data-buffer-relative"
        || contract.data_offsets[0] >= contract.data_offsets[1]
        || contract.absolute_byte_range[0] >= contract.absolute_byte_range[1]
        || contract.byte_size != contract.absolute_byte_range[1] - contract.absolute_byte_range[0]
        || contract.byte_size != contract.data_offsets[1] - contract.data_offsets[0]
        || absolute_start != Some(contract.absolute_byte_range[0])
        || absolute_end != Some(contract.absolute_byte_range[1])
        || product(contract.shape.iter().copied()).and_then(|value| value.checked_mul(2))
            != Some(contract.byte_size)
    {
        return Err(invalid(
            "slice contract has an invalid range, shape, or BF16 size",
        ));
    }
    validate_safe_path(&contract.source_file, "slice source file")?;
    if !paths.contains(contract.source_file.as_str()) {
        return Err(invalid("slice source file is not locked"));
    }
    if contract.normalization.kind != NormalizationKind::RmsNorm
        || contract.normalization.scale_mode != ScaleMode::OffsetOne
        || contract.normalization.effective_scale != "1 + raw_weight"
        || contract.normalization.activation_dtype != TensorDType::Bf16
        || contract.normalization.weight_dtype != TensorDType::Bf16
        || contract.normalization.accumulation_dtype != AccumulationDType::Fp32
        || contract.normalization.output_dtype != TensorDType::Bf16
        || contract.normalization.epsilon != text_config.rms_norm_eps
    {
        return Err(invalid(
            "slice normalization contract is not the explicit BF16 RMSNorm contract",
        ));
    }
    validate_decimal_epsilon(&contract.normalization.epsilon)
}

fn validate_tokenizer_contract(
    contract: &TokenizerContract,
    paths: &BTreeSet<&str>,
) -> Result<(), ModelError> {
    validate_clean_text(
        &contract.stop_identity.config_eos.token,
        "config EOS token",
        4096,
    )?;
    validate_clean_text(
        &contract.stop_identity.tokenizer_eos.token,
        "tokenizer EOS token",
        4096,
    )?;
    if contract.files.is_empty()
        || !unique_strings(&contract.files)
        || contract.vocab_size == 0
        || contract.special_token_ids.is_empty()
    {
        return Err(invalid("invalid tokenizer contract"));
    }
    for path in &contract.files {
        validate_safe_path(path, "tokenizer file")?;
        if !paths.contains(path.as_str()) {
            return Err(invalid(format!("tokenizer file is not locked: {path}")));
        }
    }
    validate_safe_path(&contract.chat_template_path, "chat template path")?;
    if !paths.contains(contract.chat_template_path.as_str()) {
        return Err(invalid("chat template is not locked"));
    }
    validate_safe_path(
        &contract.stop_identity.config_eos.source_file,
        "config EOS source",
    )?;
    if !paths.contains(contract.stop_identity.config_eos.source_file.as_str()) {
        return Err(invalid("config EOS source is not locked"));
    }
    if contract.stop_identity.config_eos.token.is_empty()
        || contract.stop_identity.tokenizer_eos.token.is_empty()
        || contract.stop_identity.tokenizer_eos.source_files.is_empty()
        || !unique_strings(&contract.stop_identity.tokenizer_eos.source_files)
    {
        return Err(invalid("tokenizer EOS identity is incomplete"));
    }
    for path in &contract.stop_identity.tokenizer_eos.source_files {
        validate_safe_path(path, "tokenizer EOS source")?;
        if !paths.contains(path.as_str()) {
            return Err(invalid(format!(
                "tokenizer EOS source is not locked: {path}"
            )));
        }
    }
    validate_generation_stop_policy(&contract.generation_stop_policy)?;
    Ok(())
}

fn validate_generation_stop_policy(policy: &GenerationStopPolicyV1) -> Result<(), ModelError> {
    if policy.version != 1
        || policy.reason_version != 1
        || policy.stop_token.visible_output
        || policy.stop_token.subsequent_decode_input
        || policy.stop_token_ids.is_empty()
    {
        return Err(invalid("generation stop policy has an invalid fixed field"));
    }
    let mut ids = HashSet::new();
    if policy
        .stop_token_ids
        .iter()
        .any(|token_id| !ids.insert(*token_id))
    {
        return Err(invalid(
            "generation stop policy stop_token_ids must be non-empty and unique",
        ));
    }
    Ok(())
}

fn validate_decimal_epsilon(value: &str) -> Result<(), ModelError> {
    if value.is_empty() {
        return Err(invalid("rms_norm_eps must be explicit"));
    }
    let bytes = value.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
        cursor += 1;
    }
    if cursor == 0 {
        return Err(invalid(format!("invalid decimal epsilon: {value}")));
    }
    if cursor < bytes.len() && bytes[cursor] == b'.' {
        cursor += 1;
        let start = cursor;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor == start {
            return Err(invalid(format!("invalid decimal epsilon: {value}")));
        }
    }
    if cursor < bytes.len() && matches!(bytes[cursor], b'e' | b'E') {
        cursor += 1;
        if cursor < bytes.len() && matches!(bytes[cursor], b'+' | b'-') {
            cursor += 1;
        }
        let start = cursor;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor == start {
            return Err(invalid(format!("invalid decimal epsilon: {value}")));
        }
    }
    if cursor != bytes.len() {
        return Err(invalid(format!("invalid decimal epsilon: {value}")));
    }
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid("epsilon is not finite"))?;
    if !parsed.is_finite() || parsed <= 0.0 || parsed > 1.0 {
        return Err(invalid(format!("epsilon is outside (0, 1]: {value}")));
    }
    Ok(())
}

fn validate_repo_id(value: &str, field: &str) -> Result<(), ModelError> {
    validate_clean_text(value, field, 512)?;
    let parts: Vec<&str> = value.split('/').collect();
    if value.contains('\0') || parts.len() != 2 || parts.iter().any(|part| part.is_empty()) {
        return Err(invalid(format!("{field} is not namespace/name")));
    }
    Ok(())
}

fn validate_safe_path(value: &str, field: &str) -> Result<(), ModelError> {
    if value.is_empty()
        || value.len() > 512
        || value.starts_with('/')
        || value.chars().any(is_forbidden_control)
        || value.contains('\\')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(invalid(format!("unsafe {field}: {value:?}")));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), ModelError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{field} is not lowercase SHA-256")));
    }
    Ok(())
}

fn validate_sha40(value: &str, field: &str) -> Result<(), ModelError> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{field} is not lowercase 40-hex SHA")));
    }
    Ok(())
}

fn is_sha256_fingerprint(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_alias(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'.' || *byte == b'-'
        })
}

fn is_forbidden_control(character: char) -> bool {
    matches!(character as u32, 0x00..=0x1f | 0x7f..=0x9f)
}

fn validate_clean_text(value: &str, field: &str, max_bytes: usize) -> Result<(), ModelError> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(is_forbidden_control) {
        return Err(invalid(format!(
            "{field} is empty, oversized, or contains a control character"
        )));
    }
    Ok(())
}

fn validate_lock_json_bounds(value: &Value) -> Result<(), ModelError> {
    fn visit(value: &Value, depth: usize) -> Result<(), ModelError> {
        if depth > MAX_JSON_DEPTH {
            return Err(invalid("model-lock JSON exceeds the maximum nesting depth"));
        }
        match value {
            Value::Null | Value::Bool(_) | Value::Number(_) => Ok(()),
            Value::String(text) => {
                validate_clean_text(text, "model-lock string", MAX_JSON_STRING_BYTES)
            }
            Value::Array(values) => {
                if values.len() > MAX_JSON_COLLECTION_ITEMS {
                    return Err(invalid(
                        "model-lock JSON array exceeds the collection limit",
                    ));
                }
                for item in values {
                    visit(item, depth + 1)?;
                }
                Ok(())
            }
            Value::Object(values) => {
                if values.len() > MAX_JSON_COLLECTION_ITEMS {
                    return Err(invalid(
                        "model-lock JSON object exceeds the collection limit",
                    ));
                }
                for (key, item) in values {
                    validate_clean_text(key, "model-lock object key", MAX_JSON_STRING_BYTES)?;
                    visit(item, depth + 1)?;
                }
                Ok(())
            }
        }
    }
    visit(value, 0)
}

fn unique_strings(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values.iter().all(|value| seen.insert(value))
}

fn unique_copy<T: Copy + Eq + std::hash::Hash>(values: &[T]) -> bool {
    let mut seen = HashSet::new();
    values.iter().copied().all(|value| seen.insert(value))
}

fn product(mut values: impl Iterator<Item = u64>) -> Option<u64> {
    values.try_fold(1u64, |current, value| current.checked_mul(value))
}

fn validate_datetime(value: &str) -> Result<(), ModelError> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || !bytes[0..4].iter().all(u8::is_ascii_digit)
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(invalid("generated_at is not RFC3339 date-time"));
    }
    let digit = |start: usize, end: usize| -> Option<u32> {
        if end > bytes.len() || !bytes[start..end].iter().all(u8::is_ascii_digit) {
            None
        } else {
            std::str::from_utf8(&bytes[start..end]).ok()?.parse().ok()
        }
    };
    let month = digit(5, 7).unwrap_or(0);
    let day = digit(8, 10).unwrap_or(0);
    let hour = digit(11, 13).unwrap_or(99);
    let minute = digit(14, 16).unwrap_or(99);
    let second = digit(17, 19).unwrap_or(99);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let year = digit(0, 4).unwrap_or(0);
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    if !(1..=12).contains(&month)
        || day == 0
        || day > days_in_month
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(invalid("generated_at has an out-of-range component"));
    }
    Ok(())
}

fn parse_json(
    bytes: &[u8],
    reject_floats: bool,
    max_bytes: usize,
    purpose: &str,
) -> Result<Value, ModelError> {
    if bytes.len() > max_bytes {
        return Err(invalid(format!("{purpose} exceeds the parser byte limit")));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValueSeed {
        reject_floats,
        depth: 0,
    }
    .deserialize(&mut deserializer)
    .map_err(|error| ModelError::Json(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| ModelError::Json(error.to_string()))?;
    Ok(value)
}

struct StrictValueSeed {
    reject_floats: bool,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for StrictValueSeed {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.depth > MAX_JSON_DEPTH {
            return Err(de::Error::custom("JSON nesting exceeds the parser limit"));
        }
        deserializer.deserialize_any(StrictValueVisitor {
            reject_floats: self.reject_floats,
            depth: self.depth,
        })
    }
}

struct StrictValueVisitor {
    reject_floats: bool,
    depth: usize,
}

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a strict JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.len() > MAX_JSON_STRING_BYTES {
            return Err(E::custom("JSON string exceeds the parser limit"));
        }
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.len() > MAX_JSON_STRING_BYTES {
            return Err(E::custom("JSON string exceeds the parser limit"));
        }
        Ok(Value::String(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if i128::from(value) < JCS_SAFE_INTEGER_MIN {
            Err(E::custom("integer is outside RFC 8785 safe range"))
        } else {
            Ok(Value::Number(value.into()))
        }
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if u128::from(value) > JCS_SAFE_INTEGER_MAX {
            Err(E::custom("integer is outside RFC 8785 safe range"))
        } else {
            Ok(Value::Number(value.into()))
        }
    }

    fn visit_i128<E>(self, value: i128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if !(JCS_SAFE_INTEGER_MIN..=JCS_SAFE_INTEGER_MAX as i128).contains(&value) {
            return Err(E::custom("integer is outside RFC 8785 safe range"));
        }
        Ok(Value::Number(Number::from(value as i64)))
    }

    fn visit_u128<E>(self, value: u128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value > JCS_SAFE_INTEGER_MAX {
            return Err(E::custom("integer is outside RFC 8785 safe range"));
        }
        Ok(Value::Number(Number::from(value as u64)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if self.reject_floats {
            Err(E::custom(
                "floating-point JSON numbers are forbidden in model-lock JSON",
            ))
        } else {
            Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| E::custom("non-finite JSON number"))
        }
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictValueSeed {
            reject_floats: self.reject_floats,
            depth: self.depth + 1,
        })? {
            if values.len() == MAX_JSON_COLLECTION_ITEMS {
                return Err(de::Error::custom("JSON array exceeds the parser limit"));
            }
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut keys = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if key.len() > MAX_JSON_STRING_BYTES {
                return Err(de::Error::custom(
                    "JSON object key exceeds the parser limit",
                ));
            }
            if values.len() == MAX_JSON_COLLECTION_ITEMS {
                return Err(de::Error::custom("JSON object exceeds the parser limit"));
            }
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
            let value = map.next_value_seed(StrictValueSeed {
                reject_floats: self.reject_floats,
                depth: self.depth + 1,
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

fn fingerprint_for_value(root: &Map<String, Value>) -> Result<String, ModelError> {
    let schema_version = root
        .get("schema_version")
        .ok_or_else(|| invalid("fingerprint input lacks schema_version"))?;
    let model = root
        .get("model")
        .ok_or_else(|| invalid("fingerprint input lacks model"))?;
    let mut target = Map::new();
    target.insert("schema_version".to_owned(), schema_version.clone());
    target.insert("model".to_owned(), model.clone());
    let mut canonical = String::new();
    jcs_value(&Value::Object(target), &mut canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("sha256:{}", hex_lower(&hasher.finalize())))
}

fn jcs_value(value: &Value, output: &mut String) -> Result<(), ModelError> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::String(value) => output.push_str(
            &serde_json::to_string(value).map_err(|error| ModelError::Json(error.to_string()))?,
        ),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                if i128::from(value) < JCS_SAFE_INTEGER_MIN {
                    return Err(invalid("JCS integer is outside safe range"));
                }
                output.push_str(&value.to_string());
            } else if let Some(value) = number.as_u64() {
                if u128::from(value) > JCS_SAFE_INTEGER_MAX {
                    return Err(invalid("JCS integer is outside safe range"));
                }
                output.push_str(&value.to_string());
            } else {
                return Err(invalid(
                    "floating-point values are forbidden in fingerprint input",
                ));
            }
        }
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                jcs_value(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            let mut keys: Vec<&String> = values.keys().collect();
            keys.sort_by_key(|key| key.encode_utf16().collect::<Vec<u16>>());
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(*key)
                        .map_err(|error| ModelError::Json(error.to_string()))?,
                );
                output.push(':');
                jcs_value(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// A read-only descriptor for a tensor's bytes.  No tensor payload is stored here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    pub tensor_name: String,
    pub source_file: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub header_length_field_bytes: u64,
    pub header_length_bytes: u64,
    pub data_buffer_start: u64,
    pub data_offset_basis: String,
    pub data_offsets: [u64; 2],
    pub absolute_byte_range: [u64; 2],
    pub byte_size: u64,
}

impl TensorDescriptor {
    pub fn absolute_start(&self) -> u64 {
        self.absolute_byte_range[0]
    }

    pub fn absolute_end(&self) -> u64 {
        self.absolute_byte_range[1]
    }
}

#[derive(Debug)]
pub struct VerifiedCache {
    pub lock_fingerprint: String,
    pub files: Vec<VerifiedFile>,
    tensors: BTreeMap<String, TensorDescriptor>,
    owned_files: BTreeMap<String, OwnedVerifiedFile>,
    cache_root: PathBuf,
    root_identity: FileIdentity,
}

/// The fixed set of frontend assets that may be read from a verified cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendAssetKind {
    ConfigJson,
    TokenizerJson,
    TokenizerConfigJson,
    ChatTemplateJinja,
}

impl FrontendAssetKind {
    fn specification(self) -> (&'static str, usize) {
        match self {
            Self::ConfigJson => ("config.json", MAX_CONFIG_JSON_BYTES),
            Self::TokenizerJson => ("tokenizer.json", MAX_TOKENIZER_JSON_BYTES),
            Self::TokenizerConfigJson => ("tokenizer_config.json", MAX_TOKENIZER_CONFIG_JSON_BYTES),
            Self::ChatTemplateJinja => ("chat_template.jinja", MAX_CHAT_TEMPLATE_JINJA_BYTES),
        }
    }
}

#[derive(Debug)]
struct OwnedVerifiedFile {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    size_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    size_bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
    link_count: u64,
}

impl VerifiedCache {
    pub fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(tensor_name)
    }

    pub fn tensors(&self) -> impl Iterator<Item = &TensorDescriptor> {
        self.tensors.values()
    }

    /// Read a bounded byte range from a tensor through the descriptor that was
    /// hash-verified by `verify_model_cache`.  The range is positioned with
    /// `read_at`, so it never depends on (or mutates) a shared seek cursor.
    pub fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, ModelError> {
        let tensor = self
            .tensors
            .get(tensor_name)
            .ok_or_else(|| invalid(format!("unknown verified tensor: {tensor_name}")))?;
        let requested =
            u64::try_from(length).map_err(|_| invalid("tensor range length does not fit u64"))?;
        let end = offset
            .checked_add(requested)
            .ok_or_else(|| invalid("tensor range offset overflow"))?;
        if end > tensor.byte_size {
            return Err(invalid("tensor range exceeds the verified tensor"));
        }
        let absolute = tensor
            .absolute_start()
            .checked_add(offset)
            .ok_or_else(|| invalid("tensor absolute range offset overflow"))?;
        let file = self
            .owned_files
            .get(&tensor.source_file)
            .ok_or_else(|| invalid("verified tensor source file is unavailable"))?;
        assert_cache_root_stable(&self.cache_root, &self.root_identity, "tensor range")?;
        assert_cache_path_bindings(&self.cache_root, &self.owned_files, "tensor range")?;
        let bytes = read_owned_range(file, absolute, length, MAX_RANGE_READ_BYTES, "tensor range")?;
        assert_cache_root_stable(&self.cache_root, &self.root_identity, "tensor range")?;
        assert_cache_path_bindings(&self.cache_root, &self.owned_files, "tensor range")?;
        Ok(bytes)
    }

    /// Read one fixed, hash-verified frontend asset through its held file
    /// descriptor.  The whole asset is bounded before any buffer is allocated.
    pub fn read_frontend_asset(&self, kind: FrontendAssetKind) -> Result<Vec<u8>, ModelError> {
        let (relative, max_bytes) = kind.specification();
        let file = self
            .owned_files
            .get(relative)
            .ok_or_else(|| invalid(format!("frontend asset is not locked: {relative}")))?;
        let max_bytes_u64 = u64::try_from(max_bytes)
            .map_err(|_| invalid("frontend asset read limit does not fit u64"))?;
        if file.size_bytes > max_bytes_u64 {
            return Err(invalid(format!(
                "frontend asset {relative} exceeds the bounded read limit"
            )));
        }
        let length = usize::try_from(file.size_bytes)
            .map_err(|_| invalid("frontend asset size does not fit usize"))?;
        assert_cache_root_stable(&self.cache_root, &self.root_identity, "frontend asset read")?;
        assert_cache_path_bindings(&self.cache_root, &self.owned_files, "frontend asset read")?;
        let bytes = read_owned_range(file, 0, length, max_bytes, "frontend asset")?;
        assert_cache_root_stable(&self.cache_root, &self.root_identity, "frontend asset read")?;
        assert_cache_path_bindings(&self.cache_root, &self.owned_files, "frontend asset read")?;
        Ok(bytes)
    }
}

/// Verify every locked cache file and every index/shard metadata range.
///
/// Hashing is streaming.  Safetensors payloads are never read into memory;
/// only the little-endian header and JSON metadata are read, and successful
/// validation returns byte-range descriptors for later read-only consumers.
pub fn verify_model_cache(
    lock: &ModelLock,
    cache_root: impl AsRef<Path>,
) -> Result<VerifiedCache, ModelError> {
    let cache_root = cache_root.as_ref();
    let root_before = validate_cache_root(cache_root)?;
    let mut actual = BTreeMap::new();
    collect_cache_files(cache_root, cache_root, &mut actual)?;
    let expected: BTreeMap<&str, &LockedFile> = lock
        .model
        .files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect();
    if actual.len() != expected.len()
        || actual
            .keys()
            .any(|path| !expected.contains_key(path.as_str()))
    {
        return Err(invalid("cache file set differs from locked files"));
    }

    let mut verified = Vec::with_capacity(expected.len());
    let mut owned_files = BTreeMap::new();
    for (path, entry) in &expected {
        let owned = hash_open_cache_file(cache_root, path, entry)?;
        verified.push(VerifiedFile {
            path: (*path).to_owned(),
            size_bytes: owned.size_bytes,
            sha256: entry.sha256.clone(),
        });
        owned_files.insert((*path).to_owned(), owned);
    }
    assert_cache_root_stable(cache_root, &root_before, "hash verification")?;
    assert_cache_path_bindings(cache_root, &owned_files, "hash verification")?;

    let index_value = read_verified_json(
        &owned_files,
        lock.model.tensor_contract.index_path.as_str(),
        MAX_INDEX_JSON_BYTES,
        true,
        "safetensors index",
    )?;
    let config = read_verified_bytes(
        &owned_files,
        "config.json",
        MAX_CONFIG_JSON_BYTES,
        "model config",
    )?;
    let config_value = parse_json(&config, false, MAX_CONFIG_JSON_BYTES, "model config")?;
    let qwen_shape_inputs = validate_parsed_model_config(lock, &config_value)?;
    let tensors =
        validate_safetensors(lock, &owned_files, &index_value, qwen_shape_inputs.as_ref())?;
    validate_stop_identity(lock, &owned_files)?;
    assert_cache_root_stable(cache_root, &root_before, "semantic validation")?;
    assert_cache_path_bindings(cache_root, &owned_files, "semantic validation")?;
    let mut actual_after = BTreeMap::new();
    collect_cache_files(cache_root, cache_root, &mut actual_after)?;
    if actual_after.len() != expected.len()
        || actual_after
            .keys()
            .any(|path| !expected.contains_key(path.as_str()))
    {
        return Err(invalid("cache file set changed during validation"));
    }
    Ok(VerifiedCache {
        lock_fingerprint: lock.fingerprint.clone(),
        files: verified,
        tensors,
        owned_files,
        cache_root: cache_root.to_path_buf(),
        root_identity: root_before,
    })
}

/// Validate the selected typed fields in a source `config.json`.
///
/// The source config is not part of the lock fingerprint.  Its ordinary
/// numeric `rms_norm_eps` is permitted only as a finite value matching the
/// locked decimal string; all lock identity JSON remains integer-only.
pub fn validate_model_config(lock: &ModelLock, bytes: &[u8]) -> Result<(), ModelError> {
    let value = parse_json(bytes, false, MAX_CONFIG_JSON_BYTES, "model config")?;
    validate_parsed_model_config(lock, &value).map(|_| ())
}

fn validate_parsed_model_config(
    lock: &ModelLock,
    value: &Value,
) -> Result<Option<QwenShapeInputs>, ModelError> {
    let root = value
        .as_object()
        .ok_or_else(|| invalid("config root must be an object"))?;
    let architecture = &lock.model.architecture;
    let require_full_text_config = lock.model.repo_id == QWEN_REPO_ID;
    if let Some(value) = root.get("architectures") {
        if !json_string_array_equals(value, &architecture.architectures) {
            return Err(invalid("config architectures differ from lock"));
        }
    } else if require_full_text_config {
        return Err(invalid("config architectures are missing"));
    }
    if let Some(value) = root.get("model_type") {
        if value.as_str() != Some(architecture.model_type.as_str()) {
            return Err(invalid("config model_type differs from lock"));
        }
    } else if require_full_text_config {
        return Err(invalid("config model_type is missing"));
    }
    let text = root
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("config lacks text_config object"))?;
    let locked = &architecture.text_config;
    check_string(text, "model_type", &locked.model_type)?;
    for (key, expected) in [
        ("hidden_size", locked.hidden_size),
        ("num_hidden_layers", locked.num_hidden_layers),
        ("num_attention_heads", locked.num_attention_heads),
        ("num_key_value_heads", locked.num_key_value_heads),
        ("head_dim", locked.head_dim),
        ("intermediate_size", locked.intermediate_size),
        ("full_attention_interval", locked.full_attention_interval),
        ("vocab_size", locked.vocab_size),
        ("mtp_num_hidden_layers", locked.mtp_num_hidden_layers),
    ] {
        if require_full_text_config {
            check_u64(text, key, expected)?;
        } else {
            check_optional_u64(text, key, expected)?;
        }
    }
    if require_full_text_config || text.contains_key("tie_word_embeddings") {
        check_bool(text, "tie_word_embeddings", locked.tie_word_embeddings)?;
    }
    if let Some(value) = text.get("dtype") {
        if value.as_str() != Some("bfloat16") {
            return Err(invalid("config text dtype is not bfloat16"));
        }
    } else if require_full_text_config {
        return Err(invalid("config text dtype is missing"));
    }
    if let Some(value) = text.get("rms_norm_eps") {
        let matches = match value {
            Value::String(value) => value == &locked.rms_norm_eps,
            Value::Number(value) => value
                .as_f64()
                .map(|actual| {
                    actual.is_finite()
                        && actual == locked.rms_norm_eps.parse::<f64>().unwrap_or(-1.0)
                })
                .unwrap_or(false),
            _ => false,
        };
        if !matches {
            return Err(invalid("config rms_norm_eps differs from lock"));
        }
    } else if require_full_text_config {
        return Err(invalid("config rms_norm_eps is missing"));
    }
    if let Some(value) = text.get("layer_types") {
        let expected: Vec<&str> = locked
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerType::LinearAttention => "linear_attention",
                LayerType::FullAttention => "full_attention",
            })
            .collect();
        let actual = value
            .as_array()
            .ok_or_else(|| invalid("config layer_types is not an array"))?;
        if actual.len() != expected.len()
            || actual
                .iter()
                .zip(&expected)
                .any(|(actual, expected)| actual.as_str() != Some(*expected))
        {
            return Err(invalid("config layer_types differ from lock"));
        }
    } else if require_full_text_config {
        return Err(invalid("config layer_types is missing"));
    }
    if let Some(value) = text.get("eos_token_id") {
        if value.as_u64()
            != Some(
                lock.model
                    .tokenizer_contract
                    .stop_identity
                    .config_eos
                    .token_id,
            )
        {
            return Err(invalid("config EOS ID differs from lock"));
        }
    } else if require_full_text_config {
        return Err(invalid("config EOS ID is missing"));
    }
    if require_full_text_config {
        validate_qwen_config_constants(root)?;
    }
    if require_full_text_config {
        Ok(Some(validate_qwen_shape_inputs(root, architecture)?))
    } else {
        Ok(None)
    }
}

/// Validate every reviewed config field that is specific to the immutable
/// Qwen3.5-4B revision.  The lock is still the byte identity; these constants
/// make the semantic surface explicit without widening model-lock-v1.
fn validate_qwen_config_constants(root: &Map<String, Value>) -> Result<(), ModelError> {
    expect_exact_keys(
        root,
        &[
            "architectures",
            "image_token_id",
            "model_type",
            "text_config",
            "tie_word_embeddings",
            "transformers_version",
            "video_token_id",
            "vision_config",
            "vision_end_token_id",
            "vision_start_token_id",
        ],
        "Qwen config root",
    )?;
    expect_string_array(root, "architectures", &["Qwen3_5ForConditionalGeneration"])?;
    expect_string(root, "model_type", "qwen3_5")?;
    expect_bool(root, "tie_word_embeddings", true)?;
    expect_string(root, "transformers_version", "4.57.0.dev0")?;
    for (field, expected) in [
        ("image_token_id", 248056),
        ("video_token_id", 248057),
        ("vision_end_token_id", 248054),
        ("vision_start_token_id", 248053),
    ] {
        expect_u64(root, field, expected)?;
    }

    let vision = root
        .get("vision_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Qwen vision_config is not an object"))?;
    expect_exact_keys(
        vision,
        &[
            "deepstack_visual_indexes",
            "depth",
            "hidden_act",
            "hidden_size",
            "in_channels",
            "initializer_range",
            "intermediate_size",
            "model_type",
            "num_heads",
            "num_position_embeddings",
            "out_hidden_size",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
        ],
        "Qwen vision_config",
    )?;
    if vision
        .get("deepstack_visual_indexes")
        .and_then(Value::as_array)
        .is_none_or(|values| !values.is_empty())
    {
        return Err(invalid("Qwen vision deepstack_visual_indexes differs"));
    }
    expect_string(vision, "model_type", "qwen3_5")?;
    expect_string(vision, "hidden_act", "gelu_pytorch_tanh")?;
    expect_f64(vision, "initializer_range", 0.02)?;
    for (field, expected) in [
        ("depth", 24),
        ("hidden_size", 1024),
        ("in_channels", 3),
        ("intermediate_size", 4096),
        ("num_heads", 16),
        ("num_position_embeddings", 2304),
        ("out_hidden_size", 2560),
        ("patch_size", 16),
        ("spatial_merge_size", 2),
        ("temporal_patch_size", 2),
    ] {
        expect_u64(vision, field, expected)?;
    }

    let text = root
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Qwen text_config is not an object"))?;
    expect_exact_keys(
        text,
        &[
            "attention_bias",
            "attention_dropout",
            "attn_output_gate",
            "dtype",
            "eos_token_id",
            "full_attention_interval",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "layer_types",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "mamba_ssm_dtype",
            "max_position_embeddings",
            "mlp_only_layers",
            "model_type",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_parameters",
            "tie_word_embeddings",
            "use_cache",
            "vocab_size",
        ],
        "Qwen text_config",
    )?;
    expect_bool(text, "attention_bias", false)?;
    expect_bool(text, "attn_output_gate", true)?;
    expect_bool(text, "mtp_use_dedicated_embeddings", false)?;
    expect_bool(text, "tie_word_embeddings", true)?;
    expect_bool(text, "use_cache", true)?;
    expect_string(text, "hidden_act", "silu")?;
    expect_string(text, "mamba_ssm_dtype", "float32")?;
    expect_f64(text, "attention_dropout", 0.0)?;
    expect_f64(text, "initializer_range", 0.02)?;
    expect_f64(text, "rms_norm_eps", 0.000001)?;
    if text
        .get("mlp_only_layers")
        .and_then(Value::as_array)
        .is_none_or(|values| !values.is_empty())
    {
        return Err(invalid("Qwen text mlp_only_layers differs"));
    }
    for (field, expected) in [
        ("linear_conv_kernel_dim", 4),
        ("linear_key_head_dim", 128),
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 32),
        ("linear_value_head_dim", 128),
        ("max_position_embeddings", 262144),
    ] {
        expect_u64(text, field, expected)?;
    }
    let rope = text
        .get("rope_parameters")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Qwen text rope_parameters is not an object"))?;
    expect_exact_keys(
        rope,
        &[
            "mrope_interleaved",
            "mrope_section",
            "partial_rotary_factor",
            "rope_theta",
            "rope_type",
        ],
        "Qwen text rope_parameters",
    )?;
    expect_bool(rope, "mrope_interleaved", true)?;
    expect_u64(rope, "rope_theta", 10_000_000)?;
    expect_string(rope, "rope_type", "default")?;
    expect_f64(rope, "partial_rotary_factor", 0.25)?;
    if rope
        .get("mrope_section")
        .and_then(Value::as_array)
        .map(|values| values.iter().map(Value::as_u64).collect::<Vec<_>>())
        != Some(vec![Some(11), Some(11), Some(10)])
    {
        return Err(invalid("Qwen text mrope_section differs"));
    }
    Ok(())
}

fn expect_exact_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    field: &str,
) -> Result<(), ModelError> {
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err(invalid(format!(
            "{field} fields differ from the reviewed revision"
        )));
    }
    Ok(())
}

fn expect_string(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
) -> Result<(), ModelError> {
    if object.get(field).and_then(Value::as_str) != Some(expected) {
        return Err(invalid(format!("Qwen config field {field} differs")));
    }
    Ok(())
}

fn expect_bool(object: &Map<String, Value>, field: &str, expected: bool) -> Result<(), ModelError> {
    if object.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(invalid(format!("Qwen config field {field} differs")));
    }
    Ok(())
}

fn expect_u64(object: &Map<String, Value>, field: &str, expected: u64) -> Result<(), ModelError> {
    if object.get(field).and_then(Value::as_u64) != Some(expected) {
        return Err(invalid(format!("Qwen config field {field} differs")));
    }
    Ok(())
}

fn expect_f64(object: &Map<String, Value>, field: &str, expected: f64) -> Result<(), ModelError> {
    if !matches!(
        object.get(field),
        Some(Value::Number(value)) if value.is_f64() && value.as_f64() == Some(expected)
    ) {
        return Err(invalid(format!("Qwen config field {field} differs")));
    }
    Ok(())
}

fn expect_string_array(
    object: &Map<String, Value>,
    field: &str,
    expected: &[&str],
) -> Result<(), ModelError> {
    let values = object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("Qwen config field {field} is not an array")))?;
    if values.len() != expected.len()
        || values
            .iter()
            .zip(expected)
            .any(|(value, expected)| value.as_str() != Some(*expected))
    {
        return Err(invalid(format!("Qwen config field {field} differs")));
    }
    Ok(())
}

fn validate_stop_identity(
    lock: &ModelLock,
    files: &BTreeMap<String, OwnedVerifiedFile>,
) -> Result<(), ModelError> {
    let identity = &lock.model.tokenizer_contract.stop_identity;
    let config = read_verified_json(
        files,
        &identity.config_eos.source_file,
        MAX_CONFIG_JSON_BYTES,
        false,
        "config EOS source",
    )?;
    let config_eos = config
        .get("text_config")
        .and_then(Value::as_object)
        .and_then(|object| object.get("eos_token_id"))
        .and_then(Value::as_u64);
    if config_eos != Some(identity.config_eos.token_id) {
        return Err(invalid("config EOS identity differs from lock"));
    }
    let tokenizer_config_path = identity
        .tokenizer_eos
        .source_files
        .iter()
        .find(|path| path.ends_with("tokenizer_config.json"))
        .ok_or_else(|| invalid("tokenizer EOS sources omit tokenizer_config.json"))?;
    let tokenizer_json_path = identity
        .tokenizer_eos
        .source_files
        .iter()
        .find(|path| path.ends_with("tokenizer.json"))
        .ok_or_else(|| invalid("tokenizer EOS sources omit tokenizer.json"))?;
    let tokenizer_config = read_verified_json(
        files,
        tokenizer_config_path,
        MAX_CONFIG_JSON_BYTES,
        false,
        "tokenizer config",
    )?;
    let tokenizer_object = tokenizer_config
        .as_object()
        .ok_or_else(|| invalid("tokenizer config is not an object"))?;
    if tokenizer_object.get("eos_token").and_then(Value::as_str)
        != Some(identity.tokenizer_eos.token.as_str())
    {
        return Err(invalid("tokenizer EOS token differs from lock"));
    }
    let decoder = tokenizer_object
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("tokenizer config lacks added_tokens_decoder"))?;
    for (id, token) in [
        (
            identity.config_eos.token_id,
            identity.config_eos.token.as_str(),
        ),
        (
            identity.tokenizer_eos.token_id,
            identity.tokenizer_eos.token.as_str(),
        ),
    ] {
        if decoder
            .get(&id.to_string())
            .and_then(Value::as_object)
            .and_then(|entry| entry.get("content"))
            .and_then(Value::as_str)
            != Some(token)
        {
            return Err(invalid("tokenizer config EOS decoder differs from lock"));
        }
        if decoder
            .values()
            .filter(|entry| {
                entry
                    .as_object()
                    .and_then(|entry| entry.get("content"))
                    .and_then(Value::as_str)
                    == Some(token)
            })
            .count()
            != 1
        {
            return Err(invalid(
                "tokenizer config EOS content is missing or duplicated",
            ));
        }
    }
    let tokenizer = read_verified_json(
        files,
        tokenizer_json_path,
        MAX_TOKENIZER_JSON_BYTES,
        false,
        "tokenizer JSON",
    )?;
    let added_tokens = tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("tokenizer JSON lacks added_tokens"))?;
    for (id, token) in [
        (
            identity.config_eos.token_id,
            identity.config_eos.token.as_str(),
        ),
        (
            identity.tokenizer_eos.token_id,
            identity.tokenizer_eos.token.as_str(),
        ),
    ] {
        let id_matches = added_tokens
            .iter()
            .filter(|entry| entry.get("id").and_then(Value::as_u64) == Some(id))
            .count();
        let content_matches = added_tokens
            .iter()
            .filter(|entry| entry.get("content").and_then(Value::as_str) == Some(token))
            .count();
        let exact_matches = added_tokens
            .iter()
            .filter(|entry| {
                entry.get("id").and_then(Value::as_u64) == Some(id)
                    && entry.get("content").and_then(Value::as_str) == Some(token)
            })
            .count();
        if id_matches != 1 || content_matches != 1 || exact_matches != 1 {
            return Err(invalid(
                "tokenizer JSON EOS ID/content identity is missing or duplicated",
            ));
        }
    }
    Ok(())
}

fn check_string(object: &Map<String, Value>, key: &str, expected: &str) -> Result<(), ModelError> {
    if object.get(key).and_then(Value::as_str) != Some(expected) {
        return Err(invalid(format!("config field {key} differs from lock")));
    }
    Ok(())
}

fn check_u64(object: &Map<String, Value>, key: &str, expected: u64) -> Result<(), ModelError> {
    if object.get(key).and_then(Value::as_u64) != Some(expected) {
        return Err(invalid(format!("config field {key} differs from lock")));
    }
    Ok(())
}

fn check_optional_u64(
    object: &Map<String, Value>,
    key: &str,
    expected: u64,
) -> Result<(), ModelError> {
    if let Some(value) = object.get(key) {
        if value.as_u64() != Some(expected) {
            return Err(invalid(format!("config field {key} differs from lock")));
        }
    }
    Ok(())
}

fn check_bool(object: &Map<String, Value>, key: &str, expected: bool) -> Result<(), ModelError> {
    if object.get(key).and_then(Value::as_bool) != Some(expected) {
        return Err(invalid(format!("config field {key} differs from lock")));
    }
    Ok(())
}

fn json_string_array_equals(value: &Value, expected: &[String]) -> bool {
    value.as_array().is_some_and(|array| {
        array.len() == expected.len()
            && array
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.as_str() == Some(expected.as_str()))
    })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    total_size: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsIndex {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

fn validate_safetensors(
    lock: &ModelLock,
    files: &BTreeMap<String, OwnedVerifiedFile>,
    index_value: &Value,
    qwen_shape_inputs: Option<&QwenShapeInputs>,
) -> Result<BTreeMap<String, TensorDescriptor>, ModelError> {
    let index: SafetensorsIndex = from_value(index_value.clone())?;
    let contract = &lock.model.tensor_contract;
    if index.weight_map.len() as u64 != contract.indexed_tensor_count
        || index.weight_map.is_empty()
        || index.weight_map.values().any(|shard| shard.is_empty())
        || index.weight_map.values().collect::<BTreeSet<_>>().len() != contract.shards.len()
    {
        return Err(invalid(
            "safetensors index tensor/shard count differs from lock",
        ));
    }
    let locked_shards: BTreeSet<&str> = contract.shards.iter().map(String::as_str).collect();
    let index_shards: BTreeSet<&str> = index.weight_map.values().map(String::as_str).collect();
    if index_shards != locked_shards {
        return Err(invalid("safetensors index shard set differs from lock"));
    }
    let mut tensors = BTreeMap::new();
    let mut total_size = 0u64;
    for shard in &contract.shards {
        let bytes = read_safetensors_header(files, shard)?;
        let (header_length, file_size, header_value) = bytes;
        let header = header_value
            .as_object()
            .ok_or_else(|| invalid(format!("safetensors header is not an object: {shard}")))?;
        let data_buffer_start = 8u64
            .checked_add(header_length)
            .ok_or_else(|| invalid("safetensors data-buffer offset overflow"))?;
        if data_buffer_start > file_size {
            return Err(invalid(format!(
                "safetensors header exceeds shard: {shard}"
            )));
        }
        if let Some(metadata) = header.get("__metadata__") {
            let metadata = metadata
                .as_object()
                .ok_or_else(|| invalid(format!("invalid safetensors metadata: {shard}")))?;
            if metadata.values().any(|value| value.as_str().is_none()) {
                return Err(invalid(format!(
                    "safetensors metadata values must be strings: {shard}"
                )));
            }
        }
        let mut spans = Vec::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let object = value.as_object().ok_or_else(|| {
                invalid(format!("tensor metadata is not an object: {shard}:{name}"))
            })?;
            if object.len() != 3
                || !object.contains_key("dtype")
                || !object.contains_key("shape")
                || !object.contains_key("data_offsets")
            {
                return Err(invalid(format!(
                    "tensor metadata keys are not exact: {shard}:{name}"
                )));
            }
            let dtype: TensorDType = from_value(object["dtype"].clone())?;
            let shape = object["shape"]
                .as_array()
                .ok_or_else(|| invalid(format!("tensor shape is not an array: {shard}:{name}")))?;
            let shape: Vec<u64> = shape
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| invalid(format!("invalid tensor shape: {shard}:{name}")))
                })
                .collect::<Result<_, _>>()?;
            if shape.is_empty() || shape.contains(&0) {
                return Err(invalid(format!(
                    "tensor shape is empty/zero: {shard}:{name}"
                )));
            }
            let offsets = object["data_offsets"].as_array().ok_or_else(|| {
                invalid(format!("tensor offsets are not an array: {shard}:{name}"))
            })?;
            if offsets.len() != 2 {
                return Err(invalid(format!(
                    "tensor offsets do not have length two: {shard}:{name}"
                )));
            }
            let offsets = [
                offsets[0]
                    .as_u64()
                    .ok_or_else(|| invalid("invalid tensor offset"))?,
                offsets[1]
                    .as_u64()
                    .ok_or_else(|| invalid("invalid tensor offset"))?,
            ];
            if offsets[0] >= offsets[1] {
                return Err(invalid(format!(
                    "tensor offsets are empty/reversed: {shard}:{name}"
                )));
            }
            let expected_bytes = product(shape.iter().copied())
                .and_then(|value| value.checked_mul(dtype.byte_width()))
                .ok_or_else(|| invalid(format!("tensor size overflow: {shard}:{name}")))?;
            if offsets[1] - offsets[0] != expected_bytes {
                return Err(invalid(format!(
                    "tensor dtype/shape size mismatch: {shard}:{name}"
                )));
            }
            let absolute_start = data_buffer_start
                .checked_add(offsets[0])
                .ok_or_else(|| invalid("tensor start overflow"))?;
            let absolute_end = data_buffer_start
                .checked_add(offsets[1])
                .ok_or_else(|| invalid("tensor end overflow"))?;
            if absolute_end > file_size {
                return Err(invalid(format!(
                    "tensor range exceeds shard: {shard}:{name}"
                )));
            }
            if tensors.contains_key(name) {
                return Err(invalid(format!("duplicate tensor across shards: {name}")));
            }
            let descriptor = TensorDescriptor {
                tensor_name: name.clone(),
                source_file: shard.clone(),
                dtype,
                shape,
                header_length_field_bytes: 8,
                header_length_bytes: header_length,
                data_buffer_start,
                data_offset_basis: "data-buffer-relative".to_owned(),
                data_offsets: offsets,
                absolute_byte_range: [absolute_start, absolute_end],
                byte_size: expected_bytes,
            };
            spans.push((offsets[0], offsets[1], name.clone()));
            total_size = total_size
                .checked_add(expected_bytes)
                .ok_or_else(|| invalid("total tensor size overflow"))?;
            tensors.insert(name.clone(), descriptor);
        }
        spans.sort_by_key(|span| (span.0, span.1, span.2.clone()));
        let payload_size = file_size - data_buffer_start;
        let mut cursor = 0u64;
        for (start, end, name) in spans {
            if start != cursor {
                return Err(invalid(format!(
                    "safetensors payload has gap/overlap: {shard}:{name}"
                )));
            }
            cursor = end;
        }
        if cursor != payload_size {
            return Err(invalid(format!(
                "safetensors payload is not fully covered: {shard}"
            )));
        }
    }
    if index.metadata.total_size != total_size {
        return Err(invalid("safetensors index total_size differs from headers"));
    }
    if tensors.len() != index.weight_map.len()
        || tensors.keys().any(|name| {
            index.weight_map.get(name) != tensors.get(name).map(|tensor| &tensor.source_file)
        })
    {
        return Err(invalid("safetensors index and shard tensor names differ"));
    }
    for (name, shard) in &index.weight_map {
        if tensors.get(name).map(|tensor| tensor.source_file.as_str()) != Some(shard.as_str()) {
            return Err(invalid(format!(
                "safetensors index mapping differs: {name}"
            )));
        }
    }
    let mut counts: HashMap<&str, u64> = HashMap::new();
    for name in tensors.keys() {
        let matches: Vec<&TensorClassification> = contract
            .classifications
            .iter()
            .filter(|classification| name.starts_with(&classification.prefix))
            .collect();
        if matches.len() != 1 {
            return Err(invalid(format!(
                "unknown or multiply classified tensor: {name}"
            )));
        }
        *counts.entry(matches[0].id.as_str()).or_default() += 1;
    }
    for classification in &contract.classifications {
        if counts.get(classification.id.as_str()).copied().unwrap_or(0)
            != classification.tensor_count
        {
            return Err(invalid(format!(
                "tensor classification count differs: {}",
                classification.id
            )));
        }
    }
    if lock.model.repo_id == QWEN_REPO_ID {
        let shape_inputs = qwen_shape_inputs
            .ok_or_else(|| invalid("Qwen safetensors validation lacks parsed config shapes"))?;
        let catalog = qwen_tensor_catalog(shape_inputs)?;
        validate_qwen_header_catalog(&tensors, &contract.classifications, &catalog)?;
    }
    let slice = &lock.model.slice_contract;
    let descriptor = tensors
        .get(&slice.tensor_name)
        .ok_or_else(|| invalid("locked slice tensor is absent"))?;
    if descriptor.source_file != slice.source_file
        || descriptor.dtype != slice.dtype
        || descriptor.shape != slice.shape
        || descriptor.header_length_field_bytes != slice.header_length_field_bytes
        || descriptor.header_length_bytes != slice.header_length_bytes
        || descriptor.data_buffer_start != slice.data_buffer_start
        || descriptor.data_offsets != slice.data_offsets
        || descriptor.absolute_byte_range != slice.absolute_byte_range
        || descriptor.byte_size != slice.byte_size
    {
        return Err(invalid("locked slice does not match safetensors metadata"));
    }
    Ok(tensors)
}

fn read_safetensors_header(
    files: &BTreeMap<String, OwnedVerifiedFile>,
    shard: &str,
) -> Result<(u64, u64, Value), ModelError> {
    let file = verified_file(files, shard)?;
    let raw_length = read_owned_range(file, 0, 8, 8, "safetensors header length")?;
    let header_length = u64::from_le_bytes(
        raw_length
            .as_slice()
            .try_into()
            .map_err(|_| invalid("safetensors header length is truncated"))?,
    );
    if !(8..=MAX_SAFE_TENSOR_HEADER).contains(&header_length) {
        return Err(invalid(format!(
            "invalid safetensors header length: {shard}"
        )));
    }
    let header_bytes = read_owned_range(
        file,
        8,
        usize::try_from(header_length).map_err(|_| invalid("header length does not fit usize"))?,
        usize::try_from(MAX_SAFE_TENSOR_HEADER)
            .map_err(|_| invalid("safetensors header limit does not fit usize"))?,
        "safetensors header",
    )?;
    let value = parse_json(
        &header_bytes,
        true,
        usize::try_from(MAX_SAFE_TENSOR_HEADER)
            .map_err(|_| invalid("safetensors header limit does not fit usize"))?,
        "safetensors header",
    )?;
    assert_owned_file_stable(file, "safetensors header validation")?;
    Ok((header_length, file.size_bytes, value))
}

fn validate_cache_root(path: &Path) -> Result<FileIdentity, ModelError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid(format!(
            "cache root must be a non-symlink directory: {}",
            path.display()
        )));
    }
    Ok(metadata_identity(&metadata))
}

fn collect_cache_files(
    root: &Path,
    directory: &Path,
    output: &mut BTreeMap<String, PathBuf>,
) -> Result<(), ModelError> {
    for entry in fs::read_dir(directory).map_err(|error| io_error(directory, error))? {
        let entry = entry.map_err(|error| io_error(directory, error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
        if metadata.file_type().is_symlink() {
            return Err(invalid(format!(
                "cache contains a symlink: {}",
                path.display()
            )));
        }
        if metadata.is_dir() {
            collect_cache_files(root, &path, output)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| invalid("cache entry escaped root"))?
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            validate_safe_path(&relative, "cache file path")?;
            if output.insert(relative.clone(), path).is_some() {
                return Err(invalid(format!("duplicate cache path: {relative}")));
            }
        } else {
            return Err(invalid(format!(
                "cache contains a non-regular entry: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn open_cache_file(root: &Path, relative: &str) -> Result<File, ModelError> {
    validate_safe_path(relative, "cache relative path")?;
    if relative.contains('/') {
        return Err(invalid(
            "nested cache paths are not accepted without an openat directory walk",
        ));
    }
    let path = root.join(relative);
    let metadata = fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata_identity(&metadata).link_count != 1
    {
        return Err(invalid(format!(
            "cache path is not a single-link regular non-symlink file: {relative}"
        )));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(0o400000);
    options.open(&path).map_err(|error| io_error(&path, error))
}

fn hash_open_cache_file(
    root: &Path,
    relative: &str,
    expected: &LockedFile,
) -> Result<OwnedVerifiedFile, ModelError> {
    let path = root.join(relative);
    let before = fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
    let file = open_cache_file(root, relative)?;
    let opened = file.metadata().map_err(|error| io_error(&path, error))?;
    if metadata_identity(&before) != metadata_identity(&opened) {
        return Err(invalid(format!(
            "cache file changed while opening: {relative}"
        )));
    }
    if opened.len() != expected.size_bytes {
        return Err(invalid(format!(
            "cache size mismatch before hashing: {relative}"
        )));
    }
    let mut hasher = Sha256::new();
    let mut offset = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    while offset < opened.len() {
        let remaining = opened.len() - offset;
        let requested = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| invalid("cache read length does not fit usize"))?;
        let read = read_at_exact(&file, &mut buffer[..requested], offset, &path)?;
        if read != requested {
            return Err(invalid(format!(
                "cache file ended while hashing: {relative}"
            )));
        }
        hasher.update(&buffer[..read]);
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| invalid("cache hash offset overflow"))?;
    }
    let after = file.metadata().map_err(|error| io_error(&path, error))?;
    let path_after = fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
    if metadata_identity(&opened) != metadata_identity(&after)
        || metadata_identity(&after) != metadata_identity(&path_after)
    {
        return Err(invalid(format!(
            "cache file changed during hashing: {relative}"
        )));
    }
    let sha256 = hex_lower(&hasher.finalize());
    if offset != expected.size_bytes || sha256 != expected.sha256 {
        return Err(invalid(format!("cache size/hash mismatch: {relative}")));
    }
    Ok(OwnedVerifiedFile {
        path,
        file,
        identity: metadata_identity(&after),
        size_bytes: after.len(),
    })
}

fn verified_file<'a>(
    files: &'a BTreeMap<String, OwnedVerifiedFile>,
    relative: &str,
) -> Result<&'a OwnedVerifiedFile, ModelError> {
    files.get(relative).ok_or_else(|| {
        invalid(format!(
            "semantic read references an unhashed file: {relative}"
        ))
    })
}

fn read_verified_bytes(
    files: &BTreeMap<String, OwnedVerifiedFile>,
    relative: &str,
    max_bytes: usize,
    purpose: &str,
) -> Result<Vec<u8>, ModelError> {
    let file = verified_file(files, relative)?;
    let length = usize::try_from(file.size_bytes)
        .map_err(|_| invalid(format!("{purpose} size does not fit usize")))?;
    read_owned_range(file, 0, length, max_bytes, purpose)
}

fn read_verified_json(
    files: &BTreeMap<String, OwnedVerifiedFile>,
    relative: &str,
    max_bytes: usize,
    reject_floats: bool,
    purpose: &str,
) -> Result<Value, ModelError> {
    let bytes = read_verified_bytes(files, relative, max_bytes, purpose)?;
    parse_json(&bytes, reject_floats, max_bytes, purpose)
}

fn read_owned_range(
    owned: &OwnedVerifiedFile,
    offset: u64,
    length: usize,
    max_bytes: usize,
    purpose: &str,
) -> Result<Vec<u8>, ModelError> {
    if length > max_bytes {
        return Err(invalid(format!("{purpose} exceeds the bounded read limit")));
    }
    let end = offset
        .checked_add(u64::try_from(length).map_err(|_| invalid("range length does not fit u64"))?)
        .ok_or_else(|| invalid("range offset overflow"))?;
    if end > owned.size_bytes {
        return Err(invalid(format!(
            "{purpose} range exceeds the verified file"
        )));
    }
    assert_owned_file_stable(owned, purpose)?;
    let mut bytes = vec![0u8; length];
    read_at_exact(&owned.file, &mut bytes, offset, &owned.path)?;
    assert_owned_file_stable(owned, purpose)?;
    Ok(bytes)
}

fn read_at_exact(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
    path: &Path,
) -> Result<usize, ModelError> {
    #[cfg(unix)]
    {
        let mut total = 0usize;
        while total < buffer.len() {
            let position = offset
                .checked_add(total as u64)
                .ok_or_else(|| invalid("read offset overflow"))?;
            let read = file
                .read_at(&mut buffer[total..], position)
                .map_err(|error| io_error(path, error))?;
            if read == 0 {
                return Err(invalid(format!(
                    "unexpected EOF while reading {}",
                    path.display()
                )));
            }
            total = total
                .checked_add(read)
                .ok_or_else(|| invalid("read length overflow"))?;
        }
        Ok(total)
    }
    #[cfg(not(unix))]
    {
        let _ = (file, buffer, offset, path);
        Err(invalid("verified positional reads require Unix"))
    }
}

fn assert_owned_file_stable(
    owned: &OwnedVerifiedFile,
    operation: &str,
) -> Result<FileIdentity, ModelError> {
    let metadata = owned
        .file
        .metadata()
        .map_err(|error| io_error(&owned.path, error))?;
    let identity = metadata_identity(&metadata);
    if identity != owned.identity || identity.link_count != 1 {
        return Err(invalid(format!(
            "verified file changed during {operation}: {}",
            owned.path.display()
        )));
    }
    Ok(identity)
}

fn assert_cache_root_stable(
    root: &Path,
    expected: &FileIdentity,
    operation: &str,
) -> Result<(), ModelError> {
    let current = fs::symlink_metadata(root).map_err(|error| io_error(root, error))?;
    if current.file_type().is_symlink()
        || !current.is_dir()
        || metadata_identity(&current) != *expected
    {
        return Err(invalid(format!("cache root changed during {operation}")));
    }
    Ok(())
}

fn assert_cache_path_bindings(
    root: &Path,
    files: &BTreeMap<String, OwnedVerifiedFile>,
    operation: &str,
) -> Result<(), ModelError> {
    for (relative, owned) in files {
        assert_owned_file_stable(owned, operation)?;
        let metadata = fs::symlink_metadata(root.join(relative))
            .map_err(|error| io_error(&root.join(relative), error))?;
        if metadata.file_type().is_symlink() || metadata_identity(&metadata) != owned.identity {
            return Err(invalid(format!(
                "cache path changed during {operation}: {relative}"
            )));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn metadata_identity(metadata: &Metadata) -> FileIdentity {
    FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        size_bytes: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
        link_count: metadata.nlink(),
    }
}

#[cfg(not(unix))]
fn metadata_identity(metadata: &Metadata) -> FileIdentity {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as i64)
        .unwrap_or_default();
    FileIdentity {
        device: 0,
        inode: 0,
        size_bytes: metadata.len(),
        modified_seconds: modified,
        modified_nanoseconds: 0,
        changed_seconds: 0,
        changed_nanoseconds: 0,
        link_count: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_fingerprint_matches_reviewed_lock() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/models/locks/qwen3.5-4b-bf16.json");
        let lock = read_model_lock(path).expect("reviewed Qwen lock parses");
        assert_eq!(lock.fingerprint(), QWEN_FINGERPRINT);
        assert_eq!(lock.model.architecture.text_config.dtype, TensorDType::Bf16);
    }

    #[test]
    fn duplicate_key_and_float_fingerprint_are_rejected() {
        assert!(
            parse_model_lock(
                br#"{"schema_version":"model-lock-v1","schema_version":"model-lock-v1"}"#
            )
            .is_err()
        );
        assert!(parse_model_lock(br#"{"schema_version":1.0}"#).is_err());
    }

    #[test]
    fn generation_stop_policy_accepts_u32_max_and_rejects_overflow() {
        let base = serde_json::json!({
            "version": 1,
            "stop_token_ids": [4294967295u64],
            "evaluation": "newly_generated_after_argmax",
            "prompt_evaluation": "never_stop",
            "stop_token": {
                "visible_output": false,
                "subsequent_decode_input": false
            },
            "budget_boundary": "stop_token_wins",
            "max_new_tokens_zero": "max_new_tokens_before_decode",
            "reason_version": 1
        });
        let policy: GenerationStopPolicyV1 =
            serde_json::from_value(base.clone()).expect("u32 maximum is representable");
        validate_generation_stop_policy(&policy).expect("u32 maximum is valid");
        let mut overflow = base;
        overflow["stop_token_ids"] = serde_json::json!([4294967296u64]);
        assert!(serde_json::from_value::<GenerationStopPolicyV1>(overflow).is_err());
    }

    #[test]
    fn frontend_asset_specifications_match_locked_names_and_caps() {
        let specifications = [
            (FrontendAssetKind::ConfigJson, "config.json", 1024 * 1024),
            (
                FrontendAssetKind::TokenizerJson,
                "tokenizer.json",
                16 * 1024 * 1024,
            ),
            (
                FrontendAssetKind::TokenizerConfigJson,
                "tokenizer_config.json",
                256 * 1024,
            ),
            (
                FrontendAssetKind::ChatTemplateJinja,
                "chat_template.jinja",
                64 * 1024,
            ),
        ];

        for (kind, expected_name, expected_cap) in specifications {
            assert_eq!(kind.specification(), (expected_name, expected_cap));
        }
    }
}
