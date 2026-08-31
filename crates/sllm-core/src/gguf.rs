//! Bounded GGUF v3 parsing and the versioned sLLM tensor-recipe extension.
//!
//! The reader owns the file descriptor used for verification and payload reads.
//! It does not mmap, allocate device memory, reopen a path, or fall back to a
//! source container after an error.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};

pub const GGUF_VERSION: u32 = 3;
pub const GGUF_ALIGNMENT: u64 = 32;
pub const SLLM_GGUF_EXTENSION_VERSION: u32 = 1;
pub const SLLM_EXTENSION_VERSION_KEY: &str = "sllm.extension.version";
pub const SLLM_TENSOR_RECIPE_KEY: &str = "sllm.tensor_recipe";
pub const SLLM_TENSOR_RECIPE_SHA256_KEY: &str = "sllm.tensor_recipe.sha256";
pub const SLLM_FRONTEND_CONFIG_KEY: &str = "sllm.frontend.config_json";
pub const SLLM_FRONTEND_TOKENIZER_KEY: &str = "sllm.frontend.tokenizer_json";
pub const SLLM_FRONTEND_TOKENIZER_CONFIG_KEY: &str = "sllm.frontend.tokenizer_config_json";
pub const SLLM_FRONTEND_PREPROCESSOR_CONFIG_KEY: &str = "sllm.frontend.preprocessor_config_json";
/// Metadata namespace used by the Gemma 4 assistant GGUF.  These keys are
/// intentionally kept outside the tensor recipe: they describe the required
/// target/assistant pair and are not an alternate weight encoding.
pub const GEMMA4_MTP_ROLE_KEY: &str = "gemma4mtp.role";
pub const GEMMA4_MTP_SEMANTIC_PAIR_KEY: &str = "gemma4mtp.semantic_pair_id";
pub const GEMMA4_MTP_TARGET_FINGERPRINT_KEY: &str = "gemma4mtp.target_fingerprint";
pub const GEMMA4_MTP_ASSISTANT_FINGERPRINT_KEY: &str = "gemma4mtp.assistant_fingerprint";
pub const GEMMA4_MTP_LAYER_MAPPING_KEY: &str = "gemma4mtp.layer_mapping";
pub const GEMMA4_MTP_KV_MAPPING_KEY: &str = "gemma4mtp.kv_mapping";
pub const GEMMA4_MTP_TOKENIZER_IDENTITY_KEY: &str = "gemma4mtp.tokenizer_identity";
pub const GEMMA4_MTP_SOURCE_RANGES_KEY: &str = "gemma4mtp.source_ranges";

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MAX_HEADER_BYTES: u64 = 512 * 1024 * 1024;
const MAX_METADATA_ENTRIES: u64 = 16_384;
// DeepSeek V4 Flash contains 72,317 indexed tensors. Keep this bounded while
// allowing its reviewed catalog to be represented without weakening any byte
// range or allocation checks.
const MAX_TENSORS: u64 = 100_000;
const MAX_DIMS: u32 = 4;
const MAX_KEY_BYTES: u64 = 1_024;
const MAX_NAME_BYTES: u64 = 16 * 1024;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 2_000_000;
const HASH_CHUNK_BYTES: usize = 1024 * 1024;

#[cfg(unix)]
const O_NONBLOCK: i32 = 0o4000;
#[cfg(unix)]
const O_NOFOLLOW: i32 = 0o400000;
#[cfg(unix)]
const O_CLOEXEC: i32 = 0o2000000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufError {
    Invalid(String),
    Io { path: PathBuf, message: String },
}

impl fmt::Display for GgufError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid GGUF: {message}"),
            Self::Io { path, message } => {
                write!(formatter, "GGUF I/O error at {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for GgufError {}

fn invalid(message: impl Into<String>) -> GgufError {
    GgufError::Invalid(message.into())
}

fn io_error(path: &Path, error: impl fmt::Display) -> GgufError {
    GgufError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    I8Carrier = 24,
    Bf16 = 30,
    Mxfp4 = 39,
    Nvfp4 = 40,
}

impl GgufTensorType {
    fn from_raw(raw: u32) -> Result<Self, GgufError> {
        match raw {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            24 => Ok(Self::I8Carrier),
            30 => Ok(Self::Bf16),
            39 => Ok(Self::Mxfp4),
            40 => Ok(Self::Nvfp4),
            _ => Err(invalid(format!("unsupported tensor type {raw}"))),
        }
    }

    pub fn raw(self) -> u32 {
        self as u32
    }

    pub fn block_size(self) -> u64 {
        match self {
            Self::Mxfp4 => 32,
            Self::Nvfp4 => 64,
            Self::F32 | Self::F16 | Self::I8Carrier | Self::Bf16 => 1,
        }
    }

    pub fn type_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I8Carrier => 1,
            Self::Mxfp4 => 17,
            Self::Nvfp4 => 36,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GgufArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    /// GGUF/GGML dimension order (`ne[0]` first), not source-framework order.
    pub dimensions: Vec<u64>,
    pub tensor_type: GgufTensorType,
    pub relative_offset: u64,
    pub absolute_range: [u64; 2],
}

impl GgufTensorInfo {
    pub fn byte_length(&self) -> u64 {
        self.absolute_range[1] - self.absolute_range[0]
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GgufRecipeEncoding {
    Fp8E4m3fnChannelBf16Scale,
    Fp8E4m3fnChannelF32Scale,
    Nvfp4E2m1Block16E4m3fnF32Outer,
    Mxfp4E2m1Block32E8m0,
    Mxfp8E4m3Block32E8m0,
    Mxfp6E3m2Block32E8m0,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GgufTensorScope {
    Consumed,
    KnownUnconsumed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GgufScaleRole {
    Channel,
    Block,
    Outer,
    Input,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GgufScaleBinding {
    pub tensor: String,
    pub role: GgufScaleRole,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GgufTensorBinding {
    pub logical_tensor: String,
    pub value_tensor: String,
    pub encoding: GgufRecipeEncoding,
    pub role: String,
    pub logical_shape: Vec<u64>,
    pub scope: GgufTensorScope,
    pub scales: Vec<GgufScaleBinding>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GgufLogicalShapeBinding {
    pub tensor: String,
    pub logical_shape: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GgufStaticFp8KvBinding {
    pub layer: u32,
    pub key_decode_scale_bf16: u16,
    pub value_decode_scale_bf16: u16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GgufTensorRecipeV1 {
    pub schema_version: String,
    pub semantic_model_id: String,
    pub source_lock_fingerprints: Vec<String>,
    pub bindings: Vec<GgufTensorBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logical_shapes: Vec<GgufLogicalShapeBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub static_fp8_kv: Vec<GgufStaticFp8KvBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub known_unconsumed_tensors: Vec<String>,
}

impl GgufTensorRecipeV1 {
    pub fn canonical_json(&self) -> Result<String, GgufError> {
        serde_json::to_string(self)
            .map_err(|error| invalid(format!("serialize tensor recipe: {error}")))
    }

    pub fn digest(&self) -> Result<String, GgufError> {
        Ok(sha256_text(self.canonical_json()?.as_bytes()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufExtensionV1 {
    pub recipe: GgufTensorRecipeV1,
    pub recipe_sha256: String,
    pub frontend_assets: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone)]
pub struct VerifiedGguf {
    path: PathBuf,
    file: Arc<File>,
    file_size: u64,
    alignment: u64,
    data_offset: u64,
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
    tensor_index: BTreeMap<String, usize>,
    metadata_sha256: String,
    tensor_catalog_sha256: String,
    extension: Option<GgufExtensionV1>,
}

impl fmt::Debug for VerifiedGguf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedGguf")
            .field("path", &self.path)
            .field("file_size", &self.file_size)
            .field("alignment", &self.alignment)
            .field("data_offset", &self.data_offset)
            .field("metadata_count", &self.metadata.len())
            .field("tensor_count", &self.tensors.len())
            .field("metadata_sha256", &self.metadata_sha256)
            .field("tensor_catalog_sha256", &self.tensor_catalog_sha256)
            .field("extension", &self.extension)
            .finish()
    }
}

impl VerifiedGguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let file = open_regular_file(path)?;
        let file_size = file
            .metadata()
            .map_err(|error| io_error(path, error))?
            .len();
        let mut reader = BoundedReader::new(path, &file, file_size);

        if reader.read_exact(4)? != GGUF_MAGIC {
            return Err(invalid("magic differs"));
        }
        let version = reader.read_u32()?;
        if version != GGUF_VERSION {
            return Err(invalid(format!("unsupported version {version}")));
        }
        let tensor_count = reader.read_u64()?;
        let metadata_count = reader.read_u64()?;
        if tensor_count > MAX_TENSORS {
            return Err(invalid("tensor count exceeds bound"));
        }
        if metadata_count > MAX_METADATA_ENTRIES {
            return Err(invalid("metadata count exceeds bound"));
        }

        let metadata_start = reader.position();
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = reader.read_string(MAX_KEY_BYTES, "metadata key")?;
            if key.is_empty() {
                return Err(invalid("metadata key is empty"));
            }
            let value_type = reader.read_u32()?;
            let value = reader.read_value(value_type)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(invalid(format!("duplicate metadata key {key}")));
            }
        }
        let metadata_end = reader.position();

        let alignment = match metadata.get("general.alignment") {
            None => GGUF_ALIGNMENT,
            Some(GgufValue::U32(value)) => u64::from(*value),
            Some(_) => return Err(invalid("general.alignment has wrong type")),
        };
        if alignment != GGUF_ALIGNMENT {
            return Err(invalid(format!("alignment {alignment} is not 32")));
        }
        let architecture = match metadata.get("general.architecture") {
            Some(GgufValue::String(value)) => value.as_str(),
            Some(_) => return Err(invalid("general.architecture has wrong type")),
            None => return Err(invalid("general.architecture is missing")),
        };
        if !matches!(
            architecture,
            "qwen35"
                | "qwen35moe"
                | "gemma4"
                | "gemma4moe"
                | "gemma4mtp"
                | "deepseek4"
                | "minimax-m3"
                | "diffusion-gemma"
                | "mistral3"
        ) {
            return Err(invalid(format!("unsupported architecture {architecture}")));
        }

        let catalog_start = reader.position();
        let mut raw_tensors = Vec::with_capacity(to_usize(tensor_count, "tensor count")?);
        let mut tensor_names = BTreeSet::new();
        for _ in 0..tensor_count {
            let name = reader.read_string(MAX_NAME_BYTES, "tensor name")?;
            if name.is_empty() {
                return Err(invalid("tensor name is empty"));
            }
            if !tensor_names.insert(name.clone()) {
                return Err(invalid(format!("duplicate tensor name {name}")));
            }
            let dimension_count = reader.read_u32()?;
            if dimension_count == 0 || dimension_count > MAX_DIMS {
                return Err(invalid(format!(
                    "tensor {name} dimension count is outside 1..={MAX_DIMS}"
                )));
            }
            let mut dimensions = Vec::with_capacity(dimension_count as usize);
            for _ in 0..dimension_count {
                let dimension = reader.read_u64()?;
                if dimension == 0 {
                    return Err(invalid(format!("tensor {name} has zero dimension")));
                }
                dimensions.push(dimension);
            }
            let tensor_type = GgufTensorType::from_raw(reader.read_u32()?)?;
            let relative_offset = reader.read_u64()?;
            if relative_offset % alignment != 0 {
                return Err(invalid(format!("tensor {name} offset is misaligned")));
            }
            let byte_length = tensor_byte_length(&name, &dimensions, tensor_type)?;
            raw_tensors.push((name, dimensions, tensor_type, relative_offset, byte_length));
        }
        let catalog_end = reader.position();
        if catalog_end > MAX_HEADER_BYTES {
            return Err(invalid("header exceeds bound"));
        }
        let data_offset = if tensor_count == 0 {
            catalog_end
        } else {
            align_up(catalog_end, alignment)?
        };
        if data_offset > file_size {
            return Err(invalid("truncated tensor-data alignment padding"));
        }

        let mut tensors = Vec::with_capacity(raw_tensors.len());
        for (name, dimensions, tensor_type, relative_offset, byte_length) in raw_tensors {
            let start = data_offset
                .checked_add(relative_offset)
                .ok_or_else(|| invalid(format!("tensor {name} start overflows")))?;
            let end = start
                .checked_add(byte_length)
                .ok_or_else(|| invalid(format!("tensor {name} end overflows")))?;
            if end > file_size {
                return Err(invalid(format!("tensor {name} exceeds file")));
            }
            tensors.push(GgufTensorInfo {
                name,
                dimensions,
                tensor_type,
                relative_offset,
                absolute_range: [start, end],
            });
        }
        validate_non_overlapping(&tensors)?;

        let extension = parse_extension(&metadata, &tensors)?;
        if architecture == "gemma4mtp" {
            validate_gemma4_mtp_extension(&metadata, &tensors, extension.as_ref())?;
        }
        validate_i8_carriers(&tensors, extension.as_ref())?;

        let tensor_index = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        let metadata_sha256 =
            sha256_range(path, &file, metadata_start, metadata_end - metadata_start)?;
        let tensor_catalog_sha256 =
            sha256_range(path, &file, catalog_start, catalog_end - catalog_start)?;

        Ok(Self {
            path: path.to_path_buf(),
            file: Arc::new(file),
            file_size,
            alignment,
            data_offset,
            metadata,
            tensors,
            tensor_index,
            metadata_sha256,
            tensor_catalog_sha256,
            extension,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    pub(crate) fn owned_file(&self) -> Arc<File> {
        Arc::clone(&self.file)
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    pub fn metadata_value(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn architecture(&self) -> &str {
        match self.metadata.get("general.architecture") {
            Some(GgufValue::String(value)) => value,
            _ => unreachable!("verified GGUF always has architecture"),
        }
    }

    /// Returns whether this file is an assistant-only artifact.  An
    /// assistant GGUF is intentionally parseable for the explicit MTP
    /// verifier, but must never be treated as a standalone target model by a
    /// generic model loader.
    pub fn is_assistant_only(&self) -> bool {
        self.architecture() == "gemma4mtp"
            && matches!(
                self.metadata_value(GEMMA4_MTP_ROLE_KEY),
                Some(GgufValue::String(role)) if role == "assistant"
            )
    }

    /// Open a GGUF for a standalone target-model runtime.  MTP assistant
    /// artifacts are intentionally excluded; callers that explicitly enable
    /// MTP must validate the target/assistant pair through the MTP verifier.
    pub fn open_target(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let verified = Self::open(path)?;
        if verified.architecture() == "gemma4mtp" {
            return Err(invalid(
                "Gemma 4 MTP assistant GGUF requires its reviewed target pair",
            ));
        }
        Ok(verified)
    }

    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index
            .get(name)
            .map(|index| &self.tensors[*index])
    }

    pub fn metadata_sha256(&self) -> &str {
        &self.metadata_sha256
    }

    pub fn tensor_catalog_sha256(&self) -> &str {
        &self.tensor_catalog_sha256
    }

    pub fn extension(&self) -> Option<&GgufExtensionV1> {
        self.extension.as_ref()
    }

    pub fn frontend_asset(&self, key: &str) -> Option<&[u8]> {
        self.extension
            .as_ref()?
            .frontend_assets
            .get(key)
            .map(Vec::as_slice)
    }

    pub fn read_tensor_range(
        &self,
        tensor_name: &str,
        relative_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, GgufError> {
        let tensor = self
            .tensor(tensor_name)
            .ok_or_else(|| invalid(format!("unknown tensor {tensor_name}")))?;
        let length_u64 = u64::try_from(length).map_err(|_| invalid("read length overflows u64"))?;
        let end = relative_offset
            .checked_add(length_u64)
            .ok_or_else(|| invalid("tensor read range overflows"))?;
        if end > tensor.byte_length() {
            return Err(invalid(format!("tensor {tensor_name} read exceeds range")));
        }
        let absolute = tensor.absolute_range[0]
            .checked_add(relative_offset)
            .ok_or_else(|| invalid("tensor read offset overflows"))?;
        let mut bytes = vec![0; length];
        read_exact_at(&self.path, &self.file, absolute, &mut bytes)?;
        Ok(bytes)
    }
}

fn parse_extension(
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[GgufTensorInfo],
) -> Result<Option<GgufExtensionV1>, GgufError> {
    let extension_keys: Vec<&str> = metadata
        .keys()
        .filter(|key| key.starts_with("sllm."))
        .map(String::as_str)
        .collect();
    if extension_keys.is_empty() {
        return Ok(None);
    }
    for key in &extension_keys {
        let is_frontend = frontend_asset_name(key).is_some()
            || key
                .strip_suffix(".sha256")
                .and_then(frontend_asset_name)
                .is_some();
        if !is_frontend
            && !matches!(
                *key,
                SLLM_EXTENSION_VERSION_KEY
                    | SLLM_TENSOR_RECIPE_KEY
                    | SLLM_TENSOR_RECIPE_SHA256_KEY
                    | "sllm.source.fp8_manifest_fingerprint"
                    | "sllm.source.fp8_artifact.sha256"
                    | "sllm.source.recipe.sha256"
                    | "sllm.source.artifact.fingerprint"
                    | "sllm.source.semantic.repository"
                    | "sllm.source.semantic.revision"
                    | "sllm.source.recipe.producer"
                    | "sllm.kv.fp8.scheme"
                    | "sllm.kv.fp8.implicit_decode_scale_bf16"
            )
        {
            return Err(invalid(format!("unknown sLLM extension key {key}")));
        }
    }
    let version = match metadata.get(SLLM_EXTENSION_VERSION_KEY) {
        Some(GgufValue::U32(value)) => *value,
        Some(_) => return Err(invalid("sLLM extension version has wrong type")),
        None => return Err(invalid("sLLM extension version is missing")),
    };
    if version != SLLM_GGUF_EXTENSION_VERSION {
        return Err(invalid(format!("unknown sLLM extension version {version}")));
    }
    let recipe_json = match metadata.get(SLLM_TENSOR_RECIPE_KEY) {
        Some(GgufValue::String(value)) => value,
        Some(_) => return Err(invalid("sLLM tensor recipe has wrong type")),
        None => return Err(invalid("sLLM tensor recipe is missing")),
    };
    let expected_digest = match metadata.get(SLLM_TENSOR_RECIPE_SHA256_KEY) {
        Some(GgufValue::String(value)) => value,
        Some(_) => return Err(invalid("sLLM tensor recipe digest has wrong type")),
        None => return Err(invalid("sLLM tensor recipe digest is missing")),
    };
    let recipe: GgufTensorRecipeV1 = serde_json::from_str(recipe_json)
        .map_err(|error| invalid(format!("sLLM tensor recipe JSON: {error}")))?;
    if recipe.schema_version != "sllm-gguf-tensor-recipe-v1" {
        return Err(invalid("unknown tensor recipe schema"));
    }
    if recipe.semantic_model_id.is_empty() {
        return Err(invalid("tensor recipe semantic model identity is empty"));
    }
    if recipe.source_lock_fingerprints.is_empty()
        || recipe
            .source_lock_fingerprints
            .iter()
            .any(|value| !valid_sha256(value))
    {
        return Err(invalid("tensor recipe source fingerprints are invalid"));
    }
    if recipe.canonical_json()? != *recipe_json {
        return Err(invalid("tensor recipe JSON is not canonical"));
    }
    let actual_digest = recipe.digest()?;
    if actual_digest != *expected_digest {
        return Err(invalid(format!(
            "tensor recipe digest differs: expected {expected_digest}, computed {actual_digest}"
        )));
    }
    validate_recipe_bindings(&recipe, tensors)?;
    let frontend_assets = parse_frontend_assets(metadata)?;
    Ok(Some(GgufExtensionV1 {
        recipe,
        recipe_sha256: actual_digest,
        frontend_assets,
    }))
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Gemma4MtpSourceRange {
    name: String,
    source_file: String,
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
    absolute_byte_range: [u64; 2],
}

/// Validate the structural safety boundary shared by the generic GGUF reader
/// and the explicit Gemma MTP verifier.  This does not admit an assistant as
/// a target and does not replace target-lock validation; it guarantees that a
/// malformed or identity-less assistant artifact cannot pass as a pair member.
fn validate_gemma4_mtp_extension(
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[GgufTensorInfo],
    extension: Option<&GgufExtensionV1>,
) -> Result<(), GgufError> {
    let extension = extension.ok_or_else(|| invalid("Gemma 4 MTP GGUF extension is missing"))?;
    let role = metadata_string(metadata, GEMMA4_MTP_ROLE_KEY)?;
    if role != "assistant" {
        return Err(invalid("Gemma 4 MTP GGUF role is not assistant"));
    }
    let target_fingerprint = metadata_string(metadata, GEMMA4_MTP_TARGET_FINGERPRINT_KEY)?;
    let assistant_fingerprint = metadata_string(metadata, GEMMA4_MTP_ASSISTANT_FINGERPRINT_KEY)?;
    if !valid_sha256(target_fingerprint) || !valid_sha256(assistant_fingerprint) {
        return Err(invalid("Gemma 4 MTP pair fingerprint is invalid"));
    }
    if extension.recipe.source_lock_fingerprints.as_slice()
        != [
            target_fingerprint.to_owned(),
            assistant_fingerprint.to_owned(),
        ]
    {
        return Err(invalid(
            "Gemma 4 MTP recipe source fingerprints differ from metadata",
        ));
    }
    let semantic_pair = metadata_string(metadata, GEMMA4_MTP_SEMANTIC_PAIR_KEY)?;
    if semantic_pair != extension.recipe.semantic_model_id || semantic_pair.is_empty() {
        return Err(invalid("Gemma 4 MTP semantic pair identity differs"));
    }
    let catalog = metadata_string(metadata, "gemma4mtp.tensor_catalog_sha256")?;
    if !valid_sha256(catalog) {
        return Err(invalid("Gemma 4 MTP catalog fingerprint is invalid"));
    }
    for key in [
        "gemma4mtp.source_model_sha256",
        "gemma4mtp.source_header_sha256",
    ] {
        if !valid_sha256(metadata_string(metadata, key)?) {
            return Err(invalid(format!(
                "Gemma 4 MTP source fingerprint is invalid: {key}"
            )));
        }
    }
    validate_u32_array(metadata, GEMMA4_MTP_LAYER_MAPPING_KEY, &[0, 1, 2, 3])?;
    validate_u32_array(metadata, GEMMA4_MTP_KV_MAPPING_KEY, &[46, 46, 46, 47])?;
    let layer_types = match metadata.get("gemma4mtp.layer_types") {
        Some(GgufValue::Array(GgufArray::String(values))) => values,
        _ => return Err(invalid("Gemma 4 MTP layer types metadata is invalid")),
    };
    if layer_types.as_slice()
        != [
            "sliding_attention".to_owned(),
            "sliding_attention".to_owned(),
            "sliding_attention".to_owned(),
            "full_attention".to_owned(),
        ]
    {
        return Err(invalid("Gemma 4 MTP layer types metadata differs"));
    }
    let tokenizer_identity = metadata_string(metadata, GEMMA4_MTP_TOKENIZER_IDENTITY_KEY)?;
    let tokenizer_value: Value = serde_json::from_str(tokenizer_identity)
        .map_err(|error| invalid(format!("Gemma 4 MTP tokenizer identity JSON: {error}")))?;
    if serde_json::to_string(&tokenizer_value)
        .map_err(|error| invalid(format!("serialize tokenizer identity: {error}")))?
        != tokenizer_identity
    {
        return Err(invalid("Gemma 4 MTP tokenizer identity is not canonical"));
    }
    let source_ranges = metadata_string(metadata, GEMMA4_MTP_SOURCE_RANGES_KEY)?;
    let source_ranges: Vec<Gemma4MtpSourceRange> = serde_json::from_str(source_ranges)
        .map_err(|error| invalid(format!("Gemma 4 MTP source ranges JSON: {error}")))?;
    if serde_json::to_string(&source_ranges)
        .map_err(|error| invalid(format!("serialize source ranges: {error}")))?
        != source_ranges_json(metadata, GEMMA4_MTP_SOURCE_RANGES_KEY)?
    {
        return Err(invalid("Gemma 4 MTP source ranges are not canonical"));
    }
    if source_ranges.len() != tensors.len() {
        return Err(invalid("Gemma 4 MTP source range count differs"));
    }
    let mut names = BTreeSet::new();
    for (source, tensor) in source_ranges.iter().zip(tensors) {
        if source.name != tensor.name
            || source.source_file.is_empty()
            || source.dtype != "BF16"
            || source.shape.is_empty()
            || source.shape.contains(&0)
            || !names.insert(source.name.as_str())
            || tensor.tensor_type != GgufTensorType::Bf16
        {
            return Err(invalid("Gemma 4 MTP source range tensor identity differs"));
        }
        let mut physical_shape = source.shape.clone();
        physical_shape.reverse();
        let logical_elements = source
            .shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| invalid("Gemma 4 MTP source range shape overflows"))?;
        let expected_bytes = logical_elements
            .checked_mul(2)
            .ok_or_else(|| invalid("Gemma 4 MTP source range byte size overflows"))?;
        if tensor.dimensions != physical_shape
            || source.data_offsets[0] >= source.data_offsets[1]
            || source.absolute_byte_range[0] >= source.absolute_byte_range[1]
            || source.data_offsets[1] - source.data_offsets[0]
                != source.absolute_byte_range[1] - source.absolute_byte_range[0]
            || tensor.byte_length() != expected_bytes
        {
            return Err(invalid("Gemma 4 MTP source range shape or size differs"));
        }
    }
    Ok(())
}

fn metadata_string<'a>(
    metadata: &'a BTreeMap<String, GgufValue>,
    key: &str,
) -> Result<&'a str, GgufError> {
    match metadata.get(key) {
        Some(GgufValue::String(value)) if !value.is_empty() => Ok(value),
        _ => Err(invalid(format!(
            "Gemma 4 MTP metadata {key} is missing or invalid"
        ))),
    }
}

fn source_ranges_json<'a>(
    metadata: &'a BTreeMap<String, GgufValue>,
    key: &str,
) -> Result<&'a str, GgufError> {
    metadata_string(metadata, key)
}

fn validate_u32_array(
    metadata: &BTreeMap<String, GgufValue>,
    key: &str,
    expected: &[u32],
) -> Result<(), GgufError> {
    match metadata.get(key) {
        Some(GgufValue::Array(GgufArray::U32(values))) if values == expected => Ok(()),
        _ => Err(invalid(format!("Gemma 4 MTP metadata {key} differs"))),
    }
}

fn frontend_asset_name(key: &str) -> Option<&'static str> {
    match key {
        SLLM_FRONTEND_CONFIG_KEY => Some("config.json"),
        SLLM_FRONTEND_TOKENIZER_KEY => Some("tokenizer.json"),
        SLLM_FRONTEND_TOKENIZER_CONFIG_KEY => Some("tokenizer_config.json"),
        SLLM_FRONTEND_PREPROCESSOR_CONFIG_KEY => Some("preprocessor_config.json"),
        "sllm.frontend.generation_config_json" => Some("generation_config.json"),
        "sllm.source.hf_quant_config_json" => Some("hf_quant_config.json"),
        _ => None,
    }
}

fn parse_frontend_assets(
    metadata: &BTreeMap<String, GgufValue>,
) -> Result<BTreeMap<String, Vec<u8>>, GgufError> {
    let mut assets = BTreeMap::new();
    for key in [
        SLLM_FRONTEND_CONFIG_KEY,
        SLLM_FRONTEND_TOKENIZER_KEY,
        SLLM_FRONTEND_TOKENIZER_CONFIG_KEY,
        SLLM_FRONTEND_PREPROCESSOR_CONFIG_KEY,
        "sllm.frontend.generation_config_json",
        "sllm.source.hf_quant_config_json",
    ] {
        let digest_key = format!("{key}.sha256");
        match (metadata.get(key), metadata.get(&digest_key)) {
            (None, None) => continue,
            (Some(GgufValue::String(contents)), Some(GgufValue::String(expected))) => {
                let actual = sha256_text(contents.as_bytes());
                if actual != *expected {
                    return Err(invalid(format!("frontend asset digest differs for {key}")));
                }
                let name = frontend_asset_name(key).expect("known frontend key");
                assets.insert(name.to_owned(), contents.as_bytes().to_vec());
            }
            (Some(_), Some(_)) => {
                return Err(invalid(format!("frontend asset {key} has wrong type")));
            }
            _ => return Err(invalid(format!("frontend asset {key} is missing its pair"))),
        }
    }
    Ok(assets)
}

fn validate_recipe_bindings(
    recipe: &GgufTensorRecipeV1,
    tensors: &[GgufTensorInfo],
) -> Result<(), GgufError> {
    let tensor_map: BTreeMap<&str, &GgufTensorInfo> = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let mut logical_names = BTreeSet::new();
    let mut value_names = BTreeSet::new();
    for binding in &recipe.bindings {
        if binding.logical_tensor.is_empty() || binding.role.is_empty() {
            return Err(invalid(
                "tensor recipe binding contains an empty identifier",
            ));
        }
        if !logical_names.insert(binding.logical_tensor.as_str()) {
            return Err(invalid(format!(
                "duplicate logical tensor binding {}",
                binding.logical_tensor
            )));
        }
        if !value_names.insert(binding.value_tensor.as_str()) {
            return Err(invalid(format!(
                "duplicate value tensor binding {}",
                binding.value_tensor
            )));
        }
        if binding.logical_shape.is_empty() || binding.logical_shape.contains(&0) {
            return Err(invalid(format!(
                "binding {} has invalid logical shape",
                binding.logical_tensor
            )));
        }
        let value = tensor_map
            .get(binding.value_tensor.as_str())
            .ok_or_else(|| invalid(format!("missing value tensor {}", binding.value_tensor)))?;
        match binding.encoding {
            GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale
            | GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale => {
                if value.tensor_type != GgufTensorType::I8Carrier {
                    return Err(invalid(format!(
                        "FP8 value tensor {} is not the I8 carrier",
                        value.name
                    )));
                }
                let channel_scales = binding
                    .scales
                    .iter()
                    .filter(|scale| scale.role == GgufScaleRole::Channel)
                    .count();
                if channel_scales != 1 || binding.scales.len() != 1 {
                    return Err(invalid(format!(
                        "FP8 binding {} requires exactly one channel scale",
                        binding.logical_tensor
                    )));
                }
            }
            GgufRecipeEncoding::Nvfp4E2m1Block16E4m3fnF32Outer => {
                if value.tensor_type != GgufTensorType::Nvfp4 {
                    return Err(invalid(format!(
                        "NVFP4 value tensor {} has wrong type",
                        value.name
                    )));
                }
                let outer = binding
                    .scales
                    .iter()
                    .filter(|scale| scale.role == GgufScaleRole::Outer)
                    .count();
                let input = binding
                    .scales
                    .iter()
                    .filter(|scale| scale.role == GgufScaleRole::Input)
                    .count();
                if outer != 1 || input > 1 || binding.scales.len() != outer + input {
                    return Err(invalid(format!(
                        "NVFP4 binding {} requires one outer and at most one input scale",
                        binding.logical_tensor
                    )));
                }
            }
            GgufRecipeEncoding::Mxfp4E2m1Block32E8m0 => {
                if value.tensor_type != GgufTensorType::Mxfp4 {
                    return Err(invalid(format!(
                        "MXFP4 value tensor {} has wrong type",
                        value.name
                    )));
                }
                if !binding.scales.is_empty() {
                    return Err(invalid(format!(
                        "MXFP4 binding {} must embed its E8M0 block scales",
                        binding.logical_tensor
                    )));
                }
            }
            GgufRecipeEncoding::Mxfp8E4m3Block32E8m0 | GgufRecipeEncoding::Mxfp6E3m2Block32E8m0 => {
                if value.tensor_type != GgufTensorType::I8Carrier {
                    return Err(invalid(format!(
                        "MX value tensor {} is not the I8 carrier",
                        value.name
                    )));
                }
                if binding.logical_shape.len() != 2 || binding.logical_shape[1] % 32 != 0 {
                    return Err(invalid(format!(
                        "MX binding {} requires rank-two [N,K] with K divisible by 32",
                        binding.logical_tensor
                    )));
                }
                let blocks = binding.logical_shape[0]
                    .checked_mul(binding.logical_shape[1] / 32)
                    .ok_or_else(|| invalid("MX block count overflowed"))?;
                let expected_values = binding.logical_shape[0]
                    .checked_mul(binding.logical_shape[1])
                    .and_then(|elements| match binding.encoding {
                        GgufRecipeEncoding::Mxfp8E4m3Block32E8m0 => Some(elements),
                        GgufRecipeEncoding::Mxfp6E3m2Block32E8m0 => {
                            elements.checked_mul(3).map(|bytes| bytes / 4)
                        }
                        _ => unreachable!(),
                    })
                    .ok_or_else(|| invalid("MX value byte count overflowed"))?;
                if value.byte_length() != expected_values {
                    return Err(invalid(format!(
                        "MX value tensor {} has {} bytes, expected {expected_values}",
                        value.name,
                        value.byte_length()
                    )));
                }
                let block_scales: Vec<_> = binding
                    .scales
                    .iter()
                    .filter(|scale| scale.role == GgufScaleRole::Block)
                    .collect();
                if block_scales.len() != 1 || binding.scales.len() != 1 {
                    return Err(invalid(format!(
                        "MX binding {} requires exactly one E8M0 block-scale plane",
                        binding.logical_tensor
                    )));
                }
                let scale = tensor_map
                    .get(block_scales[0].tensor.as_str())
                    .ok_or_else(|| invalid("MX block-scale tensor is absent"))?;
                let scale_binding_count = recipe
                    .bindings
                    .iter()
                    .flat_map(|candidate| candidate.scales.iter())
                    .filter(|candidate| candidate.tensor == scale.name)
                    .count();
                let aliases_value_plane = recipe
                    .bindings
                    .iter()
                    .any(|candidate| candidate.value_tensor == scale.name);
                if scale_binding_count != 1 || aliases_value_plane {
                    return Err(invalid(format!(
                        "MX block-scale tensor {} must be one exclusive plane",
                        scale.name
                    )));
                }
                if scale.tensor_type != GgufTensorType::I8Carrier || scale.byte_length() != blocks {
                    return Err(invalid(format!(
                        "MX block-scale tensor {} must contain {blocks} E8M0 bytes",
                        scale.name
                    )));
                }
            }
        }
        let mut scale_roles = BTreeSet::new();
        let mut scale_names = BTreeSet::new();
        for scale in &binding.scales {
            if !scale_roles.insert(scale.role) {
                return Err(invalid(format!(
                    "binding {} has duplicate scale role",
                    binding.logical_tensor
                )));
            }
            if !scale_names.insert(scale.tensor.as_str()) {
                return Err(invalid(format!(
                    "binding {} has duplicate scale tensor",
                    binding.logical_tensor
                )));
            }
            let scale_tensor = tensor_map
                .get(scale.tensor.as_str())
                .ok_or_else(|| invalid(format!("missing scale tensor {}", scale.tensor)))?;
            let valid_type = match scale.role {
                GgufScaleRole::Channel => match binding.encoding {
                    GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale => {
                        scale_tensor.tensor_type == GgufTensorType::Bf16
                    }
                    GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale => {
                        scale_tensor.tensor_type == GgufTensorType::F32
                    }
                    _ => false,
                },
                GgufScaleRole::Outer | GgufScaleRole::Input => {
                    matches!(
                        scale_tensor.tensor_type,
                        GgufTensorType::F32 | GgufTensorType::Bf16
                    )
                }
                GgufScaleRole::Block => match binding.encoding {
                    GgufRecipeEncoding::Mxfp8E4m3Block32E8m0
                    | GgufRecipeEncoding::Mxfp6E3m2Block32E8m0 => {
                        scale_tensor.tensor_type == GgufTensorType::I8Carrier
                    }
                    _ => matches!(
                        scale_tensor.tensor_type,
                        GgufTensorType::I8Carrier | GgufTensorType::Bf16
                    ),
                },
            };
            if !valid_type {
                return Err(invalid(format!(
                    "scale tensor {} has wrong type for {:?}",
                    scale.tensor, scale.role
                )));
            }
        }
    }
    let mut shape_names = BTreeSet::new();
    for binding in &recipe.logical_shapes {
        if binding.tensor.is_empty()
            || binding.logical_shape.len() <= MAX_DIMS as usize
            || binding.logical_shape.contains(&0)
            || !shape_names.insert(binding.tensor.as_str())
            || logical_names.contains(binding.tensor.as_str())
        {
            return Err(invalid("logical-shape override is invalid or ambiguous"));
        }
        let tensor = tensor_map
            .get(binding.tensor.as_str())
            .ok_or_else(|| invalid("logical-shape override tensor is absent"))?;
        let logical_elements = binding
            .logical_shape
            .iter()
            .try_fold(1_u64, |product, value| {
                product
                    .checked_mul(*value)
                    .ok_or_else(|| invalid("logical-shape override overflows"))
            })?;
        let physical_elements = tensor.dimensions.iter().try_fold(1_u64, |product, value| {
            product
                .checked_mul(*value)
                .ok_or_else(|| invalid("physical tensor shape overflows"))
        })?;
        if logical_elements != physical_elements {
            return Err(invalid(
                "logical-shape override changes tensor element count",
            ));
        }
    }
    let mut kv_layers = BTreeSet::new();
    for binding in &recipe.static_fp8_kv {
        if !kv_layers.insert(binding.layer) {
            return Err(invalid(format!(
                "duplicate static FP8 KV layer {}",
                binding.layer
            )));
        }
        for scale in [
            binding.key_decode_scale_bf16,
            binding.value_decode_scale_bf16,
        ] {
            let value = f32::from_bits(u32::from(scale) << 16);
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid("static FP8 KV scale is non-positive or non-finite"));
            }
        }
    }
    let bound_names: BTreeSet<&str> = recipe
        .bindings
        .iter()
        .map(|binding| binding.value_tensor.as_str())
        .collect();
    let mut known = BTreeSet::new();
    for name in &recipe.known_unconsumed_tensors {
        if name.is_empty()
            || !known.insert(name.as_str())
            || !tensor_map.contains_key(name.as_str())
            || bound_names.contains(name.as_str())
        {
            return Err(invalid(format!(
                "known-unconsumed tensor binding is invalid: {name}"
            )));
        }
    }
    Ok(())
}

fn validate_i8_carriers(
    tensors: &[GgufTensorInfo],
    extension: Option<&GgufExtensionV1>,
) -> Result<(), GgufError> {
    let i8_names: BTreeSet<&str> = tensors
        .iter()
        .filter(|tensor| tensor.tensor_type == GgufTensorType::I8Carrier)
        .map(|tensor| tensor.name.as_str())
        .collect();
    if i8_names.is_empty() {
        return Ok(());
    }
    let extension = extension.ok_or_else(|| invalid("I8 carrier requires the sLLM extension"))?;
    let bound: BTreeSet<&str> = extension
        .recipe
        .bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.encoding,
                GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale
                    | GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale
                    | GgufRecipeEncoding::Mxfp8E4m3Block32E8m0
                    | GgufRecipeEncoding::Mxfp6E3m2Block32E8m0
            )
        })
        .flat_map(|binding| {
            std::iter::once(binding.value_tensor.as_str()).chain(
                binding
                    .scales
                    .iter()
                    .filter(|scale| scale.role == GgufScaleRole::Block)
                    .map(|scale| scale.tensor.as_str()),
            )
        })
        .collect();
    if i8_names != bound {
        return Err(invalid(
            "I8 carrier tensors and FP8/MX recipe bindings differ",
        ));
    }
    Ok(())
}

fn tensor_byte_length(
    name: &str,
    dimensions: &[u64],
    tensor_type: GgufTensorType,
) -> Result<u64, GgufError> {
    let block_size = tensor_type.block_size();
    if dimensions[0] % block_size != 0 {
        return Err(invalid(format!(
            "tensor {name} first dimension is not divisible by block size {block_size}"
        )));
    }
    let element_count = dimensions.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| invalid(format!("tensor {name} element count overflows")))
    })?;
    element_count
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(tensor_type.type_size()))
        .ok_or_else(|| invalid(format!("tensor {name} byte length overflows")))
}

fn validate_non_overlapping(tensors: &[GgufTensorInfo]) -> Result<(), GgufError> {
    let mut ranges: Vec<(&str, [u64; 2])> = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.absolute_range))
        .collect();
    ranges.sort_by_key(|(_, range)| (range[0], range[1]));
    for pair in ranges.windows(2) {
        if pair[1].1[0] < pair[0].1[1] {
            return Err(invalid(format!(
                "tensor ranges overlap: {} and {}",
                pair[0].0, pair[1].0
            )));
        }
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| invalid("alignment overflows"))
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn sha256_text(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn sha256_range(path: &Path, file: &File, start: u64, length: u64) -> Result<String, GgufError> {
    let mut hasher = Sha256::new();
    let mut offset = 0_u64;
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    while offset < length {
        let remaining = length - offset;
        let chunk = usize::try_from(remaining.min(HASH_CHUNK_BYTES as u64))
            .map_err(|_| invalid("hash chunk length overflows usize"))?;
        read_exact_at(path, file, start + offset, &mut buffer[..chunk])?;
        hasher.update(&buffer[..chunk]);
        offset += chunk as u64;
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn open_regular_file(path: &Path) -> Result<File, GgufError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
    let file = options.open(path).map_err(|error| io_error(path, error))?;
    let metadata = file.metadata().map_err(|error| io_error(path, error))?;
    if !metadata.is_file() {
        return Err(invalid("GGUF path is not a regular file"));
    }
    #[cfg(unix)]
    if metadata.nlink() == 0 {
        return Err(invalid("GGUF file is unlinked"));
    }
    Ok(file)
}

#[cfg(unix)]
fn read_exact_at(
    path: &Path,
    file: &File,
    mut offset: u64,
    mut output: &mut [u8],
) -> Result<(), GgufError> {
    while !output.is_empty() {
        let read = file
            .read_at(output, offset)
            .map_err(|error| io_error(path, error))?;
        if read == 0 {
            return Err(invalid("truncated file"));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| invalid("read offset overflows"))?;
        output = &mut output[read..];
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(
    path: &Path,
    file: &File,
    offset: u64,
    output: &mut [u8],
) -> Result<(), GgufError> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = file.try_clone().map_err(|error| io_error(path, error))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| io_error(path, error))?;
    file.read_exact(output)
        .map_err(|error| io_error(path, error))
}

fn to_usize(value: u64, label: &str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| invalid(format!("{label} does not fit usize")))
}

struct BoundedReader<'a> {
    path: &'a Path,
    file: &'a File,
    file_size: u64,
    position: u64,
}

impl<'a> BoundedReader<'a> {
    fn new(path: &'a Path, file: &'a File, file_size: u64) -> Self {
        Self {
            path,
            file,
            file_size,
            position: 0,
        }
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn read_exact(&mut self, length: usize) -> Result<Vec<u8>, GgufError> {
        let end = self
            .position
            .checked_add(length as u64)
            .ok_or_else(|| invalid("header position overflows"))?;
        if end > self.file_size || end > MAX_HEADER_BYTES {
            return Err(invalid("truncated or oversized header"));
        }
        let mut bytes = vec![0; length];
        read_exact_at(self.path, self.file, self.position, &mut bytes)?;
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let bytes: [u8; 2] = self.read_exact(2)?.try_into().expect("length is fixed");
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let bytes: [u8; 4] = self.read_exact(4)?.try_into().expect("length is fixed");
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let bytes: [u8; 8] = self.read_exact(8)?.try_into().expect("length is fixed");
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let value = f32::from_bits(self.read_u32()?);
        if !value.is_finite() {
            return Err(invalid("non-finite f32 metadata value"));
        }
        Ok(value)
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let value = f64::from_bits(self.read_u64()?);
        if !value.is_finite() {
            return Err(invalid("non-finite f64 metadata value"));
        }
        Ok(value)
    }

    fn read_bool(&mut self) -> Result<bool, GgufError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(invalid(format!("invalid boolean value {value}"))),
        }
    }

    fn read_string(&mut self, bound: u64, label: &str) -> Result<String, GgufError> {
        let length = self.read_u64()?;
        if length > bound {
            return Err(invalid(format!("{label} exceeds byte bound")));
        }
        let bytes = self.read_exact(to_usize(length, label)?)?;
        String::from_utf8(bytes).map_err(|_| invalid(format!("{label} is not UTF-8")))
    }

    fn read_value(&mut self, value_type: u32) -> Result<GgufValue, GgufError> {
        match value_type {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_bool()?)),
            8 => Ok(GgufValue::String(
                self.read_string(MAX_STRING_BYTES, "metadata string")?,
            )),
            9 => self.read_array().map(GgufValue::Array),
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            _ => Err(invalid(format!("unknown metadata type {value_type}"))),
        }
    }

    fn read_array(&mut self) -> Result<GgufArray, GgufError> {
        let element_type = self.read_u32()?;
        if element_type == 9 || element_type > 12 {
            return Err(invalid(format!(
                "invalid array element type {element_type}"
            )));
        }
        let length = self.read_u64()?;
        if length > MAX_ARRAY_ITEMS {
            return Err(invalid("metadata array exceeds item bound"));
        }
        let length = to_usize(length, "metadata array length")?;
        macro_rules! read_values {
            ($method:ident) => {{
                let mut values = Vec::with_capacity(length);
                for _ in 0..length {
                    values.push(self.$method()?);
                }
                values
            }};
        }
        Ok(match element_type {
            0 => GgufArray::U8(read_values!(read_u8)),
            1 => GgufArray::I8(read_values!(read_i8)),
            2 => GgufArray::U16(read_values!(read_u16)),
            3 => GgufArray::I16(read_values!(read_i16)),
            4 => GgufArray::U32(read_values!(read_u32)),
            5 => GgufArray::I32(read_values!(read_i32)),
            6 => GgufArray::F32(read_values!(read_f32)),
            7 => GgufArray::Bool(read_values!(read_bool)),
            8 => {
                let mut values = Vec::with_capacity(length);
                for _ in 0..length {
                    values.push(self.read_string(MAX_STRING_BYTES, "array string")?);
                }
                GgufArray::String(values)
            }
            10 => GgufArray::U64(read_values!(read_u64)),
            11 => GgufArray::I64(read_values!(read_i64)),
            12 => GgufArray::F64(read_values!(read_f64)),
            _ => unreachable!("array type was validated"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mx_recipe_fixture(
        encoding: GgufRecipeEncoding,
        logical_shape: Vec<u64>,
        value_bytes: u64,
        scale_bytes: u64,
    ) -> (GgufTensorRecipeV1, Vec<GgufTensorInfo>) {
        let recipe = GgufTensorRecipeV1 {
            schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
            semantic_model_id: "mx-fixture".to_owned(),
            source_lock_fingerprints: vec![format!("sha256:{}", "1".repeat(64))],
            bindings: vec![GgufTensorBinding {
                logical_tensor: "linear.weight".to_owned(),
                value_tensor: "linear.weight.values".to_owned(),
                encoding,
                role: "linear".to_owned(),
                logical_shape,
                scope: GgufTensorScope::Consumed,
                scales: vec![GgufScaleBinding {
                    tensor: "linear.weight.e8m0".to_owned(),
                    role: GgufScaleRole::Block,
                }],
            }],
            logical_shapes: Vec::new(),
            static_fp8_kv: Vec::new(),
            known_unconsumed_tensors: Vec::new(),
        };
        let tensors = vec![
            GgufTensorInfo {
                name: "linear.weight.values".to_owned(),
                dimensions: vec![value_bytes],
                tensor_type: GgufTensorType::I8Carrier,
                relative_offset: 0,
                absolute_range: [0, value_bytes],
            },
            GgufTensorInfo {
                name: "linear.weight.e8m0".to_owned(),
                dimensions: vec![scale_bytes],
                tensor_type: GgufTensorType::I8Carrier,
                relative_offset: value_bytes,
                absolute_range: [value_bytes, value_bytes + scale_bytes],
            },
        ];
        (recipe, tensors)
    }

    #[test]
    fn mx_weight_activation_recipes_validate_exact_resident_planes() {
        for (encoding, value_bytes, serialized) in [
            (
                GgufRecipeEncoding::Mxfp8E4m3Block32E8m0,
                3 * 64,
                "mxfp8-e4m3-block32-e8m0",
            ),
            (
                GgufRecipeEncoding::Mxfp6E3m2Block32E8m0,
                3 * 64 * 3 / 4,
                "mxfp6-e3m2-block32-e8m0",
            ),
        ] {
            let (recipe, tensors) = mx_recipe_fixture(encoding, vec![3, 64], value_bytes, 6);
            validate_recipe_bindings(&recipe, &tensors).expect("valid OCP MX recipe");
            assert_eq!(
                serde_json::to_value(encoding).unwrap(),
                serde_json::Value::String(serialized.to_owned())
            );
        }
    }

    #[test]
    fn mx_weight_activation_recipes_reject_non_block_k_and_plane_lengths() {
        let (nonaligned, tensors) =
            mx_recipe_fixture(GgufRecipeEncoding::Mxfp8E4m3Block32E8m0, vec![3, 33], 99, 3);
        assert!(
            validate_recipe_bindings(&nonaligned, &tensors)
                .unwrap_err()
                .to_string()
                .contains("K divisible by 32")
        );

        let (bad_values, tensors) = mx_recipe_fixture(
            GgufRecipeEncoding::Mxfp6E3m2Block32E8m0,
            vec![3, 64],
            143,
            6,
        );
        assert!(
            validate_recipe_bindings(&bad_values, &tensors)
                .unwrap_err()
                .to_string()
                .contains("expected 144")
        );

        let (bad_scales, tensors) = mx_recipe_fixture(
            GgufRecipeEncoding::Mxfp8E4m3Block32E8m0,
            vec![3, 64],
            192,
            5,
        );
        assert!(
            validate_recipe_bindings(&bad_scales, &tensors)
                .unwrap_err()
                .to_string()
                .contains("must contain 6 E8M0 bytes")
        );

        let (mut aliased, tensors) =
            mx_recipe_fixture(GgufRecipeEncoding::Mxfp8E4m3Block32E8m0, vec![1, 32], 32, 1);
        aliased.bindings[0].scales[0].tensor = "linear.weight.values".to_owned();
        assert!(
            validate_recipe_bindings(&aliased, &tensors)
                .unwrap_err()
                .to_string()
                .contains("exclusive plane")
        );
    }

    #[test]
    fn gemma4_mtp_extension_is_fail_closed_without_pair_metadata() {
        let metadata = BTreeMap::from([
            (
                "general.architecture".to_owned(),
                GgufValue::String("gemma4mtp".to_owned()),
            ),
            (
                GEMMA4_MTP_ROLE_KEY.to_owned(),
                GgufValue::String("assistant".to_owned()),
            ),
        ]);
        let error = validate_gemma4_mtp_extension(&metadata, &[], None)
            .expect_err("assistant metadata without a recipe must fail closed");
        assert!(error.to_string().contains("extension is missing"));
    }

    #[test]
    fn gemma4_mtp_source_range_digest_uses_the_explicit_pair_namespace() {
        assert_eq!(GEMMA4_MTP_SOURCE_RANGES_KEY, "gemma4mtp.source_ranges");
        assert_eq!(GEMMA4_MTP_KV_MAPPING_KEY, "gemma4mtp.kv_mapping");
    }

    #[test]
    fn gemma4_mtp_extension_accepts_a_structurally_bound_bf16_source_range() {
        let target = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
        let assistant = "sha256:2222222222222222222222222222222222222222222222222222222222222222";
        let semantic_pair = "gemma4mtp-pair:sha256:1111111111111111111111111111111111111111111111111111111111111111:sha256:2222222222222222222222222222222222222222222222222222222222222222";
        let tensor = GgufTensorInfo {
            name: "model.norm.weight".to_owned(),
            dimensions: vec![2, 2],
            tensor_type: GgufTensorType::Bf16,
            relative_offset: 0,
            absolute_range: [0, 8],
        };
        let ranges = serde_json::to_string(&vec![Gemma4MtpSourceRange {
            name: tensor.name.clone(),
            source_file: "model.safetensors".to_owned(),
            dtype: "BF16".to_owned(),
            shape: vec![2, 2],
            data_offsets: [0, 8],
            absolute_byte_range: [5368, 5376],
        }])
        .expect("source ranges JSON");
        let metadata = BTreeMap::from([
            (
                GEMMA4_MTP_ROLE_KEY.to_owned(),
                GgufValue::String("assistant".to_owned()),
            ),
            (
                GEMMA4_MTP_TARGET_FINGERPRINT_KEY.to_owned(),
                GgufValue::String(target.to_owned()),
            ),
            (
                GEMMA4_MTP_ASSISTANT_FINGERPRINT_KEY.to_owned(),
                GgufValue::String(assistant.to_owned()),
            ),
            (
                GEMMA4_MTP_SEMANTIC_PAIR_KEY.to_owned(),
                GgufValue::String(semantic_pair.to_owned()),
            ),
            (
                "gemma4mtp.tensor_catalog_sha256".to_owned(),
                GgufValue::String(format!("sha256:{}", "3".repeat(64))),
            ),
            (
                "gemma4mtp.source_model_sha256".to_owned(),
                GgufValue::String(format!("sha256:{}", "4".repeat(64))),
            ),
            (
                "gemma4mtp.source_header_sha256".to_owned(),
                GgufValue::String(format!("sha256:{}", "5".repeat(64))),
            ),
            (
                GEMMA4_MTP_LAYER_MAPPING_KEY.to_owned(),
                GgufValue::Array(GgufArray::U32(vec![0, 1, 2, 3])),
            ),
            (
                GEMMA4_MTP_KV_MAPPING_KEY.to_owned(),
                GgufValue::Array(GgufArray::U32(vec![46, 46, 46, 47])),
            ),
            (
                "gemma4mtp.layer_types".to_owned(),
                GgufValue::Array(GgufArray::String(vec![
                    "sliding_attention".to_owned(),
                    "sliding_attention".to_owned(),
                    "sliding_attention".to_owned(),
                    "full_attention".to_owned(),
                ])),
            ),
            (
                GEMMA4_MTP_TOKENIZER_IDENTITY_KEY.to_owned(),
                GgufValue::String("{}".to_owned()),
            ),
            (
                GEMMA4_MTP_SOURCE_RANGES_KEY.to_owned(),
                GgufValue::String(ranges),
            ),
        ]);
        let extension = GgufExtensionV1 {
            recipe: GgufTensorRecipeV1 {
                schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
                semantic_model_id: semantic_pair.to_owned(),
                source_lock_fingerprints: vec![target.to_owned(), assistant.to_owned()],
                bindings: Vec::new(),
                logical_shapes: Vec::new(),
                static_fp8_kv: Vec::new(),
                known_unconsumed_tensors: Vec::new(),
            },
            recipe_sha256: String::new(),
            frontend_assets: BTreeMap::new(),
        };
        validate_gemma4_mtp_extension(&metadata, &[tensor], Some(&extension))
            .expect("structurally bound MTP extension");
    }
}
