//! Bounded, opt-in debug artifacts.
//!
//! The writer only accepts an explicit allow-list of metadata fields.  Raw
//! prompts, responses, credentials, payloads, pointers, and device addresses
//! are rejected before an artifact is written.  A dump is published by rename
//! only after its JSON has been serialized, so an interrupted write cannot be
//! mistaken for a valid report.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEBUG_DUMP_SCHEMA_VERSION: &str = "sllm-phase46-debug-dump-v1";
pub const DEBUG_DUMP_STRUCT_SIZE_V1: u32 = 8;

#[derive(Clone, Debug)]
pub struct DebugDumpConfig {
    pub enabled: bool,
    pub output_dir: PathBuf,
    pub file_name: String,
    pub max_bytes: usize,
    pub max_tensors: usize,
    pub max_tokens: usize,
    pub max_top_k: usize,
    pub max_layers: usize,
    pub max_positions: usize,
}

impl Default for DebugDumpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_dir: PathBuf::from("."),
            file_name: "sllm-debug.json".to_owned(),
            max_bytes: 16 * 1024 * 1024,
            max_tensors: 128,
            max_tokens: 4096,
            max_top_k: 64,
            max_layers: 256,
            max_positions: 4096,
        }
    }
}

impl DebugDumpConfig {
    fn validate(&self) -> Result<(), DebugDumpError> {
        let hard = Self::default();
        if self.file_name.is_empty()
            || self.file_name == "."
            || self.file_name == ".."
            || Path::new(&self.file_name).components().count() != 1
            || self.max_bytes == 0
            || self.max_bytes > hard.max_bytes
            || self.max_tensors == 0
            || self.max_tensors > hard.max_tensors
            || self.max_tokens == 0
            || self.max_tokens > hard.max_tokens
            || self.max_top_k == 0
            || self.max_top_k > hard.max_top_k
            || self.max_layers == 0
            || self.max_layers > hard.max_layers
            || self.max_positions == 0
            || self.max_positions > hard.max_positions
        {
            return Err(DebugDumpError::OverLimit(
                "invalid debug dump limits".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum DebugDumpError {
    Disabled,
    Invalid(String),
    OverLimit(String),
    Forbidden(String),
    Io(io::Error),
    Serialization(String),
}

impl fmt::Display for DebugDumpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => f.write_str("debug dump is disabled"),
            Self::Invalid(message) => write!(f, "invalid debug dump: {message}"),
            Self::OverLimit(message) => write!(f, "debug dump exceeds bound: {message}"),
            Self::Forbidden(message) => write!(f, "forbidden debug dump field: {message}"),
            Self::Io(error) => write!(f, "debug dump I/O failed: {error}"),
            Self::Serialization(error) => write!(f, "debug dump serialization failed: {error}"),
        }
    }
}

impl std::error::Error for DebugDumpError {}

impl From<io::Error> for DebugDumpError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugDumpArtifact {
    pub path: Option<PathBuf>,
    pub bytes: usize,
    pub digest: Option<String>,
    pub tensor_count: usize,
    pub token_count: usize,
    pub logit_count: usize,
}

/// A writer that remains disabled and side-effect free unless `enabled` is
/// explicitly set.  Keep the writer alive until `finish`; dropping it removes
/// any partial file.
pub struct DebugDumpWriter {
    config: DebugDumpConfig,
    final_path: Option<PathBuf>,
    partial_path: Option<PathBuf>,
    metadata: Map<String, Value>,
    manifest: Option<Value>,
    tensors: Vec<Value>,
    tokens: Vec<i64>,
    logits: Vec<Value>,
    committed: bool,
}

impl DebugDumpWriter {
    pub fn new(config: DebugDumpConfig) -> Result<Self, DebugDumpError> {
        config.validate()?;
        if !config.enabled {
            return Ok(Self {
                config,
                final_path: None,
                partial_path: None,
                metadata: Map::new(),
                manifest: None,
                tensors: Vec::new(),
                tokens: Vec::new(),
                logits: Vec::new(),
                committed: false,
            });
        }
        fs::create_dir_all(&config.output_dir)?;
        let final_path = config.output_dir.join(&config.file_name);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let partial_path = config.output_dir.join(format!(
            ".{}.partial-{}-{}",
            config.file_name,
            std::process::id(),
            nonce
        ));
        // Reserve the name so two writers cannot silently overwrite each
        // other's evidence.
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial_path)?
            .sync_all()?;
        Ok(Self {
            config,
            final_path: Some(final_path),
            partial_path: Some(partial_path),
            metadata: Map::new(),
            manifest: None,
            tensors: Vec::new(),
            tokens: Vec::new(),
            logits: Vec::new(),
            committed: false,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn set_metadata(&mut self, metadata: Value) -> Result<(), DebugDumpError> {
        if !self.config.enabled {
            return Ok(());
        }
        let object = metadata
            .as_object()
            .ok_or_else(|| DebugDumpError::Invalid("metadata must be an object".to_owned()))?;
        for (key, value) in object {
            validate_metadata_key(key)?;
            validate_metadata_value(key, value)?;
            if self.metadata.contains_key(key) {
                return Err(DebugDumpError::Invalid(format!(
                    "duplicate metadata field {key}"
                )));
            }
            self.metadata.insert(key.clone(), value.clone());
        }
        self.check_size()?;
        Ok(())
    }

    /// Bind a validated common run manifest.  The manifest is kept as an
    /// identity envelope; evaluator-specific metrics remain outside it.
    pub fn set_manifest(
        &mut self,
        manifest: &crate::tool_manifest::ToolRunManifestV1,
    ) -> Result<(), DebugDumpError> {
        if !self.config.enabled {
            return Ok(());
        }
        manifest
            .validate()
            .map_err(|error| DebugDumpError::Invalid(format!("invalid tool manifest: {error}")))?;
        if self.manifest.is_some() {
            return Err(DebugDumpError::Invalid(
                "duplicate tool manifest".to_owned(),
            ));
        }
        self.manifest = Some(
            serde_json::to_value(manifest)
                .map_err(|error| DebugDumpError::Serialization(error.to_string()))?,
        );
        self.check_size()
    }

    /// Add token IDs.  Token IDs are not prompts and are included only when a
    /// caller explicitly enables this writer.
    pub fn add_tokens(&mut self, values: &[i64]) -> Result<(), DebugDumpError> {
        if !self.config.enabled {
            return Ok(());
        }
        if values.iter().any(|value| *value < 0) {
            return Err(DebugDumpError::Invalid("token ID is negative".to_owned()));
        }
        let total = self
            .tokens
            .len()
            .checked_add(values.len())
            .ok_or_else(|| DebugDumpError::OverLimit("token count overflow".to_owned()))?;
        if total > self.config.max_tokens {
            return Err(DebugDumpError::OverLimit("token count".to_owned()));
        }
        self.tokens.extend_from_slice(values);
        self.check_size()
    }

    /// Add an intermediate tensor or an explicitly selected packed KV plane.
    /// Values are bounded by the tensor and byte limits before publication.
    #[allow(clippy::too_many_arguments)]
    pub fn add_tensor(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        layout: &str,
        endianness: &str,
        values: &[f32],
        quantization: Option<&str>,
        scale_plane: Option<&str>,
    ) -> Result<(), DebugDumpError> {
        if !self.config.enabled {
            return Ok(());
        }
        validate_public_text(name, "tensor name")?;
        validate_public_text(dtype, "dtype")?;
        validate_public_text(layout, "layout")?;
        validate_public_text(endianness, "endianness")?;
        if let Some(quantization) = quantization {
            validate_public_text(quantization, "quantization")?;
        }
        if let Some(scale_plane) = scale_plane {
            validate_public_text(scale_plane, "scale_plane")?;
        }
        if dtype.eq_ignore_ascii_case("fp16")
            && name.to_ascii_lowercase().contains("kv")
            && (quantization.is_some() || scale_plane.is_some())
        {
            return Err(DebugDumpError::Forbidden(
                "packed KV tensor cannot be labeled as FP16".to_owned(),
            ));
        }
        if shape.is_empty() || shape.contains(&0) {
            return Err(DebugDumpError::Invalid(
                "tensor shape must be non-empty".to_owned(),
            ));
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension));
        if expected != Some(values.len()) {
            return Err(DebugDumpError::Invalid(
                "tensor value count does not match shape".to_owned(),
            ));
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DebugDumpError::Invalid(format!(
                    "tensor value {index} is non-finite"
                )));
            }
        }
        if self.tensors.len() >= self.config.max_tensors {
            return Err(DebugDumpError::OverLimit("tensor count".to_owned()));
        }
        let tensor = json!({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "layout": layout,
            "endianness": endianness,
            "quantization": quantization,
            "scale_plane": scale_plane,
            "values": values,
        });
        self.tensors.push(tensor);
        self.check_size()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_intermediate_tensor(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        layout: &str,
        endianness: &str,
        values: &[f32],
        quantization: Option<&str>,
        scale_plane: Option<&str>,
    ) -> Result<(), DebugDumpError> {
        self.add_tensor(
            name,
            dtype,
            shape,
            layout,
            endianness,
            values,
            quantization,
            scale_plane,
        )
    }

    /// Store only top-k logits at one layer/position, never an unrestricted
    /// vocabulary dump.
    pub fn add_logits(
        &mut self,
        layer: usize,
        position: usize,
        values: &[f32],
        top_k: usize,
    ) -> Result<(), DebugDumpError> {
        if !self.config.enabled {
            return Ok(());
        }
        if layer >= self.config.max_layers {
            return Err(DebugDumpError::OverLimit("layer".to_owned()));
        }
        if position >= self.config.max_positions {
            return Err(DebugDumpError::OverLimit("position".to_owned()));
        }
        if self.logits.len() >= self.config.max_positions {
            return Err(DebugDumpError::OverLimit("logit sample count".to_owned()));
        }
        if top_k == 0 || top_k > self.config.max_top_k {
            return Err(DebugDumpError::OverLimit("top-k".to_owned()));
        }
        if values.is_empty() {
            return Err(DebugDumpError::Invalid("logits are empty".to_owned()));
        }
        for value in values {
            if !value.is_finite() {
                return Err(DebugDumpError::Invalid("logit is non-finite".to_owned()));
            }
        }
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|left, right| {
            values[*right]
                .partial_cmp(&values[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        indices.truncate(top_k.min(values.len()));
        let entries = indices
            .into_iter()
            .map(|index| json!({ "token_index": index, "logit": values[index] }))
            .collect::<Vec<_>>();
        self.logits.push(json!({
            "layer": layer,
            "position": position,
            "top_k": entries,
        }));
        self.check_size()
    }

    pub fn finish(mut self) -> Result<DebugDumpArtifact, DebugDumpError> {
        if !self.config.enabled {
            self.committed = true;
            return Ok(DebugDumpArtifact {
                path: None,
                bytes: 0,
                digest: None,
                tensor_count: 0,
                token_count: 0,
                logit_count: 0,
            });
        }
        self.check_size()?;
        let tensor_count = self.tensors.len();
        let token_count = self.tokens.len();
        let logit_count = self.logits.len();
        let manifest = self.manifest.as_ref().ok_or_else(|| {
            DebugDumpError::Invalid("an identity-bound tool manifest is required".to_owned())
        })?;
        let object = json!({
            "$schema": "https://sllm.dev/schema/phase46-debug-dump-v1.schema.json",
            "schema_version": DEBUG_DUMP_SCHEMA_VERSION,
            "struct_size": DEBUG_DUMP_STRUCT_SIZE_V1,
            "manifest": manifest,
            "metadata": self.metadata,
            "tokens": self.tokens,
            "tensors": self.tensors,
            "logits": self.logits,
            "extensions": {},
        });
        let bytes = serde_json::to_vec(&object)
            .map_err(|error| DebugDumpError::Serialization(error.to_string()))?;
        if bytes.len() > self.config.max_bytes {
            return Err(DebugDumpError::OverLimit("serialized bytes".to_owned()));
        }
        let partial = self
            .partial_path
            .as_ref()
            .ok_or_else(|| DebugDumpError::Invalid("missing partial path".to_owned()))?;
        let final_path = self
            .final_path
            .as_ref()
            .ok_or_else(|| DebugDumpError::Invalid("missing final path".to_owned()))?;
        {
            let mut file = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(partial)?;
            file.write_all(&bytes)?;
            file.flush()?;
            file.sync_all()?;
        }
        fs::hard_link(partial, final_path)?;
        if let Err(error) = fs::remove_file(partial) {
            let _ = fs::remove_file(final_path);
            return Err(DebugDumpError::Io(error));
        }
        if let Err(error) =
            File::open(&self.config.output_dir).and_then(|directory| directory.sync_all())
        {
            let _ = fs::remove_file(final_path);
            let _ = File::open(&self.config.output_dir).and_then(|directory| directory.sync_all());
            return Err(DebugDumpError::Io(error));
        }
        self.committed = true;
        let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
        Ok(DebugDumpArtifact {
            path: Some(final_path.clone()),
            bytes: bytes.len(),
            digest: Some(digest),
            tensor_count,
            token_count,
            logit_count,
        })
    }

    fn check_size(&self) -> Result<(), DebugDumpError> {
        let value = json!({
            "$schema": "https://sllm.dev/schema/phase46-debug-dump-v1.schema.json",
            "schema_version": DEBUG_DUMP_SCHEMA_VERSION,
            "struct_size": DEBUG_DUMP_STRUCT_SIZE_V1,
            "manifest": self.manifest,
            "metadata": self.metadata,
            "tokens": self.tokens,
            "tensors": self.tensors,
            "logits": self.logits,
            "extensions": {},
        });
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| DebugDumpError::Serialization(error.to_string()))?;
        if bytes.len() > self.config.max_bytes {
            return Err(DebugDumpError::OverLimit("serialized bytes".to_owned()));
        }
        Ok(())
    }
}

impl Drop for DebugDumpWriter {
    fn drop(&mut self) {
        if !self.committed {
            if let Some(path) = &self.partial_path {
                let _ = fs::remove_file(path);
            }
        }
    }
}

const ALLOWED_METADATA: &[&str] = &[
    "schema_version",
    "run_id",
    "model_id",
    "model_fingerprint",
    "target",
    "backend",
    "device",
    "encoding",
    "kv_encoding",
    "dtype",
    "shape",
    "layout",
    "endianness",
    "quantization",
    "quantization_descriptor",
    "scale_plane",
    "submission_id",
    "token_count",
    "token_digest",
    "artifact_digest",
    "policy_digest",
    "dataset_digest",
    "manifest_sha256",
    "provider",
];

fn validate_metadata_key(key: &str) -> Result<(), DebugDumpError> {
    if !ALLOWED_METADATA.contains(&key) {
        return Err(DebugDumpError::Forbidden(key.to_owned()));
    }
    if key.contains("prompt")
        || key.contains("response")
        || key.contains("secret")
        || key.contains("auth")
        || key.contains("api")
        || key.contains("key")
        || key.contains("pointer")
        || key.contains("address")
        || key.contains("payload")
    {
        return Err(DebugDumpError::Forbidden(key.to_owned()));
    }
    Ok(())
}

fn validate_metadata_value(key: &str, value: &Value) -> Result<(), DebugDumpError> {
    const DIGEST_KEYS: &[&str] = &[
        "model_fingerprint",
        "token_digest",
        "artifact_digest",
        "policy_digest",
        "dataset_digest",
        "manifest_sha256",
    ];
    if DIGEST_KEYS.contains(&key) {
        let valid = value.as_str().is_some_and(|digest| {
            digest.len() == 71
                && digest.starts_with("sha256:")
                && digest[7..]
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
        if !valid {
            return Err(DebugDumpError::Invalid(format!(
                "metadata field {key} must be a sha256-prefixed lowercase digest"
            )));
        }
    }
    if key == "token_count" && value.as_u64().is_none() {
        return Err(DebugDumpError::Invalid(
            "metadata field token_count must be a nonnegative integer".to_owned(),
        ));
    }
    const STRING_KEYS: &[&str] = &[
        "schema_version",
        "run_id",
        "model_id",
        "target",
        "backend",
        "device",
        "encoding",
        "kv_encoding",
        "dtype",
        "layout",
        "endianness",
        "submission_id",
        "provider",
    ];
    if STRING_KEYS.contains(&key) && value.as_str().is_none() {
        return Err(DebugDumpError::Invalid(format!(
            "metadata field {key} must be a string"
        )));
    }
    match key {
        "shape" if !value.is_array() && !value.is_object() => {
            return Err(DebugDumpError::Invalid(
                "metadata field shape must be an array or object".to_owned(),
            ));
        }
        "quantization" | "quantization_descriptor" | "scale_plane"
            if !value.is_null() && !value.is_string() && !value.is_object() =>
        {
            return Err(DebugDumpError::Invalid(format!(
                "metadata field {key} has an invalid type"
            )));
        }
        _ => {}
    }
    if let Some(string) = value.as_str() {
        validate_public_text(string, key)?;
    }
    validate_nested_metadata(value)?;
    Ok(())
}

fn validate_nested_metadata(value: &Value) -> Result<(), DebugDumpError> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                let lower = key.to_ascii_lowercase();
                if lower.contains("prompt")
                    || lower.contains("response")
                    || lower.contains("secret")
                    || lower.contains("auth")
                    || lower.contains("api")
                    || lower.contains("password")
                    || lower.contains("credential")
                    || lower.contains("pointer")
                    || lower.contains("address")
                    || lower.contains("payload")
                {
                    return Err(DebugDumpError::Forbidden(key.clone()));
                }
                if let Some(string) = value.as_str() {
                    validate_public_text(string, key)?;
                }
                validate_nested_metadata(value)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_nested_metadata(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_public_text(value: &str, label: &str) -> Result<(), DebugDumpError> {
    if value.is_empty() || value.contains('\0') {
        return Err(DebugDumpError::Invalid(format!(
            "{label} must be non-empty public text"
        )));
    }
    let lower = value.to_ascii_lowercase();
    for forbidden in [
        "prompt",
        "response",
        "authorization",
        "api_key",
        "apikey",
        "secret",
        "password",
        "credential",
        "pointer",
        "device_address",
        "payload",
    ] {
        if lower.contains(forbidden) {
            return Err(DebugDumpError::Forbidden(format!(
                "{label} contains {forbidden}"
            )));
        }
    }
    Ok(())
}
