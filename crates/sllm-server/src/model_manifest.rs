//! Offline, host-side model manifest validation for the Phase 45 registry.
//!
//! The manifest is deliberately a small data-only description.  It contains
//! aliases and already-produced local artifacts, never URLs, credentials, or
//! payload bytes.  Every path is checked before a caller can pass the result
//! to model/GPU admission.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{File, Metadata, symlink_metadata};
use std::io::{ErrorKind, Read};
use std::path::{Component, Path, PathBuf};

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use sllm_core::KvCacheEncoding;

pub const MODEL_MANIFEST_SCHEMA_VERSION_V1: &str = "sllm-model-manifest-v1";
pub const MAX_MODEL_MANIFEST_BYTES_V1: usize = 1024 * 1024;
pub const MAX_MODEL_MANIFEST_MODELS_V1: usize = 64;
pub const MAX_MODEL_MANIFEST_ARTIFACTS_V1: usize = 8;
pub const MAX_MODEL_MANIFEST_ADAPTERS_V1: usize = MAX_MODEL_MANIFEST_ARTIFACTS_V1;
pub const MAX_MODEL_MANIFEST_CONTROL_VECTORS_V1: usize = MAX_MODEL_MANIFEST_ARTIFACTS_V1;
pub const MAX_MODEL_MANIFEST_ALIAS_BYTES_V1: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelManifestErrorV1 {
    Io,
    TooLarge,
    NotRegularFile,
    Symlink,
    PathRace,
    InvalidJson,
    DuplicateField,
    UnknownField,
    UnsupportedVersion,
    InvalidValue,
    DuplicateAlias,
    AliasLimit,
    ArtifactLimit,
    ArtifactOrder,
}

impl fmt::Display for ModelManifestErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Io => "manifest I/O failed",
            Self::TooLarge => "manifest exceeds the 1 MiB limit",
            Self::NotRegularFile => "manifest or artifact is not a regular file",
            Self::Symlink => "manifest or artifact symlink is not allowed",
            Self::PathRace => "manifest metadata changed during read",
            Self::InvalidJson => "manifest JSON is invalid",
            Self::DuplicateField => "manifest contains a duplicate field",
            Self::UnknownField => "manifest contains an unknown field",
            Self::UnsupportedVersion => "manifest version is unsupported",
            Self::InvalidValue => "manifest value is invalid",
            Self::DuplicateAlias => "manifest contains a duplicate alias",
            Self::AliasLimit => "manifest model alias limit exceeded",
            Self::ArtifactLimit => "manifest artifact limit exceeded",
            Self::ArtifactOrder => "manifest artifact aliases are not sorted and unique",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ModelManifestErrorV1 {}

#[derive(Clone, Eq, PartialEq)]
pub struct ModelArtifactManifestV1 {
    alias: String,
    lock: PathBuf,
    payload: PathBuf,
}

impl fmt::Debug for ModelArtifactManifestV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelArtifactManifestV1")
            .field("alias", &self.alias)
            .field("lock", &"<redacted-path>")
            .field("payload", &"<redacted-path>")
            .finish()
    }
}

impl ModelArtifactManifestV1 {
    pub fn alias(&self) -> &str {
        &self.alias
    }

    pub fn lock(&self) -> &Path {
        &self.lock
    }

    pub fn payload(&self) -> &Path {
        &self.payload
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModelManifestEntryV1 {
    alias: String,
    gguf: PathBuf,
    derived_lock: PathBuf,
    device_index: u32,
    target: String,
    kv_cache_encoding: Option<KvCacheEncoding>,
    declared_resident_bytes: u64,
    preload: bool,
    adapters: Vec<ModelArtifactManifestV1>,
    control_vectors: Vec<ModelArtifactManifestV1>,
}

impl fmt::Debug for ModelManifestEntryV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelManifestEntryV1")
            .field("alias", &self.alias)
            .field("gguf", &"<redacted-path>")
            .field("derived_lock", &"<redacted-path>")
            .field("device_index", &self.device_index)
            .field("target", &self.target)
            .field(
                "kv_cache_encoding",
                &self.kv_cache_encoding.map(KvCacheEncoding::canonical_name),
            )
            .field("declared_resident_bytes", &self.declared_resident_bytes)
            .field("preload", &self.preload)
            .field("adapters", &self.adapters.len())
            .field("control_vectors", &self.control_vectors.len())
            .finish()
    }
}

impl ModelManifestEntryV1 {
    pub fn alias(&self) -> &str {
        &self.alias
    }

    pub fn gguf(&self) -> &Path {
        &self.gguf
    }

    pub fn derived_lock(&self) -> &Path {
        &self.derived_lock
    }

    pub const fn device_index(&self) -> u32 {
        self.device_index
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub const fn kv_cache_encoding(&self) -> Option<KvCacheEncoding> {
        self.kv_cache_encoding
    }

    pub const fn declared_resident_bytes(&self) -> u64 {
        self.declared_resident_bytes
    }

    pub const fn preload(&self) -> bool {
        self.preload
    }

    pub fn adapters(&self) -> &[ModelArtifactManifestV1] {
        &self.adapters
    }

    pub fn control_vectors(&self) -> &[ModelArtifactManifestV1] {
        &self.control_vectors
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModelManifestV1 {
    schema_version: String,
    models: Vec<ModelManifestEntryV1>,
}

impl fmt::Debug for ModelManifestV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelManifestV1")
            .field("schema_version", &self.schema_version)
            .field("models", &self.models)
            .finish()
    }
}

impl ModelManifestV1 {
    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn models(&self) -> &[ModelManifestEntryV1] {
        &self.models
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireManifestV1 {
    schema_version: String,
    models: Vec<WireModelManifestV1>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireModelManifestV1 {
    alias: String,
    gguf: String,
    derived_lock: String,
    device_index: u32,
    target: String,
    #[serde(default)]
    kv_cache_encoding: Option<String>,
    declared_resident_bytes: u64,
    preload: bool,
    #[serde(default)]
    adapters: Vec<WireArtifactManifestV1>,
    #[serde(default)]
    control_vectors: Vec<WireArtifactManifestV1>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireArtifactManifestV1 {
    alias: String,
    lock: String,
    payload: String,
}

/// Read and validate a manifest with metadata-before/after race checks.
pub fn parse_model_manifest_v1(
    path: impl AsRef<Path>,
) -> Result<ModelManifestV1, ModelManifestErrorV1> {
    let path = path.as_ref();
    let before = checked_file_metadata(path)?;
    if before.len > MAX_MODEL_MANIFEST_BYTES_V1 as u64 {
        return Err(ModelManifestErrorV1::TooLarge);
    }

    let mut file = File::open(path).map_err(|_| ModelManifestErrorV1::Io)?;
    let opened = metadata_fingerprint(&file.metadata().map_err(|_| ModelManifestErrorV1::Io)?);
    if opened != before.fingerprint {
        return Err(ModelManifestErrorV1::PathRace);
    }
    let mut bytes = Vec::with_capacity(before.len as usize);
    let read_limit = (MAX_MODEL_MANIFEST_BYTES_V1 as u64)
        .checked_add(1)
        .ok_or(ModelManifestErrorV1::TooLarge)?;
    let read = file
        .by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|_| ModelManifestErrorV1::Io)?;
    if read > MAX_MODEL_MANIFEST_BYTES_V1 {
        return Err(ModelManifestErrorV1::TooLarge);
    }

    let after = checked_file_metadata(path)?;
    if after.fingerprint != before.fingerprint || after.fingerprint != opened {
        return Err(ModelManifestErrorV1::PathRace);
    }
    parse_model_manifest_bytes_v1(&bytes)
}

/// Alias kept explicit for callers that use "read" terminology.
pub fn read_model_manifest_v1(
    path: impl AsRef<Path>,
) -> Result<ModelManifestV1, ModelManifestErrorV1> {
    parse_model_manifest_v1(path)
}

/// Parse already-read bytes.  File/path admission must use
/// [`parse_model_manifest_v1`] so the host-side metadata and no-symlink checks
/// cannot be skipped.
fn parse_model_manifest_bytes_v1(bytes: &[u8]) -> Result<ModelManifestV1, ModelManifestErrorV1> {
    if bytes.len() > MAX_MODEL_MANIFEST_BYTES_V1 {
        return Err(ModelManifestErrorV1::TooLarge);
    }
    let value = parse_strict_value(bytes)?;
    let wire = serde_json::from_value::<WireManifestV1>(value)
        .map_err(|error| map_wire_error(&error.to_string()))?;
    build_manifest(wire)
}

fn build_manifest(wire: WireManifestV1) -> Result<ModelManifestV1, ModelManifestErrorV1> {
    if wire.schema_version != MODEL_MANIFEST_SCHEMA_VERSION_V1 {
        return Err(ModelManifestErrorV1::UnsupportedVersion);
    }
    if wire.models.len() > MAX_MODEL_MANIFEST_MODELS_V1 {
        return Err(ModelManifestErrorV1::AliasLimit);
    }
    if wire.models.is_empty() {
        return Err(ModelManifestErrorV1::InvalidValue);
    }
    let mut aliases = BTreeSet::new();
    let mut models = Vec::with_capacity(wire.models.len());
    for model in wire.models {
        validate_alias(&model.alias)?;
        if !aliases.insert(model.alias.clone()) {
            return Err(ModelManifestErrorV1::DuplicateAlias);
        }
        if model.declared_resident_bytes == 0 || !valid_target(&model.target) {
            return Err(ModelManifestErrorV1::InvalidValue);
        }
        let kv_cache_encoding = model
            .kv_cache_encoding
            .as_deref()
            .map(parse_kv_cache_encoding)
            .transpose()?;
        let gguf = validate_artifact_path(&model.gguf)?;
        let derived_lock = validate_artifact_path(&model.derived_lock)?;
        let mut artifact_aliases = BTreeSet::new();
        let adapters = build_artifacts(model.adapters, &mut artifact_aliases)?;
        let control_vectors = build_artifacts(model.control_vectors, &mut artifact_aliases)?;
        models.push(ModelManifestEntryV1 {
            alias: model.alias,
            gguf,
            derived_lock,
            device_index: model.device_index,
            target: model.target,
            kv_cache_encoding,
            declared_resident_bytes: model.declared_resident_bytes,
            preload: model.preload,
            adapters,
            control_vectors,
        });
    }
    Ok(ModelManifestV1 {
        schema_version: wire.schema_version,
        models,
    })
}

fn build_artifacts(
    artifacts: Vec<WireArtifactManifestV1>,
    aliases: &mut BTreeSet<String>,
) -> Result<Vec<ModelArtifactManifestV1>, ModelManifestErrorV1> {
    if artifacts.len() > MAX_MODEL_MANIFEST_ARTIFACTS_V1 {
        return Err(ModelManifestErrorV1::ArtifactLimit);
    }
    let mut previous: Option<String> = None;
    let mut result = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        validate_alias(&artifact.alias)?;
        if previous
            .as_deref()
            .is_some_and(|value| value >= artifact.alias.as_str())
        {
            return Err(ModelManifestErrorV1::ArtifactOrder);
        }
        if !aliases.insert(artifact.alias.clone()) {
            return Err(ModelManifestErrorV1::DuplicateAlias);
        }
        previous = Some(artifact.alias.clone());
        result.push(ModelArtifactManifestV1 {
            alias: artifact.alias,
            lock: validate_artifact_path(&artifact.lock)?,
            payload: validate_artifact_path(&artifact.payload)?,
        });
    }
    Ok(result)
}

fn validate_alias(alias: &str) -> Result<(), ModelManifestErrorV1> {
    if alias.is_empty()
        || alias.len() > MAX_MODEL_MANIFEST_ALIAS_BYTES_V1
        || !alias
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ModelManifestErrorV1::InvalidValue);
    }
    Ok(())
}

fn valid_target(target: &str) -> bool {
    matches!(
        target,
        "gfx1030" | "gfx1201" | "gfx942" | "gfx942:sramecc+:xnack-"
    )
}

fn parse_kv_cache_encoding(value: &str) -> Result<KvCacheEncoding, ModelManifestErrorV1> {
    match value {
        "fp16" => Ok(KvCacheEncoding::Fp16),
        "fp8" => Ok(KvCacheEncoding::Fp8E4M3Fn),
        "fp8-static" => Ok(KvCacheEncoding::Fp8E4M3FnStatic),
        "nvfp4" => Ok(KvCacheEncoding::Nvfp4),
        "kv-mxfp8-e4" => Ok(KvCacheEncoding::Mxfp8E4),
        "kv-mxfp8-e5" => Ok(KvCacheEncoding::Mxfp8E5),
        _ => Err(ModelManifestErrorV1::InvalidValue),
    }
}

#[cfg(test)]
mod kv_cache_encoding_tests {
    use super::*;

    #[test]
    fn model_entry_accepts_canonical_names_and_rejects_derived_aliases() {
        for (name, expected) in [
            ("fp16", KvCacheEncoding::Fp16),
            ("fp8", KvCacheEncoding::Fp8E4M3Fn),
            ("fp8-static", KvCacheEncoding::Fp8E4M3FnStatic),
            ("nvfp4", KvCacheEncoding::Nvfp4),
            ("kv-mxfp8-e4", KvCacheEncoding::Mxfp8E4),
            ("kv-mxfp8-e5", KvCacheEncoding::Mxfp8E5),
        ] {
            assert_eq!(parse_kv_cache_encoding(name), Ok(expected));
        }
        for alias in [
            "fp8-e4-block16",
            "kv-fp8-e4m3-block16",
            "kv-fp8-e5m2-block16",
            "KV-FP8-E4-BLOCK16",
            "mxfp8-e4",
            "kv-mxfp8-e4-block32",
            "KV-MXFP8-E4",
        ] {
            assert_eq!(
                parse_kv_cache_encoding(alias),
                Err(ModelManifestErrorV1::InvalidValue)
            );
        }
    }
}

fn validate_artifact_path(value: &str) -> Result<PathBuf, ModelManifestErrorV1> {
    if value.is_empty() || value.contains('\0') || is_network_form(value) || value.starts_with("//")
    {
        return Err(ModelManifestErrorV1::InvalidValue);
    }
    let path = Path::new(value);
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err(ModelManifestErrorV1::InvalidValue);
    }
    let mut current = PathBuf::from(Path::new("/"));
    let components = path.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        if matches!(component, Component::RootDir) {
            continue;
        }
        let Component::Normal(part) = component else {
            return Err(ModelManifestErrorV1::InvalidValue);
        };
        current.push(part);
        let metadata = symlink_metadata(&current).map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                ModelManifestErrorV1::InvalidValue
            } else {
                ModelManifestErrorV1::Io
            }
        })?;
        if metadata.file_type().is_symlink() {
            return Err(ModelManifestErrorV1::Symlink);
        }
        let last = index + 1 == components.len();
        if last {
            if !metadata.is_file() {
                return Err(ModelManifestErrorV1::NotRegularFile);
            }
        } else if !metadata.is_dir() {
            return Err(ModelManifestErrorV1::InvalidValue);
        }
    }
    Ok(path.to_owned())
}

fn is_network_form(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    if lower.contains("://") || lower.starts_with("//") {
        return true;
    }
    lower.split_once(':').is_some_and(|(scheme, _)| {
        matches!(
            scheme,
            "http" | "https" | "ftp" | "file" | "tcp" | "udp" | "ws" | "wss" | "s3"
        )
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MetadataFingerprint {
    dev: u64,
    ino: u64,
    mode: u32,
    len: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
}

struct CheckedMetadata {
    fingerprint: MetadataFingerprint,
    len: u64,
}

fn checked_file_metadata(path: &Path) -> Result<CheckedMetadata, ModelManifestErrorV1> {
    validate_manifest_path_components(path)?;
    let metadata = symlink_metadata(path).map_err(|_| ModelManifestErrorV1::Io)?;
    if metadata.file_type().is_symlink() {
        return Err(ModelManifestErrorV1::Symlink);
    }
    if !metadata.is_file() {
        return Err(ModelManifestErrorV1::NotRegularFile);
    }
    let fingerprint = metadata_fingerprint(&metadata);
    Ok(CheckedMetadata {
        fingerprint,
        len: metadata.len(),
    })
}

fn validate_manifest_path_components(path: &Path) -> Result<(), ModelManifestErrorV1> {
    let mut current = PathBuf::new();
    let components = path.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        match component {
            Component::Prefix(_) => return Err(ModelManifestErrorV1::InvalidValue),
            Component::RootDir => current.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(ModelManifestErrorV1::InvalidValue);
            }
            Component::Normal(part) => {
                current.push(part);
                let metadata = symlink_metadata(&current).map_err(|error| {
                    if error.kind() == ErrorKind::NotFound {
                        ModelManifestErrorV1::InvalidValue
                    } else {
                        ModelManifestErrorV1::Io
                    }
                })?;
                if metadata.file_type().is_symlink() {
                    return Err(ModelManifestErrorV1::Symlink);
                }
                if index + 1 < components.len() && !metadata.is_dir() {
                    return Err(ModelManifestErrorV1::InvalidValue);
                }
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn metadata_fingerprint(metadata: &Metadata) -> MetadataFingerprint {
    use std::os::unix::fs::MetadataExt;
    MetadataFingerprint {
        dev: metadata.dev(),
        ino: metadata.ino(),
        mode: metadata.mode(),
        len: metadata.len(),
        mtime: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
        ctime: metadata.ctime(),
        ctime_nsec: metadata.ctime_nsec(),
    }
}

#[cfg(not(unix))]
fn metadata_fingerprint(metadata: &Metadata) -> MetadataFingerprint {
    MetadataFingerprint {
        dev: 0,
        ino: 0,
        mode: u32::from(metadata.permissions().readonly()),
        len: metadata.len(),
        mtime: 0,
        mtime_nsec: 0,
        ctime: 0,
        ctime_nsec: 0,
    }
}

fn map_wire_error(message: &str) -> ModelManifestErrorV1 {
    if message.contains("unknown field") {
        ModelManifestErrorV1::UnknownField
    } else {
        ModelManifestErrorV1::InvalidJson
    }
}

fn parse_strict_value(bytes: &[u8]) -> Result<Value, ModelManifestErrorV1> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue::deserialize(&mut deserializer)
        .map_err(|error| {
            if error.to_string().contains("duplicate") {
                ModelManifestErrorV1::DuplicateField
            } else {
                ModelManifestErrorV1::InvalidJson
            }
        })?
        .0;
    deserializer
        .end()
        .map_err(|_| ModelManifestErrorV1::InvalidJson)?;
    Ok(value)
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor).map(Self)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object members")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("JSON number is not finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictValue::deserialize(deserializer).map(|value| value.0)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictValue>()? {
            values.push(value.0);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom("duplicate object field"));
            }
            let value = map.next_value::<StrictValue>()?;
            values.insert(key, value.0);
        }
        Ok(Value::Object(values))
    }
}
