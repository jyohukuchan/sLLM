//! Shared identity, result-envelope, and atomic-publication contracts.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

pub const TOOL_RUN_SCHEMA_VERSION_V1: &str = "sllm-phase46-tool-run-v1";
pub const TOOL_RUN_STRUCT_SIZE_V1: u32 = 13;
pub const TOOL_JSON_CANONICALIZATION_V1: &str = "sllm-sorted-json-v1";
const HASH_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_ID_BYTES: usize = 256;
const MAX_ARGUMENTS: usize = 512;
const MAX_ARGUMENT_BYTES: usize = 16 * 1024;
const MAX_IDENTITIES: usize = 16_384;
const MAX_EXTENSIONS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolError(String);

impl ToolError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ToolError {}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
pub enum ToolRunStateV1 {
    Pass,
    Fail,
    InsufficientEvidence,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolFileIdentityV1 {
    pub role: String,
    pub logical_name: String,
    pub size_bytes: u64,
    pub sha256: String,
}

impl ToolFileIdentityV1 {
    pub fn from_path(
        role: impl Into<String>,
        logical_name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self, ToolError> {
        let path = path.as_ref();
        let before = fs::symlink_metadata(path)
            .map_err(|error| io_error(path, "stat identity input", error))?;
        if !before.file_type().is_file() || before.file_type().is_symlink() {
            return Err(ToolError::invalid(format!(
                "identity input is not a regular non-symlink file: {}",
                path.display()
            )));
        }
        let mut file =
            File::open(path).map_err(|error| io_error(path, "open identity input", error))?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
        let mut size_bytes = 0_u64;
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| io_error(path, "read identity input", error))?;
            if count == 0 {
                break;
            }
            size_bytes = size_bytes
                .checked_add(
                    u64::try_from(count)
                        .map_err(|_| ToolError::invalid("identity read length does not fit u64"))?,
                )
                .ok_or_else(|| ToolError::invalid("identity size overflowed"))?;
            digest.update(&buffer[..count]);
        }
        let after = file
            .metadata()
            .map_err(|error| io_error(path, "restat identity input", error))?;
        if before.len() != after.len() || size_bytes != after.len() {
            return Err(ToolError::invalid(format!(
                "identity input changed while hashing: {}",
                path.display()
            )));
        }
        let identity = Self {
            role: role.into(),
            logical_name: logical_name.into(),
            size_bytes,
            sha256: format!("{:x}", digest.finalize()),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn for_bytes(
        role: impl Into<String>,
        logical_name: impl Into<String>,
        bytes: &[u8],
    ) -> Result<Self, ToolError> {
        let identity = Self {
            role: role.into(),
            logical_name: logical_name.into(),
            size_bytes: u64::try_from(bytes.len())
                .map_err(|_| ToolError::invalid("byte identity size does not fit u64"))?,
            sha256: sha256_bytes(bytes),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<(), ToolError> {
        validate_id("file identity role", &self.role)?;
        validate_id("file identity logical name", &self.logical_name)?;
        if self.size_bytes == 0 || !valid_sha256(&self.sha256) {
            return Err(ToolError::invalid(
                "file identity size or SHA-256 is invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolIdentityV1 {
    pub repository: String,
    pub commit: String,
    pub package: String,
    pub version: String,
    pub executable_sha256: String,
    pub arguments: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

impl ToolIdentityV1 {
    fn validate(&self) -> Result<(), ToolError> {
        validate_id("tool repository", &self.repository)?;
        validate_id("tool commit", &self.commit)?;
        validate_id("tool package", &self.package)?;
        validate_id("tool version", &self.version)?;
        if self.commit.len() != 40
            || !self
                .commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || self.commit.bytes().all(|byte| byte == b'0')
            || !valid_sha256(&self.executable_sha256)
        {
            return Err(ToolError::invalid(
                "tool commit or executable SHA-256 is invalid",
            ));
        }
        if self.arguments.len() > MAX_ARGUMENTS
            || self
                .arguments
                .iter()
                .any(|value| value.len() > MAX_ARGUMENT_BYTES || value.contains('\0'))
        {
            return Err(ToolError::invalid(
                "tool arguments exceed the bounded contract",
            ));
        }
        validate_string_map("tool environment", &self.environment)
    }
}

pub fn rust_toolchain_environment() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("os".to_owned(), std::env::consts::OS.to_owned()),
        ("arch".to_owned(), std::env::consts::ARCH.to_owned()),
        ("rustc".to_owned(), env!("SLLM_RUSTC_VERBOSE").to_owned()),
    ])
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRecipeIdentityV1 {
    pub id: String,
    pub version: String,
    pub config_sha256: String,
}

impl ToolRecipeIdentityV1 {
    fn validate(&self) -> Result<(), ToolError> {
        validate_id("recipe id", &self.id)?;
        validate_id("recipe version", &self.version)?;
        if !valid_sha256(&self.config_sha256) {
            return Err(ToolError::invalid("recipe config SHA-256 is invalid"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRunManifestV1 {
    pub schema_version: String,
    pub struct_size: u32,
    pub canonicalization: String,
    pub operation: String,
    pub state: ToolRunStateV1,
    pub selected_count: u64,
    pub tool: ToolIdentityV1,
    pub recipe: ToolRecipeIdentityV1,
    pub sources: Vec<ToolFileIdentityV1>,
    pub outputs: Vec<ToolFileIdentityV1>,
    pub raw_evidence: Vec<ToolFileIdentityV1>,
    pub identities: BTreeMap<String, String>,
    pub metrics: BTreeMap<String, Value>,
    /// Additive optional fields live only in this envelope.  New required
    /// fields require a new schema version.
    pub extensions: BTreeMap<String, Value>,
}

impl ToolRunManifestV1 {
    pub fn validate(&self) -> Result<(), ToolError> {
        if self.schema_version != TOOL_RUN_SCHEMA_VERSION_V1
            || self.struct_size != TOOL_RUN_STRUCT_SIZE_V1
            || self.canonicalization != TOOL_JSON_CANONICALIZATION_V1
        {
            return Err(ToolError::invalid(
                "unknown tool-run schema or canonicalization",
            ));
        }
        validate_id("operation", &self.operation)?;
        if self.selected_count == 0 {
            return Err(ToolError::invalid("tool run selected zero items"));
        }
        self.tool.validate()?;
        self.recipe.validate()?;
        if self.sources.is_empty() || self.sources.len() > MAX_IDENTITIES {
            return Err(ToolError::invalid(
                "tool run source identity count is invalid",
            ));
        }
        if self.state == ToolRunStateV1::Pass && self.outputs.is_empty() {
            return Err(ToolError::invalid(
                "passing tool run has no output identity",
            ));
        }
        if self.outputs.len() > MAX_IDENTITIES || self.raw_evidence.len() > MAX_IDENTITIES {
            return Err(ToolError::invalid("tool run identity count exceeds limit"));
        }
        let mut keys = BTreeSet::new();
        for identity in self
            .sources
            .iter()
            .chain(self.outputs.iter())
            .chain(self.raw_evidence.iter())
        {
            identity.validate()?;
            if !keys.insert((identity.role.as_str(), identity.logical_name.as_str())) {
                return Err(ToolError::invalid("duplicate role/logical-name identity"));
            }
        }
        validate_string_map("run identities", &self.identities)?;
        if self.extensions.len() > MAX_EXTENSIONS {
            return Err(ToolError::invalid("too many additive extensions"));
        }
        for (name, value) in self.metrics.iter().chain(self.extensions.iter()) {
            validate_id("metric/extension name", name)?;
            validate_json_value(value, 0)?;
        }
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, ToolError> {
        self.validate()?;
        canonical_json_bytes(
            &serde_json::to_value(self)
                .map_err(|error| ToolError::invalid(format!("serialize tool run: {error}")))?,
        )
    }

    pub fn sha256(&self) -> Result<String, ToolError> {
        Ok(sha256_bytes(&self.canonical_json()?))
    }
}

/// A directory transaction.  Every output is written below one fresh staging
/// directory and becomes visible by one directory rename.
#[derive(Debug)]
pub struct AtomicBundleV1 {
    final_path: PathBuf,
    staging_path: PathBuf,
    committed: bool,
}

impl AtomicBundleV1 {
    pub fn create(final_path: impl AsRef<Path>) -> Result<Self, ToolError> {
        let final_path = final_path.as_ref().to_path_buf();
        let parent = final_path
            .parent()
            .ok_or_else(|| ToolError::invalid("bundle path has no parent"))?;
        let name = final_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| ToolError::invalid("bundle name is invalid UTF-8"))?;
        if final_path.exists() {
            return Err(ToolError::invalid("bundle output already exists"));
        }
        let staging_path = parent.join(format!(".{name}.phase46-partial"));
        if staging_path.exists() {
            return Err(ToolError::invalid(format!(
                "stale bundle staging path exists: {}",
                staging_path.display()
            )));
        }
        fs::create_dir(&staging_path)
            .map_err(|error| io_error(&staging_path, "create bundle staging directory", error))?;
        Ok(Self {
            final_path,
            staging_path,
            committed: false,
        })
    }

    pub fn staging_root(&self) -> &Path {
        &self.staging_path
    }

    pub fn path(&self, relative: impl AsRef<Path>) -> Result<PathBuf, ToolError> {
        let relative = relative.as_ref();
        validate_relative_path(relative)?;
        let output = self.staging_path.join(relative);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| io_error(parent, "create bundle subdirectory", error))?;
        }
        Ok(output)
    }

    pub fn write_bytes(
        &self,
        relative: impl AsRef<Path>,
        bytes: &[u8],
    ) -> Result<PathBuf, ToolError> {
        let output = self.path(relative)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|error| io_error(&output, "create bundle member", error))?;
        file.write_all(bytes)
            .map_err(|error| io_error(&output, "write bundle member", error))?;
        file.sync_all()
            .map_err(|error| io_error(&output, "sync bundle member", error))?;
        Ok(output)
    }

    pub fn write_json<T: Serialize>(
        &self,
        relative: impl AsRef<Path>,
        value: &T,
    ) -> Result<PathBuf, ToolError> {
        let json = serde_json::to_value(value)
            .map_err(|error| ToolError::invalid(format!("serialize bundle JSON: {error}")))?;
        self.write_bytes(relative, &canonical_json_bytes(&json)?)
    }

    pub fn commit(mut self) -> Result<PathBuf, ToolError> {
        if count_regular_files(&self.staging_path)? == 0 {
            return Err(ToolError::invalid("bundle contains no regular files"));
        }
        sync_tree(&self.staging_path)?;
        if self.final_path.exists() {
            return Err(ToolError::invalid("bundle output appeared before commit"));
        }
        rename_noreplace(&self.staging_path, &self.final_path)
            .map_err(|error| io_error(&self.final_path, "publish bundle", error))?;
        let parent = self
            .final_path
            .parent()
            .ok_or_else(|| ToolError::invalid("bundle parent disappeared"))?;
        if let Err(error) = sync_directory(parent) {
            let _ = fs::remove_dir_all(&self.final_path);
            let _ = sync_directory(parent);
            return Err(error);
        }
        self.committed = true;
        Ok(self.final_path.clone())
    }
}

impl Drop for AtomicBundleV1 {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.staging_path);
        }
    }
}

pub fn atomic_write_json<T: Serialize>(
    output: impl AsRef<Path>,
    value: &T,
) -> Result<ToolFileIdentityV1, ToolError> {
    let output = output.as_ref();
    let parent = output
        .parent()
        .ok_or_else(|| ToolError::invalid("JSON output has no parent"))?;
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| ToolError::invalid("JSON output name is invalid UTF-8"))?;
    if output.exists() {
        return Err(ToolError::invalid("JSON output already exists"));
    }
    let partial = parent.join(format!(".{name}.phase46-partial"));
    if partial.exists() {
        return Err(ToolError::invalid("stale JSON staging path exists"));
    }
    let json_value = serde_json::to_value(value)
        .map_err(|error| ToolError::invalid(format!("serialize JSON output: {error}")))?;
    let bytes = canonical_json_bytes(&json_value)?;
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|error| io_error(&partial, "create JSON staging file", error))?;
        file.write_all(&bytes)
            .map_err(|error| io_error(&partial, "write JSON staging file", error))?;
        file.sync_all()
            .map_err(|error| io_error(&partial, "sync JSON staging file", error))?;
        if output.exists() {
            return Err(ToolError::invalid("JSON output appeared before commit"));
        }
        let identity = ToolFileIdentityV1::from_path("report", name, &partial)?;
        fs::hard_link(&partial, output)
            .map_err(|error| io_error(output, "publish JSON output without replacement", error))?;
        fs::remove_file(&partial)
            .map_err(|error| io_error(&partial, "remove JSON staging link", error))?;
        if let Err(error) = sync_directory(parent) {
            let _ = fs::remove_file(output);
            let _ = sync_directory(parent);
            return Err(error);
        }
        Ok(identity)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

#[cfg(target_os = "linux")]
fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    // SAFETY: both path pointers are live NUL-terminated byte strings for the
    // duration of the call; AT_FDCWD makes them relative to the current cwd.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "atomic no-replace directory publication requires Linux renameat2",
    ))
}

pub fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, ToolError> {
    validate_json_value(value, 0)?;
    let canonical = sort_json(value);
    let mut bytes = serde_json::to_vec(&canonical)
        .map_err(|error| ToolError::invalid(format!("serialize canonical JSON: {error}")))?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sort_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sort_json).collect()),
        Value::Object(values) => {
            let sorted: BTreeMap<&String, Value> = values
                .iter()
                .map(|(key, value)| (key, sort_json(value)))
                .collect();
            let mut output = serde_json::Map::new();
            for (key, value) in sorted {
                output.insert(key.clone(), value);
            }
            Value::Object(output)
        }
        _ => value.clone(),
    }
}

fn validate_json_value(value: &Value, depth: usize) -> Result<(), ToolError> {
    if depth > 64 {
        return Err(ToolError::invalid("JSON nesting exceeds 64"));
    }
    match value {
        Value::Array(values) => {
            if values.len() > 1_000_000 {
                return Err(ToolError::invalid("JSON array exceeds limit"));
            }
            for value in values {
                validate_json_value(value, depth + 1)?;
            }
        }
        Value::Object(values) => {
            if values.len() > 65_536 {
                return Err(ToolError::invalid("JSON object exceeds limit"));
            }
            for (key, value) in values {
                validate_id("JSON object key", key)?;
                validate_json_value(value, depth + 1)?;
            }
        }
        Value::String(value) if value.len() > 16 * 1024 * 1024 || value.contains('\0') => {
            return Err(ToolError::invalid(
                "JSON string exceeds limit or contains NUL",
            ));
        }
        Value::Number(number) => {
            if number.as_f64().is_some_and(|value| !value.is_finite()) {
                return Err(ToolError::invalid("JSON number is non-finite"));
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<(), ToolError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(ToolError::invalid(
            "bundle member path is empty or absolute",
        ));
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(ToolError::invalid("bundle member path is not normalized"));
        }
    }
    Ok(())
}

fn count_regular_files(root: &Path) -> Result<usize, ToolError> {
    let mut count = 0_usize;
    for entry in
        fs::read_dir(root).map_err(|error| io_error(root, "read bundle directory", error))?
    {
        let entry = entry.map_err(|error| io_error(root, "read bundle entry", error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| io_error(&path, "stat bundle entry", error))?;
        if metadata.file_type().is_symlink() {
            return Err(ToolError::invalid("bundle contains a symlink"));
        }
        if metadata.is_dir() {
            count = count
                .checked_add(count_regular_files(&path)?)
                .ok_or_else(|| ToolError::invalid("bundle file count overflowed"))?;
        } else if metadata.is_file() {
            count = count
                .checked_add(1)
                .ok_or_else(|| ToolError::invalid("bundle file count overflowed"))?;
        } else {
            return Err(ToolError::invalid("bundle contains a non-regular entry"));
        }
    }
    Ok(count)
}

fn sync_tree(root: &Path) -> Result<(), ToolError> {
    for entry in
        fs::read_dir(root).map_err(|error| io_error(root, "read bundle for sync", error))?
    {
        let entry = entry.map_err(|error| io_error(root, "read bundle sync entry", error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| io_error(&path, "stat bundle sync entry", error))?;
        if metadata.is_dir() {
            sync_tree(&path)?;
        } else if metadata.is_file() {
            File::open(&path)
                .and_then(|file| file.sync_all())
                .map_err(|error| io_error(&path, "sync bundle file", error))?;
        } else {
            return Err(ToolError::invalid("bundle contains an unsupported entry"));
        }
    }
    sync_directory(root)
}

fn sync_directory(path: &Path) -> Result<(), ToolError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| io_error(path, "sync directory", error))
}

fn validate_string_map(label: &str, values: &BTreeMap<String, String>) -> Result<(), ToolError> {
    if values.len() > 4_096 {
        return Err(ToolError::invalid(format!("{label} exceeds entry limit")));
    }
    for (key, value) in values {
        validate_id(label, key)?;
        if value.len() > MAX_ARGUMENT_BYTES || value.contains('\0') {
            return Err(ToolError::invalid(format!("{label} value is invalid")));
        }
    }
    Ok(())
}

fn validate_id(label: &str, value: &str) -> Result<(), ToolError> {
    if value.is_empty() || value.len() > MAX_ID_BYTES || value.contains(['\0', '\n', '\r']) {
        return Err(ToolError::invalid(format!("{label} is invalid")));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn io_error(path: &Path, action: &str, error: impl fmt::Display) -> ToolError {
    ToolError::invalid(format!("{action} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sllm-phase46-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn manifest(selected_count: u64) -> ToolRunManifestV1 {
        ToolRunManifestV1 {
            schema_version: TOOL_RUN_SCHEMA_VERSION_V1.to_owned(),
            struct_size: TOOL_RUN_STRUCT_SIZE_V1,
            canonicalization: TOOL_JSON_CANONICALIZATION_V1.to_owned(),
            operation: "test".to_owned(),
            state: ToolRunStateV1::Pass,
            selected_count,
            tool: ToolIdentityV1 {
                repository: "https://github.com/89chin/sLLM".to_owned(),
                commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                package: "sllm-tools".to_owned(),
                version: "0.1.0".to_owned(),
                executable_sha256: sha256_bytes(b"test executable"),
                arguments: vec!["test".to_owned()],
                environment: BTreeMap::from([("os".to_owned(), "linux".to_owned())]),
            },
            recipe: ToolRecipeIdentityV1 {
                id: "fixture".to_owned(),
                version: "v1".to_owned(),
                config_sha256: sha256_bytes(b"recipe"),
            },
            sources: vec![ToolFileIdentityV1::for_bytes("source", "source.bin", b"x").unwrap()],
            outputs: vec![ToolFileIdentityV1::for_bytes("output", "result.bin", b"y").unwrap()],
            raw_evidence: Vec::new(),
            identities: BTreeMap::new(),
            metrics: BTreeMap::from([("finite".to_owned(), Value::from(1.5))]),
            extensions: BTreeMap::new(),
        }
    }

    #[test]
    fn manifest_is_deterministic_and_rejects_zero_selection() {
        let valid = manifest(1);
        assert_eq!(
            valid.canonical_json().unwrap(),
            valid.canonical_json().unwrap()
        );
        assert_eq!(valid.sha256().unwrap().len(), 64);
        assert!(manifest(0).validate().is_err());
        let mut duplicate = valid.clone();
        duplicate.outputs = duplicate.sources.clone();
        assert!(duplicate.validate().is_err());
        let mut unknown_commit = valid;
        unknown_commit.tool.commit = "0".repeat(40);
        assert!(unknown_commit.validate().is_err());
    }

    #[test]
    fn bundle_publishes_once_and_cleans_failed_staging() {
        let root = temp_path("bundle");
        fs::create_dir(&root).unwrap();
        let final_path = root.join("result");
        {
            let bundle = AtomicBundleV1::create(&final_path).unwrap();
            bundle.write_bytes("nested/value.bin", b"abc").unwrap();
            assert!(!final_path.exists());
            bundle.commit().unwrap();
        }
        assert_eq!(
            fs::read(final_path.join("nested/value.bin")).unwrap(),
            b"abc"
        );
        assert!(AtomicBundleV1::create(&final_path).is_err());

        let abandoned = root.join("abandoned");
        let staging = {
            let bundle = AtomicBundleV1::create(&abandoned).unwrap();
            let staging = bundle.staging_root().to_path_buf();
            bundle.write_bytes("x", b"x").unwrap();
            staging
        };
        assert!(!staging.exists());
        assert!(!abandoned.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stale_staging_and_path_escape_are_rejected() {
        let root = temp_path("stale");
        fs::create_dir(&root).unwrap();
        let final_path = root.join("result");
        fs::create_dir(root.join(".result.phase46-partial")).unwrap();
        assert!(AtomicBundleV1::create(&final_path).is_err());
        fs::remove_dir_all(root.join(".result.phase46-partial")).unwrap();
        let bundle = AtomicBundleV1::create(&final_path).unwrap();
        assert!(bundle.path("../escape").is_err());
        drop(bundle);
        fs::remove_dir_all(root).unwrap();
    }
}
