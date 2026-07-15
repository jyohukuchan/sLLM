// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed identity and tensor-set admission for the Qwen3.5 AQ4 QKV/Z SQ8 overlay.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use crate::format_id::FORMAT_SQ8_0;
use crate::sq::{SqFp8Artifact, read_sq_fp8_artifact};

pub const QWEN35_AQ4_SQ8_OVERLAY_BINDING_SCHEMA: &str = "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2";
pub const QWEN35_AQ4_SQ8_OVERLAY_IMPLEMENTATION_ID: &str = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1";
pub const QWEN35_AQ4_SQ8_OVERLAY_EXECUTION_PROFILE: &str =
    "rdna4_aq4_resident_sq8_linear_qkv_z_overlay";
pub const QWEN35_AQ4_SQ8_OVERLAY_TENSOR_COUNT: usize = 48;
pub const QWEN35_AQ4_SQ8_OVERLAY_SCALE_BLOCK_COLS: u64 = 256;
const CONTENT_DOMAIN: &[u8] = b"ullm.qwen35-aq4-sq8-overlay-content.v1\0";
const TENSOR_SET_DOMAIN: &[u8] = b"ullm.qwen35-aq4-sq8-overlay-tensor-set.v1\0";
const HASH_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Qwen35Aq4Sq8OverlayLoadConfig {
    pub artifact_dir: PathBuf,
    pub binding_manifest: PathBuf,
    pub expected_binding_manifest_sha256: String,
    pub expected_content_sha256: String,
    pub expected_package_manifest_sha256: String,
    pub expected_source_model_dir: PathBuf,
    pub row_chunk: usize,
}

#[derive(Debug)]
pub struct ValidatedQwen35Aq4Sq8Overlay {
    pub artifact: SqFp8Artifact,
    pub binding_manifest_sha256: String,
    pub content_sha256: String,
    pub tensor_set_sha256: String,
    pub tensor_names: Vec<String>,
    pub resident_bytes: u64,
    pub row_chunk: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4Sq8OverlayIdentity {
    pub binding_manifest_sha256: String,
    pub content_sha256: String,
    pub tensor_set_sha256: String,
    pub tensor_names: Vec<String>,
}

impl ValidatedQwen35Aq4Sq8Overlay {
    pub fn identity(&self) -> Qwen35Aq4Sq8OverlayIdentity {
        Qwen35Aq4Sq8OverlayIdentity {
            binding_manifest_sha256: self.binding_manifest_sha256.clone(),
            content_sha256: self.content_sha256.clone(),
            tensor_set_sha256: self.tensor_set_sha256.clone(),
            tensor_names: self.tensor_names.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingManifest {
    schema_version: String,
    format_id: String,
    overlay_format_id: String,
    implementation_id: String,
    sq_manifest: BoundFile,
    content_sha256: String,
    tensor_set_sha256: String,
    tensor_names: Vec<String>,
    scale: BindingScale,
    artifact_policy: BindingArtifactPolicy,
    source: BindingSource,
    package: BindingPackage,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingScale {
    granularity: String,
    block_cols: u64,
    dtype: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingSource {
    model_dir: String,
    config_sha256: String,
    index_sha256: String,
    shards: Vec<BindingSourceShard>,
    tensors: Vec<BindingSourceTensor>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingSourceShard {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingSourceTensor {
    name: String,
    source: BindingLogicalTensor,
    overlay: BindingOverlayTensor,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingLogicalTensor {
    file: String,
    dtype: String,
    shape: Vec<u64>,
    logical_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingOverlayTensor {
    payload: BoundSizedFile,
    scale: BoundSizedFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundSizedFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingArtifactPolicy {
    uid: u32,
    gid: u32,
    directory_mode: String,
    file_mode: String,
    regular_file_nlink: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingPackage {
    root: String,
    manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceIndex {
    metadata: SourceIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceIndexMetadata {
    total_size: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsTensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn contained_file(root: &Path, relative: &str, label: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(format!("{label} path must be a non-empty relative path"));
    }
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize {}: {error}", root.display()))?;
    let path = root.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize {}: {error}", path.display()))?;
    if !canonical.starts_with(&canonical_root) {
        return Err(format!("{label} path escapes artifact directory"));
    }
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|error| format!("failed to stat {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular file"));
    }
    Ok(path)
}

fn sha256_file_range(path: &Path, offset: u64, bytes: u64) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("failed to seek {}: {error}", path.display()))?;
    let mut remaining = bytes;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("bounded hash read fits usize");
        let count = file
            .read(&mut buffer[..wanted])
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if count == 0 {
            return Err(format!("{} tensor payload is truncated", path.display()));
        }
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn safetensors_headers(
    path: &Path,
) -> Result<(u64, BTreeMap<String, SafetensorsTensorHeader>), String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open source shard {}: {error}", path.display()))?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| format!("failed to read source shard header length: {error}"))?;
    let header_len = u64::from_le_bytes(length_bytes);
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| "source shard header length exceeds usize".to_string())?;
    if header_len_usize == 0 || header_len_usize > MAX_SAFETENSORS_HEADER_BYTES {
        return Err("source shard safetensors header length is invalid".into());
    }
    let mut header = vec![0_u8; header_len_usize];
    file.read_exact(&mut header)
        .map_err(|error| format!("failed to read source shard header: {error}"))?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|error| format!("failed to parse source shard header: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "source shard safetensors header must be an object".to_string())?;
    let mut tensors = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor: SafetensorsTensorHeader = serde_json::from_value(value.clone())
            .map_err(|error| format!("source tensor {name} header differs: {error}"))?;
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(format!("source shard contains duplicate tensor {name}"));
        }
    }
    let data_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| "source shard data start overflows".to_string())?;
    Ok((data_start, tensors))
}

fn parse_mode(value: &str, expected: &str, label: &str) -> Result<u32, String> {
    if value != expected {
        return Err(format!(
            "SQ8 overlay artifact {label} policy must be {expected}"
        ));
    }
    u32::from_str_radix(value, 8)
        .map_err(|_| format!("SQ8 overlay artifact {label} policy is invalid"))
}

fn validate_artifact_immutability(
    artifact_root: &Path,
    binding_path: &Path,
    artifact: &SqFp8Artifact,
    policy: &BindingArtifactPolicy,
) -> Result<(), String> {
    let directory_mode = parse_mode(&policy.directory_mode, "0555", "directory mode")?;
    let file_mode = parse_mode(&policy.file_mode, "0444", "file mode")?;
    if policy.regular_file_nlink != 1 {
        return Err("SQ8 overlay artifact regular-file nlink policy must be 1".into());
    }
    let canonical_root = artifact_root
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize SQ8 overlay root: {error}"))?;
    let mut expected_files = BTreeSet::from([
        PathBuf::from("binding.json"),
        PathBuf::from("sq_manifest.json"),
    ]);
    for entry in &artifact.manifest.fp8_tensors {
        expected_files.insert(PathBuf::from(&entry.payload_file));
        expected_files.insert(PathBuf::from(&entry.scale_file));
    }
    let expected_binding = binding_path
        .strip_prefix(&canonical_root)
        .map_err(|_| "SQ8 overlay binding is outside canonical artifact root".to_string())?;
    if expected_binding != Path::new("binding.json") {
        return Err("SQ8 overlay binding must be artifact_dir/binding.json".into());
    }
    let mut found_files = BTreeSet::new();
    let mut pending = vec![canonical_root.clone()];
    while let Some(directory) = pending.pop() {
        let metadata = std::fs::symlink_metadata(&directory)
            .map_err(|error| format!("failed to stat {}: {error}", directory.display()))?;
        if !metadata.file_type().is_dir()
            || metadata.uid() != policy.uid
            || metadata.gid() != policy.gid
            || metadata.permissions().mode() & 0o777 != directory_mode
        {
            return Err(format!(
                "SQ8 overlay artifact directory policy differs: {}",
                directory.display()
            ));
        }
        for item in std::fs::read_dir(&directory)
            .map_err(|error| format!("failed to enumerate {}: {error}", directory.display()))?
        {
            let item = item.map_err(|error| format!("failed to enumerate artifact: {error}"))?;
            let path = item.path();
            let metadata = std::fs::symlink_metadata(&path)
                .map_err(|error| format!("failed to stat {}: {error}", path.display()))?;
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "SQ8 overlay artifact contains symlink: {}",
                    path.display()
                ));
            }
            if metadata.file_type().is_dir() {
                pending.push(path);
                continue;
            }
            if !metadata.file_type().is_file()
                || metadata.nlink() != policy.regular_file_nlink
                || metadata.uid() != policy.uid
                || metadata.gid() != policy.gid
                || metadata.permissions().mode() & 0o777 != file_mode
            {
                return Err(format!(
                    "SQ8 overlay artifact file policy differs: {}",
                    path.display()
                ));
            }
            let relative = path
                .strip_prefix(&canonical_root)
                .map_err(|_| "SQ8 overlay artifact traversal escaped root".to_string())?
                .to_path_buf();
            if !found_files.insert(relative) {
                return Err("SQ8 overlay artifact contains duplicate file path".into());
            }
        }
    }
    if found_files != expected_files {
        return Err("SQ8 overlay artifact file inventory differs".into());
    }
    Ok(())
}

pub fn qwen35_aq4_sq8_overlay_tensor_names(
    linear_layer_indices: &[usize],
) -> Result<Vec<String>, String> {
    let mut unique = BTreeSet::new();
    for &layer in linear_layer_indices {
        if !unique.insert(layer) {
            return Err(format!("linear layer index {layer} is duplicated"));
        }
    }
    let mut names = Vec::with_capacity(linear_layer_indices.len() * 2);
    for layer in unique {
        names.push(format!(
            "model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"
        ));
        names.push(format!(
            "model.language_model.layers.{layer}.linear_attn.in_proj_z.weight"
        ));
    }
    Ok(names)
}

pub fn qwen35_aq4_sq8_overlay_tensor_set_sha256(names: &[String]) -> String {
    let mut names = names.to_vec();
    names.sort();
    let mut digest = Sha256::new();
    digest.update(TENSOR_SET_DOMAIN);
    for name in names {
        digest.update(name.as_bytes());
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

fn artifact_content_sha256(artifact: &SqFp8Artifact, names: &[String]) -> Result<String, String> {
    let mut entries = artifact.manifest.fp8_tensors.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    let expected = names.iter().cloned().collect::<BTreeSet<_>>();
    let actual = entries
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<BTreeSet<_>>();
    if entries.len() != actual.len() {
        return Err("SQ8 overlay contains duplicate tensor names".into());
    }
    if actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(format!(
            "SQ8 overlay tensor set differs: missing={missing:?} unexpected={unexpected:?}"
        ));
    }
    let mut digest = Sha256::new();
    digest.update(CONTENT_DOMAIN);
    for entry in entries {
        let payload_sha = entry
            .payload_sha256
            .as_deref()
            .filter(|value| is_sha256(value))
            .ok_or_else(|| format!("SQ8 overlay {} payload SHA-256 is missing", entry.name))?;
        let scale_sha = entry
            .scale_sha256
            .as_deref()
            .filter(|value| is_sha256(value))
            .ok_or_else(|| format!("SQ8 overlay {} scale SHA-256 is missing", entry.name))?;
        let payload = contained_file(
            &artifact.artifact_dir,
            &entry.payload_file,
            &format!("SQ8 overlay {} payload", entry.name),
        )?;
        let scale = contained_file(
            &artifact.artifact_dir,
            &entry.scale_file,
            &format!("SQ8 overlay {} scale", entry.name),
        )?;
        if sha256_file(&payload)? != payload_sha {
            return Err(format!(
                "SQ8 overlay {} payload SHA-256 differs",
                entry.name
            ));
        }
        if sha256_file(&scale)? != scale_sha {
            return Err(format!("SQ8 overlay {} scale SHA-256 differs", entry.name));
        }
        digest.update(entry.name.as_bytes());
        digest.update(b"\0");
        digest.update(payload_sha.as_bytes());
        digest.update(b"\0");
        digest.update(scale_sha.as_bytes());
        digest.update(b"\n");
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn source_provenance_maps<'a>(
    source: &'a BindingSource,
    expected_names: &[String],
) -> Result<
    (
        BTreeMap<String, &'a BindingSourceTensor>,
        BTreeMap<String, &'a BindingSourceShard>,
    ),
    String,
> {
    let expected_set = expected_names.iter().cloned().collect::<BTreeSet<_>>();
    let mut tensor_by_name = BTreeMap::new();
    for tensor in &source.tensors {
        if tensor_by_name.insert(tensor.name.clone(), tensor).is_some() {
            return Err(format!(
                "SQ8 overlay source provenance duplicates tensor {}",
                tensor.name
            ));
        }
    }
    if tensor_by_name.keys().cloned().collect::<BTreeSet<_>>() != expected_set {
        return Err("SQ8 overlay source provenance tensor set differs".into());
    }
    let mut shard_by_path = BTreeMap::new();
    for shard in &source.shards {
        if shard_by_path.insert(shard.path.clone(), shard).is_some() {
            return Err(format!(
                "SQ8 overlay source provenance duplicates shard {}",
                shard.path
            ));
        }
    }
    let referenced_shards = source
        .tensors
        .iter()
        .map(|tensor| tensor.source.file.clone())
        .collect::<BTreeSet<_>>();
    if shard_by_path.keys().cloned().collect::<BTreeSet<_>>() != referenced_shards {
        return Err("SQ8 overlay source shard set differs from tensor provenance".into());
    }
    Ok((tensor_by_name, shard_by_path))
}

fn validate_source_provenance(
    source_root: &Path,
    source: &BindingSource,
    artifact: &SqFp8Artifact,
    expected_names: &[String],
) -> Result<(), String> {
    let index_bytes = std::fs::read(source_root.join("model.safetensors.index.json"))
        .map_err(|error| format!("failed to read source model index: {error}"))?;
    let index: SourceIndex = serde_json::from_slice(&index_bytes)
        .map_err(|error| format!("failed to parse source model index: {error}"))?;
    if index.metadata.total_size == 0 {
        return Err("SQ8 overlay source model index total_size must be positive".into());
    }
    let (tensor_by_name, shard_by_path) = source_provenance_maps(source, expected_names)?;

    let artifact_by_name = artifact
        .manifest
        .fp8_tensors
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let mut shard_headers = BTreeMap::new();
    for (relative, shard) in &shard_by_path {
        if !is_sha256(&shard.sha256) {
            return Err(format!(
                "SQ8 overlay source shard {relative} SHA-256 is invalid"
            ));
        }
        let path = contained_file(source_root, relative, "SQ8 overlay source shard")?;
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|error| format!("failed to stat source shard {}: {error}", path.display()))?;
        if metadata.len() != shard.bytes || sha256_file(&path)? != shard.sha256 {
            return Err(format!(
                "SQ8 overlay source shard {relative} identity differs"
            ));
        }
        let headers = safetensors_headers(&path)?;
        shard_headers.insert(relative.clone(), (path, headers));
    }

    for name in expected_names {
        let binding = tensor_by_name
            .get(name)
            .ok_or_else(|| format!("SQ8 overlay source tensor {name} is missing"))?;
        let artifact_entry = artifact_by_name
            .get(name.as_str())
            .ok_or_else(|| format!("SQ8 overlay artifact tensor {name} is missing"))?;
        let expected_shape = if name.ends_with(".linear_attn.in_proj_qkv.weight") {
            vec![8192_u64, 4096_u64]
        } else {
            vec![4096_u64, 4096_u64]
        };
        if binding.source.dtype != "BF16"
            || binding.source.shape != expected_shape
            || !is_sha256(&binding.source.logical_sha256)
            || index.weight_map.get(name) != Some(&binding.source.file)
        {
            return Err(format!("SQ8 overlay source tensor {name} metadata differs"));
        }
        let (shard_path, (data_start, headers)) = shard_headers
            .get(&binding.source.file)
            .ok_or_else(|| format!("SQ8 overlay source tensor {name} shard is missing"))?;
        let header = headers
            .get(name)
            .ok_or_else(|| format!("SQ8 overlay source shard omits tensor {name}"))?;
        let logical_bytes = binding
            .source
            .shape
            .iter()
            .try_fold(2_u64, |bytes, dimension| bytes.checked_mul(*dimension))
            .ok_or_else(|| format!("SQ8 overlay source tensor {name} byte length overflows"))?;
        if header.dtype != binding.source.dtype
            || header.shape != binding.source.shape
            || header.data_offsets[1] < header.data_offsets[0]
            || header.data_offsets[1] - header.data_offsets[0] != logical_bytes
        {
            return Err(format!(
                "SQ8 overlay source tensor {name} safetensors header differs"
            ));
        }
        let absolute_offset = data_start
            .checked_add(header.data_offsets[0])
            .ok_or_else(|| format!("SQ8 overlay source tensor {name} offset overflows"))?;
        if sha256_file_range(shard_path, absolute_offset, logical_bytes)?
            != binding.source.logical_sha256
        {
            return Err(format!(
                "SQ8 overlay source tensor {name} logical SHA-256 differs"
            ));
        }
        let payload_sha = artifact_entry
            .payload_sha256
            .as_deref()
            .ok_or_else(|| format!("SQ8 overlay tensor {name} payload SHA-256 is missing"))?;
        let scale_sha = artifact_entry
            .scale_sha256
            .as_deref()
            .ok_or_else(|| format!("SQ8 overlay tensor {name} scale SHA-256 is missing"))?;
        if binding.overlay.payload.path != artifact_entry.payload_file
            || binding.overlay.payload.bytes != artifact_entry.payload_bytes
            || binding.overlay.payload.sha256 != payload_sha
            || binding.overlay.scale.path != artifact_entry.scale_file
            || binding.overlay.scale.bytes != artifact_entry.scale_bytes
            || binding.overlay.scale.sha256 != scale_sha
        {
            return Err(format!(
                "SQ8 overlay tensor {name} source-to-overlay mapping differs"
            ));
        }
    }
    Ok(())
}

pub fn validate_qwen35_aq4_sq8_overlay(
    config: &Qwen35Aq4Sq8OverlayLoadConfig,
    package_dir: &Path,
    linear_layer_indices: &[usize],
) -> Result<ValidatedQwen35Aq4Sq8Overlay, String> {
    if config.row_chunk == 0 {
        return Err("Qwen3.5 AQ4 SQ8 overlay row_chunk must be positive".into());
    }
    for (label, value) in [
        (
            "binding manifest",
            config.expected_binding_manifest_sha256.as_str(),
        ),
        ("content", config.expected_content_sha256.as_str()),
        (
            "package manifest",
            config.expected_package_manifest_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            return Err(format!(
                "Qwen3.5 AQ4 SQ8 overlay expected {label} SHA-256 is invalid"
            ));
        }
    }
    let expected_names = qwen35_aq4_sq8_overlay_tensor_names(linear_layer_indices)?;
    if expected_names.len() != QWEN35_AQ4_SQ8_OVERLAY_TENSOR_COUNT {
        return Err(format!(
            "Qwen3.5 AQ4 SQ8 overlay requires exactly {} tensors from 24 linear layers, got {}",
            QWEN35_AQ4_SQ8_OVERLAY_TENSOR_COUNT,
            expected_names.len()
        ));
    }
    let binding_path = config
        .binding_manifest
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize SQ8 overlay binding: {error}"))?;
    let artifact_root = config
        .artifact_dir
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize SQ8 overlay artifact: {error}"))?;
    if !binding_path.starts_with(&artifact_root) {
        return Err("SQ8 overlay binding is outside artifact directory".into());
    }
    let binding_sha = sha256_file(&binding_path)?;
    if binding_sha != config.expected_binding_manifest_sha256 {
        return Err("SQ8 overlay binding manifest SHA-256 differs".into());
    }
    let binding_bytes = std::fs::read(&binding_path)
        .map_err(|error| format!("failed to read SQ8 overlay binding: {error}"))?;
    let binding: BindingManifest = serde_json::from_slice(&binding_bytes)
        .map_err(|error| format!("failed to parse SQ8 overlay binding: {error}"))?;
    if binding.schema_version != QWEN35_AQ4_SQ8_OVERLAY_BINDING_SCHEMA
        || binding.format_id != "AQ4_0"
        || binding.overlay_format_id != FORMAT_SQ8_0
        || binding.implementation_id != QWEN35_AQ4_SQ8_OVERLAY_IMPLEMENTATION_ID
    {
        return Err("SQ8 overlay binding identity is unsupported".into());
    }
    if binding.content_sha256 != config.expected_content_sha256
        || binding.package.manifest_sha256 != config.expected_package_manifest_sha256
    {
        return Err("SQ8 overlay served identity differs from binding".into());
    }
    if binding.scale.granularity != "row_block"
        || binding.scale.block_cols != QWEN35_AQ4_SQ8_OVERLAY_SCALE_BLOCK_COLS
        || binding.scale.dtype != "f32"
    {
        return Err("SQ8 overlay binding must use row_block256 f32 scales".into());
    }
    let binding_names = binding
        .tensor_names
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let expected_set = expected_names.iter().cloned().collect::<BTreeSet<_>>();
    if binding.tensor_names.len() != binding_names.len() || binding_names != expected_set {
        return Err("SQ8 overlay binding tensor set differs from exact QKV/Z set".into());
    }
    let tensor_set_sha = qwen35_aq4_sq8_overlay_tensor_set_sha256(&expected_names);
    if binding.tensor_set_sha256 != tensor_set_sha {
        return Err("SQ8 overlay binding tensor-set SHA-256 differs".into());
    }
    let package_root = package_dir
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize AQ4 package: {error}"))?;
    let bound_package = Path::new(&binding.package.root)
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize bound AQ4 package: {error}"))?;
    if bound_package != package_root
        || sha256_file(&package_dir.join("manifest.json"))?
            != config.expected_package_manifest_sha256
    {
        return Err("SQ8 overlay AQ4 package identity differs".into());
    }
    let source_root = config
        .expected_source_model_dir
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize overlay source model: {error}"))?;
    let bound_source = Path::new(&binding.source.model_dir)
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize bound source model: {error}"))?;
    if source_root != bound_source
        || !is_sha256(&binding.source.config_sha256)
        || !is_sha256(&binding.source.index_sha256)
        || sha256_file(&source_root.join("config.json"))? != binding.source.config_sha256
        || sha256_file(&source_root.join("model.safetensors.index.json"))?
            != binding.source.index_sha256
    {
        return Err("SQ8 overlay source-model identity differs".into());
    }
    let sq_manifest_path = contained_file(
        &config.artifact_dir,
        &binding.sq_manifest.path,
        "SQ8 overlay manifest",
    )?;
    if !is_sha256(&binding.sq_manifest.sha256)
        || sha256_file(&sq_manifest_path)? != binding.sq_manifest.sha256
    {
        return Err("SQ8 overlay sq_manifest SHA-256 differs".into());
    }
    if sq_manifest_path
        .file_name()
        .and_then(|value| value.to_str())
        != Some("sq_manifest.json")
        || sq_manifest_path.parent() != Some(config.artifact_dir.as_path())
    {
        return Err("SQ8 overlay sq_manifest must be artifact_dir/sq_manifest.json".into());
    }
    let artifact = read_sq_fp8_artifact(&config.artifact_dir)?;
    if artifact.manifest.candidate.id != FORMAT_SQ8_0
        || artifact.manifest.candidate.format_id.as_deref() != Some(FORMAT_SQ8_0)
        || artifact.manifest.storage.fp8_tensor_count != QWEN35_AQ4_SQ8_OVERLAY_TENSOR_COUNT as u64
    {
        return Err("SQ8 overlay artifact candidate/count identity differs".into());
    }
    validate_artifact_immutability(
        &artifact_root,
        &binding_path,
        &artifact,
        &binding.artifact_policy,
    )?;
    for entry in &artifact.manifest.fp8_tensors {
        let is_qkv = entry.name.ends_with(".linear_attn.in_proj_qkv.weight");
        let is_z = entry.name.ends_with(".linear_attn.in_proj_z.weight");
        let expected_shape = if is_qkv {
            [8192_u64, 4096_u64]
        } else if is_z {
            [4096_u64, 4096_u64]
        } else {
            return Err(format!("SQ8 overlay tensor {} is not QKV/Z", entry.name));
        };
        let expected_family = if is_qkv {
            "linear_attn_qkv"
        } else {
            "linear_attn_z"
        };
        if entry.source_dtype != "BF16"
            || entry.family != expected_family
            || entry.shape.as_slice() != expected_shape
            || entry.scale_granularity != "row_block"
            || entry.scale_block_cols != Some(QWEN35_AQ4_SQ8_OVERLAY_SCALE_BLOCK_COLS)
        {
            return Err(format!(
                "SQ8 overlay tensor {} dtype/layout differs",
                entry.name
            ));
        }
    }
    let content_sha = artifact_content_sha256(&artifact, &expected_names)?;
    if content_sha != binding.content_sha256 {
        return Err("SQ8 overlay content SHA-256 differs".into());
    }
    validate_source_provenance(&source_root, &binding.source, &artifact, &expected_names)?;
    let resident_bytes = artifact
        .manifest
        .storage
        .fp8_payload_bytes
        .checked_add(artifact.manifest.storage.fp8_scale_bytes)
        .ok_or_else(|| "SQ8 overlay resident bytes overflow".to_string())?;
    Ok(ValidatedQwen35Aq4Sq8Overlay {
        artifact,
        binding_manifest_sha256: binding_sha,
        content_sha256: content_sha,
        tensor_set_sha256: tensor_set_sha,
        tensor_names: expected_names,
        resident_bytes,
        row_chunk: config.row_chunk,
    })
}

pub fn revalidate_qwen35_aq4_sq8_overlay_after_load(
    config: &Qwen35Aq4Sq8OverlayLoadConfig,
    validated: &ValidatedQwen35Aq4Sq8Overlay,
) -> Result<(), String> {
    let artifact_root = config
        .artifact_dir
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize SQ8 overlay artifact: {error}"))?;
    let binding_path = config
        .binding_manifest
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize SQ8 overlay binding: {error}"))?;
    let binding_sha = sha256_file(&binding_path)?;
    if binding_sha != config.expected_binding_manifest_sha256
        || binding_sha != validated.binding_manifest_sha256
    {
        return Err("SQ8 overlay binding changed during model load".into());
    }
    let binding_bytes = std::fs::read(&binding_path)
        .map_err(|error| format!("failed to reread SQ8 overlay binding: {error}"))?;
    let binding: BindingManifest = serde_json::from_slice(&binding_bytes)
        .map_err(|error| format!("failed to reparse SQ8 overlay binding: {error}"))?;
    let artifact = read_sq_fp8_artifact(&artifact_root)?;
    validate_artifact_immutability(
        &artifact_root,
        &binding_path,
        &artifact,
        &binding.artifact_policy,
    )?;
    let names = artifact
        .manifest
        .fp8_tensors
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<Vec<_>>();
    let tensor_set_sha = qwen35_aq4_sq8_overlay_tensor_set_sha256(&names);
    let content_sha = artifact_content_sha256(&artifact, &validated.tensor_names)?;
    if names.len() != QWEN35_AQ4_SQ8_OVERLAY_TENSOR_COUNT
        || tensor_set_sha != validated.tensor_set_sha256
        || tensor_set_sha != binding.tensor_set_sha256
        || content_sha != validated.content_sha256
        || content_sha != binding.content_sha256
        || content_sha != config.expected_content_sha256
    {
        return Err("SQ8 overlay identity changed during model load".into());
    }
    let sq_manifest_path = contained_file(
        &artifact_root,
        &binding.sq_manifest.path,
        "SQ8 overlay manifest",
    )?;
    if sha256_file(&sq_manifest_path)? != binding.sq_manifest.sha256 {
        return Err("SQ8 overlay manifest changed during model load".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION_SOURCE_ROOT: &str =
        "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B";
    const PRODUCTION_OVERLAY_ROOT: &str = "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/artifacts/sq8-linear-qkv-z-rowblock256-v0.1";

    fn source_tensor(name: &str, file: &str) -> BindingSourceTensor {
        BindingSourceTensor {
            name: name.into(),
            source: BindingLogicalTensor {
                file: file.into(),
                dtype: "BF16".into(),
                shape: vec![1, 1],
                logical_sha256: "a".repeat(64),
            },
            overlay: BindingOverlayTensor {
                payload: BoundSizedFile {
                    path: format!("fp8/{name}.bin"),
                    bytes: 1,
                    sha256: "b".repeat(64),
                },
                scale: BoundSizedFile {
                    path: format!("scales/{name}.bin"),
                    bytes: 4,
                    sha256: "c".repeat(64),
                },
            },
        }
    }

    fn source_with(tensors: Vec<BindingSourceTensor>, shard_paths: &[&str]) -> BindingSource {
        BindingSource {
            model_dir: "/source".into(),
            config_sha256: "d".repeat(64),
            index_sha256: "e".repeat(64),
            shards: shard_paths
                .iter()
                .map(|path| BindingSourceShard {
                    path: (*path).into(),
                    bytes: 1,
                    sha256: "f".repeat(64),
                })
                .collect(),
            tensors,
        }
    }

    #[test]
    fn production_tensor_set_is_exact_and_stable() {
        let layers = (0..32).filter(|layer| layer % 4 != 3).collect::<Vec<_>>();
        let names = qwen35_aq4_sq8_overlay_tensor_names(&layers).unwrap();
        assert_eq!(names.len(), 48);
        assert!(
            names.contains(
                &"model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string()
            )
        );
        assert!(
            names.contains(
                &"model.language_model.layers.30.linear_attn.in_proj_z.weight".to_string()
            )
        );
        assert!(!names.iter().any(|name| name.contains("layers.3.")));
        assert_eq!(
            qwen35_aq4_sq8_overlay_tensor_set_sha256(&names),
            qwen35_aq4_sq8_overlay_tensor_set_sha256(
                &names.iter().rev().cloned().collect::<Vec<_>>()
            )
        );
    }

    #[test]
    fn duplicated_linear_layer_is_rejected() {
        assert!(qwen35_aq4_sq8_overlay_tensor_names(&[0, 0]).is_err());
    }

    #[test]
    fn source_index_accepts_production_schema() {
        let index: SourceIndex = serde_json::from_str(
            r#"{
                "metadata":{"total_size":19306216416},
                "weight_map":{"tensor.weight":"model-00001-of-00002.safetensors"}
            }"#,
        )
        .unwrap();
        assert_eq!(index.metadata.total_size, 19_306_216_416);
        assert_eq!(
            index.weight_map.get("tensor.weight").map(String::as_str),
            Some("model-00001-of-00002.safetensors")
        );
    }

    #[test]
    fn source_index_rejects_missing_invalid_and_unknown_metadata() {
        assert!(
            serde_json::from_str::<SourceIndex>(
                r#"{"weight_map":{"tensor.weight":"model.safetensors"}}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<SourceIndex>(
                r#"{
                    "metadata":{"total_size":"19306216416"},
                    "weight_map":{"tensor.weight":"model.safetensors"}
                }"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<SourceIndex>(
                r#"{
                    "metadata":{"total_size":19306216416,"unknown":true},
                    "weight_map":{"tensor.weight":"model.safetensors"}
                }"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<SourceIndex>(
                r#"{
                    "metadata":{"total_size":19306216416},
                    "weight_map":{"tensor.weight":"model.safetensors"},
                    "unknown":true
                }"#
            )
            .is_err()
        );
    }

    #[test]
    fn production_source_provenance_validates_cpu_when_available() {
        let source_root = Path::new(PRODUCTION_SOURCE_ROOT);
        let artifact_root = Path::new(PRODUCTION_OVERLAY_ROOT);
        if !source_root.is_dir() || !artifact_root.is_dir() {
            return;
        }
        let binding: BindingManifest =
            serde_json::from_slice(&std::fs::read(artifact_root.join("binding.json")).unwrap())
                .unwrap();
        let artifact = read_sq_fp8_artifact(artifact_root).unwrap();
        let layers = (0..32).filter(|layer| layer % 4 != 3).collect::<Vec<_>>();
        let names = qwen35_aq4_sq8_overlay_tensor_names(&layers).unwrap();
        validate_source_provenance(source_root, &binding.source, &artifact, &names).unwrap();
    }

    #[test]
    fn source_provenance_rejects_missing_tensor() {
        let source = source_with(
            vec![source_tensor("a", "one.safetensors")],
            &["one.safetensors"],
        );
        assert!(source_provenance_maps(&source, &["a".into(), "b".into()]).is_err());
    }

    #[test]
    fn source_provenance_rejects_duplicate_tensor() {
        let source = source_with(
            vec![
                source_tensor("a", "one.safetensors"),
                source_tensor("a", "one.safetensors"),
            ],
            &["one.safetensors"],
        );
        assert!(source_provenance_maps(&source, &["a".into()]).is_err());
    }

    #[test]
    fn source_provenance_rejects_mismatched_shard_set() {
        let source = source_with(
            vec![source_tensor("a", "one.safetensors")],
            &["two.safetensors"],
        );
        assert!(source_provenance_maps(&source, &["a".into()]).is_err());
    }
}
