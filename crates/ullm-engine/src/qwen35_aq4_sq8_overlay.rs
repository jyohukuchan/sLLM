// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed identity and tensor-set admission for the Qwen3.5 AQ4 QKV/Z SQ8 overlay.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::format_id::FORMAT_SQ8_0;
use crate::sq::{SqFp8Artifact, read_sq_fp8_artifact};

pub const QWEN35_AQ4_SQ8_OVERLAY_BINDING_SCHEMA: &str = "ullm.qwen35_aq4_sq8_qkv_z_overlay.v1";
pub const QWEN35_AQ4_SQ8_OVERLAY_IMPLEMENTATION_ID: &str = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1";
pub const QWEN35_AQ4_SQ8_OVERLAY_EXECUTION_PROFILE: &str =
    "rdna4_aq4_resident_sq8_linear_qkv_z_overlay";
pub const QWEN35_AQ4_SQ8_OVERLAY_TENSOR_COUNT: usize = 48;
pub const QWEN35_AQ4_SQ8_OVERLAY_SCALE_BLOCK_COLS: u64 = 256;
const CONTENT_DOMAIN: &[u8] = b"ullm.qwen35-aq4-sq8-overlay-content.v1\0";
const TENSOR_SET_DOMAIN: &[u8] = b"ullm.qwen35-aq4-sq8-overlay-tensor-set.v1\0";
const HASH_CHUNK_BYTES: usize = 1024 * 1024;

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
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingPackage {
    root: String,
    manifest_sha256: String,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
