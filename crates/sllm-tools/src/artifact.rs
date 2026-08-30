//! Host-side artifact conversion and quantization contracts.
//!
//! The implementation in this module is deliberately offline and bounded.  It
//! never downloads a model and it never turns a failed conversion into a CPU
//! inference result.  GGUF splitting operates on the verified file byte
//! stream, while conversion manifests carry enough catalog information for a
//! later consumer to verify every part before publication.

use crate::tool_manifest::{
    ToolFileIdentityV1, ToolIdentityV1, ToolRecipeIdentityV1, ToolRunManifestV1, ToolRunStateV1,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sllm_core::{
    VerifiedGguf, decode_e4m3fn, decode_e8m0, encode_e2m1, quantize_e4m3fn_outer_rows,
    quantize_nvfp4_weights, repack_mxfp4_standard, repack_nvfp4_standard,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_PARTS: usize = 65_536;
const MAX_MATRIX_ELEMENTS: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactError(pub String);

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ArtifactError {}

fn invalid(message: impl Into<String>) -> ArtifactError {
    ArtifactError(message.into())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, ArtifactError> {
    serde_json::to_vec(value).map_err(|error| invalid(format!("canonical JSON: {error}")))
}

fn tool_hash(value: &[u8]) -> String {
    sha256_bytes(value).trim_start_matches("sha256:").to_owned()
}

fn write_tool_run_manifest(
    stage: &Path,
    operation: &str,
    recipe_id: &str,
    recipe_config: &[u8],
    sources: Vec<ToolFileIdentityV1>,
    output_paths: &[(&str, &Path)],
    selected_count: u64,
) -> Result<(), ArtifactError> {
    let outputs = output_paths
        .iter()
        .map(|(logical_name, path)| ToolFileIdentityV1::from_path("output", *logical_name, path))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid(format!("tool output identity: {error}")))?;
    let executable = std::env::current_exe()
        .map_err(|error| invalid(format!("resolve tool executable: {error}")))?;
    let executable_sha256 =
        ToolFileIdentityV1::from_path("tool-binary", "sllm-artifact", executable)
            .map_err(|error| invalid(format!("tool executable identity: {error}")))?
            .sha256;
    let mut environment = crate::tool_manifest::rust_toolchain_environment();
    environment.insert(String::from("offline"), String::from("true"));
    let manifest = ToolRunManifestV1 {
        schema_version: crate::tool_manifest::TOOL_RUN_SCHEMA_VERSION_V1.to_owned(),
        struct_size: crate::tool_manifest::TOOL_RUN_STRUCT_SIZE_V1,
        canonicalization: crate::tool_manifest::TOOL_JSON_CANONICALIZATION_V1.to_owned(),
        operation: operation.to_owned(),
        state: ToolRunStateV1::Pass,
        selected_count,
        tool: ToolIdentityV1 {
            repository: "https://github.com/89chin/sLLM".to_owned(),
            commit: env!("SLLM_GIT_COMMIT").to_owned(),
            package: "sllm-tools".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            executable_sha256,
            arguments: vec![operation.to_owned()],
            environment,
        },
        recipe: ToolRecipeIdentityV1 {
            id: recipe_id.to_owned(),
            version: "v1".to_owned(),
            config_sha256: tool_hash(recipe_config),
        },
        sources,
        outputs,
        raw_evidence: Vec::new(),
        identities: BTreeMap::new(),
        metrics: BTreeMap::new(),
        extensions: BTreeMap::new(),
    };
    let bytes = manifest
        .canonical_json()
        .map_err(|error| invalid(format!("tool run manifest: {error}")))?;
    atomic_write(&stage.join("run-manifest.json"), &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ArtifactError> {
    if path.exists() {
        return Err(invalid(format!(
            "output already exists: {}",
            path.display()
        )));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| invalid(format!("create output directory: {error}")))?;
    let tmp = parent.join(format!(
        ".{}.sllm-partial-{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        unique_suffix()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|error| invalid(format!("{}: {error}", tmp.display())))?;
        file.write_all(bytes)
            .map_err(|error| invalid(format!("{}: {error}", tmp.display())))?;
        file.sync_all()
            .map_err(|error| invalid(format!("{}: {error}", tmp.display())))?;
        fs::rename(&tmp, path).map_err(|error| invalid(format!("{}: {error}", path.display())))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[allow(dead_code)]
fn publish_operation_bundle(
    output_dir: &Path,
    operation: &str,
    recipe_id: &str,
    recipe_config: &[u8],
    sources: Vec<ToolFileIdentityV1>,
    members: Vec<(String, Vec<u8>)>,
    selected_count: u64,
) -> Result<(), ArtifactError> {
    if output_dir.exists() || members.is_empty() || selected_count == 0 {
        return Err(invalid("operation bundle target or selection is invalid"));
    }
    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| invalid(format!("create bundle parent: {error}")))?;
    let stage = parent.join(format!(
        ".{}.sllm-stage-{}",
        output_dir.file_name().unwrap_or_default().to_string_lossy(),
        unique_suffix()
    ));
    fs::create_dir(&stage).map_err(|error| invalid(format!("create bundle stage: {error}")))?;
    let result = (|| {
        let mut paths = Vec::with_capacity(members.len());
        for (name, bytes) in members {
            if name.is_empty() || name.contains('/') || name.contains('\\') {
                return Err(invalid("bundle member name is invalid"));
            }
            let path = stage.join(&name);
            atomic_write(&path, &bytes)?;
            paths.push((name, path));
        }
        let borrowed: Vec<(&str, &Path)> = paths
            .iter()
            .map(|(name, path)| (name.as_str(), path.as_path()))
            .collect();
        write_tool_run_manifest(
            &stage,
            operation,
            recipe_id,
            recipe_config,
            sources,
            &borrowed,
            selected_count,
        )?;
        fs::rename(&stage, output_dir)
            .map_err(|error| invalid(format!("publish operation bundle: {error}")))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&stage);
    }
    result
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
        ^ u128::from(std::process::id())
}

/// Reviewed conversion capability.  Generic architecture and arbitrary bit
/// width dispatch are intentionally not exposed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityV1 {
    pub schema_version: String,
    pub architecture: String,
    pub tensor_catalog: String,
    pub dtypes: Vec<String>,
    pub recipes: Vec<String>,
}

pub fn reviewed_capability(architecture: &str) -> Result<CapabilityV1, ArtifactError> {
    if architecture != "qwen35" && architecture != "Qwen3_5ForConditionalGeneration" {
        return Err(invalid(format!("unsupported architecture: {architecture}")));
    }
    Ok(CapabilityV1 {
        schema_version: "sllm-capability-v1".to_owned(),
        architecture: "qwen35".to_owned(),
        tensor_catalog: "qwen35-text-reviewed-v1".to_owned(),
        dtypes: vec![
            "BF16".into(),
            "FP8-E4M3FN".into(),
            "NVFP4-E2M1".into(),
            "MXFP4-E2M1".into(),
        ],
        recipes: vec![
            "bf16".into(),
            "fp8-e4m3fn-channel-f32-scale".into(),
            "nvfp4-e2m1-block16-e4m3fn-f32-outer".into(),
            "mxfp4-e2m1-block32-e8m0".into(),
        ],
    })
}

pub fn capabilities_json(architecture: &str) -> Result<String, ArtifactError> {
    let capability = reviewed_capability(architecture)?;
    serde_json::to_string(&capability).map_err(|error| invalid(format!("capability JSON: {error}")))
}

pub fn dispatch_capability(
    architecture: &str,
    dtype: &str,
    recipe: &str,
) -> Result<CapabilityV1, ArtifactError> {
    let capability = reviewed_capability(architecture)?;
    if !capability
        .dtypes
        .iter()
        .any(|item| item.eq_ignore_ascii_case(dtype))
    {
        return Err(invalid(format!("unsupported dtype: {dtype}")));
    }
    if !capability.recipes.iter().any(|item| item == recipe) {
        return Err(invalid(format!(
            "unsupported quantization recipe: {recipe}"
        )));
    }
    let dtype_ok = match recipe {
        "bf16" => dtype.eq_ignore_ascii_case("BF16"),
        "fp8-e4m3fn-channel-f32-scale" => dtype.eq_ignore_ascii_case("FP8-E4M3FN"),
        "nvfp4-e2m1-block16-e4m3fn-f32-outer" => dtype.eq_ignore_ascii_case("NVFP4-E2M1"),
        "mxfp4-e2m1-block32-e8m0" => dtype.eq_ignore_ascii_case("MXFP4-E2M1"),
        _ => false,
    };
    if !dtype_ok {
        return Err(invalid(format!(
            "dtype does not match recipe: {dtype}/{recipe}"
        )));
    }
    Ok(capability)
}

/// Stable aliases used by callers that treat split/merge as file operations.
pub fn split_gguf_file(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    max_part_bytes: u64,
) -> Result<SplitManifestV1, ArtifactError> {
    split_gguf(input, output_dir, max_part_bytes)
}

pub fn merge_gguf_parts(
    manifest_path: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<String, ArtifactError> {
    merge_gguf(manifest_path, output)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitTensorV1 {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub absolute_range: [u64; 2],
    pub byte_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitPartV1 {
    pub index: usize,
    pub path: String,
    pub byte_range: [u64; 2],
    pub size_bytes: u64,
    pub sha256: String,
    pub tensor_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitSourceV1 {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub metadata_sha256: String,
    pub tensor_catalog_sha256: String,
    pub architecture: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitManifestV1 {
    pub schema_version: String,
    pub source: SplitSourceV1,
    pub tensors: Vec<SplitTensorV1>,
    pub parts: Vec<SplitPartV1>,
    pub semantic_digest: String,
}

impl SplitManifestV1 {
    pub fn canonical_json(&self) -> Result<Vec<u8>, ArtifactError> {
        canonical_json(self)
    }
    pub fn validate(&self) -> Result<(), ArtifactError> {
        if self.schema_version != "sllm-gguf-split-v1"
            || self.tensors.is_empty()
            || self.parts.is_empty()
            || self.parts.len() > MAX_PARTS
        {
            return Err(invalid("invalid GGUF split manifest shape"));
        }
        if self.source.size_bytes == 0
            || self.source.size_bytes > MAX_ARTIFACT_BYTES
            || !valid_sha256(&self.source.sha256)
            || !valid_sha256(&self.source.metadata_sha256)
            || !valid_sha256(&self.source.tensor_catalog_sha256)
            || !valid_sha256(&self.semantic_digest)
        {
            return Err(invalid("invalid GGUF split source identity"));
        }
        let mut names = BTreeSet::new();
        let mut part_paths = BTreeSet::new();
        let mut cursor = 0_u64;
        for (expected_index, part) in self.parts.iter().enumerate() {
            if part.index != expected_index
                || part.path.is_empty()
                || part.path.contains('/')
                || part.path.contains('\\')
                || part.byte_range[0] != cursor
                || part.byte_range[1] <= part.byte_range[0]
                || part.byte_range[1] - part.byte_range[0] != part.size_bytes
                || !valid_sha256(&part.sha256)
                || part.tensor_names.is_empty()
                || !part_paths.insert(&part.path)
            {
                return Err(invalid("GGUF split part ordering or identity is invalid"));
            }
            cursor = part.byte_range[1];
            for name in &part.tensor_names {
                if !names.insert(name) {
                    return Err(invalid("duplicate tensor in split parts"));
                }
            }
        }
        if cursor != self.source.size_bytes || names.len() != self.tensors.len() {
            return Err(invalid(
                "GGUF split part coverage differs from tensor catalog",
            ));
        }
        let mut catalog_names = BTreeSet::new();
        let tensor_ranges: BTreeMap<&str, [u64; 2]> = self
            .tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.absolute_range))
            .collect();
        for tensor in &self.tensors {
            if tensor.name.is_empty()
                || tensor.dimensions.is_empty()
                || tensor.dimensions.contains(&0)
                || tensor.absolute_range[1] <= tensor.absolute_range[0]
                || tensor.byte_size != tensor.absolute_range[1] - tensor.absolute_range[0]
                || !catalog_names.insert(&tensor.name)
            {
                return Err(invalid("invalid GGUF split tensor catalog"));
            }
        }
        if catalog_names != names {
            return Err(invalid("GGUF split tensor names differ"));
        }
        for part in &self.parts {
            for name in &part.tensor_names {
                let range = tensor_ranges
                    .get(name.as_str())
                    .ok_or_else(|| invalid("split part references unknown tensor"))?;
                if range[0] < part.byte_range[0]
                    || range[1] > part.byte_range[1]
                    || range[1] <= range[0]
                {
                    return Err(invalid("split part tensor range is inconsistent"));
                }
            }
        }
        Ok(())
    }
}

/// Split a verified GGUF stream without ever cutting through a tensor range.
/// Parts are contiguous byte ranges (the first part includes the GGUF header),
/// so merge can recover the exact source bytes and digest.
pub fn split_gguf(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    max_part_bytes: u64,
) -> Result<SplitManifestV1, ArtifactError> {
    if max_part_bytes == 0 || max_part_bytes > MAX_ARTIFACT_BYTES {
        return Err(invalid("max_part_bytes is outside the bounded range"));
    }
    let input = input.as_ref();
    let verified = VerifiedGguf::open(input)
        .map_err(|error| invalid(format!("GGUF verification: {error}")))?;
    if verified.tensors().is_empty() {
        return Err(invalid("cannot split a GGUF with zero tensors"));
    }
    if verified.file_size() > MAX_ARTIFACT_BYTES {
        return Err(invalid("GGUF exceeds bounded size"));
    }
    let bytes = fs::read(input).map_err(|error| invalid(format!("read GGUF: {error}")))?;
    let output_dir = output_dir.as_ref();
    if output_dir.exists() {
        return Err(invalid(format!(
            "output directory already exists: {}",
            output_dir.display()
        )));
    }
    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| invalid(format!("create split parent: {error}")))?;
    let stage = parent.join(format!(
        ".{}.sllm-stage-{}",
        output_dir.file_name().unwrap_or_default().to_string_lossy(),
        unique_suffix()
    ));
    fs::create_dir(&stage).map_err(|error| invalid(format!("create split stage: {error}")))?;
    let result = (|| {
        let mut parts = Vec::new();
        let mut start = 0_u64;
        let mut previous_end = 0_u64;
        let mut names = Vec::new();
        let mut index = 0_usize;
        for tensor in verified.tensors() {
            let end = tensor.absolute_range[1];
            if !names.is_empty() && end - start > max_part_bytes {
                parts.push(write_part(
                    &stage,
                    index,
                    start,
                    previous_end,
                    &bytes,
                    std::mem::take(&mut names),
                )?);
                index += 1;
                start = previous_end;
            }
            if end - start > max_part_bytes {
                return Err(invalid(
                    "one tensor exceeds max_part_bytes and cannot be split",
                ));
            }
            names.push(tensor.name.clone());
            previous_end = end;
        }
        if names.is_empty() {
            return Err(invalid("GGUF tensor catalog is empty"));
        }
        if bytes.len() as u64 - start > max_part_bytes {
            return Err(invalid("GGUF trailing bytes exceed max_part_bytes"));
        }
        parts.push(write_part(
            &stage,
            index,
            start,
            bytes.len() as u64,
            &bytes,
            names,
        )?);
        if parts.len() > MAX_PARTS {
            return Err(invalid("GGUF part count exceeds bound"));
        }
        let tensors = verified
            .tensors()
            .iter()
            .map(|tensor| SplitTensorV1 {
                name: tensor.name.clone(),
                dimensions: tensor.dimensions.clone(),
                tensor_type: tensor.tensor_type.raw(),
                absolute_range: tensor.absolute_range,
                byte_size: tensor.byte_length(),
            })
            .collect::<Vec<_>>();
        let manifest = SplitManifestV1 {
            schema_version: "sllm-gguf-split-v1".into(),
            source: SplitSourceV1 {
                path: input.display().to_string(),
                size_bytes: verified.file_size(),
                sha256: sha256_bytes(&bytes),
                metadata_sha256: verified.metadata_sha256().to_owned(),
                tensor_catalog_sha256: verified.tensor_catalog_sha256().to_owned(),
                architecture: verified.architecture().to_owned(),
            },
            tensors,
            parts: parts.clone(),
            semantic_digest: sha256_bytes(&bytes),
        };
        manifest.validate()?;
        atomic_write(&stage.join("manifest.json"), &manifest.canonical_json()?)?;
        let source_identity = ToolFileIdentityV1::from_path("source", "source.gguf", input)
            .map_err(|error| invalid(format!("tool source identity: {error}")))?;
        let mut output_paths_owned: Vec<(String, PathBuf)> = parts
            .iter()
            .map(|part| (part.path.clone(), stage.join(&part.path)))
            .collect();
        output_paths_owned.push(("manifest.json".to_owned(), stage.join("manifest.json")));
        let output_paths: Vec<(&str, &Path)> = output_paths_owned
            .iter()
            .map(|(name, path)| (name.as_str(), path.as_path()))
            .collect();
        write_tool_run_manifest(
            &stage,
            "split",
            "gguf-split",
            b"sllm-gguf-split-v1",
            vec![source_identity],
            &output_paths,
            verified.tensors().len() as u64,
        )?;
        fs::rename(&stage, output_dir)
            .map_err(|error| invalid(format!("publish split directory: {error}")))?;
        Ok(manifest)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&stage);
    }
    result
}

fn write_part(
    stage: &Path,
    index: usize,
    start: u64,
    end: u64,
    bytes: &[u8],
    tensor_names: Vec<String>,
) -> Result<SplitPartV1, ArtifactError> {
    if end <= start || end as usize > bytes.len() {
        return Err(invalid("invalid split byte range"));
    }
    let path = format!("part-{index:05}.gguf");
    let slice = &bytes[start as usize..end as usize];
    atomic_write(&stage.join(&path), slice)?;
    Ok(SplitPartV1 {
        index,
        path,
        byte_range: [start, end],
        size_bytes: slice.len() as u64,
        sha256: sha256_bytes(slice),
        tensor_names,
    })
}

pub fn read_split_manifest(path: impl AsRef<Path>) -> Result<SplitManifestV1, ArtifactError> {
    let bytes = fs::read(path).map_err(|error| invalid(format!("read split manifest: {error}")))?;
    let manifest: SplitManifestV1 = serde_json::from_slice(&bytes)
        .map_err(|error| invalid(format!("split manifest JSON: {error}")))?;
    if canonical_json(&manifest)? != bytes {
        return Err(invalid("split manifest is not canonical JSON"));
    }
    manifest.validate()?;
    Ok(manifest)
}

pub fn merge_gguf(
    manifest_path: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<String, ArtifactError> {
    let manifest_path = manifest_path.as_ref();
    let manifest = read_split_manifest(manifest_path)?;
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let output = output.as_ref();
    if output.exists() {
        return Err(invalid(format!(
            "output already exists: {}",
            output.display()
        )));
    }
    let mut bytes = Vec::with_capacity(manifest.source.size_bytes as usize);
    let mut cursor = 0_u64;
    let mut catalog_names = BTreeSet::new();
    for part in &manifest.parts {
        if part.byte_range[0] != cursor {
            return Err(invalid("split parts are out of order"));
        }
        let path = root.join(&part.path);
        let data = fs::read(&path)
            .map_err(|error| invalid(format!("read split part {}: {error}", part.path)))?;
        if data.len() as u64 != part.size_bytes || sha256_bytes(&data) != part.sha256 {
            return Err(invalid(format!("split part {} digest differs", part.path)));
        }
        for name in &part.tensor_names {
            if !catalog_names.insert(name) {
                return Err(invalid("duplicate tensor in split parts"));
            }
        }
        bytes.extend_from_slice(&data);
        cursor = part.byte_range[1];
    }
    if bytes.len() as u64 != manifest.source.size_bytes
        || sha256_bytes(&bytes) != manifest.source.sha256
    {
        return Err(invalid("merged GGUF source digest differs"));
    }
    atomic_write(output, &bytes)?;
    let verified = VerifiedGguf::open(output)
        .map_err(|error| invalid(format!("merged GGUF verification: {error}")))?;
    if verified.metadata_sha256() != manifest.source.metadata_sha256
        || verified.tensor_catalog_sha256() != manifest.source.tensor_catalog_sha256
        || verified.tensors().len() != manifest.tensors.len()
        || verified
            .tensors()
            .iter()
            .zip(&manifest.tensors)
            .any(|(actual, expected)| {
                actual.name != expected.name
                    || actual.dimensions != expected.dimensions
                    || actual.tensor_type.raw() != expected.tensor_type
                    || actual.absolute_range != expected.absolute_range
            })
    {
        let _ = fs::remove_file(output);
        return Err(invalid("merged GGUF semantic catalog differs"));
    }
    Ok(manifest.semantic_digest)
}

pub fn merge_gguf_bundle(
    manifest_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<String, ArtifactError> {
    let manifest_path = manifest_path.as_ref();
    let manifest = read_split_manifest(manifest_path)?;
    let output_dir = output_dir.as_ref();
    if output_dir.exists() {
        return Err(invalid("merge output directory already exists"));
    }
    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| invalid(format!("create merge parent: {error}")))?;
    let stage = parent.join(format!(
        ".{}.sllm-stage-{}",
        output_dir.file_name().unwrap_or_default().to_string_lossy(),
        unique_suffix()
    ));
    fs::create_dir(&stage).map_err(|error| invalid(format!("create merge stage: {error}")))?;
    let result = (|| {
        let merged_path = stage.join("model.gguf");
        let digest = merge_gguf(manifest_path, &merged_path)?;
        let report = canonical_json(&serde_json::json!({
            "schema_version": "sllm-gguf-merge-v1",
            "identity_mode": "byte-exact-and-semantic-catalog",
            "source_sha256": manifest.source.sha256,
            "semantic_digest": digest,
            "part_count": manifest.parts.len(),
            "tensor_count": manifest.tensors.len(),
        }))?;
        let report_path = stage.join("merge-report.json");
        atomic_write(&report_path, &report)?;
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let mut sources = vec![
            ToolFileIdentityV1::from_path("split-manifest", "manifest.json", manifest_path)
                .map_err(|error| invalid(format!("merge manifest identity: {error}")))?,
        ];
        for part in &manifest.parts {
            sources.push(
                ToolFileIdentityV1::from_path("split-part", &part.path, root.join(&part.path))
                    .map_err(|error| invalid(format!("merge part identity: {error}")))?,
            );
        }
        write_tool_run_manifest(
            &stage,
            "merge",
            "gguf-merge",
            b"sllm-gguf-merge-v1",
            sources,
            &[
                ("model.gguf", &merged_path),
                ("merge-report.json", &report_path),
            ],
            manifest.tensors.len() as u64,
        )?;
        fs::rename(&stage, output_dir)
            .map_err(|error| invalid(format!("publish merge bundle: {error}")))?;
        Ok(digest)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&stage);
    }
    result
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraSourceTargetV1 {
    pub tensor_name: String,
    pub target_shape: [u64; 2],
    pub rank: u64,
    pub dtype: String,
    pub a_orientation: String,
    pub b_orientation: String,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraSourceV1 {
    pub schema_version: String,
    pub artifact_id: String,
    pub base_model_fingerprint: String,
    pub base_weight_plan_digest: String,
    pub alpha: f32,
    pub targets: Vec<LoraSourceTargetV1>,
    pub provenance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LoraConversionManifestV1 {
    pub schema_version: String,
    pub source_schema_version: String,
    pub source_sha256: String,
    pub source_size_bytes: u64,
    pub base_model_fingerprint: String,
    pub base_weight_plan_digest: String,
    pub artifact_id: String,
    pub provenance: String,
    pub orientation: String,
    pub payload_sha256: String,
    pub payload_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LoraConversionResultV1 {
    pub lock_json: Vec<u8>,
    pub payload: Vec<u8>,
    pub manifest: LoraConversionManifestV1,
}

pub fn convert_lora(source: &LoraSourceV1) -> Result<LoraConversionResultV1, ArtifactError> {
    if source.schema_version != "sllm-lora-source-v1"
        || source.artifact_id.is_empty()
        || source.artifact_id.len() > 128
        || source.targets.is_empty()
        || source.targets.len() > 256
        || !source.alpha.is_finite()
        || source.alpha <= 0.0
        || !valid_sha256(&source.base_model_fingerprint)
        || !valid_sha256(&source.base_weight_plan_digest)
        || source.provenance.is_empty()
    {
        return Err(invalid("invalid LoRA source identity"));
    }
    let mut targets = source.targets.clone();
    targets.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    if targets.iter().any(|target| target.tensor_name.is_empty())
        || targets
            .windows(2)
            .any(|window| window[0].tensor_name == window[1].tensor_name)
    {
        return Err(invalid("LoRA target names must be unique and nonempty"));
    }
    let mut payload = Vec::new();
    let mut lock_targets = Vec::new();
    for target in &targets {
        let output = usize::try_from(target.target_shape[0])
            .map_err(|_| invalid("LoRA output shape overflows usize"))?;
        let input = usize::try_from(target.target_shape[1])
            .map_err(|_| invalid("LoRA input shape overflows usize"))?;
        let rank =
            usize::try_from(target.rank).map_err(|_| invalid("LoRA rank overflows usize"))?;
        if output == 0 || input == 0 || !(1..=256).contains(&target.rank) || target.dtype != "BF16"
        {
            return Err(invalid("LoRA target dtype, shape, or rank is unsupported"));
        }
        let a_count = input
            .checked_mul(rank)
            .ok_or_else(|| invalid("LoRA A size overflow"))?;
        let b_count = rank
            .checked_mul(output)
            .ok_or_else(|| invalid("LoRA B size overflow"))?;
        let a = normalize_lora_matrix(&target.a, a_count, &target.a_orientation, input, rank, "A")?;
        let b =
            normalize_lora_matrix(&target.b, b_count, &target.b_orientation, output, rank, "B")?;
        let a_offset = payload.len() as u64;
        for value in a {
            let encoded = f32_to_bf16(value);
            if !bf16_is_finite(encoded) {
                return Err(invalid("LoRA A value overflows BF16"));
            }
            payload.extend_from_slice(&encoded.to_le_bytes());
        }
        let b_offset = payload.len() as u64;
        for value in b {
            let encoded = f32_to_bf16(value);
            if !bf16_is_finite(encoded) {
                return Err(invalid("LoRA B value overflows BF16"));
            }
            payload.extend_from_slice(&encoded.to_le_bytes());
        }
        lock_targets.push(serde_json::json!({"tensor_name": target.tensor_name, "dtype":"BF16", "target_shape":[target.target_shape[0],target.target_shape[1]], "rank":target.rank, "a_offset":a_offset, "a_size":(a_count*2) as u64, "b_offset":b_offset, "b_size":(b_count*2) as u64}));
    }
    let payload_sha256 = sha256_bytes(&payload);
    let lock = serde_json::json!({"schema_version":"sllm-adapter-lock-v1", "kind":"lora", "artifact_id":source.artifact_id, "alpha":source.alpha, "base_model_fingerprint":source.base_model_fingerprint, "base_weight_plan_digest":source.base_weight_plan_digest, "payload_sha256":payload_sha256, "payload_size":payload.len() as u64, "targets":lock_targets});
    let lock_json = canonical_json(&lock)?;
    let source_json = canonical_json(source)?;
    let manifest = LoraConversionManifestV1 {
        schema_version: "sllm-lora-conversion-v1".into(),
        source_schema_version: source.schema_version.clone(),
        source_sha256: sha256_bytes(&source_json),
        source_size_bytes: source_json.len() as u64,
        base_model_fingerprint: source.base_model_fingerprint.clone(),
        base_weight_plan_digest: source.base_weight_plan_digest.clone(),
        artifact_id: source.artifact_id.clone(),
        provenance: source.provenance.clone(),
        orientation: "A=input×rank; B=rank×output; BF16 little-endian".into(),
        payload_sha256,
        payload_size: payload.len() as u64,
    };
    Ok(LoraConversionResultV1 {
        lock_json,
        payload,
        manifest,
    })
}

fn normalize_lora_matrix(
    values: &[f32],
    expected: usize,
    orientation: &str,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<Vec<f32>, ArtifactError> {
    if values.len() != expected || !values.iter().all(|value| value.is_finite()) {
        return Err(invalid(format!(
            "LoRA {label} matrix is malformed or nonfinite"
        )));
    }
    match orientation {
        "input-rank" if label == "A" => Ok(values.to_vec()),
        "rank-output" if label == "B" => Ok(values.to_vec()),
        "rank-input" if label == "A" => {
            let mut output = vec![0.0; expected];
            for row in 0..rows {
                for col in 0..cols {
                    output[row * cols + col] = values[col * rows + row];
                }
            }
            Ok(output)
        }
        "output-rank" if label == "B" => {
            let mut output = vec![0.0; expected];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = values[row * cols + col];
                }
            }
            Ok(output)
        }
        _ => Err(invalid(format!("LoRA {label} orientation is unsupported"))),
    }
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn bf16_is_finite(value: u16) -> bool {
    (value & 0x7f80) != 0x7f80
}

pub fn write_lora_bundle(
    result: &LoraConversionResultV1,
    output_dir: impl AsRef<Path>,
) -> Result<(), ArtifactError> {
    let source = ToolFileIdentityV1 {
        role: "source".to_owned(),
        logical_name: "source.json".to_owned(),
        size_bytes: result.manifest.source_size_bytes,
        sha256: result
            .manifest
            .source_sha256
            .strip_prefix("sha256:")
            .unwrap_or(&result.manifest.source_sha256)
            .to_owned(),
    };
    write_lora_bundle_inner(result, output_dir, source)
}

/// Publish a LoRA bundle while binding the run manifest to the exact source
/// bytes read by the caller (including any JSON whitespace).
pub fn write_lora_bundle_from_source(
    result: &LoraConversionResultV1,
    source_bytes: &[u8],
    output_dir: impl AsRef<Path>,
) -> Result<(), ArtifactError> {
    let source = ToolFileIdentityV1::for_bytes("source", "source.json", source_bytes)
        .map_err(|error| invalid(format!("LoRA source identity: {error}")))?;
    write_lora_bundle_inner(result, output_dir, source)
}

fn write_lora_bundle_inner(
    result: &LoraConversionResultV1,
    output_dir: impl AsRef<Path>,
    source: ToolFileIdentityV1,
) -> Result<(), ArtifactError> {
    let output_dir = output_dir.as_ref();
    if output_dir.exists() {
        return Err(invalid("LoRA output directory already exists"));
    }
    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| invalid(format!("create LoRA parent: {error}")))?;
    let stage = parent.join(format!(
        ".{}.sllm-stage-{}",
        output_dir.file_name().unwrap_or_default().to_string_lossy(),
        unique_suffix()
    ));
    fs::create_dir(&stage).map_err(|error| invalid(format!("create LoRA stage: {error}")))?;
    let write = (|| {
        atomic_write(&stage.join("adapter.lock.json"), &result.lock_json)?;
        atomic_write(&stage.join("adapter.payload"), &result.payload)?;
        atomic_write(
            &stage.join("manifest.json"),
            &canonical_json(&result.manifest)?,
        )?;
        let output_paths_owned = [
            (
                "adapter.lock.json".to_owned(),
                stage.join("adapter.lock.json"),
            ),
            ("adapter.payload".to_owned(), stage.join("adapter.payload")),
            ("manifest.json".to_owned(), stage.join("manifest.json")),
        ];
        let output_paths: Vec<(&str, &Path)> = output_paths_owned
            .iter()
            .map(|(name, path)| (name.as_str(), path.as_path()))
            .collect();
        write_tool_run_manifest(
            &stage,
            "lora-convert",
            "lora-conversion",
            b"sllm-lora-conversion-v1",
            vec![source],
            &output_paths,
            result.manifest.payload_size,
        )?;
        fs::rename(&stage, output_dir)
            .map_err(|error| invalid(format!("publish LoRA bundle: {error}")))
    })();
    if write.is_err() {
        let _ = fs::remove_dir_all(&stage);
    }
    write
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RepackManifestV1 {
    pub schema_version: String,
    pub recipe_version: String,
    pub encoding: String,
    pub rows: usize,
    pub columns: usize,
    pub logical_digest: String,
    pub physical_digest: String,
    pub bytes: usize,
    pub scale_bytes: usize,
    pub tail_policy: String,
}

pub fn repack_tensor(
    encoding: &str,
    packed_values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<(Vec<u8>, RepackManifestV1), ArtifactError> {
    if scales.iter().any(|scale| match encoding {
        "mxfp4" => *scale == 255,
        "nvfp4" => !decode_e4m3fn(*scale).is_finite(),
        _ => false,
    }) {
        return Err(invalid("repack scale plane contains a nonfinite value"));
    }
    let output = match encoding {
        "mxfp4" => repack_mxfp4_standard(packed_values, scales, rows, columns),
        "nvfp4" => repack_nvfp4_standard(packed_values, scales, rows, columns),
        _ => return Err(invalid("unsupported repack encoding")),
    }
    .map_err(|error| invalid(format!("repack: {error}")))?;
    let mut logical = Vec::with_capacity(packed_values.len() + scales.len() + 16);
    logical.extend_from_slice(&(packed_values.len() as u64).to_le_bytes());
    logical.extend_from_slice(packed_values);
    logical.extend_from_slice(&(scales.len() as u64).to_le_bytes());
    logical.extend_from_slice(scales);
    let manifest = RepackManifestV1 {
        schema_version: "sllm-repack-v1".into(),
        recipe_version: "sllm-repack-v1".into(),
        encoding: encoding.into(),
        rows,
        columns,
        logical_digest: sha256_bytes(&logical),
        physical_digest: sha256_bytes(&output),
        bytes: output.len(),
        scale_bytes: scales.len(),
        tail_policy: "reject-non-aligned-standard-block".into(),
    };
    Ok((output, manifest))
}

pub fn repack_mxfp4(
    packed_values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<(Vec<u8>, RepackManifestV1), ArtifactError> {
    repack_tensor("mxfp4", packed_values, scales, rows, columns)
}

pub fn repack_nvfp4(
    packed_values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<(Vec<u8>, RepackManifestV1), ArtifactError> {
    repack_tensor("nvfp4", packed_values, scales, rows, columns)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuantizedTensorV1 {
    pub schema_version: String,
    pub recipe: String,
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<u8>,
    pub scales: Vec<f32>,
    pub scale_bytes: Vec<u8>,
    pub tensor_scale: Option<f32>,
    pub values_sha256: String,
    pub scales_sha256: String,
}

pub fn quantize_tensor(
    recipe: &str,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedTensorV1, ArtifactError> {
    if input.len()
        != rows
            .checked_mul(columns)
            .ok_or_else(|| invalid("matrix size overflow"))?
        || input.is_empty()
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("quantization input shape or finiteness is invalid"));
    }
    let (values, scales, scale_bytes, tensor_scale) = match recipe {
        "fp8-e4m3fn-channel-f32-scale" => {
            let quantized = quantize_e4m3fn_outer_rows(input, rows, columns)
                .map_err(|error| invalid(format!("FP8 quantization: {error}")))?;
            let mut bytes = Vec::with_capacity(quantized.scales.len() * 4);
            for scale in &quantized.scales {
                bytes.extend_from_slice(&scale.to_le_bytes());
            }
            (quantized.values, quantized.scales, bytes, None)
        }
        "nvfp4-e2m1-block16-e4m3fn-f32-outer" => {
            let quantized = quantize_nvfp4_weights(input, rows, columns)
                .map_err(|error| invalid(format!("NVFP4 quantization: {error}")))?;
            let scales = quantized
                .block_scales
                .iter()
                .map(|scale| decode_e4m3fn(*scale))
                .collect();
            (
                quantized.packed_values,
                scales,
                quantized.block_scales,
                Some(quantized.tensor_scale),
            )
        }
        "mxfp4-e2m1-block32-e8m0" => quantize_mxfp4(input, rows, columns)?,
        _ => {
            return Err(invalid(format!(
                "unsupported quantization recipe: {recipe}"
            )));
        }
    };
    Ok(QuantizedTensorV1 {
        schema_version: "sllm-quantized-tensor-v1".into(),
        recipe: recipe.into(),
        rows,
        columns,
        values_sha256: sha256_bytes(&values),
        scales_sha256: sha256_bytes(&scale_bytes),
        values,
        scales,
        scale_bytes,
        tensor_scale,
    })
}

type QuantizedParts = (Vec<u8>, Vec<f32>, Vec<u8>, Option<f32>);

fn quantize_mxfp4(
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedParts, ArtifactError> {
    let mut values = vec![0_u8; input.len().div_ceil(2)];
    let mut scales = Vec::new();
    let mut scale_bytes = Vec::new();
    let blocks = columns.div_ceil(32);
    for row in 0..rows {
        for block in 0..blocks {
            let start = row * columns + block * 32;
            let end = (start + 32).min((row + 1) * columns);
            let amax = input[start..end]
                .iter()
                .fold(0.0_f32, |max, value| max.max(value.abs()));
            let exponent = if amax == 0.0 {
                127
            } else {
                ((amax / 6.0).log2().ceil() as i32 + 127).clamp(1, 254) as u8
            };
            let scale = decode_e8m0(exponent);
            scales.push(scale);
            scale_bytes.push(exponent);
            for (offset, value) in input[start..end].iter().enumerate() {
                let code = encode_e2m1(*value / scale);
                let index = start + offset;
                if index & 1 == 0 {
                    values[index / 2] = code;
                } else {
                    values[index / 2] |= code << 4;
                }
            }
        }
    }
    Ok((values, scales, scale_bytes, None))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ImatrixSummaryV1 {
    pub schema_version: String,
    pub rows: usize,
    pub columns: usize,
    pub sample_count: usize,
    pub seed: u64,
    pub sample_order_digest: String,
    pub accumulator: String,
    pub values: Vec<f64>,
    pub values_sha256: String,
}

pub fn compute_imatrix(
    input: &[f32],
    rows: usize,
    columns: usize,
    seed: u64,
) -> Result<ImatrixSummaryV1, ArtifactError> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("imatrix shape overflow"))?;
    if rows == 0
        || columns == 0
        || input.len() != elements
        || elements > MAX_MATRIX_ELEMENTS
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("imatrix input shape or finiteness is invalid"));
    }
    let mut values = vec![0.0_f64; columns];
    for row in input.chunks_exact(columns) {
        for (column, value) in row.iter().enumerate() {
            values[column] += f64::from(*value) * f64::from(*value);
        }
    }
    let mut sample_identity = Vec::with_capacity(8 + input.len() * 4);
    sample_identity.extend_from_slice(&seed.to_le_bytes());
    for value in input {
        sample_identity.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let sample_order_digest = sha256_bytes(&sample_identity);
    let bytes = canonical_json(&values)?;
    Ok(ImatrixSummaryV1 {
        schema_version: "sllm-imatrix-v1".into(),
        rows,
        columns,
        sample_count: rows,
        seed,
        sample_order_digest,
        accumulator: "f64-sum-squares-row-major-v1".into(),
        values_sha256: sha256_bytes(&bytes),
        values,
    })
}

/// Small command-line surface used by host contract tests.  All commands are
/// offline and reject unknown flags.
pub fn run_artifact_cli<I>(mut args: I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    let command = args
        .next()
        .ok_or_else(|| "command is required".to_owned())?;
    if command == "--help" || command == "-h" || command == "help" {
        return Ok("sllm-artifact commands:\n  capabilities [--architecture qwen35]\n  split --input MODEL.gguf --output-dir DIR --max-part-bytes N\n  merge --manifest PARTS/manifest.json --output-dir DIR\n  lora --input SOURCE.json --output-dir DIR\n  repack --encoding mxfp4|nvfp4 --values FILE --scales FILE --rows N --columns N --output-dir DIR\n  quantize --recipe RECIPE --input-json FILE --rows N --columns N --output-dir DIR\n  imatrix --input-json FILE --rows N --columns N --seed N --output-dir DIR".to_owned());
    }
    let mut flags = BTreeMap::<String, String>::new();
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(
                "sllm-artifact capabilities|split|merge|lora|repack|quantize|imatrix".to_owned(),
            );
        }
        if !flag.starts_with("--") {
            return Err(format!("unexpected argument {flag}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        if value.starts_with("--") {
            return Err(format!("missing value for {flag}"));
        }
        if flags.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    let get = |name: &str| flags.get(name).ok_or_else(|| format!("{name} is required"));
    match command.as_str() {
        "capabilities" => {
            ensure_flags(&flags, &["--architecture"])?;
            let capability = reviewed_capability(
                flags
                    .get("--architecture")
                    .map(String::as_str)
                    .unwrap_or("qwen35"),
            )
            .map_err(|error| error.to_string())?;
            serde_json::to_string(&capability).map_err(|error| error.to_string())
        }
        "split" => {
            ensure_flags(&flags, &["--input", "--output-dir", "--max-part-bytes"])?;
            let input = get("--input")?;
            let output = get("--output-dir")?;
            let max = get("--max-part-bytes")?
                .parse()
                .map_err(|_| "--max-part-bytes must be U64".to_owned())?;
            let manifest = split_gguf(input, output, max).map_err(|error| error.to_string())?;
            serde_json::to_string(&manifest).map_err(|error| error.to_string())
        }
        "merge" => {
            ensure_flags(&flags, &["--manifest", "--output-dir"])?;
            let manifest = get("--manifest")?;
            let output = get("--output-dir")?;
            let digest = merge_gguf_bundle(manifest, output).map_err(|error| error.to_string())?;
            Ok(serde_json::json!({"result":"PASS","semantic_digest":digest}).to_string())
        }
        "repack" => {
            ensure_flags(
                &flags,
                &[
                    "--encoding",
                    "--values",
                    "--scales",
                    "--rows",
                    "--columns",
                    "--output-dir",
                ],
            )?;
            let encoding = get("--encoding")?;
            let values = fs::read(get("--values")?).map_err(|error| error.to_string())?;
            let scales = fs::read(get("--scales")?).map_err(|error| error.to_string())?;
            let rows: usize = get("--rows")?
                .parse()
                .map_err(|_| "--rows must be usize".to_owned())?;
            let columns: usize = get("--columns")?
                .parse()
                .map_err(|_| "--columns must be usize".to_owned())?;
            let (output, manifest) = repack_tensor(encoding, &values, &scales, rows, columns)
                .map_err(|error| error.to_string())?;
            let sources = vec![
                ToolFileIdentityV1::from_path("packed-values", "values.bin", get("--values")?)
                    .map_err(|error| error.to_string())?,
                ToolFileIdentityV1::from_path("scales", "scales.bin", get("--scales")?)
                    .map_err(|error| error.to_string())?,
            ];
            let manifest_bytes = canonical_json(&manifest).map_err(|error| error.to_string())?;
            let selected = u64::try_from(
                rows.checked_mul(columns)
                    .ok_or("repack element count overflow")?,
            )
            .map_err(|_| "repack element count overflow".to_owned())?;
            publish_operation_bundle(
                Path::new(get("--output-dir")?),
                "repack",
                encoding,
                encoding.as_bytes(),
                sources,
                vec![
                    ("repacked.bin".to_owned(), output),
                    ("repack-manifest.json".to_owned(), manifest_bytes),
                ],
                selected,
            )
            .map_err(|error| error.to_string())?;
            Ok(serde_json::to_string(&manifest).map_err(|error| error.to_string())?)
        }
        "lora" => {
            ensure_flags(&flags, &["--input", "--output-dir"])?;
            let source_path = get("--input")?;
            let bytes = fs::read(source_path).map_err(|error| error.to_string())?;
            let source: LoraSourceV1 = serde_json::from_slice(&bytes)
                .map_err(|error| format!("LoRA source JSON: {error}"))?;
            let result = convert_lora(&source).map_err(|error| error.to_string())?;
            write_lora_bundle_from_source(&result, &bytes, get("--output-dir")?)
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({"result":"PASS","payload_sha256":result.manifest.payload_sha256,"payload_size":result.manifest.payload_size}).to_string())
        }
        "quantize" => {
            ensure_flags(
                &flags,
                &[
                    "--recipe",
                    "--input-json",
                    "--rows",
                    "--columns",
                    "--output-dir",
                ],
            )?;
            let values = read_float_json(get("--input-json")?)?;
            let rows: usize = get("--rows")?
                .parse()
                .map_err(|_| "--rows must be usize".to_owned())?;
            let columns: usize = get("--columns")?
                .parse()
                .map_err(|_| "--columns must be usize".to_owned())?;
            let artifact = quantize_tensor(get("--recipe")?, &values, rows, columns)
                .map_err(|error| error.to_string())?;
            let bytes = canonical_json(&artifact).map_err(|error| error.to_string())?;
            let source =
                ToolFileIdentityV1::from_path("tensor-source", "input.json", get("--input-json")?)
                    .map_err(|error| error.to_string())?;
            let selected = u64::try_from(
                rows.checked_mul(columns)
                    .ok_or("quantization element count overflow")?,
            )
            .map_err(|_| "quantization element count overflow".to_owned())?;
            publish_operation_bundle(
                Path::new(get("--output-dir")?),
                "quantize",
                get("--recipe")?,
                get("--recipe")?.as_bytes(),
                vec![source],
                vec![("quantized-tensor.json".to_owned(), bytes)],
                selected,
            )
            .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({"result":"PASS","values_sha256":artifact.values_sha256,"scales_sha256":artifact.scales_sha256}).to_string())
        }
        "imatrix" => {
            ensure_flags(
                &flags,
                &[
                    "--input-json",
                    "--rows",
                    "--columns",
                    "--seed",
                    "--output-dir",
                ],
            )?;
            let values = read_float_json(get("--input-json")?);
            let values = values?;
            let rows: usize = get("--rows")?
                .parse()
                .map_err(|_| "--rows must be usize".to_owned())?;
            let columns: usize = get("--columns")?
                .parse()
                .map_err(|_| "--columns must be usize".to_owned())?;
            let seed: u64 = get("--seed")?
                .parse()
                .map_err(|_| "--seed must be U64".to_owned())?;
            let summary =
                compute_imatrix(&values, rows, columns, seed).map_err(|error| error.to_string())?;
            let bytes = canonical_json(&summary).map_err(|error| error.to_string())?;
            let source =
                ToolFileIdentityV1::from_path("calibration", "input.json", get("--input-json")?)
                    .map_err(|error| error.to_string())?;
            publish_operation_bundle(
                Path::new(get("--output-dir")?),
                "imatrix",
                "imatrix",
                b"sllm-imatrix-v1",
                vec![source],
                vec![("imatrix.json".to_owned(), bytes)],
                summary.sample_count as u64,
            )
            .map_err(|error| error.to_string())?;
            Ok(
                serde_json::json!({"result":"PASS","values_sha256":summary.values_sha256})
                    .to_string(),
            )
        }
        _ => Err(format!("unsupported artifact command: {command}")),
    }
}

fn ensure_flags(flags: &BTreeMap<String, String>, allowed: &[&str]) -> Result<(), String> {
    if let Some(unknown) = flags.keys().find(|flag| !allowed.contains(&flag.as_str())) {
        return Err(format!("unknown argument {unknown}"));
    }
    Ok(())
}

fn read_float_json(path: &str) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("input JSON: {error}"))?;
    let array = value
        .get("values")
        .unwrap_or(&value)
        .as_array()
        .ok_or_else(|| "input JSON must be an array or {values:[...] }".to_owned())?;
    if array.len() > MAX_MATRIX_ELEMENTS {
        return Err("input matrix exceeds bound".to_owned());
    }
    array
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("input value {index} is not numeric"))?;
            if !number.is_finite() || number < f64::from(f32::MIN) || number > f64::from(f32::MAX) {
                return Err(format!("input value {index} is nonfinite or outside f32"));
            }
            Ok(number as f32)
        })
        .collect()
}
