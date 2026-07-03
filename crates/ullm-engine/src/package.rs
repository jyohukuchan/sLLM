// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageSummary {
    pub package_dir: PathBuf,
    pub schema_version: Option<String>,
    pub source_model_dir: Option<String>,
    pub quantized_tensors: usize,
    pub passthrough_tensors: usize,
    pub codebooks: usize,
    pub quantized_elements: u64,
    pub passthrough_elements: u64,
    pub referenced_files: usize,
    pub referenced_file_bytes: u64,
    pub missing_referenced_files: usize,
    pub declared_passthrough_payload_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferencedFile {
    pub relative_path: String,
    pub absolute_path: PathBuf,
    pub bytes: u64,
    pub role: ReferencedFileRole,
    pub owner_index: Option<usize>,
    pub owner_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorPayloadBundle {
    pub tensor_index: usize,
    pub tensor_name: String,
    pub dtype: Option<String>,
    pub shape: Vec<u64>,
    pub family: Option<String>,
    pub candidate_id: Option<String>,
    pub scale_format: Option<String>,
    pub group_size: Option<usize>,
    pub tensor_scale: Option<f32>,
    pub index_encoding: Option<String>,
    pub scale_encoding: Option<String>,
    pub elements: u64,
    pub groups: u64,
    pub index_file: ReferencedFile,
    pub scale_file: ReferencedFile,
    pub codebook_file: ReferencedFile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorSelector {
    First,
    Index(usize),
    Name(String),
}

impl TensorSelector {
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            None | Some("") => Self::First,
            Some(value) => value
                .parse::<usize>()
                .map(Self::Index)
                .unwrap_or_else(|_| Self::Name(value.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferencedFileRole {
    Smallest,
    TensorIndex,
    TensorScale,
    TensorCodebook,
    Codebook,
    Passthrough,
}

impl ReferencedFileRole {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "smallest" => Some(Self::Smallest),
            "tensor-index" | "index" | "idx4" => Some(Self::TensorIndex),
            "tensor-scale" | "scale" | "scale-u8" => Some(Self::TensorScale),
            "tensor-codebook" => Some(Self::TensorCodebook),
            "codebook" => Some(Self::Codebook),
            "passthrough" | "raw" => Some(Self::Passthrough),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smallest => "smallest",
            Self::TensorIndex => "tensor-index",
            Self::TensorScale => "tensor-scale",
            Self::TensorCodebook => "tensor-codebook",
            Self::Codebook => "codebook",
            Self::Passthrough => "passthrough",
        }
    }
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: Option<String>,
    source_model_dir: Option<String>,
    #[serde(default)]
    tensors: Vec<QuantizedTensor>,
    #[serde(default)]
    passthrough_tensors: Vec<PassthroughTensor>,
    #[serde(default)]
    codebooks: Vec<Codebook>,
}

#[derive(Debug, Deserialize)]
struct QuantizedTensor {
    name: Option<String>,
    dtype: Option<String>,
    #[serde(default)]
    shape: Vec<u64>,
    family: Option<String>,
    candidate_id: Option<String>,
    scale_format: Option<String>,
    group_size: Option<usize>,
    tensor_scale: Option<f32>,
    index_encoding: Option<String>,
    scale_encoding: Option<String>,
    #[serde(default)]
    elements: u64,
    #[serde(default)]
    groups: u64,
    index_file: Option<String>,
    scale_file: Option<String>,
    codebook_file: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PassthroughTensor {
    name: Option<String>,
    #[serde(default)]
    elements: u64,
    #[serde(default)]
    payload_bytes: u64,
    payload_file: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Codebook {
    family: Option<String>,
    candidate_id: Option<String>,
    file: Option<String>,
}

pub fn inspect_package(path: impl AsRef<Path>) -> Result<PackageSummary, String> {
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    let referenced = referenced_paths(&manifest);

    let mut referenced_file_bytes = 0_u64;
    let mut missing_referenced_files = 0_usize;
    for relative in &referenced {
        let path = package_dir.join(relative);
        match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => {
                referenced_file_bytes = referenced_file_bytes.saturating_add(metadata.len());
            }
            _ => {
                missing_referenced_files += 1;
            }
        }
    }

    Ok(PackageSummary {
        package_dir: package_dir.to_path_buf(),
        schema_version: manifest.schema_version,
        source_model_dir: manifest.source_model_dir,
        quantized_tensors: manifest.tensors.len(),
        passthrough_tensors: manifest.passthrough_tensors.len(),
        codebooks: manifest.codebooks.len(),
        quantized_elements: manifest.tensors.iter().map(|tensor| tensor.elements).sum(),
        passthrough_elements: manifest
            .passthrough_tensors
            .iter()
            .map(|tensor| tensor.elements)
            .sum(),
        referenced_files: referenced.len(),
        referenced_file_bytes,
        missing_referenced_files,
        declared_passthrough_payload_bytes: manifest
            .passthrough_tensors
            .iter()
            .map(|tensor| tensor.payload_bytes)
            .sum(),
    })
}

pub fn select_smallest_existing_referenced_file(
    path: impl AsRef<Path>,
) -> Result<ReferencedFile, String> {
    select_existing_referenced_file(path, ReferencedFileRole::Smallest)
}

pub fn select_existing_referenced_file(
    path: impl AsRef<Path>,
    role: ReferencedFileRole,
) -> Result<ReferencedFile, String> {
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    match role {
        ReferencedFileRole::Smallest => select_referenced_file_from_candidates(
            package_dir,
            role,
            referenced_file_candidates(package_dir, &manifest),
        ),
        _ => select_referenced_file_from_candidates(
            package_dir,
            role,
            referenced_file_candidates(package_dir, &manifest)
                .into_iter()
                .filter(|candidate| candidate.role == role),
        ),
    }
}

pub fn select_tensor_payload_bundle(
    path: impl AsRef<Path>,
    selector: &TensorSelector,
) -> Result<TensorPayloadBundle, String> {
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    let tensor_index = select_tensor_index(&manifest, selector)?;
    let tensor = manifest
        .tensors
        .get(tensor_index)
        .ok_or_else(|| format!("tensor index {tensor_index} is out of range"))?;
    tensor_payload_bundle_from_manifest(package_dir, tensor_index, tensor)
}

pub fn list_tensor_payload_bundles(
    path: impl AsRef<Path>,
) -> Result<Vec<TensorPayloadBundle>, String> {
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    let mut bundles = Vec::with_capacity(manifest.tensors.len());
    for (tensor_index, tensor) in manifest.tensors.iter().enumerate() {
        bundles.push(tensor_payload_bundle_from_manifest(
            package_dir,
            tensor_index,
            tensor,
        )?);
    }
    Ok(bundles)
}

fn tensor_payload_bundle_from_manifest(
    package_dir: &Path,
    tensor_index: usize,
    tensor: &QuantizedTensor,
) -> Result<TensorPayloadBundle, String> {
    let tensor_name = tensor
        .name
        .clone()
        .unwrap_or_else(|| format!("tensor#{tensor_index}"));

    let index_file = required_tensor_file(
        package_dir,
        tensor,
        tensor_index,
        &tensor_name,
        ReferencedFileRole::TensorIndex,
        tensor.index_file.as_deref(),
    )?;
    let scale_file = required_tensor_file(
        package_dir,
        tensor,
        tensor_index,
        &tensor_name,
        ReferencedFileRole::TensorScale,
        tensor.scale_file.as_deref(),
    )?;
    let codebook_file = required_tensor_file(
        package_dir,
        tensor,
        tensor_index,
        &tensor_name,
        ReferencedFileRole::TensorCodebook,
        tensor.codebook_file.as_deref(),
    )?;

    Ok(TensorPayloadBundle {
        tensor_index,
        tensor_name,
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
        family: tensor.family.clone(),
        candidate_id: tensor.candidate_id.clone(),
        scale_format: tensor.scale_format.clone(),
        group_size: tensor.group_size,
        tensor_scale: tensor.tensor_scale,
        index_encoding: tensor.index_encoding.clone(),
        scale_encoding: tensor.scale_encoding.clone(),
        elements: tensor.elements,
        groups: tensor.groups,
        index_file,
        scale_file,
        codebook_file,
    })
}

fn select_tensor_index(manifest: &Manifest, selector: &TensorSelector) -> Result<usize, String> {
    match selector {
        TensorSelector::First => {
            if manifest.tensors.is_empty() {
                Err("package contains no quantized tensors".to_string())
            } else {
                Ok(0)
            }
        }
        TensorSelector::Index(index) => {
            if *index < manifest.tensors.len() {
                Ok(*index)
            } else {
                Err(format!(
                    "tensor index {index} is out of range for {} quantized tensors",
                    manifest.tensors.len()
                ))
            }
        }
        TensorSelector::Name(name) => select_tensor_index_by_name(manifest, name),
    }
}

fn select_tensor_index_by_name(manifest: &Manifest, name: &str) -> Result<usize, String> {
    if let Some((index, _)) = manifest
        .tensors
        .iter()
        .enumerate()
        .find(|(_, tensor)| tensor.name.as_deref() == Some(name))
    {
        return Ok(index);
    }

    let matches: Vec<usize> = manifest
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(index, tensor)| {
            tensor
                .name
                .as_ref()
                .filter(|tensor_name| tensor_name.contains(name))
                .map(|_| index)
        })
        .collect();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(format!("no quantized tensor matched selector \"{name}\"")),
        _ => Err(format!(
            "tensor selector \"{name}\" matched {} tensors; use an exact name or numeric index",
            matches.len()
        )),
    }
}

fn required_tensor_file(
    package_dir: &Path,
    _tensor: &QuantizedTensor,
    tensor_index: usize,
    tensor_name: &str,
    role: ReferencedFileRole,
    relative: Option<&str>,
) -> Result<ReferencedFile, String> {
    let relative = relative.ok_or_else(|| {
        format!(
            "tensor {tensor_index} ({tensor_name}) does not declare {}",
            role.as_str()
        )
    })?;
    referenced_file_candidate(
        package_dir,
        relative,
        role,
        Some(tensor_index),
        Some(tensor_name.to_string()),
    )
    .ok_or_else(|| {
        format!(
            "tensor {tensor_index} ({tensor_name}) does not reference an existing non-empty {} file",
            role.as_str()
        )
    })
}

fn select_referenced_file_from_candidates(
    package_dir: &Path,
    role: ReferencedFileRole,
    candidates: impl IntoIterator<Item = ReferencedFile>,
) -> Result<ReferencedFile, String> {
    let mut selected: Option<ReferencedFile> = None;
    for candidate in candidates {
        if selected.as_ref().is_none_or(|current| {
            candidate.bytes < current.bytes
                || (candidate.bytes == current.bytes
                    && candidate.relative_path.as_str() < current.relative_path.as_str())
        }) {
            selected = Some(candidate);
        }
    }
    selected.ok_or_else(|| missing_role_error(package_dir, role))
}

fn referenced_file_candidates(package_dir: &Path, manifest: &Manifest) -> Vec<ReferencedFile> {
    let mut candidates = Vec::new();
    for (index, tensor) in manifest.tensors.iter().enumerate() {
        for (role, relative) in [
            (
                ReferencedFileRole::TensorIndex,
                tensor.index_file.as_deref(),
            ),
            (
                ReferencedFileRole::TensorScale,
                tensor.scale_file.as_deref(),
            ),
            (
                ReferencedFileRole::TensorCodebook,
                tensor.codebook_file.as_deref(),
            ),
        ] {
            if let Some(relative) = relative {
                if let Some(candidate) = referenced_file_candidate(
                    package_dir,
                    relative,
                    role,
                    Some(index),
                    tensor.name.clone(),
                ) {
                    candidates.push(candidate);
                }
            }
        }
    }
    for (index, tensor) in manifest.passthrough_tensors.iter().enumerate() {
        if let Some(relative) = tensor.payload_file.as_deref() {
            if let Some(candidate) = referenced_file_candidate(
                package_dir,
                relative,
                ReferencedFileRole::Passthrough,
                Some(index),
                tensor.name.clone(),
            ) {
                candidates.push(candidate);
            }
        }
    }
    for (index, codebook) in manifest.codebooks.iter().enumerate() {
        if let Some(relative) = codebook.file.as_deref() {
            let owner_name = match (&codebook.family, &codebook.candidate_id) {
                (Some(family), Some(candidate)) => Some(format!("{family}:{candidate}")),
                (Some(family), None) => Some(family.clone()),
                (None, Some(candidate)) => Some(candidate.clone()),
                (None, None) => None,
            };
            if let Some(candidate) = referenced_file_candidate(
                package_dir,
                relative,
                ReferencedFileRole::Codebook,
                Some(index),
                owner_name,
            ) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

fn referenced_file_candidate(
    package_dir: &Path,
    relative: &str,
    role: ReferencedFileRole,
    owner_index: Option<usize>,
    owner_name: Option<String>,
) -> Option<ReferencedFile> {
    if relative.is_empty() {
        return None;
    }
    let absolute = package_dir.join(relative);
    let Ok(metadata) = fs::metadata(&absolute) else {
        return None;
    };
    if !metadata.is_file() || metadata.len() == 0 {
        return None;
    }
    Some(ReferencedFile {
        relative_path: relative.to_string(),
        absolute_path: absolute,
        bytes: metadata.len(),
        role,
        owner_index,
        owner_name,
    })
}

fn missing_role_error(package_dir: &Path, role: ReferencedFileRole) -> String {
    format!(
        "package {} does not reference any existing non-empty {} payload file",
        package_dir.display(),
        role.as_str()
    )
}

fn read_manifest(package_dir: &Path) -> Result<Manifest, String> {
    let manifest_path = package_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    serde_json::from_str(&manifest_text)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))
}

fn referenced_paths(manifest: &Manifest) -> BTreeSet<String> {
    let mut referenced = BTreeSet::new();
    for tensor in &manifest.tensors {
        insert_optional_path(&mut referenced, &tensor.index_file);
        insert_optional_path(&mut referenced, &tensor.scale_file);
        insert_optional_path(&mut referenced, &tensor.codebook_file);
    }
    for tensor in &manifest.passthrough_tensors {
        insert_optional_path(&mut referenced, &tensor.payload_file);
    }
    for codebook in &manifest.codebooks {
        insert_optional_path(&mut referenced, &codebook.file);
    }
    referenced
}

fn insert_optional_path(paths: &mut BTreeSet<String>, value: &Option<String>) {
    if let Some(value) = value {
        if !value.is_empty() {
            paths.insert(value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn inspects_minimal_package_without_loading_payloads() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-package-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [1_u8, 2, 3]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [4_u8, 5]).unwrap();
        fs::write(root.join("tensors/b.raw"), [6_u8, 7, 8, 9]).unwrap();
        fs::write(root.join("codebooks/c.f32"), [0_u8; 8]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "schema_version": "test",
              "source_model_dir": "/model",
              "tensors": [{
                "name": "tensor-a",
                "elements": 6,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/c.f32"
              }],
              "passthrough_tensors": [{
                "name": "tensor-b",
                "elements": 4,
                "payload_bytes": 4,
                "payload_file": "tensors/b.raw"
              }],
              "codebooks": [{
                "family": "test-family",
                "candidate_id": "test-candidate",
                "file": "codebooks/c.f32"
              }]
            }"#,
        )
        .unwrap();

        let summary = inspect_package(&root).unwrap();
        assert_eq!(summary.schema_version.as_deref(), Some("test"));
        assert_eq!(summary.quantized_tensors, 1);
        assert_eq!(summary.passthrough_tensors, 1);
        assert_eq!(summary.codebooks, 1);
        assert_eq!(summary.quantized_elements, 6);
        assert_eq!(summary.passthrough_elements, 4);
        assert_eq!(summary.referenced_files, 4);
        assert_eq!(summary.referenced_file_bytes, 17);
        assert_eq!(summary.missing_referenced_files, 0);
        assert_eq!(summary.declared_passthrough_payload_bytes, 4);

        let selected = select_smallest_existing_referenced_file(&root).unwrap();
        assert_eq!(selected.relative_path, "tensors/a.scale_u8");
        assert_eq!(selected.absolute_path, root.join("tensors/a.scale_u8"));
        assert_eq!(selected.bytes, 2);
        assert_eq!(selected.role, ReferencedFileRole::TensorScale);

        let index =
            select_existing_referenced_file(&root, ReferencedFileRole::TensorIndex).unwrap();
        assert_eq!(index.relative_path, "tensors/a.idx4");
        assert_eq!(index.role, ReferencedFileRole::TensorIndex);
        assert_eq!(index.owner_index, Some(0));
        assert_eq!(index.owner_name.as_deref(), Some("tensor-a"));

        let scale =
            select_existing_referenced_file(&root, ReferencedFileRole::TensorScale).unwrap();
        assert_eq!(scale.relative_path, "tensors/a.scale_u8");
        assert_eq!(scale.role, ReferencedFileRole::TensorScale);

        let tensor_codebook =
            select_existing_referenced_file(&root, ReferencedFileRole::TensorCodebook).unwrap();
        assert_eq!(tensor_codebook.relative_path, "codebooks/c.f32");
        assert_eq!(tensor_codebook.role, ReferencedFileRole::TensorCodebook);
        assert_eq!(tensor_codebook.owner_name.as_deref(), Some("tensor-a"));

        let codebook =
            select_existing_referenced_file(&root, ReferencedFileRole::Codebook).unwrap();
        assert_eq!(codebook.relative_path, "codebooks/c.f32");
        assert_eq!(codebook.role, ReferencedFileRole::Codebook);
        assert_eq!(
            codebook.owner_name.as_deref(),
            Some("test-family:test-candidate")
        );

        let passthrough =
            select_existing_referenced_file(&root, ReferencedFileRole::Passthrough).unwrap();
        assert_eq!(passthrough.relative_path, "tensors/b.raw");
        assert_eq!(passthrough.role, ReferencedFileRole::Passthrough);
        assert_eq!(passthrough.owner_name.as_deref(), Some("tensor-b"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn smallest_referenced_file_ignores_missing_and_empty_files() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-package-select-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::write(root.join("tensors/empty.idx4"), []).unwrap();
        fs::write(root.join("tensors/payload.raw"), [1_u8, 2, 3, 4, 5]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "index_file": "tensors/empty.idx4",
                "scale_file": "tensors/missing.scale_u8"
              }],
              "passthrough_tensors": [{
                "payload_file": "tensors/payload.raw"
              }]
            }"#,
        )
        .unwrap();

        let selected = select_smallest_existing_referenced_file(&root).unwrap();
        assert_eq!(selected.relative_path, "tensors/payload.raw");
        assert_eq!(selected.bytes, 5);
        let err =
            select_existing_referenced_file(&root, ReferencedFileRole::TensorIndex).unwrap_err();
        assert!(err.contains("tensor-index"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn referenced_file_role_parser_accepts_cli_tokens() {
        assert_eq!(
            ReferencedFileRole::parse("smallest"),
            Some(ReferencedFileRole::Smallest)
        );
        assert_eq!(
            ReferencedFileRole::parse("idx4"),
            Some(ReferencedFileRole::TensorIndex)
        );
        assert_eq!(
            ReferencedFileRole::parse("scale-u8"),
            Some(ReferencedFileRole::TensorScale)
        );
        assert_eq!(
            ReferencedFileRole::parse("tensor-codebook"),
            Some(ReferencedFileRole::TensorCodebook)
        );
        assert_eq!(
            ReferencedFileRole::parse("raw"),
            Some(ReferencedFileRole::Passthrough)
        );
        assert_eq!(ReferencedFileRole::parse("unknown"), None);
    }

    #[test]
    fn selects_tensor_payload_bundle_by_index_and_name() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-tensor-bundle-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [1_u8, 2, 3, 4]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [5_u8, 6]).unwrap();
        fs::write(root.join("codebooks/a.f32"), [7_u8; 64]).unwrap();
        fs::write(root.join("tensors/b.idx4"), [8_u8, 9, 10, 11]).unwrap();
        fs::write(root.join("tensors/b.scale_u8"), [12_u8, 13]).unwrap();
        fs::write(root.join("codebooks/b.f32"), [14_u8; 64]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "shape": [2, 4],
                "family": "attn_q",
                "candidate_id": "aq4_test",
                "scale_format": "e4m3",
                "group_size": 4,
                "tensor_scale": 1.25,
                "index_encoding": "idx4_low_nibble_first",
                "scale_encoding": "u8_scale_table_index",
                "elements": 8,
                "groups": 2,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }, {
                "name": "layer.0.attn.k_proj.weight",
                "dtype": "BF16",
                "family": "attn_k",
                "candidate_id": "aq4_test",
                "elements": 8,
                "groups": 2,
                "index_file": "tensors/b.idx4",
                "scale_file": "tensors/b.scale_u8",
                "codebook_file": "codebooks/b.f32"
              }]
            }"#,
        )
        .unwrap();

        let first = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap();
        assert_eq!(first.tensor_index, 0);
        assert_eq!(first.tensor_name, "layer.0.attn.q_proj.weight");
        assert_eq!(first.dtype.as_deref(), Some("BF16"));
        assert_eq!(first.shape, vec![2, 4]);
        assert_eq!(first.family.as_deref(), Some("attn_q"));
        assert_eq!(first.candidate_id.as_deref(), Some("aq4_test"));
        assert_eq!(first.scale_format.as_deref(), Some("e4m3"));
        assert_eq!(first.group_size, Some(4));
        assert_eq!(first.tensor_scale, Some(1.25));
        assert_eq!(
            first.index_encoding.as_deref(),
            Some("idx4_low_nibble_first")
        );
        assert_eq!(
            first.scale_encoding.as_deref(),
            Some("u8_scale_table_index")
        );
        assert_eq!(first.elements, 8);
        assert_eq!(first.groups, 2);
        assert_eq!(first.index_file.relative_path, "tensors/a.idx4");
        assert_eq!(first.scale_file.relative_path, "tensors/a.scale_u8");
        assert_eq!(first.codebook_file.relative_path, "codebooks/a.f32");

        let by_index = select_tensor_payload_bundle(&root, &TensorSelector::Index(1)).unwrap();
        assert_eq!(by_index.tensor_index, 1);
        assert_eq!(by_index.tensor_name, "layer.0.attn.k_proj.weight");
        assert!(by_index.shape.is_empty());
        assert_eq!(by_index.scale_format, None);
        assert_eq!(by_index.group_size, None);
        assert_eq!(by_index.tensor_scale, None);

        let by_exact = select_tensor_payload_bundle(
            &root,
            &TensorSelector::Name("layer.0.attn.k_proj.weight".to_string()),
        )
        .unwrap();
        assert_eq!(by_exact.tensor_index, 1);

        let by_unique_substring =
            select_tensor_payload_bundle(&root, &TensorSelector::Name("q_proj".to_string()))
                .unwrap();
        assert_eq!(by_unique_substring.tensor_index, 0);

        let ambiguous =
            select_tensor_payload_bundle(&root, &TensorSelector::Name("layer.0".to_string()))
                .unwrap_err();
        assert!(ambiguous.contains("matched 2 tensors"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lists_tensor_payload_bundles_in_manifest_order() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-tensor-bundle-list-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [1_u8]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [2_u8]).unwrap();
        fs::write(root.join("codebooks/a.f32"), [3_u8]).unwrap();
        fs::write(root.join("tensors/b.idx4"), [4_u8, 5]).unwrap();
        fs::write(root.join("tensors/b.scale_u8"), [6_u8]).unwrap();
        fs::write(root.join("codebooks/b.f32"), [7_u8]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "tensor-a",
                "shape": [1, 2],
                "family": "family-a",
                "candidate_id": "candidate-a",
                "scale_format": "e4m3",
                "group_size": 2,
                "tensor_scale": 0.5,
                "elements": 2,
                "groups": 1,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }, {
                "name": "tensor-b",
                "family": "family-b",
                "candidate_id": "candidate-b",
                "elements": 4,
                "groups": 1,
                "index_file": "tensors/b.idx4",
                "scale_file": "tensors/b.scale_u8",
                "codebook_file": "codebooks/b.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundles = list_tensor_payload_bundles(&root).unwrap();
        assert_eq!(bundles.len(), 2);
        assert_eq!(bundles[0].tensor_index, 0);
        assert_eq!(bundles[0].tensor_name, "tensor-a");
        assert_eq!(bundles[0].shape, vec![1, 2]);
        assert_eq!(bundles[0].family.as_deref(), Some("family-a"));
        assert_eq!(bundles[0].candidate_id.as_deref(), Some("candidate-a"));
        assert_eq!(bundles[0].scale_format.as_deref(), Some("e4m3"));
        assert_eq!(bundles[0].group_size, Some(2));
        assert_eq!(bundles[0].tensor_scale, Some(0.5));
        assert_eq!(bundles[0].index_file.relative_path, "tensors/a.idx4");
        assert_eq!(bundles[1].tensor_index, 1);
        assert_eq!(bundles[1].tensor_name, "tensor-b");
        assert_eq!(bundles[1].family.as_deref(), Some("family-b"));
        assert_eq!(bundles[1].candidate_id.as_deref(), Some("candidate-b"));
        assert_eq!(bundles[1].index_file.relative_path, "tensors/b.idx4");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tensor_selector_parser_accepts_index_and_name() {
        assert_eq!(TensorSelector::parse(None), TensorSelector::First);
        assert_eq!(TensorSelector::parse(Some("")), TensorSelector::First);
        assert_eq!(TensorSelector::parse(Some("12")), TensorSelector::Index(12));
        assert_eq!(
            TensorSelector::parse(Some("layer.0.attn.q_proj.weight")),
            TensorSelector::Name("layer.0.attn.q_proj.weight".to_string())
        );
    }

    #[test]
    fn tensor_payload_bundle_rejects_missing_payload_file() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-tensor-bundle-missing-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [1_u8, 2]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "bad",
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/missing.scale_u8",
                "codebook_file": "codebooks/missing.f32"
              }]
            }"#,
        )
        .unwrap();

        let err = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap_err();
        assert!(err.contains("tensor-scale"));

        fs::remove_dir_all(root).unwrap();
    }
}
