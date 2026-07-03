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
    #[serde(default)]
    elements: u64,
    index_file: Option<String>,
    scale_file: Option<String>,
    codebook_file: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PassthroughTensor {
    #[serde(default)]
    elements: u64,
    #[serde(default)]
    payload_bytes: u64,
    payload_file: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Codebook {
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
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    let referenced = referenced_paths(&manifest);
    let mut selected: Option<ReferencedFile> = None;
    for relative in referenced {
        let absolute = package_dir.join(&relative);
        let Ok(metadata) = fs::metadata(&absolute) else {
            continue;
        };
        if !metadata.is_file() || metadata.len() == 0 {
            continue;
        }
        let candidate = ReferencedFile {
            relative_path: relative,
            absolute_path: absolute,
            bytes: metadata.len(),
        };
        if selected
            .as_ref()
            .is_none_or(|current| candidate.bytes < current.bytes)
        {
            selected = Some(candidate);
        }
    }
    selected.ok_or_else(|| {
        format!(
            "package {} does not reference any existing non-empty payload file",
            package_dir.display()
        )
    })
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
                "elements": 6,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/c.f32"
              }],
              "passthrough_tensors": [{
                "elements": 4,
                "payload_bytes": 4,
                "payload_file": "tensors/b.raw"
              }],
              "codebooks": [{
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

        fs::remove_dir_all(root).unwrap();
    }
}
