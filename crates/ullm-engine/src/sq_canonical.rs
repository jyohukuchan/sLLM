// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::format_id::FORMAT_SQ8_0;
use crate::qwen3_names::qwen3_tensor_name_alias;
use crate::sq::fp8_e4m3fn_to_f32;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

pub const SQ8_CANONICAL_ARTIFACT_SCHEMA_VERSION: &str = "sq-fp8-artifact-v0.2";
pub const SQ8_CANONICAL_ARTIFACT_KIND: &str = "canonical";
pub const SQ8_CANONICAL_IMPORT_MODE: &str = "fp8_checkpoint";
pub const SQ8_CANONICAL_RAW_ENCODING: &str = "raw_safetensors_payload";
pub const SQ8_CANONICAL_WEIGHT_DTYPE: &str = "F8_E4M3";
pub const SQ8_CANONICAL_SCALE_DTYPE: &str = "BF16";
pub const SQ8_CANONICAL_SCALE_LAYOUT: &str = "block_2d";
pub const SQ8_CANONICAL_SCALE_ORDER: &str = "row_major";
pub const SQ8_CANONICAL_SCALE_SEMANTIC: &str = "dequant_multiplier";
pub const SQ8_CANONICAL_BLOCK_SHAPE: [u64; 2] = [128, 128];
pub const SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES: usize = 64 * 1024 * 1024;
pub const SQ8_CANONICAL_MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
pub const SQ8_CANONICAL_MAX_RECONSTRUCT_ROW_ELEMENTS: usize = 1024 * 1024;
pub const SQ8_CANONICAL_MAX_SCALE_READ_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalArtifactManifest {
    pub schema_version: String,
    pub artifact_kind: String,
    pub format_id: String,
    pub source: Sq8CanonicalSource,
    #[serde(rename = "import")]
    pub import_contract: Sq8CanonicalImport,
    pub integrity: Sq8CanonicalIntegrity,
    pub coverage: Sq8CanonicalCoverage,
    pub storage: Sq8CanonicalStorage,
    #[serde(default)]
    pub quantized_tensors: Vec<Sq8CanonicalTensorPair>,
    #[serde(default)]
    pub passthrough_tensors: Vec<Sq8CanonicalPassthroughTensor>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalSource {
    pub model_name: String,
    pub config_file: String,
    pub config_sha256: String,
    pub index_file: Option<String>,
    pub index_sha256: Option<String>,
    pub quantization: Sq8CanonicalSourceQuantization,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalSourceQuantization {
    pub quant_method: String,
    pub format: String,
    pub activation_scheme: String,
    pub weight_block_shape: [u64; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalImport {
    pub mode: String,
    pub encoding: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalIntegrity {
    pub content_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalCoverage {
    pub scope: String,
    pub source_tensor_count: u64,
    pub source_fp8_weight_count: u64,
    pub source_scale_count: u64,
    pub paired_tensor_count: u64,
    pub selected_pair_count: u64,
    pub unpaired_tensor_count: u64,
    pub passthrough_tensor_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalStorage {
    pub weight_payload_bytes: u64,
    pub scale_payload_bytes: u64,
    pub total_payload_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalTensorPair {
    pub name: String,
    pub family: String,
    pub shape: [u64; 2],
    pub elements: u64,
    pub weight: Sq8CanonicalWeightPayload,
    pub scale: Sq8CanonicalBlockScalePayload,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalWeightPayload {
    pub dtype: String,
    pub encoding: String,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    pub source_file: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalBlockScalePayload {
    pub name: String,
    pub dtype: String,
    pub encoding: String,
    pub file: String,
    pub shape: [u64; 2],
    pub elements: u64,
    pub bytes: u64,
    pub sha256: String,
    pub source_file: String,
    pub layout: String,
    pub block_shape: [u64; 2],
    pub order: String,
    pub semantic: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8CanonicalPassthroughTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub source_file: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Sq8CanonicalArtifact {
    artifact_dir: PathBuf,
    manifest: Sq8CanonicalArtifactManifest,
    verified: Sq8CanonicalVerifiedState,
}

#[derive(Debug, Clone)]
struct Sq8CanonicalVerifiedState {
    checksum_report: Sq8CanonicalChecksumReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sq8CanonicalChecksumReport {
    pub selected_pair_count: u64,
    pub weight_payload_bytes: u64,
    pub scale_payload_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8CanonicalTensorPayloadPaths {
    pub weight: PathBuf,
    pub scale: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8CanonicalTensorChecksumReport {
    pub tensor_name: String,
    pub weight_payload_bytes: u64,
    pub scale_payload_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8CanonicalReconstructedBlock {
    pub tensor_name: String,
    pub block_row: usize,
    pub block_col: usize,
    pub start_row: usize,
    pub start_col: usize,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

impl Sq8CanonicalArtifact {
    pub fn artifact_dir(&self) -> &Path {
        &self.artifact_dir
    }

    pub fn manifest(&self) -> &Sq8CanonicalArtifactManifest {
        &self.manifest
    }

    pub fn checksum_report(&self) -> Sq8CanonicalChecksumReport {
        self.verified.checksum_report
    }

    pub fn tensor_pair(&self, tensor_name: &str) -> Result<&Sq8CanonicalTensorPair, String> {
        find_tensor_pair(self, tensor_name)
    }

    pub fn tensor_payload_paths(
        &self,
        tensor_name: &str,
    ) -> Result<Sq8CanonicalTensorPayloadPaths, String> {
        let pair = find_tensor_pair(self, tensor_name)?;
        Ok(Sq8CanonicalTensorPayloadPaths {
            weight: canonical_artifact_file(
                &self.artifact_dir,
                &pair.weight.file,
                &format!("{} weight", pair.name),
            )?,
            scale: canonical_artifact_file(
                &self.artifact_dir,
                &pair.scale.file,
                &format!("{} scale", pair.name),
            )?,
        })
    }

    pub fn verify_tensor_payloads(
        &self,
        tensor_name: &str,
        chunk_bytes: usize,
    ) -> Result<Sq8CanonicalTensorChecksumReport, String> {
        if chunk_bytes == 0 {
            return Err("SQ8 canonical checksum chunk_bytes must be greater than zero".to_string());
        }
        let pair = find_tensor_pair(self, tensor_name)?;
        verify_sq8_canonical_tensor_pair_payloads(
            &self.artifact_dir,
            pair,
            chunk_bytes.min(SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES),
        )?;
        Ok(Sq8CanonicalTensorChecksumReport {
            tensor_name: pair.name.clone(),
            weight_payload_bytes: pair.weight.bytes,
            scale_payload_bytes: pair.scale.bytes,
        })
    }

    pub fn read_tensor_scales_f32(
        &self,
        tensor_name: &str,
        chunk_bytes: usize,
    ) -> Result<Vec<f32>, String> {
        if chunk_bytes == 0 {
            return Err("SQ8 canonical scale chunk_bytes must be greater than zero".to_string());
        }
        let pair = find_tensor_pair(self, tensor_name)?;
        if pair.scale.bytes > SQ8_CANONICAL_MAX_SCALE_READ_BYTES {
            return Err(format!(
                "SQ8 canonical tensor {} scale payload has {} bytes, exceeding compact read limit {SQ8_CANONICAL_MAX_SCALE_READ_BYTES}",
                pair.name, pair.scale.bytes
            ));
        }
        let scale_path = canonical_artifact_file(
            &self.artifact_dir,
            &pair.scale.file,
            &format!("{} scale", pair.name),
        )?;
        read_verified_positive_bf16_payload(
            &scale_path,
            pair.scale.bytes,
            &pair.scale.sha256,
            chunk_bytes.min(SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES),
            &format!("{} scale", pair.name),
        )
    }
}

pub fn read_sq8_canonical_artifact(path: impl AsRef<Path>) -> Result<Sq8CanonicalArtifact, String> {
    let artifact_dir = path.as_ref();
    let manifest_path = artifact_dir.join("sq_manifest.json");
    let manifest_bytes = std::fs::metadata(&manifest_path)
        .map_err(|err| format!("failed to stat {}: {err}", manifest_path.display()))?
        .len();
    if manifest_bytes > SQ8_CANONICAL_MAX_MANIFEST_BYTES {
        return Err(format!(
            "SQ8 canonical manifest {} is too large: {manifest_bytes} bytes exceeds {}",
            manifest_path.display(),
            SQ8_CANONICAL_MAX_MANIFEST_BYTES
        ));
    }
    let payload = std::fs::read_to_string(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    verify_manifest_content_sha256(&payload, &manifest_path)?;
    let manifest: Sq8CanonicalArtifactManifest = serde_json::from_str(&payload)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))?;
    validate_sq8_canonical_manifest(artifact_dir, &manifest)?;
    let checksum_report = verify_sq8_canonical_manifest_payloads(
        artifact_dir,
        &manifest,
        SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES,
    )?;
    Ok(Sq8CanonicalArtifact {
        artifact_dir: artifact_dir.to_path_buf(),
        manifest,
        verified: Sq8CanonicalVerifiedState { checksum_report },
    })
}

pub fn validate_sq8_canonical_manifest(
    artifact_dir: &Path,
    manifest: &Sq8CanonicalArtifactManifest,
) -> Result<(), String> {
    if manifest.schema_version != SQ8_CANONICAL_ARTIFACT_SCHEMA_VERSION {
        return Err(format!(
            "SQ8 canonical schema_version must be {SQ8_CANONICAL_ARTIFACT_SCHEMA_VERSION}, got {}",
            manifest.schema_version
        ));
    }
    if manifest.artifact_kind != SQ8_CANONICAL_ARTIFACT_KIND {
        return Err(format!(
            "SQ8 canonical artifact_kind must be {SQ8_CANONICAL_ARTIFACT_KIND}, got {}",
            manifest.artifact_kind
        ));
    }
    if manifest.format_id != FORMAT_SQ8_0 {
        return Err(format!(
            "SQ8 canonical format_id must be {FORMAT_SQ8_0}, got {}",
            manifest.format_id
        ));
    }
    validate_source_contract(&manifest.source)?;
    if manifest.import_contract.mode != SQ8_CANONICAL_IMPORT_MODE {
        return Err(format!(
            "SQ8 canonical import.mode must be {SQ8_CANONICAL_IMPORT_MODE}, got {}",
            manifest.import_contract.mode
        ));
    }
    if manifest.import_contract.encoding != SQ8_CANONICAL_RAW_ENCODING {
        return Err(format!(
            "SQ8 canonical import.encoding must be {SQ8_CANONICAL_RAW_ENCODING}, got {}",
            manifest.import_contract.encoding
        ));
    }
    validate_sha256(
        &manifest.integrity.content_sha256,
        "SQ8 canonical integrity.content_sha256",
    )?;
    validate_coverage(manifest)?;

    let mut tensor_names = BTreeSet::new();
    let mut scale_names = BTreeSet::new();
    let mut artifact_files = BTreeSet::new();
    let mut weight_payload_bytes = 0_u64;
    let mut scale_payload_bytes = 0_u64;
    validate_name_order(
        manifest
            .quantized_tensors
            .iter()
            .map(|pair| pair.name.as_str()),
        "quantized_tensors",
    )?;
    validate_name_order(
        manifest
            .passthrough_tensors
            .iter()
            .map(|tensor| tensor.name.as_str()),
        "passthrough_tensors",
    )?;
    for (index, pair) in manifest.quantized_tensors.iter().enumerate() {
        if !tensor_names.insert(pair.name.as_str()) {
            return Err(format!(
                "SQ8 canonical tensor pair name is duplicated at index {index}: {}",
                pair.name
            ));
        }
        if !scale_names.insert(pair.scale.name.as_str()) {
            return Err(format!(
                "SQ8 canonical scale name is duplicated at index {index}: {}",
                pair.scale.name
            ));
        }
        validate_tensor_pair(artifact_dir, index, pair, &mut artifact_files)?;
        weight_payload_bytes = weight_payload_bytes
            .checked_add(pair.weight.bytes)
            .ok_or_else(|| "SQ8 canonical weight payload byte sum overflows".to_string())?;
        scale_payload_bytes = scale_payload_bytes
            .checked_add(pair.scale.bytes)
            .ok_or_else(|| "SQ8 canonical scale payload byte sum overflows".to_string())?;
    }
    for name in &tensor_names {
        if let Some(alias) = qwen3_tensor_name_alias(name)
            && tensor_names.contains(alias.as_str())
        {
            return Err(format!(
                "SQ8 canonical quantized tensors contain both Qwen3 namespace aliases: {name:?} and {alias:?}"
            ));
        }
    }

    let mut passthrough_names = BTreeSet::new();
    for (index, tensor) in manifest.passthrough_tensors.iter().enumerate() {
        if tensor.name.is_empty() || !passthrough_names.insert(tensor.name.as_str()) {
            return Err(format!(
                "SQ8 canonical passthrough tensor name is empty or duplicated at index {index}: {:?}",
                tensor.name
            ));
        }
        if tensor_names.contains(tensor.name.as_str()) || scale_names.contains(tensor.name.as_str())
        {
            return Err(format!(
                "SQ8 canonical tensor {} appears both as a quantized pair component and passthrough",
                tensor.name
            ));
        }
        if tensor.dtype.is_empty() {
            return Err(format!(
                "SQ8 canonical passthrough tensor {} has an empty dtype",
                tensor.name
            ));
        }
        let elements = checked_dynamic_shape_elements(
            &tensor.shape,
            &format!("passthrough tensor {}", tensor.name),
        )?;
        if tensor.elements != elements {
            return Err(format!(
                "SQ8 canonical passthrough tensor {} element count mismatch: shape={elements} elements={}",
                tensor.name, tensor.elements
            ));
        }
        validate_source_shard_name(
            &tensor.source_file,
            &format!("passthrough tensor {}", tensor.name),
        )?;
    }

    if manifest.storage.weight_payload_bytes != weight_payload_bytes {
        return Err(format!(
            "SQ8 canonical storage weight_payload_bytes mismatch: storage={} entries={weight_payload_bytes}",
            manifest.storage.weight_payload_bytes
        ));
    }
    if manifest.storage.scale_payload_bytes != scale_payload_bytes {
        return Err(format!(
            "SQ8 canonical storage scale_payload_bytes mismatch: storage={} entries={scale_payload_bytes}",
            manifest.storage.scale_payload_bytes
        ));
    }
    let total_payload_bytes = weight_payload_bytes
        .checked_add(scale_payload_bytes)
        .ok_or_else(|| "SQ8 canonical total payload byte sum overflows".to_string())?;
    if manifest.storage.total_payload_bytes != total_payload_bytes {
        return Err(format!(
            "SQ8 canonical storage total_payload_bytes mismatch: storage={} entries={total_payload_bytes}",
            manifest.storage.total_payload_bytes
        ));
    }
    Ok(())
}

pub fn verify_sq8_canonical_artifact_checksums(
    artifact: &Sq8CanonicalArtifact,
    chunk_bytes: usize,
) -> Result<Sq8CanonicalChecksumReport, String> {
    verify_sq8_canonical_manifest_payloads(&artifact.artifact_dir, &artifact.manifest, chunk_bytes)
}

fn verify_sq8_canonical_manifest_payloads(
    artifact_dir: &Path,
    manifest: &Sq8CanonicalArtifactManifest,
    chunk_bytes: usize,
) -> Result<Sq8CanonicalChecksumReport, String> {
    if chunk_bytes == 0 {
        return Err("SQ8 canonical checksum chunk_bytes must be greater than zero".to_string());
    }
    let chunk_bytes = chunk_bytes.min(SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES);
    let mut weight_payload_bytes = 0_u64;
    let mut scale_payload_bytes = 0_u64;
    for pair in &manifest.quantized_tensors {
        verify_sq8_canonical_tensor_pair_payloads(artifact_dir, pair, chunk_bytes)?;
        weight_payload_bytes = weight_payload_bytes
            .checked_add(pair.weight.bytes)
            .ok_or_else(|| "SQ8 canonical verified weight byte sum overflows".to_string())?;
        scale_payload_bytes = scale_payload_bytes
            .checked_add(pair.scale.bytes)
            .ok_or_else(|| "SQ8 canonical verified scale byte sum overflows".to_string())?;
    }
    Ok(Sq8CanonicalChecksumReport {
        selected_pair_count: manifest.coverage.selected_pair_count,
        weight_payload_bytes,
        scale_payload_bytes,
    })
}

fn verify_sq8_canonical_tensor_pair_payloads(
    artifact_dir: &Path,
    pair: &Sq8CanonicalTensorPair,
    chunk_bytes: usize,
) -> Result<(), String> {
    let weight_path = canonical_artifact_file(
        artifact_dir,
        &pair.weight.file,
        &format!("{} weight", pair.name),
    )?;
    verify_payload_file(
        &weight_path,
        pair.weight.bytes,
        &pair.weight.sha256,
        chunk_bytes,
        PayloadValidation::Fp8E4m3,
        &format!("{} weight", pair.name),
    )?;

    let scale_path = canonical_artifact_file(
        artifact_dir,
        &pair.scale.file,
        &format!("{} scale", pair.name),
    )?;
    verify_payload_file(
        &scale_path,
        pair.scale.bytes,
        &pair.scale.sha256,
        chunk_bytes,
        PayloadValidation::PositiveBf16,
        &format!("{} scale", pair.name),
    )?;
    Ok(())
}

pub fn reconstruct_sq8_canonical_tensor_row_f32(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    row_index: usize,
) -> Result<Vec<f32>, String> {
    let pair = find_tensor_pair(artifact, tensor_name)?;
    let rows = to_usize(pair.shape[0], &format!("{tensor_name} rows"))?;
    let cols = to_usize(pair.shape[1], &format!("{tensor_name} cols"))?;
    if cols > SQ8_CANONICAL_MAX_RECONSTRUCT_ROW_ELEMENTS {
        return Err(format!(
            "SQ8 canonical tensor {tensor_name} row has {cols} elements, exceeding reconstruction limit {SQ8_CANONICAL_MAX_RECONSTRUCT_ROW_ELEMENTS}"
        ));
    }
    if row_index >= rows {
        return Err(format!(
            "SQ8 canonical tensor {tensor_name} row {row_index} is out of range for {rows} rows"
        ));
    }
    let block_rows = to_usize(pair.scale.block_shape[0], "SQ8 canonical scale block rows")?;
    let block_cols = to_usize(pair.scale.block_shape[1], "SQ8 canonical scale block cols")?;
    let scale_cols = to_usize(pair.scale.shape[1], "SQ8 canonical scale grid cols")?;
    verify_sq8_canonical_tensor_pair_payloads(
        &artifact.artifact_dir,
        pair,
        SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES,
    )
    .map_err(|err| {
        format!("SQ8 canonical tensor {tensor_name} failed pre-reconstruction verification: {err}")
    })?;

    let weight_path = canonical_artifact_file(
        &artifact.artifact_dir,
        &pair.weight.file,
        &format!("{tensor_name} weight"),
    )?;
    let weight_offset = row_index
        .checked_mul(cols)
        .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} row offset overflows"))?;
    let weight_bytes = read_exact_at(&weight_path, weight_offset as u64, cols)?;

    let scale_path = canonical_artifact_file(
        &artifact.artifact_dir,
        &pair.scale.file,
        &format!("{tensor_name} scale"),
    )?;
    let scale_row = row_index / block_rows;
    let scale_offset = scale_row
        .checked_mul(scale_cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} scale row offset overflows"))?;
    let scale_bytes = read_exact_at(
        &scale_path,
        scale_offset as u64,
        scale_cols
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} scale read overflows"))?,
    )?;
    let scales = decode_positive_bf16_values(&scale_bytes, &format!("{tensor_name} scale row"))?;

    let mut values = Vec::with_capacity(cols);
    for (col, byte) in weight_bytes.into_iter().enumerate() {
        let weight = checked_fp8_value(byte, tensor_name, row_index, col)?;
        values.push(weight * scales[col / block_cols]);
    }
    Ok(values)
}

pub fn reconstruct_sq8_canonical_tensor_block_f32(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    block_row: usize,
    block_col: usize,
) -> Result<Sq8CanonicalReconstructedBlock, String> {
    let pair = find_tensor_pair(artifact, tensor_name)?;
    let rows = to_usize(pair.shape[0], &format!("{tensor_name} rows"))?;
    let cols = to_usize(pair.shape[1], &format!("{tensor_name} cols"))?;
    let scale_rows = to_usize(pair.scale.shape[0], "SQ8 canonical scale grid rows")?;
    let scale_cols = to_usize(pair.scale.shape[1], "SQ8 canonical scale grid cols")?;
    if block_row >= scale_rows || block_col >= scale_cols {
        return Err(format!(
            "SQ8 canonical tensor {tensor_name} block [{block_row},{block_col}] is out of range for scale grid [{scale_rows},{scale_cols}]"
        ));
    }
    let block_rows = to_usize(pair.scale.block_shape[0], "SQ8 canonical scale block rows")?;
    let block_cols = to_usize(pair.scale.block_shape[1], "SQ8 canonical scale block cols")?;
    let start_row = block_row
        .checked_mul(block_rows)
        .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} block row offset overflows"))?;
    let start_col = block_col
        .checked_mul(block_cols)
        .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} block col offset overflows"))?;
    let block_row_count = block_rows.min(rows - start_row);
    let block_col_count = block_cols.min(cols - start_col);
    verify_sq8_canonical_tensor_pair_payloads(
        &artifact.artifact_dir,
        pair,
        SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES,
    )
    .map_err(|err| {
        format!("SQ8 canonical tensor {tensor_name} failed pre-reconstruction verification: {err}")
    })?;

    let scale_path = canonical_artifact_file(
        &artifact.artifact_dir,
        &pair.scale.file,
        &format!("{tensor_name} scale"),
    )?;
    let scale_index = block_row
        .checked_mul(scale_cols)
        .and_then(|value| value.checked_add(block_col))
        .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} scale index overflows"))?;
    let scale_offset = scale_index
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} scale offset overflows"))?;
    let scale_bytes = read_exact_at(&scale_path, scale_offset as u64, std::mem::size_of::<u16>())?;
    let scale = decode_positive_bf16_values(&scale_bytes, &format!("{tensor_name} scale"))?[0];

    let weight_path = canonical_artifact_file(
        &artifact.artifact_dir,
        &pair.weight.file,
        &format!("{tensor_name} weight"),
    )?;
    let mut weight_file = File::open(&weight_path)
        .map_err(|err| format!("failed to open {}: {err}", weight_path.display()))?;
    let value_count = block_row_count
        .checked_mul(block_col_count)
        .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} block size overflows"))?;
    let mut values = Vec::with_capacity(value_count);
    let mut row_bytes = vec![0_u8; block_col_count];
    for local_row in 0..block_row_count {
        let row = start_row + local_row;
        let offset = row
            .checked_mul(cols)
            .and_then(|value| value.checked_add(start_col))
            .ok_or_else(|| format!("SQ8 canonical tensor {tensor_name} block offset overflows"))?;
        weight_file
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|err| {
                format!(
                    "failed to seek {} to {offset}: {err}",
                    weight_path.display()
                )
            })?;
        weight_file.read_exact(&mut row_bytes).map_err(|err| {
            format!(
                "failed to read {} bytes from {} at {offset}: {err}",
                row_bytes.len(),
                weight_path.display()
            )
        })?;
        for (local_col, byte) in row_bytes.iter().copied().enumerate() {
            let col = start_col + local_col;
            values.push(checked_fp8_value(byte, tensor_name, row, col)? * scale);
        }
    }
    Ok(Sq8CanonicalReconstructedBlock {
        tensor_name: pair.name.clone(),
        block_row,
        block_col,
        start_row,
        start_col,
        rows: block_row_count,
        cols: block_col_count,
        values,
    })
}

fn validate_source_contract(source: &Sq8CanonicalSource) -> Result<(), String> {
    if source.model_name.is_empty() {
        return Err("SQ8 canonical source.model_name must not be empty".to_string());
    }
    if source.config_file.is_empty() {
        return Err("SQ8 canonical source.config_file must not be empty".to_string());
    }
    validate_sha256(&source.config_sha256, "SQ8 canonical source.config_sha256")?;
    match (&source.index_file, &source.index_sha256) {
        (Some(file), Some(sha256)) => {
            if file.is_empty() {
                return Err("SQ8 canonical source.index_file must not be empty".to_string());
            }
            validate_sha256(sha256, "SQ8 canonical source.index_sha256")?;
        }
        (None, None) => {}
        _ => {
            return Err(
                "SQ8 canonical source.index_file and index_sha256 must both be present or absent"
                    .to_string(),
            );
        }
    }
    if source.quantization.quant_method != "fp8" {
        return Err(format!(
            "SQ8 canonical source quant_method must be fp8, got {}",
            source.quantization.quant_method
        ));
    }
    if source.quantization.format != "e4m3" {
        return Err(format!(
            "SQ8 canonical source format must be e4m3, got {}",
            source.quantization.format
        ));
    }
    if source.quantization.activation_scheme != "dynamic" {
        return Err(format!(
            "SQ8 canonical source activation_scheme must be dynamic, got {}",
            source.quantization.activation_scheme
        ));
    }
    if source.quantization.weight_block_shape != SQ8_CANONICAL_BLOCK_SHAPE {
        return Err(format!(
            "SQ8 canonical source weight_block_shape must be {:?}, got {:?}",
            SQ8_CANONICAL_BLOCK_SHAPE, source.quantization.weight_block_shape
        ));
    }
    Ok(())
}

fn validate_coverage(manifest: &Sq8CanonicalArtifactManifest) -> Result<(), String> {
    let coverage = &manifest.coverage;
    if coverage.scope != "selected_tensors" && coverage.scope != "full_model" {
        return Err(format!(
            "SQ8 canonical coverage.scope must be selected_tensors or full_model, got {}",
            coverage.scope
        ));
    }
    if coverage.unpaired_tensor_count != 0 {
        return Err(format!(
            "SQ8 canonical coverage has {} unpaired tensors",
            coverage.unpaired_tensor_count
        ));
    }
    if coverage.paired_tensor_count == 0 || coverage.selected_pair_count == 0 {
        return Err(
            "SQ8 canonical coverage requires at least one paired and selected tensor".to_string(),
        );
    }
    if coverage.source_fp8_weight_count != coverage.paired_tensor_count
        || coverage.source_scale_count != coverage.paired_tensor_count
    {
        return Err(format!(
            "SQ8 canonical source pair coverage mismatch: weights={} scales={} paired={}",
            coverage.source_fp8_weight_count,
            coverage.source_scale_count,
            coverage.paired_tensor_count
        ));
    }
    if coverage.selected_pair_count > coverage.paired_tensor_count {
        return Err(format!(
            "SQ8 canonical selected_pair_count {} exceeds paired_tensor_count {}",
            coverage.selected_pair_count, coverage.paired_tensor_count
        ));
    }
    if usize::try_from(coverage.selected_pair_count).ok() != Some(manifest.quantized_tensors.len())
    {
        return Err(format!(
            "SQ8 canonical selected pair count mismatch: coverage={} entries={}",
            coverage.selected_pair_count,
            manifest.quantized_tensors.len()
        ));
    }
    if usize::try_from(coverage.passthrough_tensor_count).ok()
        != Some(manifest.passthrough_tensors.len())
    {
        return Err(format!(
            "SQ8 canonical passthrough count mismatch: coverage={} entries={}",
            coverage.passthrough_tensor_count,
            manifest.passthrough_tensors.len()
        ));
    }
    let source_tensor_count = coverage
        .source_fp8_weight_count
        .checked_add(coverage.source_scale_count)
        .and_then(|value| value.checked_add(coverage.passthrough_tensor_count))
        .ok_or_else(|| "SQ8 canonical source tensor count overflows".to_string())?;
    if coverage.source_tensor_count != source_tensor_count {
        return Err(format!(
            "SQ8 canonical source_tensor_count mismatch: coverage={} accounted={source_tensor_count}",
            coverage.source_tensor_count
        ));
    }
    if coverage.scope == "full_model"
        && coverage.selected_pair_count != coverage.paired_tensor_count
    {
        return Err(format!(
            "SQ8 canonical full_model coverage requires every pair selected: selected={} paired={}",
            coverage.selected_pair_count, coverage.paired_tensor_count
        ));
    }
    Ok(())
}

fn validate_tensor_pair<'a>(
    artifact_dir: &Path,
    index: usize,
    pair: &'a Sq8CanonicalTensorPair,
    artifact_files: &mut BTreeSet<&'a str>,
) -> Result<(), String> {
    if pair.name.is_empty() || !pair.name.ends_with(".weight") {
        return Err(format!(
            "SQ8 canonical tensor pair {index} name must be a non-empty .weight name"
        ));
    }
    if pair.family.is_empty() {
        return Err(format!(
            "SQ8 canonical tensor pair {} has an empty family",
            pair.name
        ));
    }
    let expected_scale_name = format!("{}_scale_inv", pair.name);
    if pair.scale.name != expected_scale_name {
        return Err(format!(
            "SQ8 canonical tensor {} scale name mismatch: expected {expected_scale_name}, got {}",
            pair.name, pair.scale.name
        ));
    }
    let elements = checked_shape_elements(pair.shape, &format!("{} weight", pair.name))?;
    if pair.elements != elements {
        return Err(format!(
            "SQ8 canonical tensor {} element count mismatch: shape={elements} elements={}",
            pair.name, pair.elements
        ));
    }
    if pair.weight.dtype != SQ8_CANONICAL_WEIGHT_DTYPE {
        return Err(format!(
            "SQ8 canonical tensor {} weight dtype must be {SQ8_CANONICAL_WEIGHT_DTYPE}, got {}",
            pair.name, pair.weight.dtype
        ));
    }
    if pair.weight.encoding != SQ8_CANONICAL_RAW_ENCODING {
        return Err(format!(
            "SQ8 canonical tensor {} weight encoding must be {SQ8_CANONICAL_RAW_ENCODING}, got {}",
            pair.name, pair.weight.encoding
        ));
    }
    if pair.weight.bytes != elements {
        return Err(format!(
            "SQ8 canonical tensor {} weight bytes must equal elements for F8: bytes={} elements={elements}",
            pair.name, pair.weight.bytes
        ));
    }
    validate_sha256(
        &pair.weight.sha256,
        &format!("SQ8 canonical tensor {} weight sha256", pair.name),
    )?;
    validate_source_shard_name(&pair.weight.source_file, &format!("{} weight", pair.name))?;

    if pair.scale.dtype != SQ8_CANONICAL_SCALE_DTYPE {
        return Err(format!(
            "SQ8 canonical tensor {} scale dtype must be {SQ8_CANONICAL_SCALE_DTYPE}, got {}",
            pair.name, pair.scale.dtype
        ));
    }
    if pair.scale.encoding != SQ8_CANONICAL_RAW_ENCODING {
        return Err(format!(
            "SQ8 canonical tensor {} scale encoding must be {SQ8_CANONICAL_RAW_ENCODING}, got {}",
            pair.name, pair.scale.encoding
        ));
    }
    if pair.scale.layout != SQ8_CANONICAL_SCALE_LAYOUT
        || pair.scale.order != SQ8_CANONICAL_SCALE_ORDER
        || pair.scale.semantic != SQ8_CANONICAL_SCALE_SEMANTIC
    {
        return Err(format!(
            "SQ8 canonical tensor {} scale contract must be layout={SQ8_CANONICAL_SCALE_LAYOUT} order={SQ8_CANONICAL_SCALE_ORDER} semantic={SQ8_CANONICAL_SCALE_SEMANTIC}",
            pair.name
        ));
    }
    if pair.scale.block_shape != SQ8_CANONICAL_BLOCK_SHAPE {
        return Err(format!(
            "SQ8 canonical tensor {} scale block_shape must be {:?}, got {:?}",
            pair.name, SQ8_CANONICAL_BLOCK_SHAPE, pair.scale.block_shape
        ));
    }
    let expected_scale_shape = [
        ceil_div(pair.shape[0], pair.scale.block_shape[0], &pair.name)?,
        ceil_div(pair.shape[1], pair.scale.block_shape[1], &pair.name)?,
    ];
    if pair.scale.shape != expected_scale_shape {
        return Err(format!(
            "SQ8 canonical tensor {} scale shape mismatch: expected {:?}, got {:?}",
            pair.name, expected_scale_shape, pair.scale.shape
        ));
    }
    let scale_elements = checked_shape_elements(pair.scale.shape, &format!("{} scale", pair.name))?;
    if pair.scale.elements != scale_elements {
        return Err(format!(
            "SQ8 canonical tensor {} scale element count mismatch: shape={scale_elements} elements={}",
            pair.name, pair.scale.elements
        ));
    }
    let expected_scale_bytes = scale_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            format!(
                "SQ8 canonical tensor {} scale byte count overflows",
                pair.name
            )
        })?;
    if pair.scale.bytes != expected_scale_bytes {
        return Err(format!(
            "SQ8 canonical tensor {} scale bytes mismatch: expected {expected_scale_bytes}, got {}",
            pair.name, pair.scale.bytes
        ));
    }
    validate_sha256(
        &pair.scale.sha256,
        &format!("SQ8 canonical tensor {} scale sha256", pair.name),
    )?;
    validate_source_shard_name(&pair.scale.source_file, &format!("{} scale", pair.name))?;

    for (relative, expected_bytes, label) in [
        (
            pair.weight.file.as_str(),
            pair.weight.bytes,
            format!("{} weight", pair.name),
        ),
        (
            pair.scale.file.as_str(),
            pair.scale.bytes,
            format!("{} scale", pair.name),
        ),
    ] {
        if !artifact_files.insert(relative) {
            return Err(format!(
                "SQ8 canonical artifact payload path is duplicated: {relative}"
            ));
        }
        let path = canonical_artifact_file(artifact_dir, relative, &label)?;
        let actual_bytes = std::fs::metadata(&path)
            .map_err(|err| format!("failed to stat {}: {err}", path.display()))?
            .len();
        if actual_bytes != expected_bytes {
            return Err(format!(
                "SQ8 canonical {label} byte length mismatch: manifest={expected_bytes} file={actual_bytes}"
            ));
        }
    }
    Ok(())
}

fn verify_manifest_content_sha256(payload: &str, manifest_path: &Path) -> Result<(), String> {
    let mut manifest: Value = serde_json::from_str(payload)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))?;
    let expected = manifest
        .get("integrity")
        .and_then(Value::as_object)
        .and_then(|integrity| integrity.get("content_sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            format!(
                "SQ8 canonical manifest {} is missing string integrity.content_sha256",
                manifest_path.display()
            )
        })?
        .to_string();
    validate_sha256(&expected, "SQ8 canonical integrity.content_sha256")?;
    manifest
        .as_object_mut()
        .ok_or_else(|| {
            format!(
                "SQ8 canonical manifest {} root must be an object",
                manifest_path.display()
            )
        })?
        .remove("integrity");
    // serde_json's default map is key-sorted. The manifest contract contains only JSON
    // values whose compact encoding matches Python's sort_keys/separators canonical form.
    let canonical = serde_json::to_vec(&manifest).map_err(|err| {
        format!(
            "failed to canonicalize SQ8 manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    let mut digest = Sha256::new();
    digest.update(canonical);
    let actual = format!("{:x}", digest.finalize());
    if actual != expected {
        return Err(format!(
            "SQ8 canonical manifest content checksum mismatch: manifest={expected} computed={actual}"
        ));
    }
    Ok(())
}

fn validate_name_order<'a>(
    names: impl Iterator<Item = &'a str>,
    label: &str,
) -> Result<(), String> {
    let mut previous: Option<&str> = None;
    for name in names {
        if let Some(previous) = previous
            && previous > name
        {
            return Err(format!(
                "SQ8 canonical {label} must be sorted by name: {previous:?} appears before {name:?}"
            ));
        }
        previous = Some(name);
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{label} must be 64 lowercase hexadecimal characters, got {value:?}"
        ));
    }
    Ok(())
}

fn validate_source_shard_name(value: &str, label: &str) -> Result<(), String> {
    let path = Path::new(value);
    let mut components = path.components();
    if value.is_empty()
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(format!(
            "SQ8 canonical {label} source_file must be a shard basename, got {value:?}"
        ));
    }
    Ok(())
}

fn canonical_artifact_file(
    artifact_dir: &Path,
    relative: &str,
    label: &str,
) -> Result<PathBuf, String> {
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "SQ8 canonical {label} path must be a normalized relative path inside the artifact: {relative:?}"
        ));
    }
    let root = std::fs::canonicalize(artifact_dir).map_err(|err| {
        format!(
            "failed to canonicalize SQ8 artifact directory {}: {err}",
            artifact_dir.display()
        )
    })?;
    let joined = artifact_dir.join(relative_path);
    let resolved = std::fs::canonicalize(&joined).map_err(|err| {
        format!(
            "failed to resolve SQ8 canonical {label} {}: {err}",
            joined.display()
        )
    })?;
    if !resolved.starts_with(&root) {
        return Err(format!(
            "SQ8 canonical {label} path escapes the artifact directory: {relative:?}"
        ));
    }
    if !resolved.is_file() {
        return Err(format!(
            "SQ8 canonical {label} path is not a file: {}",
            resolved.display()
        ));
    }
    Ok(resolved)
}

fn find_tensor_pair<'a>(
    artifact: &'a Sq8CanonicalArtifact,
    tensor_name: &str,
) -> Result<&'a Sq8CanonicalTensorPair, String> {
    if let Some(pair) = artifact
        .manifest
        .quantized_tensors
        .iter()
        .find(|pair| pair.name == tensor_name)
    {
        return Ok(pair);
    }
    if let Some(alias) = qwen3_tensor_name_alias(tensor_name)
        && let Some(pair) = artifact
            .manifest
            .quantized_tensors
            .iter()
            .find(|pair| pair.name == alias)
    {
        return Ok(pair);
    }
    Err(format!(
        "SQ8 canonical artifact has no tensor named {tensor_name:?}"
    ))
}

enum PayloadValidation {
    Fp8E4m3,
    PositiveBf16,
}

fn verify_payload_file(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    chunk_bytes: usize,
    validation: PayloadValidation,
    label: &str,
) -> Result<(), String> {
    let mut file = File::open(path)
        .map_err(|err| format!("failed to open {} for checksum: {err}", path.display()))?;
    let opened_bytes = file
        .metadata()
        .map_err(|err| format!("failed to stat opened {}: {err}", path.display()))?
        .len();
    if opened_bytes != expected_bytes {
        return Err(format!(
            "SQ8 canonical {label} byte length mismatch before verification: manifest={expected_bytes} file={opened_bytes}"
        ));
    }
    let file_chunk_bytes = usize::try_from(expected_bytes.min(chunk_bytes as u64))
        .map_err(|_| format!("SQ8 canonical {label} chunk length does not fit usize"))?;
    let file_chunk_bytes = match validation {
        PayloadValidation::Fp8E4m3 => file_chunk_bytes.max(1),
        PayloadValidation::PositiveBf16 => file_chunk_bytes.max(2) & !1,
    };
    let mut buffer = vec![0_u8; file_chunk_bytes];
    let mut remaining = expected_bytes;
    let mut offset = 0_u64;
    let mut digest = Sha256::new();
    while remaining > 0 {
        let read_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| format!("SQ8 canonical {label} read length does not fit usize"))?;
        file.read_exact(&mut buffer[..read_len]).map_err(|err| {
            format!(
                "failed to read SQ8 canonical {label} at byte {offset} from {}: {err}",
                path.display()
            )
        })?;
        let chunk = &buffer[..read_len];
        match validation {
            PayloadValidation::Fp8E4m3 => {
                if let Some(index) = first_non_finite_fp8_byte(chunk) {
                    return Err(format!(
                        "SQ8 canonical {label} contains non-finite E4M3 byte at offset {}",
                        offset + index as u64
                    ));
                }
            }
            PayloadValidation::PositiveBf16 => {
                if !chunk.len().is_multiple_of(std::mem::size_of::<u16>()) {
                    return Err(format!(
                        "SQ8 canonical {label} BF16 chunk has odd byte length {}",
                        chunk.len()
                    ));
                }
                for (index, raw) in chunk.chunks_exact(2).enumerate() {
                    let value = bf16_le_to_f32([raw[0], raw[1]]);
                    if !value.is_finite() || value <= 0.0 {
                        return Err(format!(
                            "SQ8 canonical {label} contains invalid BF16 scale {value} at byte offset {}",
                            offset + (index * 2) as u64
                        ));
                    }
                }
            }
        }
        digest.update(chunk);
        remaining -= read_len as u64;
        offset += read_len as u64;
    }
    let mut trailing = [0_u8; 1];
    let trailing_bytes = file.read(&mut trailing).map_err(|err| {
        format!(
            "failed to verify EOF for SQ8 canonical {label} after byte {expected_bytes} from {}: {err}",
            path.display()
        )
    })?;
    if trailing_bytes != 0 {
        return Err(format!(
            "SQ8 canonical {label} has trailing data after declared {expected_bytes} bytes"
        ));
    }
    let final_bytes = file
        .metadata()
        .map_err(|err| format!("failed to re-stat opened {}: {err}", path.display()))?
        .len();
    if final_bytes != expected_bytes {
        return Err(format!(
            "SQ8 canonical {label} byte length changed during verification: manifest={expected_bytes} before={opened_bytes} after={final_bytes}"
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "SQ8 canonical {label} checksum mismatch: manifest={expected_sha256} file={actual_sha256}"
        ));
    }
    Ok(())
}

fn read_verified_positive_bf16_payload(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    chunk_bytes: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let mut file = File::open(path)
        .map_err(|err| format!("failed to open {} for verified read: {err}", path.display()))?;
    let opened_bytes = file
        .metadata()
        .map_err(|err| format!("failed to stat opened {}: {err}", path.display()))?
        .len();
    if opened_bytes != expected_bytes {
        return Err(format!(
            "SQ8 canonical {label} byte length mismatch before verified read: manifest={expected_bytes} file={opened_bytes}"
        ));
    }
    let value_count = usize::try_from(expected_bytes / std::mem::size_of::<u16>() as u64)
        .map_err(|_| format!("SQ8 canonical {label} element count does not fit usize"))?;
    let file_chunk_bytes = usize::try_from(expected_bytes.min(chunk_bytes as u64))
        .map_err(|_| format!("SQ8 canonical {label} chunk length does not fit usize"))?
        .max(2)
        & !1;
    let mut buffer = vec![0_u8; file_chunk_bytes];
    let mut values = Vec::with_capacity(value_count);
    let mut remaining = expected_bytes;
    let mut offset = 0_u64;
    let mut digest = Sha256::new();
    while remaining > 0 {
        let read_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| format!("SQ8 canonical {label} read length does not fit usize"))?;
        file.read_exact(&mut buffer[..read_len]).map_err(|err| {
            format!(
                "failed to read SQ8 canonical {label} at byte {offset} from {}: {err}",
                path.display()
            )
        })?;
        if !read_len.is_multiple_of(std::mem::size_of::<u16>()) {
            return Err(format!(
                "SQ8 canonical {label} BF16 chunk has odd byte length {read_len}"
            ));
        }
        let chunk = &buffer[..read_len];
        digest.update(chunk);
        for (index, raw) in chunk.chunks_exact(2).enumerate() {
            let value = bf16_le_to_f32([raw[0], raw[1]]);
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "SQ8 canonical {label} contains invalid BF16 scale {value} at byte offset {}",
                    offset + (index * 2) as u64
                ));
            }
            values.push(value);
        }
        remaining -= read_len as u64;
        offset += read_len as u64;
    }
    let mut trailing = [0_u8; 1];
    let trailing_bytes = file.read(&mut trailing).map_err(|err| {
        format!(
            "failed to verify EOF for SQ8 canonical {label} after byte {expected_bytes} from {}: {err}",
            path.display()
        )
    })?;
    if trailing_bytes != 0 {
        return Err(format!(
            "SQ8 canonical {label} has trailing data after declared {expected_bytes} bytes"
        ));
    }
    let final_bytes = file
        .metadata()
        .map_err(|err| format!("failed to re-stat opened {}: {err}", path.display()))?
        .len();
    if final_bytes != expected_bytes {
        return Err(format!(
            "SQ8 canonical {label} byte length changed during verified read: manifest={expected_bytes} before={opened_bytes} after={final_bytes}"
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "SQ8 canonical {label} checksum mismatch: manifest={expected_sha256} file={actual_sha256}"
        ));
    }
    if values.len() != value_count {
        return Err(format!(
            "SQ8 canonical {label} decoded element count mismatch: expected={value_count} actual={}",
            values.len()
        ));
    }
    Ok(values)
}

fn first_non_finite_fp8_byte(bytes: &[u8]) -> Option<usize> {
    memchr::memchr2(0x7f, 0xff, bytes)
}

fn checked_shape_elements(shape: [u64; 2], label: &str) -> Result<u64, String> {
    if shape[0] == 0 || shape[1] == 0 {
        return Err(format!(
            "SQ8 canonical {label} shape must be nonzero, got {shape:?}"
        ));
    }
    shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| format!("SQ8 canonical {label} shape overflows: {shape:?}"))
}

fn checked_dynamic_shape_elements(shape: &[u64], label: &str) -> Result<u64, String> {
    shape.iter().try_fold(1_u64, |elements, dimension| {
        if *dimension == 0 {
            return Err(format!(
                "SQ8 canonical {label} shape contains a zero dimension: {shape:?}"
            ));
        }
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| format!("SQ8 canonical {label} shape overflows: {shape:?}"))
    })
}

fn ceil_div(value: u64, divisor: u64, label: &str) -> Result<u64, String> {
    if divisor == 0 {
        return Err(format!(
            "SQ8 canonical {label} block dimension must be nonzero"
        ));
    }
    value
        .checked_add(divisor - 1)
        .and_then(|sum| sum.checked_div(divisor))
        .ok_or_else(|| format!("SQ8 canonical {label} ceil division overflows"))
}

fn to_usize(value: u64, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{label} does not fit usize: {value}"))
}

fn read_exact_at(path: &Path, offset: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| format!("failed to seek {} to {offset}: {err}", path.display()))?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes).map_err(|err| {
        format!(
            "failed to read {len} bytes from {} at {offset}: {err}",
            path.display()
        )
    })?;
    Ok(bytes)
}

fn decode_positive_bf16_values(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u16>()) {
        return Err(format!(
            "SQ8 canonical {label} BF16 byte length must be even, got {}",
            bytes.len()
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 2);
    for (index, raw) in bytes.chunks_exact(2).enumerate() {
        let value = bf16_le_to_f32([raw[0], raw[1]]);
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "SQ8 canonical {label} BF16 scale {index} is invalid: {value}"
            ));
        }
        values.push(value);
    }
    Ok(values)
}

fn bf16_le_to_f32(bytes: [u8; 2]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes(bytes)) << 16)
}

fn checked_fp8_value(byte: u8, tensor_name: &str, row: usize, col: usize) -> Result<f32, String> {
    let value = fp8_e4m3fn_to_f32(byte);
    if !value.is_finite() {
        return Err(format!(
            "SQ8 canonical tensor {tensor_name} has non-finite E4M3 byte 0x{byte:02x} at [{row},{col}]"
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestArtifact {
        root: PathBuf,
    }

    impl Drop for TestArtifact {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn temp_artifact_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ullm-engine-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut digest = Sha256::new();
        digest.update(bytes);
        format!("{:x}", digest.finalize())
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        bytes
    }

    fn fixture_manifest(weight: &[u8], scale: &[u8]) -> Value {
        let mut manifest = json!({
            "schema_version": SQ8_CANONICAL_ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": SQ8_CANONICAL_ARTIFACT_KIND,
            "format_id": FORMAT_SQ8_0,
            "source": {
                "model_name": "test-sq8",
                "config_file": "config.json",
                "config_sha256": "0".repeat(64),
                "index_file": "model.safetensors.index.json",
                "index_sha256": "1".repeat(64),
                "quantization": {
                    "quant_method": "fp8",
                    "format": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_shape": [128, 128]
                }
            },
            "import": {
                "mode": SQ8_CANONICAL_IMPORT_MODE,
                "encoding": SQ8_CANONICAL_RAW_ENCODING
            },
            "integrity": {"content_sha256": "2".repeat(64)},
            "coverage": {
                "scope": "selected_tensors",
                "source_tensor_count": 2,
                "source_fp8_weight_count": 1,
                "source_scale_count": 1,
                "paired_tensor_count": 1,
                "selected_pair_count": 1,
                "unpaired_tensor_count": 0,
                "passthrough_tensor_count": 0
            },
            "storage": {
                "weight_payload_bytes": weight.len(),
                "scale_payload_bytes": scale.len(),
                "total_payload_bytes": weight.len() + scale.len()
            },
            "quantized_tensors": [{
                "name": "model.layers.0.self_attn.q_proj.weight",
                "family": "attn_q",
                "shape": [129, 130],
                "elements": 129 * 130,
                "weight": {
                    "dtype": SQ8_CANONICAL_WEIGHT_DTYPE,
                    "encoding": SQ8_CANONICAL_RAW_ENCODING,
                    "file": "weights/q.f8_e4m3",
                    "bytes": weight.len(),
                    "sha256": sha256_hex(weight),
                    "source_file": "model-00001-of-00001.safetensors"
                },
                "scale": {
                    "name": "model.layers.0.self_attn.q_proj.weight_scale_inv",
                    "dtype": SQ8_CANONICAL_SCALE_DTYPE,
                    "encoding": SQ8_CANONICAL_RAW_ENCODING,
                    "file": "scales/q.bf16",
                    "shape": [2, 2],
                    "elements": 4,
                    "bytes": scale.len(),
                    "sha256": sha256_hex(scale),
                    "source_file": "model-00001-of-00001.safetensors",
                    "layout": SQ8_CANONICAL_SCALE_LAYOUT,
                    "block_shape": [128, 128],
                    "order": SQ8_CANONICAL_SCALE_ORDER,
                    "semantic": SQ8_CANONICAL_SCALE_SEMANTIC
                }
            }],
            "passthrough_tensors": []
        });
        set_content_sha256(&mut manifest);
        manifest
    }

    fn write_fixture() -> (TestArtifact, Vec<u8>, Vec<u8>) {
        let root = temp_artifact_dir("sq8-canonical-test");
        fs::create_dir_all(root.join("weights")).unwrap();
        fs::create_dir_all(root.join("scales")).unwrap();
        let weight = vec![0x38_u8; 129 * 130];
        let scale = bf16_bytes(&[1.0, 2.0, 4.0, 8.0]);
        fs::write(root.join("weights/q.f8_e4m3"), &weight).unwrap();
        fs::write(root.join("scales/q.bf16"), &scale).unwrap();
        let manifest = fixture_manifest(&weight, &scale);
        fs::write(
            root.join("sq_manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        (TestArtifact { root }, weight, scale)
    }

    fn rewrite_manifest(root: &Path, mutate: impl FnOnce(&mut Value)) {
        let path = root.join("sq_manifest.json");
        let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        mutate(&mut manifest);
        set_content_sha256(&mut manifest);
        fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }

    fn set_content_sha256(manifest: &mut Value) {
        let mut content = manifest.clone();
        content.as_object_mut().unwrap().remove("integrity");
        let canonical = serde_json::to_vec(&content).unwrap();
        manifest["integrity"]["content_sha256"] = json!(sha256_hex(&canonical));
    }

    #[test]
    fn validates_and_verifies_streamed_canonical_payloads() {
        let (fixture, weight, scale) = write_fixture();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let report = artifact.checksum_report();
        assert_eq!(report.selected_pair_count, 1);
        assert_eq!(report.weight_payload_bytes, weight.len() as u64);
        assert_eq!(report.scale_payload_bytes, scale.len() as u64);
        assert_eq!(artifact.manifest().format_id, FORMAT_SQ8_0);
        assert_eq!(artifact.artifact_dir(), fixture.root);
        assert_eq!(artifact.tensor_pair(tensor_name).unwrap().name, tensor_name);
        let paths = artifact.tensor_payload_paths(tensor_name).unwrap();
        assert_eq!(paths.weight, fixture.root.join("weights/q.f8_e4m3"));
        assert_eq!(paths.scale, fixture.root.join("scales/q.bf16"));
        let tensor_report = artifact.verify_tensor_payloads(tensor_name, 17).unwrap();
        assert_eq!(tensor_report.tensor_name, tensor_name);
        assert_eq!(tensor_report.weight_payload_bytes, weight.len() as u64);
        assert_eq!(tensor_report.scale_payload_bytes, scale.len() as u64);
        assert_eq!(
            artifact.read_tensor_scales_f32(tensor_name, 3).unwrap(),
            [1.0, 2.0, 4.0, 8.0]
        );
    }

    #[test]
    fn manifest_canonical_json_sorts_keys_and_keeps_utf8() {
        let value = json!({"z": "\u{30e2}\u{30c7}\u{30eb}", "a": 1});
        assert_eq!(
            String::from_utf8(serde_json::to_vec(&value).unwrap()).unwrap(),
            "{\"a\":1,\"z\":\"\u{30e2}\u{30c7}\u{30eb}\"}"
        );
    }

    #[test]
    fn fast_fp8_finite_scan_matches_e4m3fn_decoder_for_every_byte() {
        for byte in 0_u8..=u8::MAX {
            let detected = first_non_finite_fp8_byte(&[byte]).is_some();
            assert_eq!(detected, !fp8_e4m3fn_to_f32(byte).is_finite());
        }
        assert_eq!(first_non_finite_fp8_byte(&[0x38, 0x7f, 0xff]), Some(1));
        assert_eq!(first_non_finite_fp8_byte(&[0x38, 0x7e, 0xfe]), None);
    }

    #[test]
    fn rejects_manifest_content_checksum_mismatch() {
        let (fixture, _, _) = write_fixture();
        let path = fixture.root.join("sq_manifest.json");
        let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest["source"]["model_name"] = json!("tampered");
        fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("manifest content checksum mismatch"));
    }

    #[test]
    fn reconstructs_full_and_edge_blocks_with_block2d_scales() {
        let (fixture, _, _) = write_fixture();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let name = "model.layers.0.self_attn.q_proj.weight";

        let first = reconstruct_sq8_canonical_tensor_block_f32(&artifact, name, 0, 0).unwrap();
        assert_eq!((first.rows, first.cols), (128, 128));
        assert!(first.values.iter().all(|value| *value == 1.0));

        let right_edge = reconstruct_sq8_canonical_tensor_block_f32(&artifact, name, 0, 1).unwrap();
        assert_eq!((right_edge.rows, right_edge.cols), (128, 2));
        assert!(right_edge.values.iter().all(|value| *value == 2.0));

        let bottom_edge =
            reconstruct_sq8_canonical_tensor_block_f32(&artifact, name, 1, 0).unwrap();
        assert_eq!((bottom_edge.rows, bottom_edge.cols), (1, 128));
        assert!(bottom_edge.values.iter().all(|value| *value == 4.0));

        let corner = reconstruct_sq8_canonical_tensor_block_f32(&artifact, name, 1, 1).unwrap();
        assert_eq!((corner.start_row, corner.start_col), (128, 128));
        assert_eq!((corner.rows, corner.cols), (1, 2));
        assert_eq!(corner.values, vec![8.0, 8.0]);

        let row = reconstruct_sq8_canonical_tensor_row_f32(&artifact, name, 128).unwrap();
        assert_eq!(row.len(), 130);
        assert!(row[..128].iter().all(|value| *value == 4.0));
        assert_eq!(&row[128..], &[8.0, 8.0]);

        let aliased = reconstruct_sq8_canonical_tensor_block_f32(
            &artifact,
            "model.language_model.layers.0.self_attn.q_proj.weight",
            1,
            1,
        )
        .unwrap();
        assert_eq!(aliased.values, vec![8.0, 8.0]);
    }

    #[test]
    fn rejects_non_two_element_shape_during_parse() {
        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["shape"] = json!([129, 130, 1]);
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("failed to parse"));
    }

    #[test]
    fn rejects_wrong_scale_shape_and_block_shape() {
        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["scale"]["shape"] = json!([1, 2]);
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("scale shape mismatch"));

        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["scale"]["block_shape"] = json!([1, 128]);
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("scale block_shape"));
    }

    #[test]
    fn rejects_path_traversal_and_symlink_escape() {
        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["weight"]["file"] = json!("../outside.f8");
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("normalized relative path"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let (fixture, weight, _) = write_fixture();
            let outside = fixture.root.with_extension("outside");
            fs::write(&outside, &weight).unwrap();
            fs::remove_file(fixture.root.join("weights/q.f8_e4m3")).unwrap();
            symlink(&outside, fixture.root.join("weights/q.f8_e4m3")).unwrap();
            let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
            assert!(err.contains("escapes the artifact directory"));
            fs::remove_file(outside).unwrap();
        }
    }

    #[test]
    fn rejects_file_length_and_checksum_mismatches() {
        let (fixture, weight, _) = write_fixture();
        fs::write(
            fixture.root.join("weights/q.f8_e4m3"),
            &weight[..weight.len() - 1],
        )
        .unwrap();
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("byte length mismatch"));

        let (fixture, mut weight, _) = write_fixture();
        weight.push(0x38);
        fs::write(fixture.root.join("weights/q.f8_e4m3"), weight).unwrap();
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("byte length mismatch"));

        let (fixture, mut weight, _) = write_fixture();
        weight[0] = 0x40;
        fs::write(fixture.root.join("weights/q.f8_e4m3"), weight).unwrap();
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("checksum mismatch"));
    }

    #[test]
    fn reconstruct_rejects_same_size_payload_change_after_read() {
        let (fixture, mut weight, _) = write_fixture();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        weight[0] = 0x40;
        fs::write(fixture.root.join("weights/q.f8_e4m3"), weight).unwrap();

        let err = reconstruct_sq8_canonical_tensor_block_f32(
            &artifact,
            "model.layers.0.self_attn.q_proj.weight",
            0,
            0,
        )
        .unwrap_err();
        assert!(err.contains("pre-reconstruction verification"));
        assert!(err.contains("checksum mismatch"));
    }

    #[test]
    fn reconstruct_rejects_appended_payload_after_read() {
        let (fixture, mut weight, _) = write_fixture();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        weight.push(0x38);
        fs::write(fixture.root.join("weights/q.f8_e4m3"), weight).unwrap();

        let err = reconstruct_sq8_canonical_tensor_block_f32(
            &artifact,
            "model.layers.0.self_attn.q_proj.weight",
            0,
            0,
        )
        .unwrap_err();
        assert!(err.contains("pre-reconstruction verification"));
        assert!(err.contains("byte length mismatch"));
    }

    #[test]
    fn rejects_invalid_coverage_and_duplicate_pairs() {
        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["coverage"]["selected_pair_count"] = json!(2);
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("selected_pair_count"));

        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            let pair = manifest["quantized_tensors"][0].clone();
            manifest["quantized_tensors"]
                .as_array_mut()
                .unwrap()
                .push(pair);
            manifest["coverage"]["source_tensor_count"] = json!(4);
            manifest["coverage"]["source_fp8_weight_count"] = json!(2);
            manifest["coverage"]["source_scale_count"] = json!(2);
            manifest["coverage"]["paired_tensor_count"] = json!(2);
            manifest["coverage"]["selected_pair_count"] = json!(2);
            let weight_bytes = manifest["storage"]["weight_payload_bytes"]
                .as_u64()
                .unwrap();
            let scale_bytes = manifest["storage"]["scale_payload_bytes"].as_u64().unwrap();
            manifest["storage"]["weight_payload_bytes"] = json!(weight_bytes * 2);
            manifest["storage"]["scale_payload_bytes"] = json!(scale_bytes * 2);
            manifest["storage"]["total_payload_bytes"] = json!((weight_bytes + scale_bytes) * 2);
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("name is duplicated"));
    }

    #[test]
    fn rejects_mismatched_pair_name_dtype_and_invalid_values() {
        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["scale"]["name"] =
                json!("model.layers.0.self_attn.k_proj.weight_scale_inv");
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("scale name mismatch"));

        let (fixture, _, _) = write_fixture();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["scale"]["dtype"] = json!("F32");
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("scale dtype"));

        let (fixture, _, _) = write_fixture();
        let mut scale = bf16_bytes(&[0.0, 2.0, 4.0, 8.0]);
        fs::write(fixture.root.join("scales/q.bf16"), &scale).unwrap();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["scale"]["sha256"] = json!(sha256_hex(&scale));
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("invalid BF16 scale"));
        scale.clear();
    }

    #[test]
    fn rejects_non_finite_fp8_bytes_even_when_checksum_matches() {
        let (fixture, mut weight, _) = write_fixture();
        let last = weight.len() - 1;
        weight[last] = 0x7f;
        fs::write(fixture.root.join("weights/q.f8_e4m3"), &weight).unwrap();
        rewrite_manifest(&fixture.root, |manifest| {
            manifest["quantized_tensors"][0]["weight"]["sha256"] = json!(sha256_hex(&weight));
        });
        let err = read_sq8_canonical_artifact(&fixture.root).unwrap_err();
        assert!(err.contains("non-finite E4M3 byte"));
    }
}
