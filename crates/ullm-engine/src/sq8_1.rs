// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Strict reader and CPU reference paths for isolated `SQ8_1` artifacts.
//!
//! `SQ8_1` deliberately has no compatibility aliases and does not share a
//! manifest or payload decoder with `SQ8_0`.  This keeps the separately
//! aligned I8 payload plane and F16 scale plane from changing an existing
//! format's ABI or release semantics.

use crate::format_id::FORMAT_SQ8_1;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub const SQ8_1_ARTIFACT_SCHEMA_VERSION: &str = "sq8_1-artifact-v0.1";
pub const SQ8_1_ARTIFACT_KIND: &str = "sq8_1_block_int8";
pub const SQ8_1_GROUP_SIZE: usize = 32;
pub const SQ8_1_PAYLOAD_ALIGNMENT_BYTES: usize = 16;
pub const SQ8_1_MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
pub const SQ8_1_VERIFY_CHUNK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneArtifactManifest {
    pub schema_version: String,
    pub artifact_kind: String,
    pub format_id: String,
    pub endianness: String,
    pub group_size: u64,
    pub source: Sq8OneSource,
    pub storage: Sq8OneStorage,
    pub tensors: Vec<Sq8OneTensorEntry>,
    pub integrity: Sq8OneIntegrity,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneSource {
    pub format_id: String,
    pub schema_version: String,
    pub manifest_sha256: String,
    pub contract: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneIntegrity {
    pub content_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneStorage {
    pub payload_bytes: u64,
    pub scale_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneTensorEntry {
    pub name: String,
    pub shape: [u64; 2],
    pub elements: u64,
    pub payload: Sq8OnePayloadPlane,
    pub scale: Sq8OneScalePlane,
    pub storage: Sq8OneTensorStorage,
    pub quantization: Sq8OneQuantizationStats,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OnePayloadPlane {
    pub file: String,
    pub dtype: String,
    pub bytes: u64,
    pub sha256: String,
    pub row_stride: u64,
    pub alignment_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneScalePlane {
    pub file: String,
    pub dtype: String,
    pub bytes: u64,
    pub sha256: String,
    pub shape: [u64; 2],
    pub order: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneTensorStorage {
    pub nominal_full_block_bpp: f64,
    pub actual_bpp: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sq8OneQuantizationStats {
    pub values: u64,
    pub blocks: u64,
    pub post_storage_clipping_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8OneChecksumReport {
    pub tensor_count: u64,
    pub payload_bytes: u64,
    pub scale_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Sq8OneArtifact {
    artifact_dir: PathBuf,
    manifest: Sq8OneArtifactManifest,
    checksum_report: Sq8OneChecksumReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8OneTensor {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub payload_row_stride: usize,
    pub payload: Vec<u8>,
    pub scales_f16_le: Vec<u8>,
}

impl Sq8OneArtifact {
    pub fn artifact_dir(&self) -> &Path {
        &self.artifact_dir
    }

    pub fn manifest(&self) -> &Sq8OneArtifactManifest {
        &self.manifest
    }

    pub fn checksum_report(&self) -> &Sq8OneChecksumReport {
        &self.checksum_report
    }

    pub fn read_tensor(&self, tensor_name: &str) -> Result<Sq8OneTensor, String> {
        let entry = self
            .manifest
            .tensors
            .iter()
            .find(|entry| entry.name == tensor_name)
            .ok_or_else(|| format!("SQ8_1 artifact has no tensor named {tensor_name:?}"))?;
        let rows = checked_usize(entry.shape[0], "SQ8_1 tensor rows")?;
        let cols = checked_usize(entry.shape[1], "SQ8_1 tensor cols")?;
        let stride = checked_usize(entry.payload.row_stride, "SQ8_1 payload row_stride")?;
        let payload_path = artifact_file(&self.artifact_dir, &entry.payload.file, "payload")?;
        let scale_path = artifact_file(&self.artifact_dir, &entry.scale.file, "scale")?;
        let tensor = Sq8OneTensor {
            name: entry.name.clone(),
            rows,
            cols,
            payload_row_stride: stride,
            payload: std::fs::read(&payload_path)
                .map_err(|err| format!("failed to read {}: {err}", payload_path.display()))?,
            scales_f16_le: std::fs::read(&scale_path)
                .map_err(|err| format!("failed to read {}: {err}", scale_path.display()))?,
        };
        validate_tensor_memory(&tensor)?;
        Ok(tensor)
    }
}

impl Sq8OneTensor {
    pub fn groups_per_row(&self) -> usize {
        groups_per_row(self.cols).expect("validated SQ8_1 tensor cols")
    }

    pub fn code(&self, row: usize, col: usize) -> Result<i8, String> {
        if row >= self.rows || col >= self.cols {
            return Err("SQ8_1 payload index is out of bounds".to_string());
        }
        Ok(self.payload[row * self.payload_row_stride + col] as i8)
    }

    pub fn scale(&self, row: usize, block: usize) -> Result<f32, String> {
        if row >= self.rows || block >= self.groups_per_row() {
            return Err("SQ8_1 scale index is out of bounds".to_string());
        }
        let offset = 2 * (row * self.groups_per_row() + block);
        Ok(f16_bits_to_f32(u16::from_le_bytes([
            self.scales_f16_le[offset],
            self.scales_f16_le[offset + 1],
        ])))
    }

    pub fn reconstruct_row(&self, row: usize) -> Result<Vec<f32>, String> {
        if row >= self.rows {
            return Err("SQ8_1 row index is out of bounds".to_string());
        }
        (0..self.cols)
            .map(|col| Ok(self.code(row, col)? as f32 * self.scale(row, col / SQ8_1_GROUP_SIZE)?))
            .collect()
    }
}

pub fn read_sq8_1_artifact(path: impl AsRef<Path>) -> Result<Sq8OneArtifact, String> {
    let input = path.as_ref();
    let input_metadata = std::fs::symlink_metadata(input)
        .map_err(|err| format!("failed to stat SQ8_1 artifact {}: {err}", input.display()))?;
    if input_metadata.file_type().is_symlink() || !input_metadata.is_dir() {
        return Err(format!(
            "SQ8_1 artifact must be a non-symlink directory: {}",
            input.display()
        ));
    }
    let artifact_dir = std::fs::canonicalize(input).map_err(|err| {
        format!(
            "failed to canonicalize SQ8_1 artifact {}: {err}",
            input.display()
        )
    })?;
    let manifest_path = artifact_dir.join("sq8_1_manifest.json");
    let manifest_metadata = std::fs::symlink_metadata(&manifest_path)
        .map_err(|err| format!("failed to stat {}: {err}", manifest_path.display()))?;
    if manifest_metadata.file_type().is_symlink() || !manifest_metadata.is_file() {
        return Err(format!(
            "SQ8_1 manifest must be a regular non-symlink file: {}",
            manifest_path.display()
        ));
    }
    if manifest_metadata.len() > SQ8_1_MAX_MANIFEST_BYTES {
        return Err(format!(
            "SQ8_1 manifest is too large: {} bytes exceeds {}",
            manifest_metadata.len(),
            SQ8_1_MAX_MANIFEST_BYTES
        ));
    }
    let manifest_text = std::fs::read_to_string(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    let manifest_value: Value = serde_json::from_str(&manifest_text)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))?;
    let manifest: Sq8OneArtifactManifest = serde_json::from_value(manifest_value.clone())
        .map_err(|err| format!("failed to decode {}: {err}", manifest_path.display()))?;
    verify_manifest_content_sha256(&manifest_value, &manifest.integrity.content_sha256)?;
    let checksum_report = validate_manifest_and_payloads(&artifact_dir, &manifest)?;
    Ok(Sq8OneArtifact {
        artifact_dir,
        manifest,
        checksum_report,
    })
}

pub fn read_sq8_1_tensor(
    artifact: impl AsRef<Path>,
    tensor_name: &str,
) -> Result<Sq8OneTensor, String> {
    read_sq8_1_artifact(artifact)?.read_tensor(tensor_name)
}

/// Default SQ8_1 path: convert signed I8 weights to F32 and apply each
/// weight block's scale.  It intentionally does not quantize activations.
pub fn matvec_w8a16(tensor: &Sq8OneTensor, activation: &[f32]) -> Result<Vec<f32>, String> {
    validate_tensor_memory(tensor)?;
    if activation.len() != tensor.cols || activation.iter().any(|value| !value.is_finite()) {
        return Err("SQ8_1 W8A16 activation must be finite and match tensor cols".to_string());
    }
    let mut output = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let mut total = 0.0_f32;
        for block in 0..tensor.groups_per_row() {
            let start = block * SQ8_1_GROUP_SIZE;
            let end = (start + SQ8_1_GROUP_SIZE).min(tensor.cols);
            let mut partial = 0.0_f32;
            for col in start..end {
                partial += tensor.code(row, col)? as f32 * activation[col];
            }
            total += partial * tensor.scale(row, block)?;
        }
        output.push(total);
    }
    Ok(output)
}

/// Opt-in SQ8_1 W8A8 reference: one signed I32 dot per K=32 block followed
/// by that block's `s_w * s_a`.  Callers select this explicitly; there is no
/// implicit W8A8 dispatch or fallback from the W8A16 default.
pub fn matvec_w8a8_explicit(tensor: &Sq8OneTensor, activation: &[f32]) -> Result<Vec<f32>, String> {
    validate_tensor_memory(tensor)?;
    if activation.len() != tensor.cols || activation.iter().any(|value| !value.is_finite()) {
        return Err("SQ8_1 W8A8 activation must be finite and match tensor cols".to_string());
    }
    let (activation_codes, activation_scales) = quantize_activation(activation)?;
    let mut output = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let mut total = 0.0_f32;
        for block in 0..tensor.groups_per_row() {
            let start = block * SQ8_1_GROUP_SIZE;
            let end = (start + SQ8_1_GROUP_SIZE).min(tensor.cols);
            let mut dot = 0_i32;
            for col in start..end {
                dot += tensor.code(row, col)? as i32 * activation_codes[col] as i32;
            }
            total += dot as f32 * tensor.scale(row, block)? * activation_scales[block];
        }
        output.push(total);
    }
    Ok(output)
}

pub fn quantize_activation(values: &[f32]) -> Result<(Vec<i8>, Vec<f32>), String> {
    if values.is_empty() {
        return Err("SQ8_1 activation vector must not be empty".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("SQ8_1 activation contains non-finite values".to_string());
    }
    let groups = groups_per_row(values.len())?;
    let mut codes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(groups);
    for block in 0..groups {
        let start = block * SQ8_1_GROUP_SIZE;
        let end = (start + SQ8_1_GROUP_SIZE).min(values.len());
        let (block_codes, scale) = quantize_block(&values[start..end])?;
        codes.extend(block_codes);
        scales.push(scale);
    }
    Ok((codes, scales))
}

fn validate_manifest_and_payloads(
    artifact_dir: &Path,
    manifest: &Sq8OneArtifactManifest,
) -> Result<Sq8OneChecksumReport, String> {
    if manifest.schema_version != SQ8_1_ARTIFACT_SCHEMA_VERSION
        || manifest.artifact_kind != SQ8_1_ARTIFACT_KIND
        || manifest.format_id != FORMAT_SQ8_1
        || manifest.endianness != "little"
        || manifest.group_size != SQ8_1_GROUP_SIZE as u64
    {
        return Err("SQ8_1 manifest format contract is invalid".to_string());
    }
    if manifest.source.format_id != "SQ8_0"
        || manifest.source.schema_version != "sq-fp8-artifact-v0.2"
        || manifest.source.contract != "reconstructed_row_major_f32_from_verified_sq8_0_canonical"
    {
        return Err("SQ8_1 source contract is invalid".to_string());
    }
    validate_sha256(
        &manifest.source.manifest_sha256,
        "SQ8_1 source.manifest_sha256",
    )?;
    validate_sha256(
        &manifest.integrity.content_sha256,
        "SQ8_1 integrity.content_sha256",
    )?;
    if manifest.tensors.is_empty() {
        return Err("SQ8_1 manifest has no tensors".to_string());
    }

    let mut previous_name = "";
    let mut names = BTreeSet::new();
    let mut files = BTreeSet::new();
    let mut payload_total = 0_u64;
    let mut scale_total = 0_u64;
    for entry in &manifest.tensors {
        if entry.name.is_empty()
            || entry.name.as_str() <= previous_name
            || !names.insert(&entry.name)
        {
            return Err("SQ8_1 tensor names must be unique and sorted".to_string());
        }
        previous_name = &entry.name;
        let rows = checked_usize(entry.shape[0], "SQ8_1 tensor rows")?;
        let cols = checked_usize(entry.shape[1], "SQ8_1 tensor cols")?;
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| "SQ8_1 tensor element count overflows".to_string())?;
        if entry.elements != elements as u64 {
            return Err(format!(
                "SQ8_1 tensor {} element count is invalid",
                entry.name
            ));
        }
        let stride = payload_row_stride(cols)?;
        if entry.payload.dtype != "I8"
            || entry.payload.alignment_bytes != SQ8_1_PAYLOAD_ALIGNMENT_BYTES as u64
            || entry.payload.row_stride != stride as u64
            || entry.scale.dtype != "F16"
            || entry.scale.order != "row_major"
            || entry.scale.shape != [rows as u64, groups_per_row(cols)? as u64]
            || entry.storage.nominal_full_block_bpp.to_bits() != 8.5_f64.to_bits()
        {
            return Err(format!(
                "SQ8_1 tensor {} plane contract is invalid",
                entry.name
            ));
        }
        let expected_bpp = actual_bpp(cols, stride)?;
        if (entry.storage.actual_bpp - expected_bpp).abs() > f64::EPSILON {
            return Err(format!(
                "SQ8_1 tensor {} bpp accounting is invalid",
                entry.name
            ));
        }
        let expected_payload_bytes = rows
            .checked_mul(stride)
            .ok_or_else(|| "SQ8_1 payload byte count overflows".to_string())?;
        let expected_scale_bytes = rows
            .checked_mul(groups_per_row(cols)?)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| "SQ8_1 scale byte count overflows".to_string())?;
        let expected_blocks = rows
            .checked_mul(groups_per_row(cols)?)
            .ok_or_else(|| "SQ8_1 block count overflows".to_string())?;
        if entry.payload.bytes != expected_payload_bytes as u64
            || entry.scale.bytes != expected_scale_bytes as u64
            || entry.quantization.values != elements as u64
            || entry.quantization.blocks != expected_blocks as u64
            || entry.quantization.post_storage_clipping_count != 0
        {
            return Err(format!("SQ8_1 tensor {} accounting is invalid", entry.name));
        }
        validate_sha256(&entry.payload.sha256, "SQ8_1 payload SHA-256")?;
        validate_sha256(&entry.scale.sha256, "SQ8_1 scale SHA-256")?;
        if !files.insert(&entry.payload.file) || !files.insert(&entry.scale.file) {
            return Err("SQ8_1 payload and scale files must be unique".to_string());
        }
        let payload_path = artifact_file(artifact_dir, &entry.payload.file, "payload")?;
        let scale_path = artifact_file(artifact_dir, &entry.scale.file, "scale")?;
        verify_file_sha256(&payload_path, entry.payload.bytes, &entry.payload.sha256)?;
        verify_file_sha256(&scale_path, entry.scale.bytes, &entry.scale.sha256)?;
        validate_tensor_files(&payload_path, &scale_path, rows, cols, stride)?;
        payload_total = payload_total
            .checked_add(expected_payload_bytes as u64)
            .ok_or_else(|| "SQ8_1 payload byte sum overflows".to_string())?;
        scale_total = scale_total
            .checked_add(expected_scale_bytes as u64)
            .ok_or_else(|| "SQ8_1 scale byte sum overflows".to_string())?;
    }
    if manifest.storage.payload_bytes != payload_total
        || manifest.storage.scale_bytes != scale_total
        || manifest.storage.total_bytes
            != payload_total
                .checked_add(scale_total)
                .ok_or_else(|| "SQ8_1 total byte count overflows".to_string())?
    {
        return Err("SQ8_1 aggregate storage accounting is invalid".to_string());
    }
    Ok(Sq8OneChecksumReport {
        tensor_count: manifest.tensors.len() as u64,
        payload_bytes: payload_total,
        scale_bytes: scale_total,
    })
}

fn validate_tensor_files(
    payload_path: &Path,
    scale_path: &Path,
    rows: usize,
    cols: usize,
    stride: usize,
) -> Result<(), String> {
    let payload = std::fs::read(payload_path)
        .map_err(|err| format!("failed to read {}: {err}", payload_path.display()))?;
    let scales = std::fs::read(scale_path)
        .map_err(|err| format!("failed to read {}: {err}", scale_path.display()))?;
    validate_planes(&payload, &scales, rows, cols, stride)
}

fn validate_tensor_memory(tensor: &Sq8OneTensor) -> Result<(), String> {
    if tensor.rows == 0
        || tensor.cols == 0
        || tensor.payload_row_stride != payload_row_stride(tensor.cols)?
    {
        return Err("SQ8_1 tensor shape or payload row stride is invalid".to_string());
    }
    validate_planes(
        &tensor.payload,
        &tensor.scales_f16_le,
        tensor.rows,
        tensor.cols,
        tensor.payload_row_stride,
    )
}

fn validate_planes(
    payload: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    stride: usize,
) -> Result<(), String> {
    let groups = groups_per_row(cols)?;
    if payload.len()
        != rows
            .checked_mul(stride)
            .ok_or_else(|| "SQ8_1 payload length overflows".to_string())?
        || scales.len()
            != rows
                .checked_mul(groups)
                .and_then(|value| value.checked_mul(2))
                .ok_or_else(|| "SQ8_1 scale length overflows".to_string())?
    {
        return Err("SQ8_1 tensor plane lengths do not match the shape".to_string());
    }
    for row in 0..rows {
        let plane = &payload[row * stride..(row + 1) * stride];
        if plane[..cols].contains(&0x80) {
            return Err("SQ8_1 payload contains forbidden I8 code -128".to_string());
        }
        if plane[cols..].iter().any(|byte| *byte != 0) {
            return Err("SQ8_1 payload physical tail padding is nonzero".to_string());
        }
    }
    for raw in scales.chunks_exact(2) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
        if !scale.is_finite() || scale <= 0.0 {
            return Err("SQ8_1 scale must be finite and strictly positive".to_string());
        }
    }
    Ok(())
}

fn verify_manifest_content_sha256(value: &Value, expected: &str) -> Result<(), String> {
    validate_sha256(expected, "SQ8_1 integrity.content_sha256")?;
    let mut without_integrity = value.clone();
    let object = without_integrity
        .as_object_mut()
        .ok_or_else(|| "SQ8_1 manifest root must be an object".to_string())?;
    if object.remove("integrity").is_none() {
        return Err("SQ8_1 manifest has no integrity object".to_string());
    }
    let canonical = serde_json::to_vec(&without_integrity)
        .map_err(|err| format!("failed to canonicalize SQ8_1 manifest: {err}"))?;
    let actual = sha256_hex(&canonical);
    if actual != expected {
        return Err("SQ8_1 manifest content SHA-256 mismatch".to_string());
    }
    Ok(())
}

fn artifact_file(root: &Path, relative: &str, label: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if relative.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "SQ8_1 {label} path must be normalized and relative: {relative:?}"
        ));
    }
    let joined = root.join(path);
    let metadata = std::fs::symlink_metadata(&joined)
        .map_err(|err| format!("failed to stat SQ8_1 {label} {}: {err}", joined.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "SQ8_1 {label} must be a regular non-symlink file: {}",
            joined.display()
        ));
    }
    let resolved = std::fs::canonicalize(&joined).map_err(|err| {
        format!(
            "failed to resolve SQ8_1 {label} {}: {err}",
            joined.display()
        )
    })?;
    if !resolved.starts_with(root) {
        return Err(format!("SQ8_1 {label} path escapes artifact: {relative:?}"));
    }
    Ok(resolved)
}

fn verify_file_sha256(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    let metadata = std::fs::metadata(path)
        .map_err(|err| format!("failed to stat {}: {err}", path.display()))?;
    if metadata.len() != expected_bytes {
        return Err(format!(
            "SQ8_1 plane byte count mismatch: {}",
            path.display()
        ));
    }
    let mut handle =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; SQ8_1_VERIFY_CHUNK_BYTES];
    loop {
        let count = handle
            .read(&mut buffer)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected_sha256 {
        return Err(format!("SQ8_1 plane SHA-256 mismatch: {}", path.display()));
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
            "{label} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_usize(value: u64, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{label} exceeds host address space"))
}

fn groups_per_row(cols: usize) -> Result<usize, String> {
    if cols == 0 {
        return Err("SQ8_1 cols must be positive".to_string());
    }
    cols.checked_add(SQ8_1_GROUP_SIZE - 1)
        .map(|value| value / SQ8_1_GROUP_SIZE)
        .ok_or_else(|| "SQ8_1 group count overflows".to_string())
}

fn payload_row_stride(cols: usize) -> Result<usize, String> {
    if cols == 0 {
        return Err("SQ8_1 cols must be positive".to_string());
    }
    cols.checked_add(SQ8_1_PAYLOAD_ALIGNMENT_BYTES - 1)
        .map(|value| value / SQ8_1_PAYLOAD_ALIGNMENT_BYTES * SQ8_1_PAYLOAD_ALIGNMENT_BYTES)
        .ok_or_else(|| "SQ8_1 payload row stride overflows".to_string())
}

fn actual_bpp(cols: usize, stride: usize) -> Result<f64, String> {
    if stride != payload_row_stride(cols)? {
        return Err("SQ8_1 row stride violates the alignment rule".to_string());
    }
    let scale_bytes = groups_per_row(cols)?
        .checked_mul(2)
        .ok_or_else(|| "SQ8_1 scale-byte count overflows".to_string())?;
    let bytes_per_row = stride
        .checked_add(scale_bytes)
        .ok_or_else(|| "SQ8_1 bytes-per-row count overflows".to_string())?;
    Ok(8.0 * bytes_per_row as f64 / cols as f64)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x03ff) as f32;
    match exponent {
        0 if mantissa == 0.0 => sign * 0.0,
        0 => sign * mantissa * 2.0_f32.powi(-24),
        0x1f if mantissa == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}

fn round_shift_ties_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let quotient = value >> shift;
    let remainder = value & ((1_u32 << shift) - 1);
    let halfway = 1_u32 << (shift - 1);
    if remainder > halfway || (remainder == halfway && quotient & 1 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

fn f32_to_f16_bits_rne(value: f32) -> u16 {
    let raw = value.to_bits();
    let sign = ((raw >> 16) & 0x8000) as u16;
    let exponent_bits = (raw >> 23) & 0xff;
    let mantissa = raw & 0x7f_ffff;
    if exponent_bits == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let exponent = exponent_bits as i32 - 127;
    if exponent > 15 {
        return sign | 0x7c00;
    }
    if exponent >= -14 {
        let mut rounded = round_shift_ties_even(mantissa, 13);
        let mut half_exponent = (exponent + 15) as u16;
        if rounded == 0x400 {
            rounded = 0;
            half_exponent += 1;
            if half_exponent >= 0x1f {
                return sign | 0x7c00;
            }
        }
        return sign | (half_exponent << 10) | rounded as u16;
    }
    if exponent < -25 {
        return sign;
    }
    let significand = mantissa | 0x80_0000;
    let subnormal = round_shift_ties_even(significand, (-exponent - 1) as u32);
    sign | subnormal as u16
}

fn ceil_f16(value: f32) -> Result<f32, String> {
    if !value.is_finite() || value <= 0.0 {
        return Err("SQ8_1 ceil-F16 scale must be finite and positive".to_string());
    }
    let mut bits = f32_to_f16_bits_rne(value);
    if bits == 0 {
        return Ok(f16_bits_to_f32(1));
    }
    if bits >= 0x7c00 {
        return Err("SQ8_1 FP16 scale overflow".to_string());
    }
    if f16_bits_to_f32(bits) < value {
        bits += 1;
        if bits >= 0x7c00 {
            return Err("SQ8_1 FP16 scale overflow".to_string());
        }
    }
    Ok(f16_bits_to_f32(bits))
}

fn quantize_block(values: &[f32]) -> Result<(Vec<i8>, f32), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("SQ8_1 quantization source block must be finite and non-empty".to_string());
    }
    let maximum = values
        .iter()
        .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
    if maximum == 0.0 {
        return Ok((vec![0; values.len()], 1.0));
    }
    let scale = ceil_f16(maximum / 127.0)?;
    let codes = values
        .iter()
        .map(|value| (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect();
    Ok((codes, scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_tensor() -> Sq8OneTensor {
        Sq8OneTensor {
            name: "fixture".to_string(),
            rows: 2,
            cols: 33,
            payload_row_stride: 48,
            payload: {
                let mut bytes = vec![0_u8; 96];
                for col in 0..33 {
                    bytes[col] = (col as i8 - 16) as u8;
                    bytes[48 + col] = (16 - col as i8) as u8;
                }
                bytes
            },
            scales_f16_le: vec![0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c],
        }
    }

    #[test]
    fn layout_and_references_cover_k32_tail() {
        let tensor = fixture_tensor();
        validate_tensor_memory(&tensor).unwrap();
        let activation = (0..33).map(|value| value as f32 / 32.0).collect::<Vec<_>>();
        let w8a16 = matvec_w8a16(&tensor, &activation).unwrap();
        assert_eq!(w8a16.len(), 2);
        assert_eq!(w8a16[0], -w8a16[1]);
        let w8a8 = matvec_w8a8_explicit(&tensor, &activation).unwrap();
        assert_eq!(w8a8.len(), 2);
        assert_eq!(w8a8[0], -w8a8[1]);
    }

    #[test]
    fn f16_scale_ceiling_is_upward_and_signed_codes_exclude_negative_128() {
        let scale = ceil_f16(1.0001).unwrap();
        assert_eq!(scale.to_bits(), 1.0009765625_f32.to_bits());
        let (codes, _) = quantize_block(&[-127.0, 0.0, 127.0]).unwrap();
        assert_eq!(codes, vec![-127, 0, 127]);
    }

    #[test]
    fn strict_format_id_does_not_alias_sq8_0() {
        assert_eq!(FORMAT_SQ8_1, "SQ8_1");
    }
}
