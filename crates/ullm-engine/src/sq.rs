// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::package::TensorSelector;
use serde::Deserialize;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream};

pub const SQ_FP8_ARTIFACT_SCHEMA_VERSION: &str = "sq-fp8-artifact-v0.1";
pub const SQ_FP8_E4M3_DTYPE: &str = "fp8_e4m3";
pub const SQ_F32_SCALE_DTYPE: &str = "f32";

#[derive(Debug, Clone, Deserialize)]
pub struct SqFp8ArtifactManifest {
    pub schema_version: String,
    pub candidate: SqFp8Candidate,
    pub source: Option<SqFp8Source>,
    pub storage: SqFp8Storage,
    #[serde(default)]
    pub fp8_tensors: Vec<SqFp8TensorEntry>,
    #[serde(default)]
    pub passthrough_tensors: Vec<SqFp8PassthroughTensorEntry>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SqFp8Candidate {
    pub id: String,
    pub weight_payload_dtype: String,
    pub activation_dtype: String,
    pub scale_granularity: String,
    pub scale_dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SqFp8Source {
    pub model_dir: Option<String>,
    pub base_package: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SqFp8Storage {
    pub fp8_tensor_count: u64,
    pub passthrough_tensor_count: u64,
    pub fp8_payload_bytes: u64,
    pub fp8_scale_bytes: u64,
    pub passthrough_source_bytes_estimate: u64,
    pub compact_resident_bytes_estimate: u64,
    pub materialized_working_set_bytes_estimate: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SqFp8TensorEntry {
    pub name: String,
    pub family: String,
    pub source_dtype: String,
    #[serde(default)]
    pub shape: Vec<u64>,
    pub elements: u64,
    pub source_file: String,
    pub payload_dtype: String,
    pub payload_file: String,
    pub payload_bytes: u64,
    pub scale_granularity: String,
    pub scale_dtype: String,
    pub scale_file: String,
    pub scale_elements: u64,
    pub scale_bytes: u64,
    pub payload_sha256: Option<String>,
    pub scale_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SqFp8PassthroughTensorEntry {
    pub name: String,
    pub dtype: String,
    #[serde(default)]
    pub shape: Vec<u64>,
    pub elements: u64,
    pub source_file: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct SqFp8Artifact {
    pub artifact_dir: PathBuf,
    pub manifest: SqFp8ArtifactManifest,
}

#[derive(Debug)]
pub struct SqFp8MaterializedRows {
    pub tensor_index: usize,
    pub tensor_name: String,
    pub rows: usize,
    pub cols: usize,
    pub start_row: usize,
    pub row_count: usize,
    pub values: Vec<f32>,
    pub buffer: RuntimeBuffer,
}

#[derive(Debug)]
pub struct SqFp8MaterializedTensor {
    pub tensor_index: usize,
    pub tensor_name: String,
    pub rows: usize,
    pub cols: usize,
    pub buffer: RuntimeBuffer,
}

pub fn read_sq_fp8_artifact(path: impl AsRef<Path>) -> Result<SqFp8Artifact, String> {
    let artifact_dir = path.as_ref();
    let manifest_path = artifact_dir.join("sq_manifest.json");
    let payload = std::fs::read_to_string(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    let manifest: SqFp8ArtifactManifest = serde_json::from_str(&payload)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))?;
    validate_sq_fp8_manifest(artifact_dir, &manifest)?;
    Ok(SqFp8Artifact {
        artifact_dir: artifact_dir.to_path_buf(),
        manifest,
    })
}

pub fn validate_sq_fp8_manifest(
    artifact_dir: &Path,
    manifest: &SqFp8ArtifactManifest,
) -> Result<(), String> {
    if manifest.schema_version != SQ_FP8_ARTIFACT_SCHEMA_VERSION {
        return Err(format!(
            "SQ FP8 artifact schema_version must be {}, got {}",
            SQ_FP8_ARTIFACT_SCHEMA_VERSION, manifest.schema_version
        ));
    }
    if manifest.candidate.weight_payload_dtype != SQ_FP8_E4M3_DTYPE {
        return Err(format!(
            "SQ FP8 candidate weight_payload_dtype must be {}, got {}",
            SQ_FP8_E4M3_DTYPE, manifest.candidate.weight_payload_dtype
        ));
    }
    if manifest.candidate.scale_dtype != SQ_F32_SCALE_DTYPE {
        return Err(format!(
            "SQ FP8 candidate scale_dtype must be {}, got {}",
            SQ_F32_SCALE_DTYPE, manifest.candidate.scale_dtype
        ));
    }
    if usize::try_from(manifest.storage.fp8_tensor_count).ok() != Some(manifest.fp8_tensors.len()) {
        return Err(format!(
            "SQ FP8 storage fp8_tensor_count mismatch: storage={} entries={}",
            manifest.storage.fp8_tensor_count,
            manifest.fp8_tensors.len()
        ));
    }
    if usize::try_from(manifest.storage.passthrough_tensor_count).ok()
        != Some(manifest.passthrough_tensors.len())
    {
        return Err(format!(
            "SQ FP8 storage passthrough_tensor_count mismatch: storage={} entries={}",
            manifest.storage.passthrough_tensor_count,
            manifest.passthrough_tensors.len()
        ));
    }

    let mut payload_bytes_sum = 0_u64;
    let mut scale_bytes_sum = 0_u64;
    for (index, tensor) in manifest.fp8_tensors.iter().enumerate() {
        validate_sq_fp8_tensor_entry(artifact_dir, index, tensor)?;
        payload_bytes_sum = payload_bytes_sum.saturating_add(tensor.payload_bytes);
        scale_bytes_sum = scale_bytes_sum.saturating_add(tensor.scale_bytes);
    }
    if payload_bytes_sum != manifest.storage.fp8_payload_bytes {
        return Err(format!(
            "SQ FP8 storage fp8_payload_bytes mismatch: storage={} entries={payload_bytes_sum}",
            manifest.storage.fp8_payload_bytes
        ));
    }
    if scale_bytes_sum != manifest.storage.fp8_scale_bytes {
        return Err(format!(
            "SQ FP8 storage fp8_scale_bytes mismatch: storage={} entries={scale_bytes_sum}",
            manifest.storage.fp8_scale_bytes
        ));
    }
    Ok(())
}

pub fn select_sq_fp8_tensor_index(
    manifest: &SqFp8ArtifactManifest,
    selector: &TensorSelector,
) -> Result<usize, String> {
    match selector {
        TensorSelector::First => {
            if manifest.fp8_tensors.is_empty() {
                Err("SQ artifact contains no FP8 tensors".to_string())
            } else {
                Ok(0)
            }
        }
        TensorSelector::Index(index) => {
            if *index < manifest.fp8_tensors.len() {
                Ok(*index)
            } else {
                Err(format!(
                    "SQ FP8 tensor index {index} is out of range for {} tensors",
                    manifest.fp8_tensors.len()
                ))
            }
        }
        TensorSelector::Name(name) => select_sq_fp8_tensor_index_by_name(manifest, name),
    }
}

pub fn materialize_sq_fp8_tensor_rows_to_runtime_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    artifact: &SqFp8Artifact,
    selector: &TensorSelector,
    start_row: usize,
    row_count: usize,
) -> Result<SqFp8MaterializedRows, String> {
    let tensor_index = select_sq_fp8_tensor_index(&artifact.manifest, selector)?;
    let tensor = artifact
        .manifest
        .fp8_tensors
        .get(tensor_index)
        .ok_or_else(|| format!("SQ FP8 tensor index {tensor_index} disappeared"))?;
    let (rows, cols) = sq_fp8_tensor_rows_cols(tensor)?;
    let values = materialize_sq_fp8_tensor_rows_to_host_f32(
        &artifact.artifact_dir,
        tensor,
        start_row,
        row_count,
    )?;
    let output_bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ FP8 materialized byte size overflows".to_string())?;
    let mut buffer = context.alloc_buffer(output_bytes).map_err(|err| {
        format!(
            "failed to allocate SQ FP8 materialized output for {}: {err}",
            tensor.name
        )
    })?;
    let mut bytes = Vec::with_capacity(output_bytes);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    buffer
        .copy_from_host(0, &bytes, Some(stream))
        .map_err(|err| {
            format!(
                "failed to copy SQ FP8 materialized rows for {} to runtime: {err}",
                tensor.name
            )
        })?;
    stream.synchronize().map_err(|err| {
        format!(
            "failed to synchronize after SQ FP8 materialized copy for {}: {err}",
            tensor.name
        )
    })?;
    Ok(SqFp8MaterializedRows {
        tensor_index,
        tensor_name: tensor.name.clone(),
        rows,
        cols,
        start_row,
        row_count,
        values,
        buffer,
    })
}

pub fn materialize_named_sq_fp8_tensor_to_runtime_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    artifact: &SqFp8Artifact,
    tensor_name: &str,
    row_chunk: usize,
) -> Result<Option<SqFp8MaterializedTensor>, String> {
    if row_chunk == 0 {
        return Err("SQ FP8 row_chunk must be greater than zero".to_string());
    }
    let Some(tensor_index) =
        find_sq_fp8_tensor_index_by_exact_name(&artifact.manifest, tensor_name)
    else {
        return Ok(None);
    };
    let tensor = artifact
        .manifest
        .fp8_tensors
        .get(tensor_index)
        .ok_or_else(|| format!("SQ FP8 tensor index {tensor_index} disappeared"))?;
    let (rows, cols) = sq_fp8_tensor_rows_cols(tensor)?;
    let output_bytes = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| format!("SQ FP8 materialized byte size overflows for {tensor_name}"))?;
    let mut buffer = context.alloc_buffer(output_bytes).map_err(|err| {
        format!("failed to allocate SQ FP8 materialized output for {tensor_name}: {err}")
    })?;
    for start_row in (0..rows).step_by(row_chunk) {
        let count = row_chunk.min(rows - start_row);
        let values = materialize_sq_fp8_tensor_rows_to_host_f32(
            &artifact.artifact_dir,
            tensor,
            start_row,
            count,
        )?;
        let row_offset_bytes = start_row
            .checked_mul(cols)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| format!("SQ FP8 materialized offset overflows for {tensor_name}"))?;
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        buffer
            .copy_from_host(row_offset_bytes, &bytes, Some(stream))
            .map_err(|err| {
                format!(
                    "failed to copy SQ FP8 materialized rows for {tensor_name} to runtime: {err}"
                )
            })?;
    }
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after SQ FP8 materialized copy for {tensor_name}: {err}")
    })?;
    Ok(Some(SqFp8MaterializedTensor {
        tensor_index,
        tensor_name: tensor.name.clone(),
        rows,
        cols,
        buffer,
    }))
}

pub fn materialize_sq_fp8_tensor_rows_to_host_f32(
    artifact_dir: &Path,
    tensor: &SqFp8TensorEntry,
    start_row: usize,
    row_count: usize,
) -> Result<Vec<f32>, String> {
    let (rows, cols) = sq_fp8_tensor_rows_cols(tensor)?;
    if row_count == 0 {
        return Err("SQ FP8 row_count must be greater than zero".to_string());
    }
    let end_row = start_row
        .checked_add(row_count)
        .ok_or_else(|| "SQ FP8 row range overflows".to_string())?;
    if end_row > rows {
        return Err(format!(
            "SQ FP8 row range out of bounds for {}: start_row={} row_count={} rows={rows}",
            tensor.name, start_row, row_count
        ));
    }

    let payload_path = artifact_relative_path(artifact_dir, &tensor.payload_file, "payload")?;
    let scale_path = artifact_relative_path(artifact_dir, &tensor.scale_file, "scale")?;
    let row_payload_bytes = cols;
    let payload_offset = start_row
        .checked_mul(row_payload_bytes)
        .ok_or_else(|| format!("SQ FP8 payload offset overflows for {}", tensor.name))?;
    let payload_bytes = row_count
        .checked_mul(row_payload_bytes)
        .ok_or_else(|| format!("SQ FP8 payload read size overflows for {}", tensor.name))?;
    let mut payload = read_exact_at(&payload_path, payload_offset as u64, payload_bytes)?;
    if payload.len() != payload_bytes {
        return Err(format!(
            "SQ FP8 payload read length mismatch for {}: expected {payload_bytes} got {}",
            tensor.name,
            payload.len()
        ));
    }

    let scales = match tensor.scale_granularity.as_str() {
        "row" => {
            let scale_offset = start_row
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("SQ FP8 scale offset overflows for {}", tensor.name))?;
            let scale_bytes = row_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("SQ FP8 scale read size overflows for {}", tensor.name))?;
            decode_f32_le_values_exact(&read_exact_at(
                &scale_path,
                scale_offset as u64,
                scale_bytes,
            )?)
            .map_err(|err| {
                format!(
                    "failed to decode SQ FP8 row scales for {}: {err}",
                    tensor.name
                )
            })?
        }
        "tensor" => {
            let values = decode_f32_le_values_exact(&read_exact_at(
                &scale_path,
                0,
                std::mem::size_of::<f32>(),
            )?)
            .map_err(|err| {
                format!(
                    "failed to decode SQ FP8 tensor scale for {}: {err}",
                    tensor.name
                )
            })?;
            vec![values[0]; row_count]
        }
        other => {
            return Err(format!(
                "SQ FP8 tensor {} uses unsupported scale_granularity {other}",
                tensor.name
            ));
        }
    };

    let mut output = Vec::with_capacity(payload.len());
    for row in 0..row_count {
        let scale = scales[row];
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "SQ FP8 tensor {} row {} has invalid scale {scale}",
                tensor.name,
                start_row + row
            ));
        }
        let row_start = row
            .checked_mul(cols)
            .ok_or_else(|| format!("SQ FP8 row offset overflows for {}", tensor.name))?;
        for byte in &payload[row_start..row_start + cols] {
            let value = fp8_e4m3fn_to_f32(*byte);
            if !value.is_finite() {
                return Err(format!(
                    "SQ FP8 tensor {} row {} contains non-finite FP8 byte 0x{byte:02x}",
                    tensor.name,
                    start_row + row
                ));
            }
            output.push(value * scale);
        }
    }
    payload.clear();
    Ok(output)
}

pub fn find_sq_fp8_tensor_index_by_exact_name(
    manifest: &SqFp8ArtifactManifest,
    tensor_name: &str,
) -> Option<usize> {
    manifest
        .fp8_tensors
        .iter()
        .position(|tensor| tensor.name == tensor_name)
}

pub fn sq_fp8_tensor_rows_cols(tensor: &SqFp8TensorEntry) -> Result<(usize, usize), String> {
    if tensor.payload_dtype != SQ_FP8_E4M3_DTYPE {
        return Err(format!(
            "SQ FP8 tensor {} payload_dtype must be {}, got {}",
            tensor.name, SQ_FP8_E4M3_DTYPE, tensor.payload_dtype
        ));
    }
    if tensor.scale_dtype != SQ_F32_SCALE_DTYPE {
        return Err(format!(
            "SQ FP8 tensor {} scale_dtype must be {}, got {}",
            tensor.name, SQ_F32_SCALE_DTYPE, tensor.scale_dtype
        ));
    }
    if tensor.shape.len() != 2 {
        return Err(format!(
            "SQ FP8 tensor {} shape must be 2D, got {:?}",
            tensor.name, tensor.shape
        ));
    }
    let rows_u64 = tensor.shape[0];
    let cols_u64 = tensor.shape[1];
    if rows_u64 == 0 || cols_u64 == 0 {
        return Err(format!(
            "SQ FP8 tensor {} has a zero dimension",
            tensor.name
        ));
    }
    let expected_elements = rows_u64
        .checked_mul(cols_u64)
        .ok_or_else(|| format!("SQ FP8 tensor {} shape overflows", tensor.name))?;
    if expected_elements != tensor.elements {
        return Err(format!(
            "SQ FP8 tensor {} shape elements mismatch: shape={} elements={}",
            tensor.name, expected_elements, tensor.elements
        ));
    }
    if tensor.payload_bytes != tensor.elements {
        return Err(format!(
            "SQ FP8 tensor {} payload_bytes must equal elements for fp8_e4m3: payload_bytes={} elements={}",
            tensor.name, tensor.payload_bytes, tensor.elements
        ));
    }
    let rows = usize::try_from(rows_u64)
        .map_err(|_| format!("SQ FP8 tensor {} row count does not fit usize", tensor.name))?;
    let cols = usize::try_from(cols_u64).map_err(|_| {
        format!(
            "SQ FP8 tensor {} column count does not fit usize",
            tensor.name
        )
    })?;
    let expected_scale_elements = match tensor.scale_granularity.as_str() {
        "row" => rows_u64,
        "tensor" => 1,
        other => {
            return Err(format!(
                "SQ FP8 tensor {} uses unsupported scale_granularity {other}",
                tensor.name
            ));
        }
    };
    if tensor.scale_elements != expected_scale_elements {
        return Err(format!(
            "SQ FP8 tensor {} scale_elements mismatch: expected {} got {}",
            tensor.name, expected_scale_elements, tensor.scale_elements
        ));
    }
    let expected_scale_bytes = tensor
        .scale_elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| format!("SQ FP8 tensor {} scale byte count overflows", tensor.name))?;
    if tensor.scale_bytes != expected_scale_bytes {
        return Err(format!(
            "SQ FP8 tensor {} scale_bytes mismatch: expected {} got {}",
            tensor.name, expected_scale_bytes, tensor.scale_bytes
        ));
    }
    Ok((rows, cols))
}

pub fn fp8_e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0_f32 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        sign * (mantissa as f32) * 2.0_f32.powi(-9)
    } else if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else {
        let significand = 1.0_f32 + (mantissa as f32) / 8.0_f32;
        sign * significand * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}

fn validate_sq_fp8_tensor_entry(
    artifact_dir: &Path,
    index: usize,
    tensor: &SqFp8TensorEntry,
) -> Result<(), String> {
    if tensor.name.is_empty() {
        return Err(format!("SQ FP8 tensor entry {index} has empty name"));
    }
    sq_fp8_tensor_rows_cols(tensor)?;
    let payload_path = artifact_relative_path(artifact_dir, &tensor.payload_file, "payload")?;
    let scale_path = artifact_relative_path(artifact_dir, &tensor.scale_file, "scale")?;
    let payload_len = std::fs::metadata(&payload_path)
        .map_err(|err| format!("failed to stat {}: {err}", payload_path.display()))?
        .len();
    if payload_len != tensor.payload_bytes {
        return Err(format!(
            "SQ FP8 tensor {} payload byte mismatch: manifest={} file={payload_len}",
            tensor.name, tensor.payload_bytes
        ));
    }
    let scale_len = std::fs::metadata(&scale_path)
        .map_err(|err| format!("failed to stat {}: {err}", scale_path.display()))?
        .len();
    if scale_len != tensor.scale_bytes {
        return Err(format!(
            "SQ FP8 tensor {} scale byte mismatch: manifest={} file={scale_len}",
            tensor.name, tensor.scale_bytes
        ));
    }
    Ok(())
}

fn select_sq_fp8_tensor_index_by_name(
    manifest: &SqFp8ArtifactManifest,
    name: &str,
) -> Result<usize, String> {
    if let Some((index, _)) = manifest
        .fp8_tensors
        .iter()
        .enumerate()
        .find(|(_, tensor)| tensor.name == name)
    {
        return Ok(index);
    }
    let matches: Vec<usize> = manifest
        .fp8_tensors
        .iter()
        .enumerate()
        .filter_map(|(index, tensor)| tensor.name.contains(name).then_some(index))
        .collect();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(format!("no SQ FP8 tensor matched selector \"{name}\"")),
        _ => Err(format!(
            "SQ FP8 tensor selector \"{name}\" matched {} tensors; use an exact name or numeric index",
            matches.len()
        )),
    }
}

fn artifact_relative_path(
    artifact_dir: &Path,
    relative: &str,
    label: &str,
) -> Result<PathBuf, String> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute() {
        return Err(format!("SQ FP8 {label} path must be relative: {relative}"));
    }
    for component in relative_path.components() {
        if matches!(component, Component::ParentDir | Component::Prefix(_)) {
            return Err(format!(
                "SQ FP8 {label} path must not escape artifact dir: {relative}"
            ));
        }
    }
    Ok(artifact_dir.join(relative_path))
}

fn read_exact_at(path: &Path, offset: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| format!("failed to seek {} to {offset}: {err}", path.display()))?;
    let mut buffer = vec![0_u8; len];
    file.read_exact(&mut buffer)
        .map_err(|err| format!("failed to read {len} bytes from {}: {err}", path.display()))?;
    Ok(buffer)
}

fn decode_f32_le_values_exact(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!(
            "f32 byte length must be a multiple of 4, got {}",
            bytes.len()
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        let mut raw = [0_u8; 4];
        raw.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(raw));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_artifact_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ullm-engine-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn fp8_e4m3fn_decode_matches_pytorch_reference_bytes() {
        let cases = [
            (0x00, 0.0_f32),
            (0x01, 0.001953125_f32),
            (0x02, 0.00390625_f32),
            (0x08, 0.015625_f32),
            (0x30, 0.5_f32),
            (0x38, 1.0_f32),
            (0xb8, -1.0_f32),
            (0x40, 2.0_f32),
            (0x77, 240.0_f32),
            (0x78, 256.0_f32),
            (0x7e, 448.0_f32),
            (0xfe, -448.0_f32),
        ];
        for (byte, expected) in cases {
            assert_eq!(fp8_e4m3fn_to_f32(byte), expected);
        }
        assert!(fp8_e4m3fn_to_f32(0x7f).is_nan());
        assert!(fp8_e4m3fn_to_f32(0xff).is_nan());
    }

    #[test]
    fn materializes_row_scaled_fp8_payload() {
        let root = temp_artifact_dir("sq-fp8-row-materialize-test");
        fs::create_dir_all(root.join("fp8")).unwrap();
        fs::create_dir_all(root.join("scales")).unwrap();
        fs::write(root.join("fp8/a.fp8_e4m3"), [0x38_u8, 0x40, 0xb8, 0x30]).unwrap();
        fs::write(root.join("scales/a.scale_f32"), f32_bytes(&[2.0, 0.5])).unwrap();
        fs::write(
            root.join("sq_manifest.json"),
            r#"{
              "schema_version": "sq-fp8-artifact-v0.1",
              "candidate": {
                "id": "sq-fp8-w8a16-r9700-v0",
                "weight_payload_dtype": "fp8_e4m3",
                "activation_dtype": "bf16_or_f32",
                "scale_granularity": "row",
                "scale_dtype": "f32"
              },
              "source": {"model_dir": null, "base_package": null},
              "storage": {
                "fp8_tensor_count": 1,
                "passthrough_tensor_count": 0,
                "fp8_payload_bytes": 4,
                "fp8_scale_bytes": 8,
                "passthrough_source_bytes_estimate": 0,
                "compact_resident_bytes_estimate": 12,
                "materialized_working_set_bytes_estimate": 16
              },
              "fp8_tensors": [{
                "name": "layer.0.mlp.gate_proj.weight",
                "family": "mlp_gate",
                "source_dtype": "F32",
                "shape": [2, 2],
                "elements": 4,
                "source_file": "/tmp/source.safetensors",
                "payload_dtype": "fp8_e4m3",
                "payload_file": "fp8/a.fp8_e4m3",
                "payload_bytes": 4,
                "scale_granularity": "row",
                "scale_dtype": "f32",
                "scale_file": "scales/a.scale_f32",
                "scale_elements": 2,
                "scale_bytes": 8
              }],
              "passthrough_tensors": []
            }"#,
        )
        .unwrap();

        let artifact = read_sq_fp8_artifact(&root).unwrap();
        let tensor = &artifact.manifest.fp8_tensors[0];
        let values = materialize_sq_fp8_tensor_rows_to_host_f32(&root, tensor, 0, 2).unwrap();
        assert_eq!(values, vec![2.0, 4.0, -0.5, 0.25]);
        let second_row = materialize_sq_fp8_tensor_rows_to_host_f32(&root, tensor, 1, 1).unwrap();
        assert_eq!(second_row, vec![-0.5, 0.25]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selector_rejects_ambiguous_substrings() {
        let manifest = SqFp8ArtifactManifest {
            schema_version: SQ_FP8_ARTIFACT_SCHEMA_VERSION.to_string(),
            candidate: SqFp8Candidate {
                id: "sq-fp8-w8a16-r9700-v0".to_string(),
                weight_payload_dtype: SQ_FP8_E4M3_DTYPE.to_string(),
                activation_dtype: "bf16_or_f32".to_string(),
                scale_granularity: "row".to_string(),
                scale_dtype: SQ_F32_SCALE_DTYPE.to_string(),
            },
            source: None,
            storage: SqFp8Storage {
                fp8_tensor_count: 2,
                passthrough_tensor_count: 0,
                fp8_payload_bytes: 0,
                fp8_scale_bytes: 0,
                passthrough_source_bytes_estimate: 0,
                compact_resident_bytes_estimate: 0,
                materialized_working_set_bytes_estimate: 0,
            },
            fp8_tensors: vec![
                SqFp8TensorEntry {
                    name: "layer.0.q_proj.weight".to_string(),
                    family: "attn_q".to_string(),
                    source_dtype: "F32".to_string(),
                    shape: vec![1, 1],
                    elements: 1,
                    source_file: "/tmp/a".to_string(),
                    payload_dtype: SQ_FP8_E4M3_DTYPE.to_string(),
                    payload_file: "fp8/a".to_string(),
                    payload_bytes: 1,
                    scale_granularity: "row".to_string(),
                    scale_dtype: SQ_F32_SCALE_DTYPE.to_string(),
                    scale_file: "scales/a".to_string(),
                    scale_elements: 1,
                    scale_bytes: 4,
                    payload_sha256: None,
                    scale_sha256: None,
                },
                SqFp8TensorEntry {
                    name: "layer.0.k_proj.weight".to_string(),
                    family: "attn_k".to_string(),
                    source_dtype: "F32".to_string(),
                    shape: vec![1, 1],
                    elements: 1,
                    source_file: "/tmp/b".to_string(),
                    payload_dtype: SQ_FP8_E4M3_DTYPE.to_string(),
                    payload_file: "fp8/b".to_string(),
                    payload_bytes: 1,
                    scale_granularity: "row".to_string(),
                    scale_dtype: SQ_F32_SCALE_DTYPE.to_string(),
                    scale_file: "scales/b".to_string(),
                    scale_elements: 1,
                    scale_bytes: 4,
                    payload_sha256: None,
                    scale_sha256: None,
                },
            ],
            passthrough_tensors: Vec::new(),
            notes: Vec::new(),
        };

        assert_eq!(
            select_sq_fp8_tensor_index(&manifest, &TensorSelector::Name("q_proj".to_string()))
                .unwrap(),
            0
        );
        let err =
            select_sq_fp8_tensor_index(&manifest, &TensorSelector::Name("layer.0".to_string()))
                .unwrap_err();
        assert!(err.contains("matched 2 tensors"));
    }
}
