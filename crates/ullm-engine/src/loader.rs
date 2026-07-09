// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::package::{
    PackageSummary, PassthroughPayloadBundle, ReferencedFile, ReferencedFileRole,
    RowScaleOverrideEntry, TensorPayloadBundle, TensorSelector, list_tensor_payload_bundles,
    select_exact_passthrough_payload_bundle, select_passthrough_payload_bundle,
    select_tensor_payload_bundle,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream};

pub const PASSTHROUGH_MAX_STREAM_CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    pub chunk_bytes: usize,
    pub verify: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            chunk_bytes: 1024 * 1024,
            verify: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedPayload {
    pub role: ReferencedFileRole,
    pub relative_path: String,
    pub bytes: u64,
    pub chunks: u64,
    pub buffer: Arc<RuntimeBuffer>,
}

#[derive(Debug)]
pub struct LoadedTensorBundle {
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
    pub index: LoadedPayload,
    pub scale: LoadedPayload,
    pub codebook: LoadedPayload,
}

#[derive(Debug, Clone)]
pub struct PassthroughF32Data {
    pub values: Vec<f32>,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct PassthroughF32Rows {
    pub values: Vec<f32>,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub row_indices: Vec<usize>,
    pub columns: usize,
}

#[derive(Debug, Clone)]
pub struct PassthroughF32RowRange {
    pub values: Vec<f32>,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub start_row: usize,
    pub row_count: usize,
    pub columns: usize,
}

#[derive(Debug)]
pub struct PassthroughBf16ResidentData {
    pub tensor_name: String,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub upload_chunks: u64,
    pub buffer: RuntimeBuffer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassthroughPayloadVerification {
    pub tensor_name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub verified_chunks: u64,
}

pub fn effective_rmsnorm_weight_values(tensor_name: &str, values: &[f32]) -> Vec<f32> {
    if uses_additive_rmsnorm_weight(tensor_name, values) {
        values.iter().map(|value| value + 1.0_f32).collect()
    } else {
        values.to_vec()
    }
}

fn uses_additive_rmsnorm_weight(tensor_name: &str, values: &[f32]) -> bool {
    let is_rmsnorm_tensor = tensor_name.ends_with(".input_layernorm.weight")
        || tensor_name.ends_with(".post_attention_layernorm.weight")
        || tensor_name.ends_with(".self_attn.q_norm.weight")
        || tensor_name.ends_with(".self_attn.k_norm.weight");
    if !is_rmsnorm_tensor || values.is_empty() {
        return false;
    }
    let mean_abs = values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32;
    mean_abs < 0.75_f32
}

#[derive(Debug)]
pub struct MaterializeConfig {
    pub scale_format: String,
    pub scale_values: Vec<f32>,
    pub group_size: usize,
    pub tensor_scale: f32,
    pub elements: usize,
    pub output_bytes: usize,
}

#[derive(Debug)]
pub struct LoadedPackage {
    pub summary: PackageSummary,
    pub loaded_tensor_count: usize,
    pub registry_indices: Vec<usize>,
    pub registry: WeightRegistry,
}

impl LoadedPackage {
    pub fn registry(&self) -> &WeightRegistry {
        &self.registry
    }

    pub fn into_registry(self) -> WeightRegistry {
        self.registry
    }

    pub fn tensor_by_name(&self, tensor_name: &str) -> Option<&LoadedTensorBundle> {
        self.registry.tensor_by_name(tensor_name)
    }

    pub fn payload_by_name_and_role(
        &self,
        tensor_name: &str,
        role: ReferencedFileRole,
    ) -> Option<&LoadedPayload> {
        self.registry.get_loaded_payload(tensor_name, role)
    }

    pub fn find_by_family_candidate(
        &self,
        family: &str,
        candidate_id: &str,
    ) -> Vec<&LoadedTensorBundle> {
        self.registry.find_by_family_candidate(family, candidate_id)
    }
}

impl LoadedTensorBundle {
    pub fn total_payload_bytes(&self) -> u64 {
        self.index.bytes + self.scale.bytes + self.codebook.bytes
    }
}

pub fn load_package_tensor_prefix(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    package_dir: impl AsRef<Path>,
    max_tensors: usize,
    options: LoadOptions,
) -> Result<LoadedPackage, String> {
    if max_tensors == 0 {
        return Err("max tensors must be greater than zero".to_string());
    }
    let package_dir = package_dir.as_ref();
    let summary = crate::package::inspect_package(package_dir)?;
    let bundles = list_tensor_payload_bundles(package_dir)?;
    if bundles.is_empty() {
        return Err(format!(
            "package {} contains no quantized tensor payload bundles",
            package_dir.display()
        ));
    }
    let selected_count = bundles.len().min(max_tensors);
    let mut registry = WeightRegistry::new();
    let registry_indices =
        registry.load_and_insert_many(context, stream, &bundles[..selected_count], options)?;
    Ok(LoadedPackage {
        summary,
        loaded_tensor_count: selected_count,
        registry_indices,
        registry,
    })
}

pub fn read_named_passthrough_f32(
    package_path: impl AsRef<Path>,
    tensor_name: &str,
    chunk_bytes: usize,
) -> Result<PassthroughF32Data, String> {
    let selector = TensorSelector::Name(tensor_name.to_string());
    let bundle = select_passthrough_payload_bundle(package_path, &selector).map_err(|err| {
        format!("failed to select package passthrough tensor {tensor_name}: {err}")
    })?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid passthrough shape for {tensor_name}: {err}"))?;
    let dtype = resolve_passthrough_dtype(&bundle, tensor_name)?.to_string();
    let values = read_passthrough_payload_f32_bytes(&bundle, chunk_bytes, &dtype)
        .map_err(|err| format!("failed to read passthrough payload for {tensor_name}: {err}"))?;
    let expected_elements = usize::try_from(bundle.elements)
        .map_err(|_| format!("passthrough tensor {tensor_name} is too large for this host"))?;
    if values.len() != expected_elements {
        return Err(format!(
            "passthrough tensor element count mismatch for {tensor_name}: expected {} got {}",
            expected_elements,
            values.len()
        ));
    }
    Ok(PassthroughF32Data {
        values,
        dtype,
        shape: bundle.shape,
    })
}

pub fn read_named_passthrough_f32_rows(
    package_path: impl AsRef<Path>,
    tensor_name: &str,
    row_indices: &[usize],
) -> Result<PassthroughF32Rows, String> {
    let selector = TensorSelector::Name(tensor_name.to_string());
    let bundle = select_passthrough_payload_bundle(package_path, &selector).map_err(|err| {
        format!("failed to select package passthrough tensor {tensor_name}: {err}")
    })?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid passthrough shape for {tensor_name}: {err}"))?;
    let dtype = resolve_passthrough_dtype(&bundle, tensor_name)?.to_string();
    let (columns, values) = read_passthrough_payload_f32_rows(&bundle, &dtype, row_indices)
        .map_err(|err| format!("failed to read passthrough rows for {tensor_name}: {err}"))?;
    Ok(PassthroughF32Rows {
        values,
        dtype,
        shape: bundle.shape,
        row_indices: row_indices.to_vec(),
        columns,
    })
}

pub fn read_named_passthrough_f32_row_range(
    package_path: impl AsRef<Path>,
    tensor_name: &str,
    start_row: usize,
    row_count: usize,
) -> Result<PassthroughF32RowRange, String> {
    let selector = TensorSelector::Name(tensor_name.to_string());
    let bundle = select_passthrough_payload_bundle(package_path, &selector).map_err(|err| {
        format!("failed to select package passthrough tensor {tensor_name}: {err}")
    })?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid passthrough shape for {tensor_name}: {err}"))?;
    let dtype = resolve_passthrough_dtype(&bundle, tensor_name)?.to_string();
    let (columns, values) = read_passthrough_payload_f32_row_range(
        &bundle, &dtype, start_row, row_count,
    )
    .map_err(|err| format!("failed to read passthrough row range for {tensor_name}: {err}"))?;
    Ok(PassthroughF32RowRange {
        values,
        dtype,
        shape: bundle.shape,
        start_row,
        row_count,
        columns,
    })
}

pub fn load_named_passthrough_bf16_resident(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    package_path: impl AsRef<Path>,
    tensor_name: &str,
    expected_shape: &[u64],
    chunk_bytes: usize,
) -> Result<PassthroughBf16ResidentData, String> {
    if chunk_bytes == 0 {
        return Err("resident BF16 passthrough chunk_bytes must be greater than zero".into());
    }
    if expected_shape.is_empty() || expected_shape.contains(&0) {
        return Err(format!(
            "resident BF16 tensor {tensor_name} expected shape must be non-empty and non-zero"
        ));
    }
    let bundle =
        select_exact_passthrough_payload_bundle(package_path, tensor_name).map_err(|err| {
            format!("failed to select resident BF16 passthrough tensor {tensor_name}: {err}")
        })?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid passthrough shape for {tensor_name}: {err}"))?;
    if bundle.tensor_name != tensor_name {
        return Err(format!(
            "resident BF16 tensor exact-name mismatch: requested={tensor_name} selected={}",
            bundle.tensor_name
        ));
    }
    if bundle.shape != expected_shape {
        return Err(format!(
            "resident BF16 tensor {tensor_name} shape mismatch: expected={expected_shape:?} actual={:?}",
            bundle.shape
        ));
    }
    if bundle.dtype.as_deref() != Some("BF16") {
        return Err(format!(
            "resident passthrough tensor {tensor_name} must explicitly declare BF16 dtype, got {:?}",
            bundle.dtype
        ));
    }
    if bundle.payload_encoding.as_deref() != Some("raw_safetensors_payload") {
        return Err(format!(
            "resident BF16 tensor {tensor_name} must explicitly declare raw_safetensors_payload encoding, got {:?}",
            bundle.payload_encoding
        ));
    }
    if bundle.payload_bytes == 0 {
        return Err(format!(
            "resident BF16 tensor {tensor_name} must explicitly declare nonzero payload_bytes"
        ));
    }

    let expected_bytes = bundle
        .elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| format!("resident BF16 tensor {tensor_name} byte count overflows"))?;
    let declared_bytes = bundle.payload_bytes;
    if declared_bytes != expected_bytes || bundle.payload_file.bytes != expected_bytes {
        return Err(format!(
            "resident BF16 tensor {tensor_name} byte mismatch: elements={} expected={expected_bytes} declared={declared_bytes} file={}",
            bundle.elements, bundle.payload_file.bytes
        ));
    }
    if expected_bytes == 0 {
        return Err(format!(
            "resident BF16 tensor {tensor_name} must not be empty"
        ));
    }
    let expected_sha256 = bundle
        .payload_sha256
        .as_deref()
        .ok_or_else(|| format!("resident BF16 tensor {tensor_name} must declare payload_sha256"))?
        .to_ascii_lowercase();
    if expected_sha256.len() != 64 || !expected_sha256.as_bytes().iter().all(u8::is_ascii_hexdigit)
    {
        return Err(format!(
            "resident BF16 tensor {tensor_name} has invalid payload_sha256 {expected_sha256}"
        ));
    }

    let buffer_bytes = usize::try_from(expected_bytes)
        .map_err(|_| format!("resident BF16 tensor {tensor_name} is too large for this host"))?;
    let mut buffer = context
        .alloc_buffer(buffer_bytes)
        .map_err(|err| format!("failed to allocate resident BF16 tensor {tensor_name}: {err}"))?;
    let mut file = File::open(&bundle.payload_file.absolute_path).map_err(|err| {
        format!(
            "failed to open resident BF16 tensor {}: {err}",
            bundle.payload_file.absolute_path.display()
        )
    })?;
    let opened_bytes = file
        .metadata()
        .map_err(|err| format!("failed to stat resident BF16 tensor {tensor_name}: {err}"))?
        .len();
    if opened_bytes != expected_bytes {
        return Err(format!(
            "resident BF16 tensor {tensor_name} changed before upload: expected={expected_bytes} actual={opened_bytes}"
        ));
    }

    let staging_bytes = passthrough_stream_chunk_bytes(expected_bytes, chunk_bytes, tensor_name)?;
    let mut staging = zeroed_staging_bytes(staging_bytes, tensor_name)?;
    let mut digest = Sha256::new();
    let mut offset = 0_usize;
    let mut chunks = 0_u64;
    while offset < buffer_bytes {
        let read_len = (buffer_bytes - offset).min(staging.len());
        file.read_exact(&mut staging[..read_len]).map_err(|err| {
            format!("failed to read resident BF16 tensor {tensor_name} at {offset}: {err}")
        })?;
        digest.update(&staging[..read_len]);
        buffer
            .copy_from_host(offset, &staging[..read_len], Some(stream))
            .map_err(|err| {
                format!("failed to upload resident BF16 tensor {tensor_name} at {offset}: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize resident BF16 tensor {tensor_name} upload at {offset}: {err}"
            )
        })?;
        offset += read_len;
        chunks += 1;
    }
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|err| format!("failed to verify resident BF16 tensor {tensor_name} EOF: {err}"))?
        != 0
    {
        return Err(format!(
            "resident BF16 tensor {tensor_name} has trailing payload data"
        ));
    }
    let final_bytes = file
        .metadata()
        .map_err(|err| format!("failed to re-stat resident BF16 tensor {tensor_name}: {err}"))?
        .len();
    if final_bytes != opened_bytes || final_bytes != expected_bytes {
        return Err(format!(
            "resident BF16 tensor {tensor_name} byte length changed during upload: expected={expected_bytes} before={opened_bytes} after={final_bytes}"
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "resident BF16 tensor {tensor_name} checksum mismatch: manifest={expected_sha256} file={actual_sha256}"
        ));
    }

    Ok(PassthroughBf16ResidentData {
        tensor_name: bundle.tensor_name,
        shape: bundle.shape,
        elements: bundle.elements,
        payload_bytes: expected_bytes,
        payload_sha256: actual_sha256,
        upload_chunks: chunks,
        buffer,
    })
}

pub fn verify_named_passthrough_payload(
    package_path: impl AsRef<Path>,
    tensor_name: &str,
    expected_dtype: &str,
    expected_shape: &[u64],
    chunk_bytes: usize,
) -> Result<PassthroughPayloadVerification, String> {
    if chunk_bytes == 0 {
        return Err(
            "passthrough payload verification chunk_bytes must be greater than zero".into(),
        );
    }
    if !matches!(expected_dtype, "BF16" | "F16" | "F32") {
        return Err(format!(
            "passthrough tensor {tensor_name} has unsupported expected dtype {expected_dtype}"
        ));
    }
    if expected_shape.is_empty() || expected_shape.contains(&0) {
        return Err(format!(
            "passthrough tensor {tensor_name} expected shape must be non-empty and non-zero"
        ));
    }
    let bundle =
        select_exact_passthrough_payload_bundle(package_path, tensor_name).map_err(|err| {
            format!("failed to select passthrough tensor {tensor_name} for verification: {err}")
        })?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid passthrough shape for {tensor_name}: {err}"))?;
    if bundle.tensor_name != tensor_name {
        return Err(format!(
            "passthrough tensor exact-name mismatch: requested={tensor_name} selected={}",
            bundle.tensor_name
        ));
    }
    if bundle.shape != expected_shape {
        return Err(format!(
            "passthrough tensor {tensor_name} shape mismatch: expected={expected_shape:?} actual={:?}",
            bundle.shape
        ));
    }
    if bundle.dtype.as_deref() != Some(expected_dtype) {
        return Err(format!(
            "passthrough tensor {tensor_name} must explicitly declare dtype {expected_dtype}, got {:?}",
            bundle.dtype
        ));
    }
    if bundle.payload_encoding.as_deref() != Some("raw_safetensors_payload") {
        return Err(format!(
            "passthrough tensor {tensor_name} must explicitly declare raw_safetensors_payload encoding, got {:?}",
            bundle.payload_encoding
        ));
    }
    if bundle.payload_bytes == 0 {
        return Err(format!(
            "passthrough tensor {tensor_name} must explicitly declare nonzero payload_bytes"
        ));
    }
    let dtype = expected_dtype.to_string();
    let declared_bytes = bundle.payload_bytes;
    if declared_bytes == 0 || declared_bytes != bundle.payload_file.bytes {
        return Err(format!(
            "passthrough tensor {tensor_name} payload byte mismatch: declared={declared_bytes} file={}",
            bundle.payload_file.bytes
        ));
    }
    let element_size = passthrough_element_size(&dtype, tensor_name)? as u64;
    let expected_bytes = bundle
        .elements
        .checked_mul(element_size)
        .ok_or_else(|| format!("passthrough tensor {tensor_name} byte count overflows"))?;
    if expected_bytes != declared_bytes {
        return Err(format!(
            "passthrough tensor {tensor_name} element byte mismatch: elements={} dtype={dtype} expected={expected_bytes} declared={declared_bytes}",
            bundle.elements
        ));
    }
    let expected_sha256 = bundle
        .payload_sha256
        .as_deref()
        .ok_or_else(|| format!("passthrough tensor {tensor_name} must declare payload_sha256"))?
        .to_ascii_lowercase();
    if expected_sha256.len() != 64 || !expected_sha256.as_bytes().iter().all(u8::is_ascii_hexdigit)
    {
        return Err(format!(
            "passthrough tensor {tensor_name} has invalid payload_sha256 {expected_sha256}"
        ));
    }

    let mut file = File::open(&bundle.payload_file.absolute_path).map_err(|err| {
        format!(
            "failed to open passthrough tensor {} for verification: {err}",
            bundle.payload_file.absolute_path.display()
        )
    })?;
    let opened_bytes = file
        .metadata()
        .map_err(|err| format!("failed to stat passthrough tensor {tensor_name}: {err}"))?
        .len();
    if opened_bytes != expected_bytes {
        return Err(format!(
            "passthrough tensor {tensor_name} changed before verification: expected={expected_bytes} actual={opened_bytes}"
        ));
    }
    let staging_len = passthrough_stream_chunk_bytes(expected_bytes, chunk_bytes, tensor_name)?;
    let mut staging = zeroed_staging_bytes(staging_len, tensor_name)?;
    let mut digest = Sha256::new();
    let mut remaining = expected_bytes;
    let mut chunks = 0_u64;
    while remaining > 0 {
        let read_len = usize::try_from(remaining.min(staging.len() as u64)).map_err(|_| {
            format!("passthrough tensor {tensor_name} read length does not fit usize")
        })?;
        file.read_exact(&mut staging[..read_len]).map_err(|err| {
            format!("failed to read passthrough tensor {tensor_name} during verification: {err}")
        })?;
        digest.update(&staging[..read_len]);
        remaining -= read_len as u64;
        chunks += 1;
    }
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|err| format!("failed to verify passthrough tensor {tensor_name} EOF: {err}"))?
        != 0
    {
        return Err(format!(
            "passthrough tensor {tensor_name} has trailing payload data"
        ));
    }
    let final_bytes = file
        .metadata()
        .map_err(|err| format!("failed to re-stat passthrough tensor {tensor_name}: {err}"))?
        .len();
    if final_bytes != opened_bytes || final_bytes != expected_bytes {
        return Err(format!(
            "passthrough tensor {tensor_name} byte length changed during verification: expected={expected_bytes} before={opened_bytes} after={final_bytes}"
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "passthrough tensor {tensor_name} checksum mismatch: manifest={expected_sha256} file={actual_sha256}"
        ));
    }

    Ok(PassthroughPayloadVerification {
        tensor_name: bundle.tensor_name,
        dtype,
        shape: bundle.shape,
        elements: bundle.elements,
        payload_bytes: expected_bytes,
        payload_sha256: actual_sha256,
        verified_chunks: chunks,
    })
}

fn passthrough_stream_chunk_bytes(
    payload_bytes: u64,
    requested_chunk_bytes: usize,
    tensor_name: &str,
) -> Result<usize, String> {
    if payload_bytes == 0 || requested_chunk_bytes == 0 {
        return Err(format!(
            "passthrough tensor {tensor_name} payload and chunk sizes must be greater than zero"
        ));
    }
    let bounded_request = requested_chunk_bytes.min(PASSTHROUGH_MAX_STREAM_CHUNK_BYTES) as u64;
    usize::try_from(payload_bytes.min(bounded_request))
        .map_err(|_| format!("passthrough tensor {tensor_name} chunk size does not fit usize"))
}

fn zeroed_staging_bytes(bytes: usize, tensor_name: &str) -> Result<Vec<u8>, String> {
    let mut staging = Vec::new();
    staging.try_reserve_exact(bytes).map_err(|err| {
        format!(
            "failed to reserve {bytes} staging bytes for passthrough tensor {tensor_name}: {err}"
        )
    })?;
    staging.resize(bytes, 0_u8);
    Ok(staging)
}

pub fn materialize_config(loaded: &LoadedTensorBundle) -> Result<MaterializeConfig, String> {
    let scale_format = loaded
        .scale_format
        .as_deref()
        .ok_or_else(|| "selected tensor does not declare scale_format".to_string())?;
    let scale_values = crate::aq::scale_values(scale_format)?;
    let group_size = match loaded.group_size {
        Some(value) if value > 0 => value,
        Some(_) | None => {
            return Err("selected tensor does not declare a valid group_size".to_string());
        }
    };
    let tensor_scale = match loaded.tensor_scale {
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(_) | None => {
            return Err("selected tensor does not declare a valid tensor_scale".to_string());
        }
    };
    if loaded.index_encoding.as_deref() != Some("idx4_low_nibble_first") {
        return Err("selected tensor uses unsupported index encoding".to_string());
    }
    if loaded.scale_encoding.as_deref() != Some("u8_scale_table_index") {
        return Err("selected tensor uses unsupported scale encoding".to_string());
    }
    let elements = usize::try_from(loaded.elements)
        .map_err(|_| "selected tensor has too many elements for this host".to_string())?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "materialized output byte size overflows".to_string())?;
    Ok(MaterializeConfig {
        scale_format: scale_format.to_string(),
        scale_values,
        group_size,
        tensor_scale,
        elements,
        output_bytes,
    })
}

pub fn matrix_shape_rows_cols(shape: &[u64], elements: usize) -> Result<(usize, usize), String> {
    let shape = match shape {
        shape if shape.len() == 2 => shape,
        _ => return Err("selected tensor shape is not 2D".to_string()),
    };
    let rows_u64 = shape[0];
    let cols_u64 = shape[1];
    if rows_u64 == 0 || cols_u64 == 0 {
        return Err("selected tensor has zero rows or columns".to_string());
    }
    let expected_elements = rows_u64
        .checked_mul(cols_u64)
        .ok_or_else(|| "selected tensor shape overflows element count".to_string())?;
    if expected_elements
        != u64::try_from(elements)
            .map_err(|_| "selected tensor has too many elements".to_string())?
    {
        return Err(format!(
            "selected tensor shape has {expected_elements} elements but materialize produced {elements}"
        ));
    }
    let rows = usize::try_from(rows_u64)
        .map_err(|_| "selected tensor row count does not fit host usize".to_string())?;
    let cols = usize::try_from(cols_u64)
        .map_err(|_| "selected tensor column count does not fit host usize".to_string())?;
    Ok((rows, cols))
}

pub fn materialize_selected_aq4_matrix(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    registry: &mut WeightRegistry,
    path: impl AsRef<Path>,
    tensor_name: &str,
    chunk_bytes: usize,
) -> Result<(usize, usize, RuntimeBuffer), String> {
    let selector = TensorSelector::Name(tensor_name.to_string());
    let bundle = select_tensor_payload_bundle(path, &selector)
        .map_err(|err| format!("failed to select tensor payloads for {tensor_name}: {err}"))?;
    let registry_index = registry
        .load_and_insert(
            context,
            stream,
            &bundle,
            LoadOptions {
                chunk_bytes,
                verify: true,
            },
        )
        .map_err(|err| format!("failed to register tensor payloads for {tensor_name}: {err}"))?;
    let loaded = registry
        .get(registry_index)
        .ok_or_else(|| "registered tensor disappeared from weight registry".to_string())?;
    let materialize = materialize_config(loaded).map_err(|err| {
        format!(
            "failed to prepare materialize config for {tensor_name} (registry index {registry_index}): {err}"
        )
    })?;
    let (rows, cols) = matrix_shape_rows_cols(&loaded.shape, materialize.elements)
        .map_err(|err| format!("invalid shape for {tensor_name}: {err}"))?;
    let mut output = context
        .alloc_buffer(materialize.output_bytes)
        .map_err(|err| {
            format!("failed to allocate materialized output for {tensor_name}: {err}")
        })?;
    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut output,
        Some(stream),
    ) {
        return Err(format!(
            "failed to materialize AQ4 tensor {tensor_name}: {err}"
        ));
    }
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize runtime stream after materializing {tensor_name}: {err}")
    })?;
    apply_row_scale_overrides_to_materialized_matrix(
        stream,
        &mut output,
        rows,
        cols,
        tensor_name,
        &bundle.row_scale_overrides,
    )?;
    Ok((rows, cols, output))
}

fn apply_row_scale_overrides_to_materialized_matrix(
    stream: &mut RuntimeStream,
    matrix: &mut RuntimeBuffer,
    rows: usize,
    cols: usize,
    tensor_name: &str,
    overrides: &[RowScaleOverrideEntry],
) -> Result<(), String> {
    if overrides.is_empty() {
        return Ok(());
    }
    let matrix_bytes_len = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            format!("row_scale_overrides matrix byte size overflows for {tensor_name}")
        })?;
    let mut matrix_bytes = vec![0_u8; matrix_bytes_len];
    matrix
        .copy_to_host(0, &mut matrix_bytes, Some(stream))
        .map_err(|err| {
            format!("failed to copy materialized {tensor_name} for row_scale_overrides: {err}")
        })?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after {tensor_name} row_scale_overrides copy: {err}")
    })?;

    let row_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("row_scale_overrides row byte size overflows for {tensor_name}"))?;
    for entry in overrides {
        if entry.row_index >= rows {
            return Err(format!(
                "row_scale_overrides row out of range for {tensor_name}: row={} rows={rows}",
                entry.row_index
            ));
        }
        let row_start = entry
            .row_index
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("row_scale_overrides row offset overflows for {tensor_name}"))?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or_else(|| format!("row_scale_overrides row end overflows for {tensor_name}"))?;
        for offset in (row_start..row_end).step_by(std::mem::size_of::<f32>()) {
            let mut raw = [0_u8; 4];
            raw.copy_from_slice(&matrix_bytes[offset..offset + 4]);
            let scaled = f32::from_le_bytes(raw) * entry.scale;
            matrix_bytes[offset..offset + 4].copy_from_slice(&scaled.to_le_bytes());
        }
    }

    matrix
        .copy_from_host(0, &matrix_bytes, Some(stream))
        .map_err(|err| {
            format!("failed to copy row-scaled materialized {tensor_name} back to runtime: {err}")
        })?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after {tensor_name} row_scale_overrides copy back: {err}")
    })
}

pub fn resolve_passthrough_dtype<'a>(
    bundle: &'a PassthroughPayloadBundle,
    tensor_name: &str,
) -> Result<&'a str, String> {
    if let Some(dtype) = bundle.dtype.as_deref() {
        return match dtype {
            "BF16" | "F16" | "F32" => Ok(dtype),
            _ => Err(format!(
                "unsupported passthrough dtype \"{dtype}\" for tensor {tensor_name}"
            )),
        };
    }

    let payload_bytes = if bundle.payload_bytes == 0 {
        bundle.payload_file.bytes
    } else {
        bundle.payload_bytes
    };
    let bf16_bytes = bundle.elements.checked_mul(2).ok_or_else(|| {
        format!("passthrough tensor {tensor_name} element count overflow while inferring dtype")
    })?;
    let f32_bytes = bundle.elements.checked_mul(4).ok_or_else(|| {
        format!("passthrough tensor {tensor_name} element count overflow while inferring dtype")
    })?;
    if payload_bytes == bf16_bytes {
        Ok("BF16")
    } else if payload_bytes == f32_bytes {
        Ok("F32")
    } else {
        Err(format!(
            "could not infer passthrough dtype for tensor {tensor_name}; declare dtype in manifest"
        ))
    }
}

pub fn validate_passthrough_shape_elements(
    bundle: &PassthroughPayloadBundle,
) -> Result<(), String> {
    if bundle.shape.is_empty() {
        return Ok(());
    }
    let mut product = 1_u64;
    for dimension in &bundle.shape {
        if *dimension == 0 {
            return Err("shape contains zero".to_string());
        }
        product = product
            .checked_mul(*dimension)
            .ok_or_else(|| "shape element count overflows u64".to_string())?;
    }
    if product != bundle.elements {
        return Err(format!(
            "shape product {} does not match element count {}",
            product, bundle.elements
        ));
    }
    Ok(())
}

fn validate_passthrough_payload_encoding(bundle: &PassthroughPayloadBundle) -> Result<(), String> {
    match bundle.payload_encoding.as_deref() {
        None | Some("raw_safetensors_payload") => Ok(()),
        Some(payload_encoding) => Err(format!(
            "passthrough tensor {} has unsupported payload encoding {payload_encoding}",
            bundle.tensor_name
        )),
    }
}

pub fn read_passthrough_payload_f32_bytes(
    bundle: &PassthroughPayloadBundle,
    chunk_bytes: usize,
    dtype: &str,
) -> Result<Vec<f32>, String> {
    validate_passthrough_payload_encoding(bundle)?;
    let mut file = File::open(&bundle.payload_file.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            bundle.payload_file.absolute_path.display()
        )
    })?;
    let payload_bytes = if bundle.payload_bytes == 0 {
        bundle.payload_file.bytes
    } else {
        bundle.payload_bytes
    };
    if payload_bytes != bundle.payload_file.bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch: declared {} actual {}",
            bundle.tensor_name, payload_bytes, bundle.payload_file.bytes
        ));
    }
    let element_size = passthrough_element_size(dtype, &bundle.tensor_name)?;
    if chunk_bytes == 0 {
        return Err("chunk bytes must be greater than zero".to_string());
    }
    let expected_bytes = usize::try_from(payload_bytes)
        .map_err(|_| "passthrough payload is too large for this host".to_string())?;
    let expected_elements = usize::try_from(bundle.elements)
        .map_err(|_| "payload element count too large".to_string())?;
    if !expected_bytes.is_multiple_of(element_size) {
        return Err(format!(
            "passthrough tensor {} payload is not aligned to {element_size}-byte elements",
            bundle.tensor_name
        ));
    }
    if expected_bytes / element_size != expected_elements {
        return Err(format!(
            "passthrough tensor {} payload has {} elements, expected {}",
            bundle.tensor_name,
            expected_bytes / element_size,
            expected_elements
        ));
    }

    let mut values = Vec::new();
    values.try_reserve(expected_elements).map_err(|err| {
        format!(
            "failed to reserve decoded passthrough values for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    let mut scratch = Vec::new();
    scratch.try_reserve(chunk_bytes).map_err(|err| {
        format!(
            "failed to reserve passthrough read buffer for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    scratch.resize(chunk_bytes, 0_u8);
    let mut read_bytes = 0_usize;
    let mut carry = Vec::new();
    carry.try_reserve(element_size - 1).map_err(|err| {
        format!(
            "failed to reserve passthrough carry buffer for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    let merge_capacity = chunk_bytes.checked_add(element_size).ok_or_else(|| {
        format!(
            "passthrough tensor {} merge buffer size overflows",
            bundle.tensor_name
        )
    })?;
    let mut merge = Vec::new();
    merge.try_reserve(merge_capacity).map_err(|err| {
        format!(
            "failed to reserve passthrough merge buffer for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    loop {
        let read = file.read(&mut scratch).map_err(|err| {
            format!(
                "failed to read {}: {err}",
                bundle.payload_file.absolute_path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        read_bytes = read_bytes.saturating_add(read);
        if read_bytes > expected_bytes {
            return Err(format!(
                "passthrough tensor {} payload is larger than declared bytes {}",
                bundle.tensor_name, expected_bytes
            ));
        }

        merge.clear();
        if carry.is_empty() {
            merge.extend_from_slice(&scratch[..read]);
        } else {
            merge.extend_from_slice(&carry);
            carry.clear();
            merge.extend_from_slice(&scratch[..read]);
        }

        let decode_end = (merge.len() / element_size) * element_size;
        for bytes in merge[..decode_end].chunks_exact(element_size) {
            values.push(decode_passthrough_element(dtype, bytes));
        }
        if decode_end < merge.len() {
            carry.extend_from_slice(&merge[decode_end..]);
        }
    }
    if read_bytes != expected_bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch while reading file: expected {} got {}",
            bundle.tensor_name, expected_bytes, read_bytes
        ));
    }
    if values.len() != expected_elements {
        return Err(format!(
            "passthrough tensor {} payload elements mismatch: expected {} got {}",
            bundle.tensor_name,
            expected_elements,
            values.len()
        ));
    }
    Ok(values)
}

pub fn read_passthrough_payload_f32_rows(
    bundle: &PassthroughPayloadBundle,
    dtype: &str,
    row_indices: &[usize],
) -> Result<(usize, Vec<f32>), String> {
    validate_passthrough_payload_encoding(bundle)?;
    let shape = match bundle.shape.as_slice() {
        [rows, columns] => (*rows, *columns),
        _ => {
            return Err(format!(
                "passthrough tensor {} must be 2D for row reads",
                bundle.tensor_name
            ));
        }
    };
    let rows = usize::try_from(shape.0).map_err(|_| {
        format!(
            "passthrough tensor {} row count is too large",
            bundle.tensor_name
        )
    })?;
    let columns = usize::try_from(shape.1).map_err(|_| {
        format!(
            "passthrough tensor {} column count is too large",
            bundle.tensor_name
        )
    })?;
    let expected_elements = rows
        .checked_mul(columns)
        .ok_or_else(|| format!("passthrough tensor {} shape overflows", bundle.tensor_name))?;
    if u64::try_from(expected_elements).ok() != Some(bundle.elements) {
        return Err(format!(
            "passthrough tensor {} shape has {expected_elements} elements, expected {}",
            bundle.tensor_name, bundle.elements
        ));
    }
    let element_size = passthrough_element_size(dtype, &bundle.tensor_name)?;
    let row_bytes = columns.checked_mul(element_size).ok_or_else(|| {
        format!(
            "passthrough tensor {} row byte size overflows",
            bundle.tensor_name
        )
    })?;
    let payload_bytes = if bundle.payload_bytes == 0 {
        bundle.payload_file.bytes
    } else {
        bundle.payload_bytes
    };
    if payload_bytes != bundle.payload_file.bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch: declared {} actual {}",
            bundle.tensor_name, payload_bytes, bundle.payload_file.bytes
        ));
    }
    let expected_bytes = u64::try_from(expected_elements)
        .ok()
        .and_then(|elements| elements.checked_mul(u64::try_from(element_size).ok()?))
        .ok_or_else(|| {
            format!(
                "passthrough tensor {} expected payload byte size overflows",
                bundle.tensor_name
            )
        })?;
    if payload_bytes != expected_bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch: expected {} got {}",
            bundle.tensor_name, expected_bytes, payload_bytes
        ));
    }

    let mut file = File::open(&bundle.payload_file.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            bundle.payload_file.absolute_path.display()
        )
    })?;
    let mut row_bytes_buf = Vec::new();
    row_bytes_buf.try_reserve(row_bytes).map_err(|err| {
        format!(
            "failed to reserve passthrough row buffer for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    row_bytes_buf.resize(row_bytes, 0_u8);
    let output_elements = row_indices.len().checked_mul(columns).ok_or_else(|| {
        format!(
            "passthrough tensor {} row output overflows",
            bundle.tensor_name
        )
    })?;
    let mut values = Vec::new();
    values.try_reserve(output_elements).map_err(|err| {
        format!(
            "failed to reserve decoded passthrough row values for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    for row_index in row_indices {
        if *row_index >= rows {
            return Err(format!(
                "passthrough tensor {} row index {} is out of range 0..{}",
                bundle.tensor_name, row_index, rows
            ));
        }
        let byte_offset = row_index
            .checked_mul(row_bytes)
            .and_then(|offset| u64::try_from(offset).ok())
            .ok_or_else(|| {
                format!(
                    "passthrough tensor {} row byte offset overflows",
                    bundle.tensor_name
                )
            })?;
        file.seek(SeekFrom::Start(byte_offset)).map_err(|err| {
            format!(
                "failed to seek {} to row {}: {err}",
                bundle.payload_file.absolute_path.display(),
                row_index
            )
        })?;
        file.read_exact(&mut row_bytes_buf).map_err(|err| {
            format!(
                "failed to read row {} from {}: {err}",
                row_index,
                bundle.payload_file.absolute_path.display()
            )
        })?;
        for bytes in row_bytes_buf.chunks_exact(element_size) {
            values.push(decode_passthrough_element(dtype, bytes));
        }
    }
    Ok((columns, values))
}

pub fn read_passthrough_payload_f32_row_range(
    bundle: &PassthroughPayloadBundle,
    dtype: &str,
    start_row: usize,
    row_count: usize,
) -> Result<(usize, Vec<f32>), String> {
    validate_passthrough_payload_encoding(bundle)?;
    if row_count == 0 {
        return Err(format!(
            "passthrough tensor {} row range count must be greater than zero",
            bundle.tensor_name
        ));
    }
    let shape = match bundle.shape.as_slice() {
        [rows, columns] => (*rows, *columns),
        _ => {
            return Err(format!(
                "passthrough tensor {} must be 2D for row range reads",
                bundle.tensor_name
            ));
        }
    };
    let rows = usize::try_from(shape.0).map_err(|_| {
        format!(
            "passthrough tensor {} row count is too large",
            bundle.tensor_name
        )
    })?;
    let columns = usize::try_from(shape.1).map_err(|_| {
        format!(
            "passthrough tensor {} column count is too large",
            bundle.tensor_name
        )
    })?;
    let end_row = start_row.checked_add(row_count).ok_or_else(|| {
        format!(
            "passthrough tensor {} row range end overflows",
            bundle.tensor_name
        )
    })?;
    if start_row >= rows || end_row > rows {
        return Err(format!(
            "passthrough tensor {} row range {}..{} is out of range 0..{}",
            bundle.tensor_name, start_row, end_row, rows
        ));
    }
    let expected_elements = rows
        .checked_mul(columns)
        .ok_or_else(|| format!("passthrough tensor {} shape overflows", bundle.tensor_name))?;
    if u64::try_from(expected_elements).ok() != Some(bundle.elements) {
        return Err(format!(
            "passthrough tensor {} shape has {expected_elements} elements, expected {}",
            bundle.tensor_name, bundle.elements
        ));
    }
    let element_size = passthrough_element_size(dtype, &bundle.tensor_name)?;
    let row_bytes = columns.checked_mul(element_size).ok_or_else(|| {
        format!(
            "passthrough tensor {} row byte size overflows",
            bundle.tensor_name
        )
    })?;
    let payload_bytes = if bundle.payload_bytes == 0 {
        bundle.payload_file.bytes
    } else {
        bundle.payload_bytes
    };
    if payload_bytes != bundle.payload_file.bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch: declared {} actual {}",
            bundle.tensor_name, payload_bytes, bundle.payload_file.bytes
        ));
    }
    let expected_bytes = u64::try_from(expected_elements)
        .ok()
        .and_then(|elements| elements.checked_mul(u64::try_from(element_size).ok()?))
        .ok_or_else(|| {
            format!(
                "passthrough tensor {} expected payload byte size overflows",
                bundle.tensor_name
            )
        })?;
    if payload_bytes != expected_bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch: expected {} got {}",
            bundle.tensor_name, expected_bytes, payload_bytes
        ));
    }

    let range_bytes = row_count.checked_mul(row_bytes).ok_or_else(|| {
        format!(
            "passthrough tensor {} row range byte size overflows",
            bundle.tensor_name
        )
    })?;
    let output_elements = row_count.checked_mul(columns).ok_or_else(|| {
        format!(
            "passthrough tensor {} row range output overflows",
            bundle.tensor_name
        )
    })?;
    let byte_offset = start_row
        .checked_mul(row_bytes)
        .and_then(|offset| u64::try_from(offset).ok())
        .ok_or_else(|| {
            format!(
                "passthrough tensor {} row range byte offset overflows",
                bundle.tensor_name
            )
        })?;

    let mut file = File::open(&bundle.payload_file.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            bundle.payload_file.absolute_path.display()
        )
    })?;
    file.seek(SeekFrom::Start(byte_offset)).map_err(|err| {
        format!(
            "failed to seek {} to row range {}..{}: {err}",
            bundle.payload_file.absolute_path.display(),
            start_row,
            end_row
        )
    })?;

    let mut range_bytes_buf = Vec::new();
    range_bytes_buf.try_reserve(range_bytes).map_err(|err| {
        format!(
            "failed to reserve passthrough row range buffer for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    range_bytes_buf.resize(range_bytes, 0_u8);
    file.read_exact(&mut range_bytes_buf).map_err(|err| {
        format!(
            "failed to read row range {}..{} from {}: {err}",
            start_row,
            end_row,
            bundle.payload_file.absolute_path.display()
        )
    })?;

    let mut values = Vec::new();
    values.try_reserve(output_elements).map_err(|err| {
        format!(
            "failed to reserve decoded passthrough row range values for tensor {}: {err}",
            bundle.tensor_name
        )
    })?;
    for bytes in range_bytes_buf.chunks_exact(element_size) {
        values.push(decode_passthrough_element(dtype, bytes));
    }
    Ok((columns, values))
}

fn passthrough_element_size(dtype: &str, tensor_name: &str) -> Result<usize, String> {
    match dtype {
        "BF16" => Ok(2),
        "F16" => Ok(2),
        "F32" => Ok(4),
        _ => Err(format!(
            "unsupported passthrough dtype {dtype} for tensor {tensor_name}"
        )),
    }
}

fn f16_to_f32(raw: u16) -> f32 {
    let sign = (u32::from(raw & 0x8000)) << 16;
    let exp = (raw >> 10) & 0x1f;
    let frac = raw & 0x03ff;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut exp_unbiased = -14_i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                exp_unbiased -= 1;
            }
            frac_norm &= 0x03ff;
            sign | (((exp_unbiased + 127) as u32) << 23) | (u32::from(frac_norm) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (u32::from(frac) << 13)
    } else {
        let exp_f32 = i32::from(exp) - 15 + 127;
        sign | ((exp_f32 as u32) << 23) | (u32::from(frac) << 13)
    };
    f32::from_bits(bits)
}

fn decode_passthrough_element(dtype: &str, bytes: &[u8]) -> f32 {
    match dtype {
        "BF16" => {
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            f32::from_bits(u32::from(raw) << 16)
        }
        "F16" => {
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            f16_to_f32(raw)
        }
        "F32" => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => unreachable!(),
    }
}

#[derive(Debug, Default)]
pub struct WeightRegistry {
    tensors: Vec<LoadedTensorBundle>,
    codebook_payloads: BTreeMap<String, LoadedPayload>,
}

impl WeightRegistry {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            codebook_payloads: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, bundle: LoadedTensorBundle) -> usize {
        self.codebook_payloads
            .entry(bundle.codebook.relative_path.clone())
            .or_insert_with(|| bundle.codebook.clone());
        let index = self.tensors.len();
        self.tensors.push(bundle);
        index
    }

    pub fn load_and_insert(
        &mut self,
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        bundle: &TensorPayloadBundle,
        options: LoadOptions,
    ) -> Result<usize, String> {
        let loaded = self.load_tensor_payload_bundle(context, stream, bundle, options)?;
        Ok(self.insert(loaded))
    }

    /// Loads and registers multiple tensor bundles in one API call.
    ///
    /// Keeping each bundle as a distinct registry row is intentional to keep a
    /// future codebook-dedup step straightforward.
    pub fn load_and_insert_many(
        &mut self,
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        bundles: &[TensorPayloadBundle],
        options: LoadOptions,
    ) -> Result<Vec<usize>, String> {
        let mut indexes = Vec::with_capacity(bundles.len());
        for bundle in bundles {
            let loaded = self.load_tensor_payload_bundle(context, stream, bundle, options)?;
            indexes.push(self.insert(loaded));
        }
        Ok(indexes)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn total_payload_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(LoadedTensorBundle::total_payload_bytes)
            .sum()
    }

    pub fn resident_payload_bytes(&self) -> u64 {
        let tensor_payload_bytes: u64 = self
            .tensors
            .iter()
            .map(|bundle| bundle.index.bytes + bundle.scale.bytes)
            .sum();
        let codebook_payload_bytes: u64 = self
            .codebook_payloads
            .values()
            .map(|payload| payload.bytes)
            .sum();
        tensor_payload_bytes + codebook_payload_bytes
    }

    pub fn codebook_payloads(&self) -> usize {
        self.codebook_payloads.len()
    }

    pub fn get(&self, index: usize) -> Option<&LoadedTensorBundle> {
        self.tensors.get(index)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&LoadedTensorBundle> {
        self.tensors
            .iter()
            .find(|bundle| bundle.tensor_name == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &LoadedTensorBundle> {
        self.tensors.iter()
    }

    pub fn find_by_family_candidate(
        &self,
        family: &str,
        candidate_id: &str,
    ) -> Vec<&LoadedTensorBundle> {
        self.tensors
            .iter()
            .filter(|bundle| {
                bundle.family.as_deref() == Some(family)
                    && bundle.candidate_id.as_deref() == Some(candidate_id)
            })
            .collect()
    }

    pub fn get_loaded_payload(
        &self,
        tensor_name: &str,
        role: ReferencedFileRole,
    ) -> Option<&LoadedPayload> {
        let bundle = self.get_by_name(tensor_name)?;
        match role {
            ReferencedFileRole::TensorIndex => Some(&bundle.index),
            ReferencedFileRole::TensorScale => Some(&bundle.scale),
            ReferencedFileRole::TensorCodebook | ReferencedFileRole::Codebook => {
                Some(&bundle.codebook)
            }
            ReferencedFileRole::Smallest | ReferencedFileRole::Passthrough => None,
        }
    }

    pub fn tensor_by_name(&self, tensor_name: &str) -> Option<&LoadedTensorBundle> {
        self.get_by_name(tensor_name)
    }

    fn load_tensor_payload_bundle(
        &mut self,
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        bundle: &TensorPayloadBundle,
        options: LoadOptions,
    ) -> Result<LoadedTensorBundle, String> {
        if options.chunk_bytes == 0 {
            return Err("chunk bytes must be greater than zero".to_string());
        }
        let index = load_payload_file(context, stream, &bundle.index_file, options)?;
        let scale = load_payload_file(context, stream, &bundle.scale_file, options)?;
        let codebook = if let Some(payload) = self
            .codebook_payloads
            .get(&bundle.codebook_file.relative_path)
        {
            payload.clone()
        } else {
            let payload = load_payload_file(context, stream, &bundle.codebook_file, options)?;
            self.codebook_payloads
                .insert(bundle.codebook_file.relative_path.clone(), payload.clone());
            payload
        };

        Ok(LoadedTensorBundle {
            tensor_index: bundle.tensor_index,
            tensor_name: bundle.tensor_name.clone(),
            dtype: bundle.dtype.clone(),
            shape: bundle.shape.clone(),
            family: bundle.family.clone(),
            candidate_id: bundle.candidate_id.clone(),
            scale_format: bundle.scale_format.clone(),
            group_size: bundle.group_size,
            tensor_scale: bundle.tensor_scale,
            index_encoding: bundle.index_encoding.clone(),
            scale_encoding: bundle.scale_encoding.clone(),
            elements: bundle.elements,
            groups: bundle.groups,
            index,
            scale,
            codebook,
        })
    }
}

pub fn load_tensor_payload_bundle(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    bundle: &TensorPayloadBundle,
    options: LoadOptions,
) -> Result<LoadedTensorBundle, String> {
    if options.chunk_bytes == 0 {
        return Err("chunk bytes must be greater than zero".to_string());
    }
    let index = load_payload_file(context, stream, &bundle.index_file, options)?;
    let scale = load_payload_file(context, stream, &bundle.scale_file, options)?;
    let codebook = load_payload_file(context, stream, &bundle.codebook_file, options)?;

    Ok(LoadedTensorBundle {
        tensor_index: bundle.tensor_index,
        tensor_name: bundle.tensor_name.clone(),
        dtype: bundle.dtype.clone(),
        shape: bundle.shape.clone(),
        family: bundle.family.clone(),
        candidate_id: bundle.candidate_id.clone(),
        scale_format: bundle.scale_format.clone(),
        group_size: bundle.group_size,
        tensor_scale: bundle.tensor_scale,
        index_encoding: bundle.index_encoding.clone(),
        scale_encoding: bundle.scale_encoding.clone(),
        elements: bundle.elements,
        groups: bundle.groups,
        index,
        scale,
        codebook,
    })
}

fn load_payload_file(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    referenced: &ReferencedFile,
    options: LoadOptions,
) -> Result<LoadedPayload, String> {
    let buffer_bytes = usize::try_from(referenced.bytes).map_err(|_| {
        format!(
            "payload {} is too large for this host: {} bytes",
            referenced.relative_path, referenced.bytes
        )
    })?;
    if buffer_bytes == 0 {
        return Err(format!("payload {} is empty", referenced.relative_path));
    }

    let mut file = File::open(&referenced.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            referenced.absolute_path.display()
        )
    })?;
    let mut buffer = context.alloc_buffer(buffer_bytes)?;
    let staging_bytes = buffer_bytes.min(options.chunk_bytes);
    let mut input = vec![0_u8; staging_bytes];
    let mut output = if options.verify {
        vec![0_u8; staging_bytes]
    } else {
        Vec::new()
    };

    let mut offset = 0_usize;
    let mut chunks = 0_u64;
    loop {
        let read = file.read(&mut input).map_err(|err| {
            format!(
                "failed to read {}: {err}",
                referenced.absolute_path.display()
            )
        })?;
        if read == 0 {
            break;
        }

        buffer.copy_from_host(offset, &input[..read], Some(stream))?;
        if options.verify {
            stream.synchronize()?;
            buffer.copy_to_host(offset, &mut output[..read], Some(stream))?;
            stream.synchronize()?;
            if input[..read] != output[..read] {
                return Err(format!(
                    "runtime load verification mismatch for {} at chunk {}",
                    referenced.relative_path, chunks
                ));
            }
        }
        offset += read;
        chunks += 1;
    }

    if !options.verify {
        stream.synchronize()?;
    }
    if offset != buffer_bytes {
        return Err(format!(
            "runtime load byte count mismatch for {}: expected {} got {}",
            referenced.relative_path, buffer_bytes, offset
        ));
    }

    Ok(LoadedPayload {
        role: referenced.role,
        relative_path: referenced.relative_path.clone(),
        bytes: referenced.bytes,
        chunks,
        buffer: Arc::new(buffer),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
    use crate::package::{TensorSelector, select_tensor_payload_bundle};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn cpu_registry_loads_tensor_payload_bundle() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [1_u8, 2, 3, 4, 5]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [6_u8, 7, 8]).unwrap();
        fs::write(root.join("codebooks/a.f32"), [9_u8; 10]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_test",
                "elements": 10,
                "groups": 3,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut registry = WeightRegistry::new();
        let registry_index = registry
            .load_and_insert(
                &mut context,
                &mut stream,
                &bundle,
                LoadOptions {
                    chunk_bytes: 4,
                    verify: true,
                },
            )
            .unwrap();

        assert_eq!(registry_index, 0);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.total_payload_bytes(), 18);
        let loaded = registry.get(0).unwrap();
        assert_eq!(loaded.tensor_name, "layer.0.attn.q_proj.weight");
        assert_eq!(loaded.index.bytes, 5);
        assert_eq!(loaded.index.chunks, 2);
        assert_eq!(loaded.scale.bytes, 3);
        assert_eq!(loaded.scale.chunks, 1);
        assert_eq!(loaded.codebook.bytes, 10);
        assert_eq!(loaded.codebook.chunks, 3);
        assert!(registry.get_by_name("layer.0.attn.q_proj.weight").is_some());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cpu_registry_loads_multiple_tensor_payload_bundles() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-multi-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/t0.idx4"), [1_u8, 2, 3]).unwrap();
        fs::write(root.join("tensors/t0.scale_u8"), [4_u8, 5]).unwrap();
        fs::write(root.join("codebooks/t0.f32"), [6_u8; 4]).unwrap();
        fs::write(root.join("tensors/t1.idx4"), [7_u8, 8]).unwrap();
        fs::write(root.join("tensors/t1.scale_u8"), [9_u8]).unwrap();
        fs::write(root.join("codebooks/t1.f32"), [10_u8; 3]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_test_a",
                "elements": 10,
                "groups": 3,
                "index_file": "tensors/t0.idx4",
                "scale_file": "tensors/t0.scale_u8",
                "codebook_file": "codebooks/t0.f32"
              }, {
                "name": "layer.1.attn.k_proj.weight",
                "dtype": "BF16",
                "family": "attn_k",
                "candidate_id": "aq4_test_b",
                "elements": 11,
                "groups": 4,
                "index_file": "tensors/t1.idx4",
                "scale_file": "tensors/t1.scale_u8",
                "codebook_file": "codebooks/t1.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle0 = select_tensor_payload_bundle(&root, &TensorSelector::Index(0)).unwrap();
        let bundle1 = select_tensor_payload_bundle(&root, &TensorSelector::Index(1)).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut registry = WeightRegistry::new();
        let indexes = registry
            .load_and_insert_many(
                &mut context,
                &mut stream,
                &[bundle0, bundle1],
                LoadOptions {
                    chunk_bytes: 2,
                    verify: true,
                },
            )
            .unwrap();

        assert_eq!(indexes, vec![0, 1]);
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.total_payload_bytes(), 15);
        assert_eq!(registry.resident_payload_bytes(), 15);
        assert_eq!(registry.codebook_payloads(), 2);
        assert!(registry.get_by_name("layer.0.attn.q_proj.weight").is_some());
        assert!(registry.get_by_name("layer.1.attn.k_proj.weight").is_some());

        let loaded0 = registry.get_by_name("layer.0.attn.q_proj.weight").unwrap();
        assert_eq!(loaded0.index.chunks, 2);
        assert_eq!(loaded0.scale.chunks, 1);
        assert_eq!(loaded0.codebook.chunks, 2);

        let loaded1 = registry.get_by_name("layer.1.attn.k_proj.weight").unwrap();
        assert_eq!(loaded1.index.chunks, 1);
        assert_eq!(loaded1.scale.chunks, 1);
        assert_eq!(loaded1.codebook.chunks, 2);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_named_passthrough_f32_decodes_bf16_f16_and_f32_payloads() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-passthrough-f32-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        fs::write(
            root.join("passthrough/bf16.raw"),
            [0x80_u8, 0x3f, 0x00, 0xc0],
        )
        .unwrap();
        fs::write(
            root.join("passthrough/f16.raw"),
            [0x00_u8, 0x3e, 0x00, 0xb4],
        )
        .unwrap();
        fs::write(
            root.join("passthrough/f32.raw"),
            [1.25_f32.to_le_bytes(), (-0.5_f32).to_le_bytes()].concat(),
        )
        .unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "model.layers.0.input_layernorm.weight",
                "shape": [2],
                "elements": 2,
                "payload_bytes": 4,
                "payload_file": "passthrough/bf16.raw"
              }, {
                "name": "model.layers.0.input_layernorm.bias",
                "dtype": "F16",
                "shape": [2],
                "elements": 2,
                "payload_bytes": 4,
                "payload_encoding": "raw_safetensors_payload",
                "payload_file": "passthrough/f16.raw"
              }, {
                "name": "model.layers.0.linear_attn.dt_bias",
                "dtype": "F32",
                "shape": [2],
                "elements": 2,
                "payload_bytes": 8,
                "payload_file": "passthrough/f32.raw"
              }]
            }"#,
        )
        .unwrap();

        let bf16 =
            read_named_passthrough_f32(&root, "model.layers.0.input_layernorm.weight", 3).unwrap();
        assert_eq!(bf16.dtype, "BF16");
        assert_eq!(bf16.shape, vec![2]);
        assert_eq!(bf16.values, vec![1.0_f32, -2.0_f32]);

        let f16 =
            read_named_passthrough_f32(&root, "model.layers.0.input_layernorm.bias", 3).unwrap();
        assert_eq!(f16.dtype, "F16");
        assert_eq!(f16.shape, vec![2]);
        assert_eq!(f16.values, vec![1.5_f32, -0.25_f32]);

        let f32 =
            read_named_passthrough_f32(&root, "model.layers.0.linear_attn.dt_bias", 3).unwrap();
        assert_eq!(f32.dtype, "F32");
        assert_eq!(f32.shape, vec![2]);
        assert_eq!(f32.values, vec![1.25_f32, -0.5_f32]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_named_passthrough_bf16_resident_streams_and_verifies_payload() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-resident-bf16-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        let payload = [0x80_u8, 0x3f, 0x00, 0xc0, 0x40, 0x40, 0x80, 0xbf];
        let payload_sha256 = format!("{:x}", Sha256::digest(payload));
        fs::write(root.join("passthrough/head.raw"), payload).unwrap();
        fs::write(
            root.join("manifest.json"),
            format!(
                r#"{{
                  "passthrough_tensors": [{{
                    "name": "lm_head.weight",
                    "dtype": "BF16",
                    "shape": [2, 2],
                    "elements": 4,
                    "payload_bytes": 8,
                    "payload_encoding": "raw_safetensors_payload",
                    "payload_sha256": "{payload_sha256}",
                    "payload_file": "passthrough/head.raw"
                  }}]
                }}"#
            ),
        )
        .unwrap();

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        for expected_shape in [&[][..], &[4][..], &[1, 4][..], &[2, 3][..]] {
            let err = load_named_passthrough_bf16_resident(
                &mut context,
                &mut stream,
                &root,
                "lm_head.weight",
                expected_shape,
                3,
            )
            .unwrap_err();
            assert!(err.contains("expected shape") || err.contains("shape mismatch"));
        }
        let verification =
            verify_named_passthrough_payload(&root, "lm_head.weight", "BF16", &[2, 2], 3).unwrap();
        assert_eq!(verification.tensor_name, "lm_head.weight");
        assert_eq!(verification.dtype, "BF16");
        assert_eq!(verification.shape, vec![2, 2]);
        assert_eq!(verification.elements, 4);
        assert_eq!(verification.payload_bytes, 8);
        assert_eq!(verification.payload_sha256, payload_sha256);
        assert_eq!(verification.verified_chunks, 3);
        let resident = load_named_passthrough_bf16_resident(
            &mut context,
            &mut stream,
            &root,
            "lm_head.weight",
            &[2, 2],
            3,
        )
        .unwrap();
        assert_eq!(resident.tensor_name, "lm_head.weight");
        assert_eq!(resident.shape, vec![2, 2]);
        assert_eq!(resident.elements, 4);
        assert_eq!(resident.payload_bytes, 8);
        assert_eq!(resident.payload_sha256, payload_sha256);
        assert_eq!(resident.upload_chunks, 3);
        let mut copied = [0_u8; 8];
        resident
            .buffer
            .copy_to_host(0, &mut copied, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(copied, payload);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resident_bf16_loader_requires_explicit_evidence_schema_and_exact_name() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-resident-bf16-schema-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        let payload = [0x80_u8, 0x3f];
        let payload_sha256 = format!("{:x}", Sha256::digest(payload));
        fs::write(root.join("passthrough/head.raw"), payload).unwrap();

        let write_manifest = |name: &str,
                              dtype: Option<&str>,
                              encoding: Option<&str>,
                              payload_bytes: Option<u64>| {
            let mut tensor = serde_json::json!({
                "name": name,
                "shape": [1, 1],
                "elements": 1,
                "payload_sha256": payload_sha256.clone(),
                "payload_file": "passthrough/head.raw"
            });
            if let Some(dtype) = dtype {
                tensor["dtype"] = serde_json::json!(dtype);
            }
            if let Some(encoding) = encoding {
                tensor["payload_encoding"] = serde_json::json!(encoding);
            }
            if let Some(payload_bytes) = payload_bytes {
                tensor["payload_bytes"] = serde_json::json!(payload_bytes);
            }
            fs::write(
                root.join("manifest.json"),
                serde_json::to_vec(&serde_json::json!({
                    "passthrough_tensors": [tensor]
                }))
                .unwrap(),
            )
            .unwrap();
        };

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        write_manifest(
            "backup.lm_head.weight",
            Some("BF16"),
            Some("raw_safetensors_payload"),
            Some(2),
        );
        let exact_name = load_named_passthrough_bf16_resident(
            &mut context,
            &mut stream,
            &root,
            "lm_head.weight",
            &[1, 1],
            1,
        )
        .unwrap_err();
        assert!(exact_name.contains("exact name"));

        write_manifest(
            "lm_head.weight",
            None,
            Some("raw_safetensors_payload"),
            Some(2),
        );
        let missing_dtype = load_named_passthrough_bf16_resident(
            &mut context,
            &mut stream,
            &root,
            "lm_head.weight",
            &[1, 1],
            1,
        )
        .unwrap_err();
        assert!(missing_dtype.contains("explicitly declare BF16 dtype"));

        write_manifest("lm_head.weight", Some("BF16"), None, Some(2));
        let missing_encoding = load_named_passthrough_bf16_resident(
            &mut context,
            &mut stream,
            &root,
            "lm_head.weight",
            &[1, 1],
            1,
        )
        .unwrap_err();
        assert!(missing_encoding.contains("raw_safetensors_payload encoding"));

        write_manifest(
            "lm_head.weight",
            Some("BF16"),
            Some("raw_safetensors_payload"),
            None,
        );
        let missing_bytes = load_named_passthrough_bf16_resident(
            &mut context,
            &mut stream,
            &root,
            "lm_head.weight",
            &[1, 1],
            1,
        )
        .unwrap_err();
        assert!(missing_bytes.contains("nonzero payload_bytes"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn passthrough_streaming_chunk_is_bounded() {
        assert_eq!(
            passthrough_stream_chunk_bytes(
                (PASSTHROUGH_MAX_STREAM_CHUNK_BYTES as u64) * 4,
                usize::MAX,
                "lm_head.weight",
            )
            .unwrap(),
            PASSTHROUGH_MAX_STREAM_CHUNK_BYTES
        );
        assert_eq!(
            passthrough_stream_chunk_bytes(17, usize::MAX, "small.weight").unwrap(),
            17
        );
        assert!(passthrough_stream_chunk_bytes(17, 0, "bad.weight").is_err());
    }

    #[test]
    fn load_named_passthrough_bf16_resident_rejects_checksum_mismatch() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-resident-bf16-checksum-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        fs::write(root.join("passthrough/head.raw"), [0x80_u8, 0x3f]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "lm_head.weight",
                "dtype": "BF16",
                "shape": [1, 1],
                "elements": 1,
                "payload_bytes": 2,
                "payload_encoding": "raw_safetensors_payload",
                "payload_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "payload_file": "passthrough/head.raw"
              }]
            }"#,
        )
        .unwrap();

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let err = load_named_passthrough_bf16_resident(
            &mut context,
            &mut stream,
            &root,
            "lm_head.weight",
            &[1, 1],
            1,
        )
        .unwrap_err();
        assert!(err.contains("checksum mismatch"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_named_passthrough_f32_rows_reads_selected_2d_rows() {
        fn bf16_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        }

        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-passthrough-f32-rows-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        fs::write(
            root.join("passthrough/embed.raw"),
            bf16_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0, -2.0]),
        )
        .unwrap();
        fs::write(
            root.join("passthrough/head.raw"),
            [
                0.25_f32.to_le_bytes(),
                0.5_f32.to_le_bytes(),
                0.75_f32.to_le_bytes(),
                1.0_f32.to_le_bytes(),
            ]
            .concat(),
        )
        .unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "model.embed_tokens.weight",
                "dtype": "BF16",
                "shape": [3, 2],
                "elements": 6,
                "payload_bytes": 12,
                "payload_encoding": "raw_safetensors_payload",
                "payload_file": "passthrough/embed.raw"
              }, {
                "name": "lm_head.weight",
                "dtype": "F32",
                "shape": [2, 2],
                "elements": 4,
                "payload_bytes": 16,
                "payload_file": "passthrough/head.raw"
              }]
            }"#,
        )
        .unwrap();

        let rows = read_named_passthrough_f32_rows(&root, "model.embed_tokens.weight", &[2, 0, 2])
            .unwrap();
        assert_eq!(rows.dtype, "BF16");
        assert_eq!(rows.shape, vec![3, 2]);
        assert_eq!(rows.row_indices, vec![2, 0, 2]);
        assert_eq!(rows.columns, 2);
        assert_eq!(rows.values, vec![-1.0, -2.0, 1.0, 2.0, -1.0, -2.0]);

        let rows = read_named_passthrough_f32_rows(&root, "lm_head.weight", &[1]).unwrap();
        assert_eq!(rows.dtype, "F32");
        assert_eq!(rows.columns, 2);
        assert_eq!(rows.values, vec![0.75, 1.0]);

        let range =
            read_named_passthrough_f32_row_range(&root, "model.embed_tokens.weight", 1, 2).unwrap();
        assert_eq!(range.dtype, "BF16");
        assert_eq!(range.shape, vec![3, 2]);
        assert_eq!(range.start_row, 1);
        assert_eq!(range.row_count, 2);
        assert_eq!(range.columns, 2);
        assert_eq!(range.values, vec![3.0, 4.0, -1.0, -2.0]);

        let err =
            read_named_passthrough_f32_rows(&root, "model.embed_tokens.weight", &[3]).unwrap_err();
        assert!(err.contains("out of range"));

        let err = read_named_passthrough_f32_row_range(&root, "model.embed_tokens.weight", 2, 2)
            .unwrap_err();
        assert!(err.contains("out of range"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_named_passthrough_f32_rows_rejects_unsupported_payload_encoding() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-passthrough-encoding-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        fs::write(
            root.join("passthrough/embed.raw"),
            [0x80_u8, 0x3f, 0x00, 0x40],
        )
        .unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "model.embed_tokens.weight",
                "dtype": "BF16",
                "shape": [1, 2],
                "elements": 2,
                "payload_bytes": 4,
                "payload_encoding": "compressed_payload",
                "payload_file": "passthrough/embed.raw"
              }]
            }"#,
        )
        .unwrap();

        let err =
            read_named_passthrough_f32_rows(&root, "model.embed_tokens.weight", &[0]).unwrap_err();
        assert!(err.contains("unsupported payload encoding"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn effective_rmsnorm_weight_values_handles_qwen35_additive_weights() {
        let additive = effective_rmsnorm_weight_values(
            "model.language_model.layers.6.input_layernorm.weight",
            &[-0.25, 0.0, 0.5],
        );
        assert_eq!(additive, vec![0.75, 1.0, 1.5]);

        let q_norm = effective_rmsnorm_weight_values(
            "model.language_model.layers.3.self_attn.q_norm.weight",
            &[0.25, -0.125],
        );
        assert_eq!(q_norm, vec![1.25, 0.875]);

        let direct = effective_rmsnorm_weight_values(
            "model.language_model.layers.4.linear_attn.norm.weight",
            &[0.25, 1.0],
        );
        assert_eq!(direct, vec![0.25, 1.0]);

        let qwen3_direct = effective_rmsnorm_weight_values(
            "model.language_model.layers.0.input_layernorm.weight",
            &[0.95, 1.05],
        );
        assert_eq!(qwen3_direct, vec![0.95, 1.05]);
    }

    #[test]
    fn materialize_config_resolves_aq4_metadata_and_matrix_shape() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-materialize-config-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [0x10_u8, 0x32]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [1_u8]).unwrap();
        fs::write(root.join("codebooks/a.f32"), [0_u8; 64]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "shape": [2, 2],
                "family": "attn_q",
                "candidate_id": "aq4_test",
                "scale_format": "e4m3",
                "group_size": 4,
                "tensor_scale": 1.0,
                "index_encoding": "idx4_low_nibble_first",
                "scale_encoding": "u8_scale_table_index",
                "elements": 4,
                "groups": 1,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let loaded = load_tensor_payload_bundle(
            &mut context,
            &mut stream,
            &bundle,
            LoadOptions {
                chunk_bytes: 8,
                verify: true,
            },
        )
        .unwrap();

        let config = materialize_config(&loaded).unwrap();
        assert_eq!(config.scale_format, "e4m3");
        assert!(!config.scale_values.is_empty());
        assert_eq!(config.group_size, 4);
        assert_eq!(config.tensor_scale, 1.0);
        assert_eq!(config.elements, 4);
        assert_eq!(config.output_bytes, 16);
        assert_eq!(
            matrix_shape_rows_cols(&loaded.shape, config.elements).unwrap(),
            (2, 2)
        );
        assert!(matrix_shape_rows_cols(&[4], config.elements).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn materialize_selected_aq4_matrix_applies_manifest_row_scale_overrides() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-row-scale-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/a.idx4"), [0x21_u8, 0x43]).unwrap();
        fs::write(root.join("tensors/a.scale_u8"), [127_u8, 127_u8]).unwrap();
        let codebook_values: Vec<f32> = (0..16).map(|value| value as f32).collect();
        fs::write(
            root.join("codebooks/a.f32"),
            encode_f32_to_bytes(&codebook_values),
        )
        .unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "tensor.weight",
                "dtype": "BF16",
                "shape": [2, 2],
                "family": "test",
                "candidate_id": "aq4_test",
                "scale_format": "e8m0",
                "group_size": 2,
                "tensor_scale": 1.0,
                "index_encoding": "idx4_low_nibble_first",
                "scale_encoding": "u8_scale_table_index",
                "elements": 4,
                "groups": 2,
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }],
              "row_scale_overrides": {
                "schema_version": "row-scale-overrides-v0.1",
                "entries": [{
                  "tensor_name": "tensor.weight",
                  "row_index": 1,
                  "scale": 10.0,
                  "source": "unit-test"
                }]
              }
            }"#,
        )
        .unwrap();

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut registry = WeightRegistry::new();
        let (rows, cols, output) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            &root,
            "tensor.weight",
            8,
        )
        .unwrap();

        assert_eq!((rows, cols), (2, 2));
        let mut output_bytes = vec![0_u8; 4 * std::mem::size_of::<f32>()];
        output
            .copy_to_host(0, &mut output_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            decode_f32_le_values(&output_bytes),
            vec![1.0_f32, 2.0, 30.0, 40.0]
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_deduplicates_shared_codebook_payloads() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-codebook-dedup-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/t0.idx4"), [1_u8, 2, 3]).unwrap();
        fs::write(root.join("tensors/t0.scale_u8"), [4_u8, 5]).unwrap();
        fs::write(root.join("tensors/t1.idx4"), [6_u8, 7]).unwrap();
        fs::write(root.join("tensors/t1.scale_u8"), [8_u8]).unwrap();
        fs::write(root.join("codebooks/shared.f32"), [9_u8; 4]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_shared",
                "elements": 10,
                "groups": 3,
                "index_file": "tensors/t0.idx4",
                "scale_file": "tensors/t0.scale_u8",
                "codebook_file": "codebooks/shared.f32"
              }, {
                "name": "layer.1.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_shared",
                "elements": 11,
                "groups": 4,
                "index_file": "tensors/t1.idx4",
                "scale_file": "tensors/t1.scale_u8",
                "codebook_file": "codebooks/shared.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle0 = select_tensor_payload_bundle(&root, &TensorSelector::Index(0)).unwrap();
        let bundle1 = select_tensor_payload_bundle(&root, &TensorSelector::Index(1)).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut registry = WeightRegistry::new();
        registry
            .load_and_insert_many(
                &mut context,
                &mut stream,
                &[bundle0, bundle1],
                LoadOptions {
                    chunk_bytes: 2,
                    verify: true,
                },
            )
            .unwrap();

        assert_eq!(registry.len(), 2);
        assert_eq!(registry.codebook_payloads(), 1);
        assert_eq!(registry.total_payload_bytes(), 16);
        assert_eq!(registry.resident_payload_bytes(), 12);
        let loaded0 = registry.get_by_name("layer.0.attn.q_proj.weight").unwrap();
        let loaded1 = registry.get_by_name("layer.1.attn.q_proj.weight").unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &loaded0.codebook.buffer,
            &loaded1.codebook.buffer
        ));
        assert_eq!(loaded0.codebook.chunks, 2);
        assert_eq!(loaded1.codebook.chunks, 2);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_package_tensor_prefix_returns_loaded_package_handle() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-package-prefix-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/t0.idx4"), [1_u8, 2]).unwrap();
        fs::write(root.join("tensors/t0.scale_u8"), [3_u8]).unwrap();
        fs::write(root.join("codebooks/t0.f32"), [4_u8; 4]).unwrap();
        fs::write(root.join("tensors/t1.idx4"), [5_u8, 6]).unwrap();
        fs::write(root.join("tensors/t1.scale_u8"), [7_u8]).unwrap();
        fs::write(root.join("codebooks/t1.f32"), [8_u8; 4]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "schema_version": "test-package",
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_a",
                "elements": 4,
                "groups": 1,
                "index_file": "tensors/t0.idx4",
                "scale_file": "tensors/t0.scale_u8",
                "codebook_file": "codebooks/t0.f32"
              }, {
                "name": "layer.0.attn.k_proj.weight",
                "dtype": "BF16",
                "family": "attn_k",
                "candidate_id": "aq4_b",
                "elements": 4,
                "groups": 1,
                "index_file": "tensors/t1.idx4",
                "scale_file": "tensors/t1.scale_u8",
                "codebook_file": "codebooks/t1.f32"
              }]
            }"#,
        )
        .unwrap();

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let loaded = load_package_tensor_prefix(
            &mut context,
            &mut stream,
            &root,
            1,
            LoadOptions {
                chunk_bytes: 2,
                verify: true,
            },
        )
        .unwrap();

        assert_eq!(
            loaded.summary.schema_version.as_deref(),
            Some("test-package")
        );
        assert_eq!(loaded.summary.quantized_tensors, 2);
        assert_eq!(loaded.loaded_tensor_count, 1);
        assert_eq!(loaded.registry_indices, vec![0]);
        assert_eq!(loaded.registry().len(), 1);
        assert_eq!(loaded.registry().total_payload_bytes(), 7);
        assert_eq!(loaded.registry().resident_payload_bytes(), 7);
        assert!(
            loaded
                .registry()
                .get_by_name("layer.0.attn.q_proj.weight")
                .is_some()
        );
        assert!(
            loaded
                .registry()
                .get_by_name("layer.0.attn.k_proj.weight")
                .is_none()
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loader_rejects_zero_chunk_size() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-zero-chunk-test-{}",
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
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "a",
                "index_file": "tensors/a.idx4",
                "scale_file": "tensors/a.scale_u8",
                "codebook_file": "codebooks/a.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let err = load_tensor_payload_bundle(
            &mut context,
            &mut stream,
            &bundle,
            LoadOptions {
                chunk_bytes: 0,
                verify: true,
            },
        )
        .unwrap_err();
        assert!(err.contains("chunk bytes"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_lookup_apis_expose_expected_views() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-lookup-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/t0.idx4"), [1_u8, 2, 3]).unwrap();
        fs::write(root.join("tensors/t0.scale_u8"), [4_u8]).unwrap();
        fs::write(root.join("codebooks/t0.f32"), [5_u8; 2]).unwrap();
        fs::write(root.join("tensors/t1.idx4"), [6_u8, 7]).unwrap();
        fs::write(root.join("tensors/t1.scale_u8"), [8_u8, 9]).unwrap();
        fs::write(root.join("codebooks/t1.f32"), [10_u8; 3]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_small",
                "elements": 10,
                "groups": 3,
                "index_file": "tensors/t0.idx4",
                "scale_file": "tensors/t0.scale_u8",
                "codebook_file": "codebooks/t0.f32"
              }, {
                "name": "layer.1.attn.k_proj.weight",
                "dtype": "BF16",
                "family": "attn_k",
                "candidate_id": "ak4_small",
                "elements": 11,
                "groups": 4,
                "index_file": "tensors/t1.idx4",
                "scale_file": "tensors/t1.scale_u8",
                "codebook_file": "codebooks/t1.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle0 = select_tensor_payload_bundle(&root, &TensorSelector::Index(0)).unwrap();
        let bundle1 = select_tensor_payload_bundle(&root, &TensorSelector::Index(1)).unwrap();
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut registry = WeightRegistry::new();
        let indexes = registry
            .load_and_insert_many(
                &mut context,
                &mut stream,
                &[bundle0, bundle1],
                LoadOptions {
                    chunk_bytes: 2,
                    verify: true,
                },
            )
            .unwrap();
        assert_eq!(indexes, vec![0, 1]);

        let mut all = registry.iter();
        assert_eq!(
            all.next().unwrap().tensor_name,
            "layer.0.attn.q_proj.weight"
        );
        assert_eq!(
            all.next().unwrap().tensor_name,
            "layer.1.attn.k_proj.weight"
        );
        assert!(all.next().is_none());

        assert!(
            registry
                .tensor_by_name("layer.0.attn.q_proj.weight")
                .is_some()
        );
        assert!(
            registry
                .find_by_family_candidate("attn_k", "ak4_small")
                .iter()
                .any(|bundle| bundle.tensor_name == "layer.1.attn.k_proj.weight")
        );
        assert!(
            registry
                .find_by_family_candidate("attn_k", "aq4_small")
                .is_empty()
        );

        let t0_index = registry
            .get_loaded_payload(
                "layer.0.attn.q_proj.weight",
                ReferencedFileRole::TensorIndex,
            )
            .unwrap();
        assert_eq!(t0_index.bytes, 3);
        let t0_scale = registry
            .get_loaded_payload(
                "layer.0.attn.q_proj.weight",
                ReferencedFileRole::TensorScale,
            )
            .unwrap();
        assert_eq!(t0_scale.bytes, 1);
        let t0_codebook = registry
            .get_loaded_payload(
                "layer.0.attn.q_proj.weight",
                ReferencedFileRole::TensorCodebook,
            )
            .unwrap();
        assert_eq!(t0_codebook.bytes, 2);

        assert!(
            registry
                .get_loaded_payload(
                    "layer.0.attn.q_proj.weight",
                    ReferencedFileRole::Passthrough
                )
                .is_none()
        );
        assert!(
            registry
                .get_loaded_payload("missing", ReferencedFileRole::TensorIndex)
                .is_none()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loaded_package_lookups_delegate_to_registry() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-loader-package-lookup-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/t0.idx4"), [1_u8, 2]).unwrap();
        fs::write(root.join("tensors/t0.scale_u8"), [3_u8]).unwrap();
        fs::write(root.join("codebooks/t0.f32"), [4_u8; 2]).unwrap();
        fs::write(root.join("tensors/t1.idx4"), [5_u8, 6]).unwrap();
        fs::write(root.join("tensors/t1.scale_u8"), [7_u8]).unwrap();
        fs::write(root.join("codebooks/t1.f32"), [8_u8; 2]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "layer.0.attn.q_proj.weight",
                "dtype": "BF16",
                "family": "attn_q",
                "candidate_id": "aq4_small",
                "elements": 10,
                "groups": 3,
                "index_file": "tensors/t0.idx4",
                "scale_file": "tensors/t0.scale_u8",
                "codebook_file": "codebooks/t0.f32"
              }, {
                "name": "layer.1.attn.k_proj.weight",
                "dtype": "BF16",
                "family": "attn_k",
                "candidate_id": "ak4_small",
                "elements": 11,
                "groups": 4,
                "index_file": "tensors/t1.idx4",
                "scale_file": "tensors/t1.scale_u8",
                "codebook_file": "codebooks/t1.f32"
              }]
            }"#,
        )
        .unwrap();

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let loaded = load_package_tensor_prefix(
            &mut context,
            &mut stream,
            &root,
            2,
            LoadOptions {
                chunk_bytes: 2,
                verify: true,
            },
        )
        .unwrap();

        assert!(
            loaded
                .tensor_by_name("layer.0.attn.q_proj.weight")
                .is_some()
        );
        assert_eq!(
            loaded.find_by_family_candidate("attn_k", "ak4_small").len(),
            1
        );
        let payload = loaded
            .payload_by_name_and_role(
                "layer.0.attn.q_proj.weight",
                ReferencedFileRole::TensorScale,
            )
            .unwrap();
        assert_eq!(payload.bytes, 1);

        assert!(
            loaded
                .payload_by_name_and_role(
                    "layer.0.attn.q_proj.weight",
                    ReferencedFileRole::Passthrough
                )
                .is_none()
        );
        assert!(loaded.tensor_by_name("missing.tensor.weight").is_none());

        fs::remove_dir_all(root).unwrap();
    }
}
