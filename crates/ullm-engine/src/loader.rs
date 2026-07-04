// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::package::{
    PackageSummary, PassthroughPayloadBundle, ReferencedFile, ReferencedFileRole,
    TensorPayloadBundle, TensorSelector, list_tensor_payload_bundles,
    select_passthrough_payload_bundle, select_tensor_payload_bundle,
};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream};

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
    Ok((rows, cols, output))
}

pub fn resolve_passthrough_dtype<'a>(
    bundle: &'a PassthroughPayloadBundle,
    tensor_name: &str,
) -> Result<&'a str, String> {
    if let Some(dtype) = bundle.dtype.as_deref() {
        return match dtype {
            "BF16" | "F32" => Ok(dtype),
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

pub fn read_passthrough_payload_f32_bytes(
    bundle: &PassthroughPayloadBundle,
    chunk_bytes: usize,
    dtype: &str,
) -> Result<Vec<f32>, String> {
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
    let element_size = match dtype {
        "BF16" => 2_usize,
        "F32" => 4_usize,
        _ => {
            return Err(format!(
                "unsupported passthrough dtype {dtype} for tensor {}",
                bundle.tensor_name
            ));
        }
    };
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

    let mut values = Vec::with_capacity(expected_elements);
    let mut scratch = vec![0_u8; chunk_bytes];
    let mut read_bytes = 0_usize;
    let mut carry = Vec::with_capacity(element_size - 1);
    let mut merge = Vec::with_capacity(chunk_bytes + element_size);
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
            let value = match dtype {
                "BF16" => {
                    let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
                    f32::from_bits(u32::from(raw) << 16)
                }
                "F32" => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _ => unreachable!(),
            };
            values.push(value);
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
    fn read_named_passthrough_f32_decodes_bf16_and_f32_payloads() {
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

        let f32 =
            read_named_passthrough_f32(&root, "model.layers.0.linear_attn.dt_bias", 3).unwrap();
        assert_eq!(f32.dtype, "F32");
        assert_eq!(f32.shape, vec![2]);
        assert_eq!(f32.values, vec![1.25_f32, -0.5_f32]);

        fs::remove_dir_all(root).unwrap();
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
