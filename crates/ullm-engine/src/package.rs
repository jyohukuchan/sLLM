// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::qwen3_names::qwen3_tensor_name_alias;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

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
    pub row_scale_overrides: Vec<RowScaleOverrideEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PassthroughPayloadBundle {
    pub tensor_index: usize,
    pub tensor_name: String,
    pub dtype: Option<String>,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub payload_bytes: u64,
    pub payload_encoding: Option<String>,
    pub payload_sha256: Option<String>,
    pub payload_file: ReferencedFile,
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

pub const ROW_SCALE_OVERRIDES_SCHEMA_VERSION: &str = "row-scale-overrides-v0.1";

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RowScaleOverridesManifest {
    pub schema_version: String,
    #[serde(default)]
    pub entries: Vec<RowScaleOverrideEntry>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RowScaleOverrideEntry {
    pub tensor_name: String,
    pub row_index: usize,
    pub scale: f32,
    #[serde(default)]
    pub source: Option<String>,
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
    #[serde(default)]
    row_scale_overrides: Option<RowScaleOverridesManifest>,
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
    dtype: Option<String>,
    #[serde(default)]
    shape: Vec<u64>,
    #[serde(default)]
    elements: u64,
    #[serde(default)]
    payload_bytes: u64,
    payload_encoding: Option<String>,
    payload_sha256: Option<String>,
    payload_file: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Codebook {
    family: Option<String>,
    candidate_id: Option<String>,
    file: Option<String>,
}

/// Bounded passthrough metadata parsed from manifest bytes whose hash was
/// already verified by the caller. This bridge never reopens the manifest.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct VerifiedPassthroughDescriptor {
    pub(crate) name: String,
    pub(crate) relative_path: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) payload_bytes: u64,
    pub(crate) encoding: String,
    pub(crate) payload_sha256: String,
}

/// Parses verified manifest bytes into exact passthrough descriptors.
///
/// An allocation-free lexical scan bounds nesting, tokens, containers, atom
/// length, and string amplification before serde. Serde's internal allocations
/// are therefore bounded but are not fully fallible allocations.
pub(crate) fn parse_verified_passthrough_descriptors(
    verified_manifest_bytes: &[u8],
    descriptor_limit: usize,
) -> Result<Vec<VerifiedPassthroughDescriptor>, String> {
    if descriptor_limit == 0 || descriptor_limit > 4_096 {
        return Err("resource: verified passthrough descriptor limit must be in 1..=4096".into());
    }
    if verified_manifest_bytes.is_empty() || verified_manifest_bytes.len() > 16 * 1024 * 1024 {
        return Err("resource: verified manifest byte length is outside supported limit".into());
    }
    scan_verified_manifest_json(verified_manifest_bytes).map_err(|failure| match failure {
        VerifiedJsonScanFailure::Invalid => "invalid: verified manifest lexical form is invalid",
        VerifiedJsonScanFailure::Resource => {
            "resource: verified manifest lexical complexity exceeds limit"
        }
    })?;
    let manifest: Manifest = serde_json::from_slice(verified_manifest_bytes)
        .map_err(|_| "verified manifest JSON is invalid".to_string())?;
    if manifest.tensors.len() > 4_096
        || manifest.passthrough_tensors.len() > descriptor_limit
        || manifest.codebooks.len() > 4_096
        || manifest
            .row_scale_overrides
            .as_ref()
            .is_some_and(|overrides| overrides.entries.len() > 4_096)
    {
        return Err("resource: verified manifest declaration count exceeds limit".into());
    }
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(manifest.passthrough_tensors.len())
        .map_err(|_| "resource: verified descriptor allocation failed".to_string())?;
    for tensor in manifest.passthrough_tensors {
        let name = required_bounded_manifest_string(tensor.name, 256, "passthrough name")?;
        let relative_path =
            required_bounded_manifest_string(tensor.payload_file, 4_096, "passthrough path")?;
        validate_verified_relative_path(&relative_path)?;
        let dtype = required_bounded_manifest_string(tensor.dtype, 64, "passthrough dtype")?;
        let encoding =
            required_bounded_manifest_string(tensor.payload_encoding, 128, "passthrough encoding")?;
        let payload_sha256 = required_bounded_manifest_string(
            tensor.payload_sha256,
            64,
            "passthrough payload digest",
        )?;
        if !is_lowercase_sha256(&payload_sha256) {
            return Err("verified passthrough payload digest is invalid".into());
        }
        if tensor.shape.is_empty() || tensor.shape.len() > crate::model_graph::MAX_TENSOR_RANK {
            return Err("verified passthrough shape rank is invalid".into());
        }
        if tensor.shape.contains(&0) {
            return Err("verified passthrough shape contains zero".into());
        }
        let elements = tensor.shape.iter().try_fold(1_u64, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| "resource: verified passthrough shape product overflows".to_string())
        })?;
        if elements > crate::model_graph::MAX_TENSOR_LOGICAL_ELEMENTS {
            return Err(
                "resource: verified passthrough shape exceeds logical element limit".into(),
            );
        }
        let element_bytes = match dtype.as_str() {
            "F32" => 4_u64,
            "BF16" | "F16" => 2_u64,
            _ => return Err("verified passthrough dtype is unsupported".into()),
        };
        let expected_bytes = elements.checked_mul(element_bytes).ok_or_else(|| {
            "resource: verified passthrough payload byte count overflows".to_string()
        })?;
        if elements != tensor.elements
            || tensor.payload_bytes == 0
            || tensor.payload_bytes != expected_bytes
            || encoding != "raw_safetensors_payload"
        {
            return Err(
                "verified passthrough element or payload byte declaration is invalid".into(),
            );
        }
        descriptors.push(VerifiedPassthroughDescriptor {
            name,
            relative_path,
            dtype,
            shape: tensor.shape,
            payload_bytes: tensor.payload_bytes,
            encoding,
            payload_sha256,
        });
    }
    descriptors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    if descriptors
        .windows(2)
        .any(|window| window[0].name == window[1].name)
    {
        return Err("verified passthrough tensor name is duplicated".into());
    }
    Ok(descriptors)
}

const VERIFIED_JSON_MAX_DEPTH: usize = 64;
const VERIFIED_JSON_MAX_TOKENS: usize = 131_072;
const VERIFIED_JSON_MAX_CONTAINERS: usize = 16_384;
const VERIFIED_JSON_MAX_STRING_BYTES: usize = 4_096;
const VERIFIED_JSON_MAX_TOTAL_STRING_BYTES: usize = 16 * 1024 * 1024;
const VERIFIED_JSON_MAX_ATOM_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerifiedJsonScanFailure {
    Invalid,
    Resource,
}

fn scan_verified_manifest_json(bytes: &[u8]) -> Result<(), VerifiedJsonScanFailure> {
    let mut stack = [0_u8; VERIFIED_JSON_MAX_DEPTH];
    let mut depth = 0_usize;
    let mut tokens = 0_usize;
    let mut containers = 0_usize;
    let mut total_string_bytes = 0_usize;
    let mut index = 0_usize;
    while index < bytes.len() {
        match bytes[index] {
            b' ' | b'\n' | b'\r' | b'\t' | b',' | b':' => index += 1,
            open @ (b'{' | b'[') => {
                if depth == VERIFIED_JSON_MAX_DEPTH {
                    return Err(VerifiedJsonScanFailure::Resource);
                }
                stack[depth] = open;
                depth += 1;
                containers = containers
                    .checked_add(1)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
                tokens = tokens
                    .checked_add(1)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
                if containers > VERIFIED_JSON_MAX_CONTAINERS || tokens > VERIFIED_JSON_MAX_TOKENS {
                    return Err(VerifiedJsonScanFailure::Resource);
                }
                index += 1;
            }
            close @ (b'}' | b']') => {
                if depth == 0
                    || (close == b'}' && stack[depth - 1] != b'{')
                    || (close == b']' && stack[depth - 1] != b'[')
                {
                    return Err(VerifiedJsonScanFailure::Invalid);
                }
                depth -= 1;
                index += 1;
            }
            b'"' => {
                index += 1;
                let mut string_bytes = 0_usize;
                let mut closed = false;
                while index < bytes.len() {
                    match bytes[index] {
                        b'"' => {
                            index += 1;
                            closed = true;
                            break;
                        }
                        0x00..=0x1f => return Err(VerifiedJsonScanFailure::Invalid),
                        b'\\' => {
                            index += 1;
                            let escape =
                                *bytes.get(index).ok_or(VerifiedJsonScanFailure::Invalid)?;
                            match escape {
                                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                                    string_bytes = string_bytes
                                        .checked_add(1)
                                        .ok_or(VerifiedJsonScanFailure::Resource)?;
                                    index += 1;
                                }
                                b'u' => {
                                    let end = index
                                        .checked_add(5)
                                        .ok_or(VerifiedJsonScanFailure::Resource)?;
                                    let digits = bytes
                                        .get(index + 1..end)
                                        .ok_or(VerifiedJsonScanFailure::Invalid)?;
                                    let unit = decode_json_hex_unit(digits)
                                        .ok_or(VerifiedJsonScanFailure::Invalid)?;
                                    let decoded_bytes = match unit {
                                        0xd800..=0xdbff => {
                                            let pair_end = end
                                                .checked_add(6)
                                                .ok_or(VerifiedJsonScanFailure::Resource)?;
                                            if bytes.get(end..end + 2) != Some(b"\\u") {
                                                return Err(VerifiedJsonScanFailure::Invalid);
                                            }
                                            let low = bytes
                                                .get(end + 2..pair_end)
                                                .and_then(decode_json_hex_unit)
                                                .ok_or(VerifiedJsonScanFailure::Invalid)?;
                                            if !(0xdc00..=0xdfff).contains(&low) {
                                                return Err(VerifiedJsonScanFailure::Invalid);
                                            }
                                            index = pair_end;
                                            4
                                        }
                                        0xdc00..=0xdfff => {
                                            return Err(VerifiedJsonScanFailure::Invalid);
                                        }
                                        _ => {
                                            index = end;
                                            char::from_u32(u32::from(unit))
                                                .ok_or(VerifiedJsonScanFailure::Invalid)?
                                                .len_utf8()
                                        }
                                    };
                                    string_bytes = string_bytes
                                        .checked_add(decoded_bytes)
                                        .ok_or(VerifiedJsonScanFailure::Resource)?;
                                }
                                _ => return Err(VerifiedJsonScanFailure::Invalid),
                            }
                        }
                        _ => {
                            string_bytes = string_bytes
                                .checked_add(1)
                                .ok_or(VerifiedJsonScanFailure::Resource)?;
                            index += 1;
                        }
                    }
                    if string_bytes > VERIFIED_JSON_MAX_STRING_BYTES {
                        return Err(VerifiedJsonScanFailure::Resource);
                    }
                }
                if !closed {
                    return Err(VerifiedJsonScanFailure::Invalid);
                }
                total_string_bytes = total_string_bytes
                    .checked_add(string_bytes)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
                tokens = tokens
                    .checked_add(1)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
                if total_string_bytes > VERIFIED_JSON_MAX_TOTAL_STRING_BYTES
                    || tokens > VERIFIED_JSON_MAX_TOKENS
                {
                    return Err(VerifiedJsonScanFailure::Resource);
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = index;
                while index < bytes.len()
                    && matches!(bytes[index], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    index += 1;
                }
                if index - start > VERIFIED_JSON_MAX_ATOM_BYTES {
                    return Err(VerifiedJsonScanFailure::Resource);
                }
                tokens = tokens
                    .checked_add(1)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
            }
            b't' if bytes.get(index..index + 4) == Some(b"true") => {
                index += 4;
                tokens = tokens
                    .checked_add(1)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
            }
            b'n' if bytes.get(index..index + 4) == Some(b"null") => {
                index += 4;
                tokens = tokens
                    .checked_add(1)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
            }
            b'f' if bytes.get(index..index + 5) == Some(b"false") => {
                index += 5;
                tokens = tokens
                    .checked_add(1)
                    .ok_or(VerifiedJsonScanFailure::Resource)?;
            }
            _ => return Err(VerifiedJsonScanFailure::Invalid),
        }
        if tokens > VERIFIED_JSON_MAX_TOKENS {
            return Err(VerifiedJsonScanFailure::Resource);
        }
    }
    if depth != 0 {
        return Err(VerifiedJsonScanFailure::Invalid);
    }
    Ok(())
}

fn decode_json_hex_unit(digits: &[u8]) -> Option<u16> {
    if digits.len() != 4 {
        return None;
    }
    let mut value = 0_u16;
    for digit in digits {
        value = value.checked_mul(16)?;
        value = value.checked_add(match digit {
            b'0'..=b'9' => u16::from(*digit - b'0'),
            b'a'..=b'f' => u16::from(*digit - b'a' + 10),
            b'A'..=b'F' => u16::from(*digit - b'A' + 10),
            _ => return None,
        })?;
    }
    Some(value)
}

fn validate_verified_relative_path(value: &str) -> Result<(), String> {
    if value.starts_with('/')
        || value.ends_with('/')
        || value.split('/').any(|part| part.is_empty())
        || !Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err("verified passthrough path is not normalized relative".into());
    }
    Ok(())
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_bounded_manifest_string(
    value: Option<String>,
    maximum_bytes: usize,
    label: &'static str,
) -> Result<String, String> {
    let value = value.ok_or_else(|| format!("verified {label} is missing"))?;
    if value.is_empty() || value.len() > maximum_bytes {
        return Err(format!("verified {label} length is invalid"));
    }
    Ok(value)
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
    tensor_payload_bundle_from_manifest(package_dir, &manifest, tensor_index, tensor)
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
            &manifest,
            tensor_index,
            tensor,
        )?);
    }
    Ok(bundles)
}

pub fn select_passthrough_payload_bundle(
    path: impl AsRef<Path>,
    selector: &TensorSelector,
) -> Result<PassthroughPayloadBundle, String> {
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    let tensor_index = select_passthrough_index(&manifest, selector)?;
    let tensor = manifest
        .passthrough_tensors
        .get(tensor_index)
        .ok_or_else(|| format!("passthrough tensor index {tensor_index} is out of range"))?;
    passthrough_payload_bundle_from_manifest(package_dir, tensor_index, tensor)
}

pub fn select_exact_passthrough_payload_bundle(
    path: impl AsRef<Path>,
    tensor_name: &str,
) -> Result<PassthroughPayloadBundle, String> {
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    let matches = manifest
        .passthrough_tensors
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.name.as_deref() == Some(tensor_name))
        .collect::<Vec<_>>();
    let (tensor_index, tensor) = match matches.as_slice() {
        [(tensor_index, tensor)] => (*tensor_index, *tensor),
        [] => {
            return Err(format!(
                "no passthrough tensor has exact name \"{tensor_name}\""
            ));
        }
        _ => {
            return Err(format!(
                "passthrough tensor exact name \"{tensor_name}\" is duplicated {} times",
                matches.len()
            ));
        }
    };
    passthrough_payload_bundle_from_manifest(package_dir, tensor_index, tensor)
}

pub fn list_passthrough_payload_bundles(
    path: impl AsRef<Path>,
) -> Result<Vec<PassthroughPayloadBundle>, String> {
    let package_dir = path.as_ref();
    let manifest = read_manifest(package_dir)?;
    let mut bundles = Vec::with_capacity(manifest.passthrough_tensors.len());
    for (tensor_index, tensor) in manifest.passthrough_tensors.iter().enumerate() {
        bundles.push(passthrough_payload_bundle_from_manifest(
            package_dir,
            tensor_index,
            tensor,
        )?);
    }
    Ok(bundles)
}

fn tensor_payload_bundle_from_manifest(
    package_dir: &Path,
    manifest: &Manifest,
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
    let row_scale_overrides =
        matching_row_scale_overrides(&manifest.row_scale_overrides, &tensor_name)?;

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
        row_scale_overrides,
    })
}

fn matching_row_scale_overrides(
    overrides: &Option<RowScaleOverridesManifest>,
    tensor_name: &str,
) -> Result<Vec<RowScaleOverrideEntry>, String> {
    let Some(overrides) = overrides else {
        return Ok(Vec::new());
    };
    if overrides.schema_version != ROW_SCALE_OVERRIDES_SCHEMA_VERSION {
        return Err(format!(
            "row_scale_overrides schema_version must be {}, got {}",
            ROW_SCALE_OVERRIDES_SCHEMA_VERSION, overrides.schema_version
        ));
    }
    let mut seen = BTreeSet::<(&str, usize)>::new();
    let mut matching = Vec::new();
    for entry in &overrides.entries {
        if entry.tensor_name.is_empty() {
            return Err("row_scale_overrides entry tensor_name must not be empty".to_string());
        }
        if !entry.scale.is_finite() || entry.scale <= 0.0 {
            return Err(format!(
                "row_scale_overrides entry for {} row {} must have a finite positive scale",
                entry.tensor_name, entry.row_index
            ));
        }
        if !seen.insert((entry.tensor_name.as_str(), entry.row_index)) {
            return Err(format!(
                "duplicate row_scale_overrides entry for {} row {}",
                entry.tensor_name, entry.row_index
            ));
        }
        if entry.tensor_name == tensor_name {
            matching.push(entry.clone());
        }
    }
    Ok(matching)
}

fn passthrough_payload_bundle_from_manifest(
    package_dir: &Path,
    tensor_index: usize,
    tensor: &PassthroughTensor,
) -> Result<PassthroughPayloadBundle, String> {
    let tensor_name = tensor
        .name
        .clone()
        .unwrap_or_else(|| format!("passthrough#{tensor_index}"));
    let payload_file = required_passthrough_file(
        package_dir,
        tensor_index,
        &tensor_name,
        tensor.payload_file.as_deref(),
    )?;
    Ok(PassthroughPayloadBundle {
        tensor_index,
        tensor_name,
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
        elements: tensor.elements,
        payload_bytes: tensor.payload_bytes,
        payload_encoding: tensor.payload_encoding.clone(),
        payload_sha256: tensor.payload_sha256.clone(),
        payload_file,
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

fn select_passthrough_index(
    manifest: &Manifest,
    selector: &TensorSelector,
) -> Result<usize, String> {
    match selector {
        TensorSelector::First => {
            if manifest.passthrough_tensors.is_empty() {
                Err("package contains no passthrough tensors".to_string())
            } else {
                Ok(0)
            }
        }
        TensorSelector::Index(index) => {
            if *index < manifest.passthrough_tensors.len() {
                Ok(*index)
            } else {
                Err(format!(
                    "passthrough tensor index {index} is out of range for {} passthrough tensors",
                    manifest.passthrough_tensors.len()
                ))
            }
        }
        TensorSelector::Name(name) => select_passthrough_index_by_name(manifest, name),
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
    if let Some(alias) = qwen3_tensor_name_alias(name) {
        if let Some((index, _)) = manifest
            .tensors
            .iter()
            .enumerate()
            .find(|(_, tensor)| tensor.name.as_deref() == Some(alias.as_str()))
        {
            return Ok(index);
        }
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

fn select_passthrough_index_by_name(manifest: &Manifest, name: &str) -> Result<usize, String> {
    if let Some((index, _)) = manifest
        .passthrough_tensors
        .iter()
        .enumerate()
        .find(|(_, tensor)| tensor.name.as_deref() == Some(name))
    {
        return Ok(index);
    }
    if let Some(alias) = qwen3_tensor_name_alias(name) {
        if let Some((index, _)) = manifest
            .passthrough_tensors
            .iter()
            .enumerate()
            .find(|(_, tensor)| tensor.name.as_deref() == Some(alias.as_str()))
        {
            return Ok(index);
        }
    }

    let matches: Vec<usize> = manifest
        .passthrough_tensors
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
        [] => Err(format!("no passthrough tensor matched selector \"{name}\"")),
        _ => Err(format!(
            "passthrough tensor selector \"{name}\" matched {} tensors; use an exact name or numeric index",
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

fn required_passthrough_file(
    package_dir: &Path,
    tensor_index: usize,
    tensor_name: &str,
    relative: Option<&str>,
) -> Result<ReferencedFile, String> {
    let relative = relative.ok_or_else(|| {
        format!("passthrough tensor {tensor_index} ({tensor_name}) does not declare payload")
    })?;
    referenced_file_candidate(
        package_dir,
        relative,
        ReferencedFileRole::Passthrough,
        Some(tensor_index),
        Some(tensor_name.to_string()),
    )
    .ok_or_else(|| {
        format!(
            "passthrough tensor {tensor_index} ({tensor_name}) does not reference an existing non-empty passthrough file"
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
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return None;
    }
    let package_root = fs::canonicalize(package_dir).ok()?;
    let absolute = fs::canonicalize(package_dir.join(relative_path)).ok()?;
    if !absolute.starts_with(&package_root) {
        return None;
    }
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
                "dtype": "BF16",
                "shape": [4],
                "elements": 4,
                "payload_bytes": 4,
                "payload_encoding": "raw_safetensors_payload",
                "payload_sha256": "unit-test-sha256",
                "payload_file": "tensors/b.raw"
              }],
              "codebooks": [{
                "family": "test-family",
                "candidate_id": "test-candidate",
                "file": "codebooks/c.f32"
              }],
              "row_scale_overrides": {
                "schema_version": "row-scale-overrides-v0.1",
                "entries": [{
                  "tensor_name": "tensor-a",
                  "row_index": 2,
                  "scale": 1.25,
                  "source": "unit-test"
                }]
              }
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

        let bundle = select_tensor_payload_bundle(&root, &TensorSelector::First).unwrap();
        assert_eq!(bundle.row_scale_overrides.len(), 1);
        assert_eq!(bundle.row_scale_overrides[0].tensor_name, "tensor-a");
        assert_eq!(bundle.row_scale_overrides[0].row_index, 2);
        assert_eq!(bundle.row_scale_overrides[0].scale, 1.25);
        assert_eq!(
            bundle.row_scale_overrides[0].source.as_deref(),
            Some("unit-test")
        );

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

        let passthrough_bundle =
            select_passthrough_payload_bundle(&root, &TensorSelector::First).unwrap();
        assert_eq!(passthrough_bundle.tensor_index, 0);
        assert_eq!(passthrough_bundle.tensor_name, "tensor-b");
        assert_eq!(passthrough_bundle.dtype.as_deref(), Some("BF16"));
        assert_eq!(passthrough_bundle.shape, vec![4]);
        assert_eq!(passthrough_bundle.elements, 4);
        assert_eq!(passthrough_bundle.payload_bytes, 4);
        assert_eq!(
            passthrough_bundle.payload_encoding.as_deref(),
            Some("raw_safetensors_payload")
        );
        assert_eq!(
            passthrough_bundle.payload_sha256.as_deref(),
            Some("unit-test-sha256")
        );
        assert_eq!(
            passthrough_bundle.payload_file.relative_path,
            "tensors/b.raw"
        );

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
    fn selects_qwen3_tensor_payload_bundle_across_namespaces() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-qwen3-tensor-namespace-selector-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        fs::write(root.join("tensors/q.idx4"), [0_u8; 4]).unwrap();
        fs::write(root.join("tensors/q.scale_u8"), [0_u8; 4]).unwrap();
        fs::write(root.join("codebooks/q.f32"), [0_u8; 64]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [{
                "name": "model.layers.0.self_attn.q_proj.weight",
                "shape": [1, 1],
                "elements": 1,
                "groups": 1,
                "index_file": "tensors/q.idx4",
                "scale_file": "tensors/q.scale_u8",
                "codebook_file": "codebooks/q.f32"
              }]
            }"#,
        )
        .unwrap();

        let bundle = select_tensor_payload_bundle(
            &root,
            &TensorSelector::Name(
                "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            ),
        )
        .unwrap();
        assert_eq!(bundle.tensor_index, 0);
        assert_eq!(bundle.tensor_name, "model.layers.0.self_attn.q_proj.weight");

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
    fn selects_passthrough_payload_bundle_by_index_and_name() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-passthrough-bundle-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        fs::write(root.join("passthrough/input.raw"), [1_u8, 2, 3, 4]).unwrap();
        fs::write(root.join("passthrough/post.raw"), [5_u8, 6, 7, 8]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "layer.0.input_layernorm.weight",
                "dtype": "BF16",
                "shape": [2],
                "elements": 2,
                "payload_bytes": 4,
                "payload_file": "passthrough/input.raw"
              }, {
                "name": "layer.0.post_attention_layernorm.weight",
                "dtype": "F32",
                "shape": [1],
                "elements": 1,
                "payload_bytes": 4,
                "payload_file": "passthrough/post.raw"
              }]
            }"#,
        )
        .unwrap();

        let first = select_passthrough_payload_bundle(&root, &TensorSelector::First).unwrap();
        assert_eq!(first.tensor_index, 0);
        assert_eq!(first.tensor_name, "layer.0.input_layernorm.weight");
        assert_eq!(first.dtype.as_deref(), Some("BF16"));
        assert_eq!(first.shape, vec![2]);
        assert_eq!(first.elements, 2);
        assert_eq!(first.payload_bytes, 4);
        assert_eq!(first.payload_file.relative_path, "passthrough/input.raw");
        assert_eq!(first.payload_file.role, ReferencedFileRole::Passthrough);
        assert_eq!(first.payload_file.owner_index, Some(0));
        assert_eq!(
            first.payload_file.owner_name.as_deref(),
            Some("layer.0.input_layernorm.weight")
        );

        let by_index = select_passthrough_payload_bundle(&root, &TensorSelector::Index(1)).unwrap();
        assert_eq!(by_index.tensor_index, 1);
        assert_eq!(by_index.dtype.as_deref(), Some("F32"));
        assert_eq!(by_index.payload_file.relative_path, "passthrough/post.raw");

        let by_exact = select_passthrough_payload_bundle(
            &root,
            &TensorSelector::Name("layer.0.post_attention_layernorm.weight".to_string()),
        )
        .unwrap();
        assert_eq!(by_exact.tensor_index, 1);

        let by_unique_substring = select_passthrough_payload_bundle(
            &root,
            &TensorSelector::Name("input_layernorm".to_string()),
        )
        .unwrap();
        assert_eq!(by_unique_substring.tensor_index, 0);

        let ambiguous =
            select_passthrough_payload_bundle(&root, &TensorSelector::Name("layer.0".to_string()))
                .unwrap_err();
        assert!(ambiguous.contains("matched 2 tensors"));

        let strict = select_exact_passthrough_payload_bundle(
            &root,
            "layer.0.post_attention_layernorm.weight",
        )
        .unwrap();
        assert_eq!(strict.tensor_index, 1);
        assert!(select_exact_passthrough_payload_bundle(&root, "input_layernorm").is_err());

        let bundles = list_passthrough_payload_bundles(&root).unwrap();
        assert_eq!(bundles.len(), 2);
        assert_eq!(bundles[0].tensor_name, "layer.0.input_layernorm.weight");
        assert_eq!(
            bundles[1].tensor_name,
            "layer.0.post_attention_layernorm.weight"
        );

        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "duplicate.weight",
                "dtype": "BF16",
                "shape": [2],
                "elements": 2,
                "payload_bytes": 4,
                "payload_file": "passthrough/input.raw"
              }, {
                "name": "duplicate.weight",
                "dtype": "BF16",
                "shape": [2],
                "elements": 2,
                "payload_bytes": 4,
                "payload_file": "passthrough/post.raw"
              }]
            }"#,
        )
        .unwrap();
        let duplicate =
            select_exact_passthrough_payload_bundle(&root, "duplicate.weight").unwrap_err();
        assert!(duplicate.contains("duplicated 2 times"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_passthrough_rejects_package_escape() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-passthrough-escape-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let outside = root.with_extension("outside.raw");
        fs::create_dir_all(&root).unwrap();
        fs::write(&outside, [0_u8, 1]).unwrap();
        let outside_name = outside.file_name().unwrap().to_str().unwrap();
        fs::write(
            root.join("manifest.json"),
            format!(
                r#"{{
                  "passthrough_tensors": [{{
                    "name": "lm_head.weight",
                    "dtype": "BF16",
                    "shape": [1, 1],
                    "elements": 1,
                    "payload_bytes": 2,
                    "payload_file": "../{outside_name}"
                  }}]
                }}"#
            ),
        )
        .unwrap();
        assert!(select_exact_passthrough_payload_bundle(&root, "lm_head.weight").is_err());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, root.join("escaped.raw")).unwrap();
            fs::write(
                root.join("manifest.json"),
                r#"{
                  "passthrough_tensors": [{
                    "name": "lm_head.weight",
                    "dtype": "BF16",
                    "shape": [1, 1],
                    "elements": 1,
                    "payload_bytes": 2,
                    "payload_file": "escaped.raw"
                  }]
                }"#,
            )
            .unwrap();
            assert!(select_exact_passthrough_payload_bundle(&root, "lm_head.weight").is_err());
        }

        fs::remove_dir_all(root).unwrap();
        fs::remove_file(outside).unwrap();
    }

    #[test]
    fn selects_qwen3_passthrough_payload_bundle_across_namespaces() {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-qwen3-passthrough-namespace-selector-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("passthrough")).unwrap();
        fs::write(root.join("passthrough/embed.raw"), [0_u8; 4]).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
              "passthrough_tensors": [{
                "name": "model.embed_tokens.weight",
                "dtype": "BF16",
                "shape": [1, 2],
                "elements": 2,
                "payload_bytes": 4,
                "payload_file": "passthrough/embed.raw"
              }]
            }"#,
        )
        .unwrap();

        let bundle = select_passthrough_payload_bundle(
            &root,
            &TensorSelector::Name("model.language_model.embed_tokens.weight".to_string()),
        )
        .unwrap();
        assert_eq!(bundle.tensor_index, 0);
        assert_eq!(bundle.tensor_name, "model.embed_tokens.weight");

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

#[cfg(test)]
mod verified_json_scan_tests {
    use super::*;
    use std::fmt::Write as _;

    #[test]
    fn scanner_bounds_depth_containers_strings_atoms_and_ignores_string_braces() {
        let mut depth_ok = "[".repeat(VERIFIED_JSON_MAX_DEPTH);
        depth_ok.push('0');
        depth_ok.push_str(&"]".repeat(VERIFIED_JSON_MAX_DEPTH));
        assert_eq!(scan_verified_manifest_json(depth_ok.as_bytes()), Ok(()));

        let mut depth_bad = "[".repeat(VERIFIED_JSON_MAX_DEPTH + 1);
        depth_bad.push('0');
        depth_bad.push_str(&"]".repeat(VERIFIED_JSON_MAX_DEPTH + 1));
        assert_eq!(
            scan_verified_manifest_json(depth_bad.as_bytes()),
            Err(VerifiedJsonScanFailure::Resource)
        );

        let string_ok = format!("\"{}\"", "a".repeat(VERIFIED_JSON_MAX_STRING_BYTES));
        assert_eq!(scan_verified_manifest_json(string_ok.as_bytes()), Ok(()));
        let string_bad = format!("\"{}\"", "a".repeat(VERIFIED_JSON_MAX_STRING_BYTES + 1));
        assert_eq!(
            scan_verified_manifest_json(string_bad.as_bytes()),
            Err(VerifiedJsonScanFailure::Resource)
        );
        let escaped_bad = format!("\"{}\"", "\\n".repeat(VERIFIED_JSON_MAX_STRING_BYTES + 1));
        assert_eq!(
            scan_verified_manifest_json(escaped_bad.as_bytes()),
            Err(VerifiedJsonScanFailure::Resource)
        );
        let escaped_ascii = format!("\"{}\"", "\\u0061".repeat(1_025));
        assert_eq!(
            scan_verified_manifest_json(escaped_ascii.as_bytes()),
            Ok(())
        );

        let exact_decoded = format!(
            "\"{}\\u00e9\\u20ac\\ud83d\\ude00\"",
            "\\u0061".repeat(4_087)
        );
        assert_eq!(
            scan_verified_manifest_json(exact_decoded.as_bytes()),
            Ok(())
        );
        let over_decoded = format!(
            "\"{}\\u00e9\\u20ac\\ud83d\\ude00\"",
            "\\u0061".repeat(4_088)
        );
        assert_eq!(
            scan_verified_manifest_json(over_decoded.as_bytes()),
            Err(VerifiedJsonScanFailure::Resource)
        );
        for invalid_surrogate in [
            br#""\ud800""#.as_slice(),
            br#""\udc00""#.as_slice(),
            br#""\udc00\ud800""#.as_slice(),
            br#""\ud800\ud800""#.as_slice(),
        ] {
            assert_eq!(
                scan_verified_manifest_json(invalid_surrogate),
                Err(VerifiedJsonScanFailure::Invalid)
            );
        }
        assert_eq!(scan_verified_manifest_json(br#"{"x":"[{}]"}"#), Ok(()));

        let atom_bad = "1".repeat(VERIFIED_JSON_MAX_ATOM_BYTES + 1);
        assert_eq!(
            scan_verified_manifest_json(atom_bad.as_bytes()),
            Err(VerifiedJsonScanFailure::Resource)
        );

        let mut containers = String::from("[");
        for index in 0..VERIFIED_JSON_MAX_CONTAINERS - 1 {
            if index != 0 {
                containers.push(',');
            }
            containers.push_str("{}");
        }
        containers.push(']');
        assert_eq!(scan_verified_manifest_json(containers.as_bytes()), Ok(()));
        containers.insert_str(containers.len() - 1, ",{}");
        assert_eq!(
            scan_verified_manifest_json(containers.as_bytes()),
            Err(VerifiedJsonScanFailure::Resource)
        );

        let mut token_heavy = String::from("[");
        for index in 0..VERIFIED_JSON_MAX_TOKENS {
            if index != 0 {
                token_heavy.push(',');
            }
            token_heavy.push('0');
        }
        token_heavy.push(']');
        assert_eq!(
            scan_verified_manifest_json(token_heavy.as_bytes()),
            Err(VerifiedJsonScanFailure::Resource)
        );
    }

    #[test]
    fn scanner_allows_4096_descriptor_scale_and_bridge_rejects_4097_as_resource() {
        let digest = "0".repeat(64);
        let mut manifest = String::from("{\"passthrough_tensors\":[");
        for index in 0..4_097 {
            if index != 0 {
                manifest.push(',');
            }
            write!(
                manifest,
                "{{\"name\":\"w{index}\",\"dtype\":\"F32\",\"shape\":[1],\"elements\":1,\"payload_bytes\":4,\"payload_encoding\":\"raw_safetensors_payload\",\"payload_sha256\":\"{digest}\",\"payload_file\":\"w/{index}.raw\"}}"
            )
            .unwrap();
        }
        manifest.push_str("]}");
        let last = manifest.rfind(",{\"name\":\"w4096\"").unwrap();
        let mut boundary = manifest[..last].to_string();
        boundary.push_str("]}");
        // The lexical scanner comfortably admits descriptor-scale JSON; exact
        // descriptor cardinality remains a post-parse resource limit.
        assert_eq!(scan_verified_manifest_json(boundary.as_bytes()), Ok(()));
        assert_eq!(scan_verified_manifest_json(manifest.as_bytes()), Ok(()));
        assert!(boundary.len() <= 16 * 1024 * 1024);
        let error = parse_verified_passthrough_descriptors(manifest.as_bytes(), 4_096).unwrap_err();
        assert!(error.starts_with("resource:"));
    }
}
