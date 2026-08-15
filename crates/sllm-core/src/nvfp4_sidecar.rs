//! Fail-closed validation for weight-only NVFP4 derived sidecars.

use crate::{Gemma4ModelLock, ModelLock, NVFP4_BLOCK_SIZE};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "sllm-nvfp4-sidecar-v1";
const VALUE_SUFFIX: &str = ".sllm_nvfp4_value";
const BLOCK_SCALE_SUFFIX: &str = ".sllm_nvfp4_block_scale";
const TENSOR_SCALE_SUFFIX: &str = ".sllm_nvfp4_tensor_scale";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Nvfp4SidecarError(String);

impl Nvfp4SidecarError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Nvfp4SidecarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid NVFP4 sidecar: {}", self.0)
    }
}

impl std::error::Error for Nvfp4SidecarError {}

/// Packed value bytes, block-scale bytes, and the mandatory FP32 tensor scale.
pub type Nvfp4TensorBytes = (Vec<u8>, Vec<u8>, [u8; 4]);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Nvfp4SidecarTensor {
    pub name: String,
    pub shape: [u64; 2],
    pub value_range: [u64; 2],
    pub block_scale_range: [u64; 2],
    pub tensor_scale_range: [u64; 2],
    pub source_sha256: String,
    pub value_sha256: String,
    pub block_scale_sha256: String,
    pub tensor_scale_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedNvfp4Sidecar {
    artifact_path: PathBuf,
    source_lock_fingerprint: String,
    manifest_fingerprint: String,
    artifact_sha256: String,
    data_start: u64,
    tensors: BTreeMap<String, Nvfp4SidecarTensor>,
}

impl VerifiedNvfp4Sidecar {
    pub fn source_lock_fingerprint(&self) -> &str {
        &self.source_lock_fingerprint
    }
    pub fn manifest_fingerprint(&self) -> &str {
        &self.manifest_fingerprint
    }
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
    pub fn tensor(&self, name: &str) -> Option<&Nvfp4SidecarTensor> {
        self.tensors.get(name)
    }
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &Nvfp4SidecarTensor> {
        self.tensors.values()
    }

    pub fn read_tensor_bytes(&self, name: &str) -> Result<Nvfp4TensorBytes, Nvfp4SidecarError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| Nvfp4SidecarError::invalid("tensor is absent"))?;
        let mut artifact = File::open(&self.artifact_path)
            .map_err(|error| Nvfp4SidecarError::invalid(format!("open artifact: {error}")))?;
        let values = read_range(&mut artifact, self.data_start, tensor.value_range)?;
        let block_scales = read_range(&mut artifact, self.data_start, tensor.block_scale_range)?;
        let tensor_scale: [u8; 4] =
            read_range(&mut artifact, self.data_start, tensor.tensor_scale_range)?
                .try_into()
                .map_err(|_| Nvfp4SidecarError::invalid("tensor scale is not four bytes"))?;
        if sha256_bytes(&values) != tensor.value_sha256
            || sha256_bytes(&block_scales) != tensor.block_scale_sha256
            || sha256_bytes(&tensor_scale) != tensor.tensor_scale_sha256
        {
            return Err(Nvfp4SidecarError::invalid(
                "tensor changed after verification",
            ));
        }
        let scale = f32::from_le_bytes(tensor_scale);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Nvfp4SidecarError::invalid(
                "tensor scale is non-positive or non-finite",
            ));
        }
        Ok((values, block_scales, tensor_scale))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    source: Source,
    format_source: FormatSource,
    tool: Tool,
    artifact: Artifact,
    tensors: Vec<TensorRecord>,
    fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    repo_id: String,
    resolved_revision: String,
    lock_fingerprint: String,
    lock_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FormatSource {
    repository: String,
    tag: String,
    commit: String,
    license: String,
    contract: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tool {
    repository: String,
    commit: String,
    path: String,
    sha256: String,
    numpy: String,
    arguments: ToolArguments,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArguments {
    tensor: Vec<String>,
    #[serde(default)]
    selection: Option<String>,
    #[serde(default)]
    tensor_scale_multipliers_sha256: Option<String>,
    #[serde(default)]
    scale_objective: Option<String>,
    #[serde(default)]
    source_repository: Option<String>,
    #[serde(default)]
    source_revision: Option<String>,
    #[serde(default)]
    source_artifact_sha256: Option<String>,
    #[serde(default)]
    global_scale_convention: Option<String>,
    #[serde(default)]
    input_global_scales_applied: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    size_bytes: u64,
    sha256: String,
    tensor_count: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRecord {
    name: String,
    logical_shape: Vec<u64>,
    source_dtype: String,
    source_sha256: String,
    value_name: String,
    value_sha256: String,
    packing: String,
    block_scale_name: String,
    block_scale_sha256: String,
    block_size: u64,
    block_axis: u64,
    block_scale_dtype: String,
    tensor_scale_name: String,
    tensor_scale_sha256: String,
    tensor_scale_dtype: String,
    rounding: String,
    saturation: String,
    zero_point: bool,
}

#[derive(Deserialize)]
struct SafeTensorMetadata {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

pub fn verify_nvfp4_sidecar(
    manifest_path: &Path,
    artifact_path: &Path,
    source_lock_path: &Path,
    source_lock: &ModelLock,
) -> Result<VerifiedNvfp4Sidecar, Nvfp4SidecarError> {
    verify_nvfp4_sidecar_identity(
        manifest_path,
        artifact_path,
        source_lock_path,
        &source_lock.model.repo_id,
        &source_lock.model.resolved_revision,
        &source_lock.fingerprint,
    )
}

pub fn verify_gemma4_nvfp4_sidecar(
    manifest_path: &Path,
    artifact_path: &Path,
    source_lock_path: &Path,
    source_lock: &Gemma4ModelLock,
) -> Result<VerifiedNvfp4Sidecar, Nvfp4SidecarError> {
    verify_nvfp4_sidecar_identity(
        manifest_path,
        artifact_path,
        source_lock_path,
        &source_lock.model.repo_id,
        &source_lock.model.resolved_revision,
        &source_lock.fingerprint,
    )
}

fn verify_nvfp4_sidecar_identity(
    manifest_path: &Path,
    artifact_path: &Path,
    source_lock_path: &Path,
    source_repo_id: &str,
    source_revision: &str,
    source_fingerprint: &str,
) -> Result<VerifiedNvfp4Sidecar, Nvfp4SidecarError> {
    let manifest_bytes = bounded_read(manifest_path, MAX_MANIFEST_BYTES, "manifest")?;
    let mut value: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("manifest JSON: {error}")))?;
    let claimed = value
        .as_object_mut()
        .and_then(|object| object.remove("fingerprint"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| Nvfp4SidecarError::invalid("manifest fingerprint is absent"))?;
    let canonical = serde_json::to_vec(&value).map_err(|error| {
        Nvfp4SidecarError::invalid(format!("manifest canonicalization: {error}"))
    })?;
    if claimed != sha256_bytes(&canonical) {
        return Err(Nvfp4SidecarError::invalid("manifest fingerprint differs"));
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("manifest schema: {error}")))?;
    if manifest.schema_version != SCHEMA
        || manifest.source.repo_id != source_repo_id
        || manifest.source.resolved_revision != source_revision
        || manifest.source.lock_fingerprint != source_fingerprint
        || manifest.source.lock_sha256 != sha256_file(source_lock_path)?
    {
        return Err(Nvfp4SidecarError::invalid("source identity differs"));
    }
    validate_format_source(&manifest.format_source)?;
    validate_tool(&manifest.tool)?;
    let artifact_name = artifact_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Nvfp4SidecarError::invalid("artifact has no UTF-8 basename"))?;
    let metadata = artifact_path
        .metadata()
        .map_err(|error| Nvfp4SidecarError::invalid(format!("artifact metadata: {error}")))?;
    if !metadata.is_file()
        || artifact_name != manifest.artifact.path
        || metadata.len() != manifest.artifact.size_bytes
        || manifest.artifact.sha256 != sha256_file(artifact_path)?
        || manifest.artifact.tensor_count == 0
        || usize::try_from(manifest.artifact.tensor_count).ok() != Some(manifest.tensors.len())
    {
        return Err(Nvfp4SidecarError::invalid(
            "artifact identity or tensor count differs",
        ));
    }
    let (data_start, header) = read_safetensors_header(artifact_path)?;
    let mut artifact = File::open(artifact_path)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("open artifact: {error}")))?;
    let mut expected_names = BTreeSet::new();
    let mut tensors = BTreeMap::new();
    for record in manifest.tensors {
        validate_record(&record)?;
        let shape: [u64; 2] = record
            .logical_shape
            .as_slice()
            .try_into()
            .map_err(|_| Nvfp4SidecarError::invalid("logical tensor must be rank two"))?;
        if tensors.contains_key(&record.name) {
            return Err(Nvfp4SidecarError::invalid("duplicate tensor record"));
        }
        let elements = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| Nvfp4SidecarError::invalid("tensor shape overflow"))?;
        let blocks = shape[0]
            .checked_mul(shape[1].div_ceil(NVFP4_BLOCK_SIZE as u64))
            .ok_or_else(|| Nvfp4SidecarError::invalid("block scale shape overflow"))?;
        let value_metadata = require_header(&header, &record.value_name, "value")?;
        let block_metadata = require_header(&header, &record.block_scale_name, "block scale")?;
        let tensor_metadata = require_header(&header, &record.tensor_scale_name, "tensor scale")?;
        if value_metadata.dtype != "U8"
            || value_metadata.shape != [elements.div_ceil(2)]
            || range_len(value_metadata.data_offsets)? != elements.div_ceil(2)
            || block_metadata.dtype != "U8"
            || block_metadata.shape != [shape[0], shape[1].div_ceil(16)]
            || range_len(block_metadata.data_offsets)? != blocks
            || tensor_metadata.dtype != "F32"
            || tensor_metadata.shape != [1]
            || range_len(tensor_metadata.data_offsets)? != 4
        {
            return Err(Nvfp4SidecarError::invalid(format!(
                "artifact metadata differs: {}",
                record.name
            )));
        }
        for name in [
            &record.value_name,
            &record.block_scale_name,
            &record.tensor_scale_name,
        ] {
            if !expected_names.insert(name.clone()) {
                return Err(Nvfp4SidecarError::invalid("duplicate artifact tensor name"));
            }
        }
        if hash_range(&mut artifact, data_start, value_metadata.data_offsets)?
            != record.value_sha256
            || hash_range(&mut artifact, data_start, block_metadata.data_offsets)?
                != record.block_scale_sha256
            || hash_range(&mut artifact, data_start, tensor_metadata.data_offsets)?
                != record.tensor_scale_sha256
        {
            return Err(Nvfp4SidecarError::invalid(format!(
                "payload hash differs: {}",
                record.name
            )));
        }
        if elements & 1 != 0 {
            let tail = read_range(
                &mut artifact,
                data_start,
                [
                    value_metadata.data_offsets[1] - 1,
                    value_metadata.data_offsets[1],
                ],
            )?;
            if tail[0] & 0xf0 != 0 {
                return Err(Nvfp4SidecarError::invalid("noncanonical packed tail"));
            }
        }
        let tensor = Nvfp4SidecarTensor {
            name: record.name.clone(),
            shape,
            value_range: value_metadata.data_offsets,
            block_scale_range: block_metadata.data_offsets,
            tensor_scale_range: tensor_metadata.data_offsets,
            source_sha256: record.source_sha256,
            value_sha256: record.value_sha256,
            block_scale_sha256: record.block_scale_sha256,
            tensor_scale_sha256: record.tensor_scale_sha256,
        };
        tensors.insert(record.name, tensor);
    }
    let actual_names: BTreeSet<_> = header
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect();
    if actual_names != expected_names {
        return Err(Nvfp4SidecarError::invalid("artifact tensor set differs"));
    }
    let verified = VerifiedNvfp4Sidecar {
        artifact_path: artifact_path.to_path_buf(),
        source_lock_fingerprint: manifest.source.lock_fingerprint,
        manifest_fingerprint: manifest.fingerprint,
        artifact_sha256: manifest.artifact.sha256,
        data_start,
        tensors,
    };
    for tensor in verified.tensors.values() {
        verified.read_tensor_bytes(&tensor.name)?;
    }
    Ok(verified)
}

fn validate_format_source(source: &FormatSource) -> Result<(), Nvfp4SidecarError> {
    if source.repository != "https://github.com/NVIDIA/TransformerEngine"
        || source.tag != "v2.18"
        || source.commit != "27486e03cfc1fa41f6932dcecdc47c71c47eac3e"
        || source.license != "BSD-3-Clause"
        || source.contract != "sllm-weight-nvfp4-v1"
    {
        return Err(Nvfp4SidecarError::invalid("format source lock differs"));
    }
    Ok(())
}

fn validate_tool(tool: &Tool) -> Result<(), Nvfp4SidecarError> {
    let optimized_scale = match (
        tool.arguments.tensor_scale_multipliers_sha256.as_deref(),
        tool.arguments.scale_objective.as_deref(),
    ) {
        (None, None) => true,
        (Some(hash), Some("sampled-weight-mse-independent-evaluation-set")) => valid_sha256(hash),
        _ => false,
    };
    let converter = tool.path == "ci/tools/convert_nvfp4_sidecar.py"
        && matches!(
            tool.arguments.selection.as_deref(),
            None | Some("gemma-mlp-144") | Some("gemma-mlp-subset")
        )
        && optimized_scale
        && tool.arguments.source_repository.is_none()
        && tool.arguments.source_revision.is_none()
        && tool.arguments.source_artifact_sha256.is_none()
        && tool.arguments.global_scale_convention.is_none()
        && tool.arguments.input_global_scales_applied.is_none();
    let importer = tool.path == "ci/tools/import_unsloth_nvfp4_sidecar.py"
        && tool.numpy == "not-used"
        && matches!(
            (
                tool.arguments.tensor.is_empty(),
                tool.arguments.selection.as_deref()
            ),
            (true, None) | (false, Some("gemma-mlp-subset"))
        )
        && tool.arguments.tensor_scale_multipliers_sha256.is_none()
        && tool.arguments.scale_objective.is_none()
        && tool.arguments.source_repository.as_deref() == Some("unsloth/gemma-4-12b-it-NVFP4")
        && tool.arguments.source_revision.as_deref()
            == Some("b1f649734b34aa5575b03d186abd1b9be3d0d5c4")
        && tool.arguments.source_artifact_sha256.as_deref()
            == Some("sha256:7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b")
        && tool.arguments.global_scale_convention.as_deref()
            == Some("multiplicative-reciprocal-of-compressed-tensors-weight-global-scale")
        && tool.arguments.input_global_scales_applied == Some(false);
    if tool.repository.is_empty()
        || tool.commit.is_empty()
        || (!converter && !importer)
        || !valid_sha256(&tool.sha256)
        || tool.numpy.is_empty()
        || !tool
            .arguments
            .tensor
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    {
        return Err(Nvfp4SidecarError::invalid("converter provenance differs"));
    }
    Ok(())
}

fn validate_record(record: &TensorRecord) -> Result<(), Nvfp4SidecarError> {
    if !record.name.starts_with("model.language_model.layers.")
        || !record.name.ends_with(".weight")
        || record.logical_shape.len() != 2
        || record.logical_shape.contains(&0)
        || record.source_dtype != "BF16"
        || record.value_name != format!("{}{}", record.name, VALUE_SUFFIX)
        || record.block_scale_name != format!("{}{}", record.name, BLOCK_SCALE_SUFFIX)
        || record.tensor_scale_name != format!("{}{}", record.name, TENSOR_SCALE_SUFFIX)
        || record.packing != "low-nibble-first-row-major"
        || record.block_size != 16
        || record.block_axis != 1
        || record.block_scale_dtype != "F8_E4M3FN"
        || record.tensor_scale_dtype != "F32"
        || record.rounding != "nearest-even"
        || record.saturation != "finite"
        || record.zero_point
        || [
            &record.source_sha256,
            &record.value_sha256,
            &record.block_scale_sha256,
            &record.tensor_scale_sha256,
        ]
        .into_iter()
        .any(|hash| !valid_sha256(hash))
    {
        return Err(Nvfp4SidecarError::invalid(
            "tensor quantization contract differs",
        ));
    }
    Ok(())
}

fn require_header<'a>(
    header: &'a BTreeMap<String, SafeTensorMetadata>,
    name: &str,
    label: &str,
) -> Result<&'a SafeTensorMetadata, Nvfp4SidecarError> {
    header
        .get(name)
        .ok_or_else(|| Nvfp4SidecarError::invalid(format!("{label} tensor is absent")))
}

fn range_len(range: [u64; 2]) -> Result<u64, Nvfp4SidecarError> {
    range[1]
        .checked_sub(range[0])
        .ok_or_else(|| Nvfp4SidecarError::invalid("range is reversed"))
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, SafeTensorMetadata>), Nvfp4SidecarError> {
    let mut file = File::open(path)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("open artifact: {error}")))?;
    let mut raw = [0_u8; 8];
    file.read_exact(&mut raw)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("read header: {error}")))?;
    let length = u64::from_le_bytes(raw);
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(Nvfp4SidecarError::invalid("header length is invalid"));
    }
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| Nvfp4SidecarError::invalid(
            "header length does not fit usize"
        ))?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("read header: {error}")))?;
    let mut value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("header JSON: {error}")))?;
    value
        .as_object_mut()
        .and_then(|object| object.remove("__metadata__"));
    let header = serde_json::from_value(value)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("header schema: {error}")))?;
    Ok((8 + length, header))
}

fn hash_range(
    file: &mut File,
    data_start: u64,
    range: [u64; 2],
) -> Result<String, Nvfp4SidecarError> {
    let bytes = read_range(file, data_start, range)?;
    Ok(sha256_bytes(&bytes))
}

fn read_range(
    file: &mut File,
    data_start: u64,
    range: [u64; 2],
) -> Result<Vec<u8>, Nvfp4SidecarError> {
    let length = range_len(range)?;
    file.seek(SeekFrom::Start(
        data_start
            .checked_add(range[0])
            .ok_or_else(|| Nvfp4SidecarError::invalid("range overflow"))?,
    ))
    .map_err(|error| Nvfp4SidecarError::invalid(format!("seek tensor: {error}")))?;
    let mut bytes = vec![
        0_u8;
        usize::try_from(length)
            .map_err(|_| Nvfp4SidecarError::invalid("range does not fit usize"))?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("read tensor: {error}")))?;
    Ok(bytes)
}

fn bounded_read(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>, Nvfp4SidecarError> {
    let metadata = path
        .metadata()
        .map_err(|error| Nvfp4SidecarError::invalid(format!("{label} metadata: {error}")))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum {
        return Err(Nvfp4SidecarError::invalid(format!(
            "{label} size is invalid"
        )));
    }
    std::fs::read(path)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("read {label}: {error}")))
}

fn sha256_file(path: &Path) -> Result<String, Nvfp4SidecarError> {
    let mut file = File::open(path)
        .map_err(|error| Nvfp4SidecarError::invalid(format!("open hash source: {error}")))?;
    let mut digest = Sha256::new();
    std::io::copy(&mut file, &mut DigestWriter(&mut digest))
        .map_err(|error| Nvfp4SidecarError::invalid(format!("hash file: {error}")))?;
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}
fn valid_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
struct DigestWriter<'a>(&'a mut Sha256);
impl std::io::Write for DigestWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
