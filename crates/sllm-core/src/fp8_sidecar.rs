//! Fail-closed validation for Phase 10 Qwen3.5 FP8 sidecars.

use crate::ModelLock;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::fs::FileExt;

const SCHEMA: &str = "sllm-qwen35-fp8-sidecar-v1";
const SCALE_SUFFIX: &str = ".sllm_fp8_scale";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8SidecarError(String);

impl Fp8SidecarError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Fp8SidecarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid FP8 sidecar: {}", self.0)
    }
}

impl std::error::Error for Fp8SidecarError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8SidecarTensor {
    pub name: String,
    pub shape: [u64; 2],
    pub value_range: [u64; 2],
    pub scale_range: [u64; 2],
    pub source_sha256: String,
    pub value_sha256: String,
    pub scale_sha256: String,
}

#[derive(Clone, Debug)]
pub struct VerifiedFp8Sidecar {
    artifact_path: PathBuf,
    artifact: Arc<File>,
    source_lock_fingerprint: String,
    manifest_fingerprint: String,
    artifact_sha256: String,
    data_start: u64,
    tensors: BTreeMap<String, Fp8SidecarTensor>,
}

impl VerifiedFp8Sidecar {
    pub fn artifact_path(&self) -> &Path {
        &self.artifact_path
    }

    pub fn source_lock_fingerprint(&self) -> &str {
        &self.source_lock_fingerprint
    }

    pub fn manifest_fingerprint(&self) -> &str {
        &self.manifest_fingerprint
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub const fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn tensor(&self, name: &str) -> Option<&Fp8SidecarTensor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &Fp8SidecarTensor> {
        self.tensors.values()
    }

    /// Read one verified value/scale pair. Verification hashes the complete
    /// artifact before this owner is created; the range hashes are checked
    /// again here so a changed sidecar cannot be uploaded after validation.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<(Vec<u8>, Vec<u8>), Fp8SidecarError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| Fp8SidecarError::invalid("FP8 tensor is absent"))?;
        let values = self.read_tensor_range(name, false, 0, range_len(tensor.value_range)?)?;
        let scales = self.read_tensor_range(name, true, 0, range_len(tensor.scale_range)?)?;
        if sha256_bytes(&values) != tensor.value_sha256
            || sha256_bytes(&scales) != tensor.scale_sha256
        {
            return Err(Fp8SidecarError::invalid(
                "FP8 tensor changed after sidecar verification",
            ));
        }
        Ok((values, scales))
    }

    pub fn read_tensor_range(
        &self,
        name: &str,
        scale: bool,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, Fp8SidecarError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| Fp8SidecarError::invalid("FP8 tensor is absent"))?;
        let range = if scale {
            tensor.scale_range
        } else {
            tensor.value_range
        };
        let plane_length = range_len(range)?;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > plane_length)
        {
            return Err(Fp8SidecarError::invalid(
                "FP8 tensor subrange exceeds its plane",
            ));
        }
        let absolute = self
            .data_start
            .checked_add(range[0])
            .and_then(|start| start.checked_add(offset))
            .ok_or_else(|| Fp8SidecarError::invalid("FP8 tensor absolute range overflows"))?;
        let mut bytes = vec![
            0_u8;
            usize::try_from(length).map_err(|_| {
                Fp8SidecarError::invalid("FP8 tensor subrange exceeds address space")
            })?
        ];
        read_exact_at(&self.artifact, absolute, &mut bytes)?;
        Ok(bytes)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    source: Source,
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
struct Tool {
    repository: String,
    commit: String,
    path: String,
    sha256: String,
    numpy: String,
    arguments: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    size_bytes: u64,
    sha256: String,
    tensor_count: u64,
    scale_tensor_count: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRecord {
    name: String,
    shape: Vec<u64>,
    source_dtype: String,
    source_sha256: String,
    value_dtype: String,
    value_sha256: String,
    scale_dtype: String,
    scale_granularity: String,
    scale_axis: u64,
    scale_sha256: String,
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

pub fn verify_fp8_sidecar(
    manifest_path: &Path,
    artifact_path: &Path,
    source_lock_path: &Path,
    source_lock: &ModelLock,
) -> Result<VerifiedFp8Sidecar, Fp8SidecarError> {
    let manifest_bytes = bounded_read(manifest_path, MAX_MANIFEST_BYTES, "manifest")?;
    let mut manifest_value: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| Fp8SidecarError::invalid(format!("manifest JSON: {error}")))?;
    let claimed_fingerprint = manifest_value
        .as_object_mut()
        .and_then(|object| object.remove("fingerprint"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| Fp8SidecarError::invalid("manifest fingerprint is absent"))?;
    let canonical = serde_json::to_vec(&manifest_value)
        .map_err(|error| Fp8SidecarError::invalid(format!("manifest canonicalization: {error}")))?;
    if claimed_fingerprint != sha256_bytes(&canonical) {
        return Err(Fp8SidecarError::invalid("manifest fingerprint differs"));
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| Fp8SidecarError::invalid(format!("manifest schema: {error}")))?;
    if manifest.schema_version != SCHEMA {
        return Err(Fp8SidecarError::invalid("manifest schema is unsupported"));
    }
    if manifest.source.repo_id != source_lock.model.repo_id
        || manifest.source.resolved_revision != source_lock.model.resolved_revision
        || manifest.source.lock_fingerprint != source_lock.fingerprint
        || manifest.source.lock_sha256 != sha256_file(source_lock_path)?
    {
        return Err(Fp8SidecarError::invalid(
            "source model lock identity differs",
        ));
    }
    validate_tool(&manifest.tool)?;
    let artifact_name = artifact_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Fp8SidecarError::invalid("artifact has no UTF-8 basename"))?;
    let artifact_metadata = artifact_path
        .metadata()
        .map_err(|error| Fp8SidecarError::invalid(format!("artifact metadata: {error}")))?;
    if !artifact_metadata.is_file()
        || artifact_name != manifest.artifact.path
        || artifact_metadata.len() != manifest.artifact.size_bytes
        || manifest.artifact.sha256 != sha256_file(artifact_path)?
    {
        return Err(Fp8SidecarError::invalid("artifact identity differs"));
    }
    if manifest.artifact.tensor_count == 0
        || manifest.artifact.tensor_count != manifest.artifact.scale_tensor_count
        || usize::try_from(manifest.artifact.tensor_count).ok() != Some(manifest.tensors.len())
    {
        return Err(Fp8SidecarError::invalid("artifact tensor counts differ"));
    }

    let (data_start, header) = read_safetensors_header(artifact_path)?;
    let mut expected_names = BTreeSet::new();
    let mut tensors = BTreeMap::new();
    let mut artifact = File::open(artifact_path)
        .map_err(|error| Fp8SidecarError::invalid(format!("open artifact: {error}")))?;
    for record in manifest.tensors {
        validate_record(&record)?;
        if !expected_names.insert(record.name.clone()) {
            return Err(Fp8SidecarError::invalid("duplicate tensor record"));
        }
        let shape: [u64; 2] = record
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Fp8SidecarError::invalid("FP8 tensor must have rank two"))?;
        let value = header
            .get(&record.name)
            .ok_or_else(|| Fp8SidecarError::invalid("FP8 value tensor is absent"))?;
        let scale_name = format!("{}{}", record.name, SCALE_SUFFIX);
        let scale = header
            .get(&scale_name)
            .ok_or_else(|| Fp8SidecarError::invalid("FP8 scale tensor is absent"))?;
        let elements = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| Fp8SidecarError::invalid("FP8 tensor shape overflow"))?;
        if value.dtype != "F8_E4M3"
            || value.shape != record.shape
            || value.data_offsets[1].checked_sub(value.data_offsets[0]) != Some(elements)
            || scale.dtype != "F32"
            || scale.shape != [shape[0]]
            || scale.data_offsets[1].checked_sub(scale.data_offsets[0]) != Some(shape[0] * 4)
        {
            return Err(Fp8SidecarError::invalid(format!(
                "FP8 value/scale metadata differs: {}",
                record.name
            )));
        }
        if hash_range(&mut artifact, data_start, value.data_offsets)? != record.value_sha256
            || hash_range(&mut artifact, data_start, scale.data_offsets)? != record.scale_sha256
        {
            return Err(Fp8SidecarError::invalid(format!(
                "FP8 value/scale payload hash differs: {}",
                record.name
            )));
        }
        tensors.insert(
            record.name.clone(),
            Fp8SidecarTensor {
                name: record.name,
                shape,
                value_range: value.data_offsets,
                scale_range: scale.data_offsets,
                source_sha256: record.source_sha256,
                value_sha256: record.value_sha256,
                scale_sha256: record.scale_sha256,
            },
        );
    }
    let header_names: BTreeSet<_> = header
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect();
    let expected_header_names: BTreeSet<_> = expected_names
        .iter()
        .flat_map(|name| [name.clone(), format!("{name}{SCALE_SUFFIX}")])
        .collect();
    if header_names != expected_header_names {
        return Err(Fp8SidecarError::invalid(
            "artifact header tensor set differs",
        ));
    }

    Ok(VerifiedFp8Sidecar {
        artifact_path: artifact_path.to_path_buf(),
        artifact: Arc::new(artifact),
        source_lock_fingerprint: manifest.source.lock_fingerprint,
        manifest_fingerprint: manifest.fingerprint,
        artifact_sha256: manifest.artifact.sha256,
        data_start,
        tensors,
    })
}

fn range_len(range: [u64; 2]) -> Result<u64, Fp8SidecarError> {
    range[1]
        .checked_sub(range[0])
        .ok_or_else(|| Fp8SidecarError::invalid("FP8 tensor range is reversed"))
}

#[cfg(unix)]
fn read_exact_at(
    file: &File,
    mut offset: u64,
    mut output: &mut [u8],
) -> Result<(), Fp8SidecarError> {
    while !output.is_empty() {
        let read = file
            .read_at(output, offset)
            .map_err(|error| Fp8SidecarError::invalid(format!("read artifact: {error}")))?;
        if read == 0 {
            return Err(Fp8SidecarError::invalid("FP8 tensor subrange is truncated"));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| Fp8SidecarError::invalid("FP8 read offset overflows"))?;
        output = &mut output[read..];
    }
    Ok(())
}

fn validate_tool(tool: &Tool) -> Result<(), Fp8SidecarError> {
    if tool.repository.is_empty()
        || tool.commit.len() < 7
        || tool.path != "ci/tools/convert_qwen35_fp8_sidecar.py"
        || !valid_sha256(&tool.sha256)
        || tool.numpy.is_empty()
        || tool.arguments.get("encoding").map(String::as_str) != Some("OCP-E4M3FN")
        || tool.arguments.get("scale").map(String::as_str) != Some("outer-dimension-f32")
        || tool.arguments.len() != 2
    {
        return Err(Fp8SidecarError::invalid("converter provenance differs"));
    }
    Ok(())
}

fn validate_record(record: &TensorRecord) -> Result<(), Fp8SidecarError> {
    if !record.name.starts_with("model.language_model.layers.")
        || !record.name.ends_with(".weight")
        || record.shape.len() != 2
        || record.shape.contains(&0)
        || record.source_dtype != "BF16"
        || record.value_dtype != "F8_E4M3FN"
        || record.scale_dtype != "F32"
        || record.scale_granularity != "outer-dimension"
        || record.scale_axis != 0
        || record.rounding != "nearest-even"
        || record.saturation != "finite-448"
        || record.zero_point
        || !valid_sha256(&record.source_sha256)
        || !valid_sha256(&record.value_sha256)
        || !valid_sha256(&record.scale_sha256)
    {
        return Err(Fp8SidecarError::invalid(
            "tensor quantization contract differs",
        ));
    }
    Ok(())
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, SafeTensorMetadata>), Fp8SidecarError> {
    let mut file = File::open(path)
        .map_err(|error| Fp8SidecarError::invalid(format!("open artifact: {error}")))?;
    let mut raw_length = [0_u8; 8];
    file.read_exact(&mut raw_length)
        .map_err(|error| Fp8SidecarError::invalid(format!("read header length: {error}")))?;
    let length = u64::from_le_bytes(raw_length);
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(Fp8SidecarError::invalid(
            "safetensors header length is invalid",
        ));
    }
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| {
            Fp8SidecarError::invalid("safetensors header length does not fit usize")
        })?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| Fp8SidecarError::invalid(format!("read header: {error}")))?;
    let mut value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| Fp8SidecarError::invalid(format!("header JSON: {error}")))?;
    value
        .as_object_mut()
        .and_then(|object| object.remove("__metadata__"));
    let header = serde_json::from_value(value)
        .map_err(|error| Fp8SidecarError::invalid(format!("header schema: {error}")))?;
    Ok((8 + length, header))
}

fn hash_range(
    file: &mut File,
    data_start: u64,
    range: [u64; 2],
) -> Result<String, Fp8SidecarError> {
    let length = range[1]
        .checked_sub(range[0])
        .ok_or_else(|| Fp8SidecarError::invalid("tensor range is reversed"))?;
    file.seek(SeekFrom::Start(
        data_start
            .checked_add(range[0])
            .ok_or_else(|| Fp8SidecarError::invalid("tensor absolute range overflow"))?,
    ))
    .map_err(|error| Fp8SidecarError::invalid(format!("seek tensor: {error}")))?;
    let mut source = file.take(length);
    let mut digest = Sha256::new();
    std::io::copy(&mut source, &mut DigestWriter(&mut digest))
        .map_err(|error| Fp8SidecarError::invalid(format!("hash tensor: {error}")))?;
    if source.limit() != 0 {
        return Err(Fp8SidecarError::invalid("tensor range is truncated"));
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
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

fn bounded_read(path: &Path, max: u64, label: &str) -> Result<Vec<u8>, Fp8SidecarError> {
    let metadata = path
        .metadata()
        .map_err(|error| Fp8SidecarError::invalid(format!("{label} metadata: {error}")))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > max {
        return Err(Fp8SidecarError::invalid(format!("{label} size is invalid")));
    }
    std::fs::read(path).map_err(|error| Fp8SidecarError::invalid(format!("read {label}: {error}")))
}

fn sha256_file(path: &Path) -> Result<String, Fp8SidecarError> {
    let mut file = File::open(path)
        .map_err(|error| Fp8SidecarError::invalid(format!("open hash source: {error}")))?;
    let mut digest = Sha256::new();
    std::io::copy(&mut file, &mut DigestWriter(&mut digest))
        .map_err(|error| Fp8SidecarError::invalid(format!("hash file: {error}")))?;
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
}
