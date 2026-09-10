//! Verified MXFP8/MXFP6 sidecars for the Qwen3.8 MTP companion.
//!
//! The sidecar owns only the eight matrix weights of the one-layer MTP
//! companion.  Norms, the shared embedding/output, and the original weight
//! load plan remain BF16/FP8 source bindings.  The source model artifact is
//! therefore still the provenance authority; this module only supplies the
//! replacement value and E8M0 scale planes.

use crate::{
    ModelLock, QuantizedTensorEncoding, UNSLOTH_QWEN38_NVFP4_MTP_SHA256,
    UNSLOTH_QWEN38_NVFP4_REPOSITORY, UNSLOTH_QWEN38_NVFP4_REVISION, VerifiedUnslothQwen38Nvfp4,
    quantize_mxfp6_e3m2, quantize_mxfp8_e4m3, validate_qwen38_mtp_artifact,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SCHEMA: &str = "sllm-qwen38-mtp-mx-sidecar-v1";
const PAYLOAD_FILE: &str = "payload.safetensors";
const MANIFEST_FILE: &str = "manifest.json";
const SCALE_SUFFIX: &str = ".sllm_mxfp_scale";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TENSOR_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const DIGEST_DOMAIN: &[u8] = b"sLLM-qwen38-mtp-combined-recipe-v1\0";

const MTP_MATRIX_NAMES: [&str; 8] = [
    "mtp.fc.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
];

fn expected_matrix_shape(name: &str) -> Option<[u64; 2]> {
    match name {
        "mtp.fc.weight" => Some([5_120, 10_240]),
        "mtp.layers.0.mlp.down_proj.weight" => Some([5_120, 17_408]),
        "mtp.layers.0.mlp.gate_proj.weight" | "mtp.layers.0.mlp.up_proj.weight" => {
            Some([17_408, 5_120])
        }
        "mtp.layers.0.self_attn.k_proj.weight" | "mtp.layers.0.self_attn.v_proj.weight" => {
            Some([1_024, 5_120])
        }
        "mtp.layers.0.self_attn.o_proj.weight" => Some([5_120, 6_144]),
        "mtp.layers.0.self_attn.q_proj.weight" => Some([12_288, 5_120]),
        _ => None,
    }
}

/// The physical MX format selected for the MTP matrix weights.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MtpWeightEncoding {
    Bf16,
    Mxfp8W8A8Block32E8M0,
    Mxfp6W6A6Block32E8M0,
}

impl MtpWeightEncoding {
    pub const fn manifest_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Mxfp8W8A8Block32E8M0 => "mxfp8-w8a8-e4m3-block32-e8m0",
            Self::Mxfp6W6A6Block32E8M0 => "mxfp6-w6a6-e3m2-block32-e8m0",
        }
    }

    fn parse(value: &str) -> Result<Self, MtpQuantizedSidecarError> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "mxfp8-w8a8-e4m3-block32-e8m0" => Ok(Self::Mxfp8W8A8Block32E8M0),
            "mxfp6-w6a6-e3m2-block32-e8m0" => Ok(Self::Mxfp6W6A6Block32E8M0),
            _ => Err(MtpQuantizedSidecarError::invalid(
                "unsupported MTP sidecar encoding",
            )),
        }
    }

    const fn value_dtype(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Mxfp8W8A8Block32E8M0 => "F8_E4M3",
            Self::Mxfp6W6A6Block32E8M0 => "U8",
        }
    }

    const fn scale_dtype(self) -> &'static str {
        "U8"
    }
}

impl fmt::Display for MtpWeightEncoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.manifest_name())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpQuantizedSidecarError(String);

impl MtpQuantizedSidecarError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
    fn io(operation: &str, error: impl fmt::Display) -> Self {
        Self::invalid(format!("{operation}: {error}"))
    }
}

impl fmt::Display for MtpQuantizedSidecarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Qwen3.8 MTP quantized sidecar: {}",
            self.0
        )
    }
}

impl std::error::Error for MtpQuantizedSidecarError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpQuantizedSidecarTensor {
    pub name: String,
    pub logical_shape: [u64; 2],
    pub value_range: [u64; 2],
    pub scale_range: [u64; 2],
    pub source_sha256: String,
    pub value_sha256: String,
    pub scale_sha256: String,
}

#[derive(Clone, Debug)]
pub struct VerifiedQwen38MtpQuantizedSidecar {
    manifest_path: PathBuf,
    payload_path: PathBuf,
    payload: Arc<File>,
    source_lock_fingerprint: String,
    source_mtp_sha256: String,
    base_recipe_digest: String,
    manifest_fingerprint: String,
    encoding: MtpWeightEncoding,
    data_start: u64,
    tensors: BTreeMap<String, MtpQuantizedSidecarTensor>,
}

impl VerifiedQwen38MtpQuantizedSidecar {
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn payload_path(&self) -> &Path {
        &self.payload_path
    }

    pub fn encoding(&self) -> MtpWeightEncoding {
        self.encoding
    }

    pub fn source_lock_fingerprint(&self) -> &str {
        &self.source_lock_fingerprint
    }

    pub fn source_mtp_sha256(&self) -> &str {
        &self.source_mtp_sha256
    }

    pub fn base_recipe_digest(&self) -> &str {
        &self.base_recipe_digest
    }

    pub fn manifest_fingerprint(&self) -> &str {
        &self.manifest_fingerprint
    }

    /// Identity used by graph and resident constructors when a quantized
    /// companion is paired with the unchanged target artifact.
    pub fn combined_recipe_digest(&self, base_recipe_digest: &str) -> String {
        combined_recipe_digest(
            base_recipe_digest,
            &self.manifest_fingerprint,
            self.encoding,
        )
    }

    pub fn tensor(&self, name: &str) -> Option<&MtpQuantizedSidecarTensor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &MtpQuantizedSidecarTensor> {
        self.tensors.values()
    }

    /// Read and hash-check one value/scale pair.  This catches payload
    /// mutation after verification before bytes reach the resident upload.
    pub fn read_tensor_bytes(
        &self,
        name: &str,
    ) -> Result<(Vec<u8>, Vec<u8>), MtpQuantizedSidecarError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar tensor is absent"))?;
        let values = read_range(&self.payload, self.data_start, tensor.value_range)?;
        let scales = read_range(&self.payload, self.data_start, tensor.scale_range)?;
        if sha256_bytes(&values) != tensor.value_sha256
            || sha256_bytes(&scales) != tensor.scale_sha256
        {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar value or scale payload changed after verification",
            ));
        }
        Ok((values, scales))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    source: Source,
    converter: String,
    encoding: String,
    base_recipe_digest: String,
    payload: Payload,
    tensors: Vec<TensorRecord>,
    fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    repository: String,
    resolved_revision: String,
    lock_fingerprint: String,
    mtp_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRecord {
    name: String,
    logical_shape: Vec<u64>,
    source_sha256: String,
    value_sha256: String,
    scale_sha256: String,
}

#[derive(Clone, Deserialize)]
struct SafeTensorMetadata {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Serialize)]
struct OutputManifest<'a> {
    schema_version: &'static str,
    source: OutputSource<'a>,
    converter: &'static str,
    encoding: &'static str,
    base_recipe_digest: &'a str,
    payload: OutputPayload<'a>,
    tensors: &'a [OutputTensor],
    fingerprint: String,
}

#[derive(Serialize)]
struct OutputSource<'a> {
    repository: &'static str,
    resolved_revision: &'static str,
    lock_fingerprint: &'a str,
    mtp_sha256: &'static str,
}

#[derive(Serialize)]
struct OutputPayload<'a> {
    path: &'a str,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Serialize)]
struct OutputTensor {
    name: String,
    logical_shape: [u64; 2],
    source_sha256: String,
    value_sha256: String,
    scale_sha256: String,
}

struct ConvertedTensor {
    record: OutputTensor,
    value_dtype: &'static str,
    value_shape: Vec<u64>,
    scale_shape: Vec<u64>,
    values: Vec<u8>,
    scales: Vec<u8>,
}

/// Verify a generated Qwen3.8 MTP MXFP sidecar against the immutable source
/// artifact and the model lock.
pub fn verify_qwen38_mtp_quantized_sidecar(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    manifest_path: &Path,
    payload_path: &Path,
) -> Result<VerifiedQwen38MtpQuantizedSidecar, MtpQuantizedSidecarError> {
    validate_qwen38_mtp_artifact(lock, artifact)
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
    let manifest_bytes = bounded_read(manifest_path, MAX_MANIFEST_BYTES, "manifest")?;
    let mut manifest_value: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| MtpQuantizedSidecarError::invalid(format!("manifest JSON: {error}")))?;
    let claimed_fingerprint = manifest_value
        .as_object_mut()
        .and_then(|object| object.remove("fingerprint"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("manifest fingerprint is absent"))?;
    let canonical = serde_json::to_vec(&manifest_value).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("manifest canonicalization: {error}"))
    })?;
    if claimed_fingerprint != sha256_bytes(&canonical) {
        return Err(MtpQuantizedSidecarError::invalid(
            "manifest fingerprint differs",
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| MtpQuantizedSidecarError::invalid(format!("manifest schema: {error}")))?;
    if manifest.fingerprint != claimed_fingerprint {
        return Err(MtpQuantizedSidecarError::invalid(
            "manifest fingerprint fields differ",
        ));
    }
    let encoding = MtpWeightEncoding::parse(&manifest.encoding)?;
    if encoding == MtpWeightEncoding::Bf16
        || manifest.schema_version != SCHEMA
        || manifest.converter != "sllm-qwen38-mtp-quantizer-v1"
        || manifest.source.repository != UNSLOTH_QWEN38_NVFP4_REPOSITORY
        || manifest.source.resolved_revision != UNSLOTH_QWEN38_NVFP4_REVISION
        || manifest.source.lock_fingerprint != lock.fingerprint()
        || manifest.source.mtp_sha256 != UNSLOTH_QWEN38_NVFP4_MTP_SHA256
        || manifest.base_recipe_digest != artifact.recipe_digest()
        || !is_sha256(&manifest.base_recipe_digest)
        || !is_sha256(&manifest.source.lock_fingerprint)
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar source or recipe identity differs",
        ));
    }
    let payload_name = payload_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload has no UTF-8 basename"))?;
    let metadata = payload_path
        .metadata()
        .map_err(|error| MtpQuantizedSidecarError::io("stat payload", error))?;
    if !metadata.is_file()
        || payload_name != manifest.payload.path
        || metadata.len() != manifest.payload.size_bytes
        || manifest.payload.size_bytes < 8
        || manifest.payload.sha256 != sha256_file(payload_path)?
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar payload identity differs",
        ));
    }
    let (data_start, header) = read_safetensors_header(payload_path)?;
    validate_payload_ranges(&header, metadata.len(), data_start)?;
    if manifest.tensors.len() != MTP_MATRIX_NAMES.len() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar tensor count differs",
        ));
    }
    let mut records = BTreeMap::new();
    let expected_names: BTreeSet<&str> = MTP_MATRIX_NAMES.into_iter().collect();
    for record in manifest.tensors {
        let shape: [u64; 2] =
            record.logical_shape.as_slice().try_into().map_err(|_| {
                MtpQuantizedSidecarError::invalid("MTP sidecar shape is not rank two")
            })?;
        if expected_matrix_shape(&record.name) != Some(shape)
            || shape[0] == 0
            || shape[1] == 0
            || shape[1] % 32 != 0
        {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar matrix shape differs from the Qwen3.8 contract",
            ));
        }
        if !expected_names.contains(record.name.as_str())
            || records.insert(record.name.clone(), record).is_some()
        {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar matrix name set differs",
            ));
        }
    }
    if records.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_names {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar matrix coverage differs",
        ));
    }
    let mut payload_file = File::open(payload_path)
        .map_err(|error| MtpQuantizedSidecarError::io("open payload", error))?;
    let mut tensors = BTreeMap::new();
    let mut header_names = BTreeSet::new();
    for (name, record) in records {
        let shape = record
            .logical_shape
            .as_slice()
            .try_into()
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP sidecar shape is invalid"))?;
        let value_name = name.as_str();
        let scale_name = format!("{name}{SCALE_SUFFIX}");
        let value = header
            .get(value_name)
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value tensor is absent"))?;
        let scale = header
            .get(scale_name.as_str())
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP scale tensor is absent"))?;
        validate_record(
            &record,
            value,
            scale,
            shape,
            encoding,
            &mut payload_file,
            data_start,
        )?;
        let descriptor = artifact
            .tensor(name.as_str())
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP source tensor is absent"))?;
        let source = artifact
            .read_source_range(&descriptor.source_name, descriptor.value_range)
            .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
        if sha256_bytes(&source) != record.source_sha256 {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar source tensor differs from the locked BF16 artifact",
            ));
        }
        let tensor = MtpQuantizedSidecarTensor {
            name: name.clone(),
            logical_shape: shape,
            value_range: value.data_offsets,
            scale_range: scale.data_offsets,
            source_sha256: record.source_sha256.clone(),
            value_sha256: record.value_sha256.clone(),
            scale_sha256: record.scale_sha256.clone(),
        };
        tensors.insert(name.clone(), tensor);
        header_names.insert(name.clone());
        header_names.insert(scale_name);
    }
    let expected_header_names: BTreeSet<String> = MTP_MATRIX_NAMES
        .into_iter()
        .flat_map(|name| [name.to_owned(), format!("{name}{SCALE_SUFFIX}")])
        .collect();
    if header_names != expected_header_names
        || header
            .keys()
            .filter(|name| name.as_str() != "__metadata__")
            .cloned()
            .collect::<BTreeSet<_>>()
            != expected_header_names
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar payload tensor set differs",
        ));
    }
    Ok(VerifiedQwen38MtpQuantizedSidecar {
        manifest_path: manifest_path.to_path_buf(),
        payload_path: payload_path.to_path_buf(),
        payload: Arc::new(
            File::open(payload_path)
                .map_err(|error| MtpQuantizedSidecarError::io("open payload", error))?,
        ),
        source_lock_fingerprint: manifest.source.lock_fingerprint,
        source_mtp_sha256: manifest.source.mtp_sha256,
        base_recipe_digest: manifest.base_recipe_digest,
        manifest_fingerprint: claimed_fingerprint,
        encoding,
        data_start,
        tensors,
    })
}

/// Convert the eight verified BF16 MTP matrices and publish a complete
/// sidecar directory atomically.  The returned object has already been
/// verified by rereading the published payload and manifest.
pub fn convert_qwen38_mtp_quantized_sidecar(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    encoding: MtpWeightEncoding,
    output_dir: &Path,
) -> Result<VerifiedQwen38MtpQuantizedSidecar, MtpQuantizedSidecarError> {
    if encoding == MtpWeightEncoding::Bf16 {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar conversion requires MXFP8 or MXFP6",
        ));
    }
    validate_qwen38_mtp_artifact(lock, artifact)
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
    if output_dir.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar output directory already exists",
        ));
    }
    let parent = output_dir
        .parent()
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no parent"))?;
    fs::create_dir_all(parent)
        .map_err(|error| MtpQuantizedSidecarError::io("create output parent", error))?;
    let name = output_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no basename"))?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar temporary output already exists",
        ));
    }
    fs::create_dir(&temporary)
        .map_err(|error| MtpQuantizedSidecarError::io("create temporary output", error))?;
    let result = convert_into_directory(lock, artifact, encoding, &temporary).and_then(|()| {
        verify_qwen38_mtp_quantized_sidecar(
            lock,
            artifact,
            &temporary.join(MANIFEST_FILE),
            &temporary.join(PAYLOAD_FILE),
        )
        .map(|_| ())?;
        fs::rename(&temporary, output_dir)
            .map_err(|error| MtpQuantizedSidecarError::io("publish MTP sidecar", error))
    });
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result?;
    let verified = verify_qwen38_mtp_quantized_sidecar(
        lock,
        artifact,
        &output_dir.join(MANIFEST_FILE),
        &output_dir.join(PAYLOAD_FILE),
    );
    if verified.is_err() {
        let _ = fs::remove_dir_all(output_dir);
    }
    verified
}

fn convert_into_directory(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    encoding: MtpWeightEncoding,
    directory: &Path,
) -> Result<(), MtpQuantizedSidecarError> {
    let mut converted = Vec::with_capacity(MTP_MATRIX_NAMES.len());
    for name in MTP_MATRIX_NAMES {
        let descriptor = artifact.tensor(name).ok_or_else(|| {
            MtpQuantizedSidecarError::invalid(format!("MTP tensor is absent: {name}"))
        })?;
        if descriptor.encoding != QuantizedTensorEncoding::UnquantizedBf16
            || descriptor.logical_shape.len() != 2
            || descriptor.logical_shape[1] % 32 != 0
        {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "MTP tensor is not a block32 BF16 matrix: {name}"
            )));
        }
        let rows = usize::try_from(descriptor.logical_shape[0])
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP row count overflows usize"))?;
        let columns = usize::try_from(descriptor.logical_shape[1])
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP column count overflows usize"))?;
        let source = artifact
            .read_source_range(&descriptor.source_name, descriptor.value_range)
            .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP source size overflows"))?;
        if source.len() != expected_bytes {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "MTP source byte size differs: {name}"
            )));
        }
        let values = source
            .chunks_exact(2)
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
            .collect::<Vec<_>>();
        let quantized = match encoding {
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0 => quantize_mxfp8_e4m3(&values, rows, columns)
                .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?,
            MtpWeightEncoding::Mxfp6W6A6Block32E8M0 => quantize_mxfp6_e3m2(&values, rows, columns)
                .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?,
            MtpWeightEncoding::Bf16 => unreachable!(),
        };
        let value_shape = match encoding {
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0 => vec![rows as u64, columns as u64],
            MtpWeightEncoding::Mxfp6W6A6Block32E8M0 => {
                vec![rows as u64, (columns * 3 / 4) as u64]
            }
            MtpWeightEncoding::Bf16 => unreachable!(),
        };
        let scales = quantized.scales().to_vec();
        converted.push(ConvertedTensor {
            record: OutputTensor {
                name: name.to_owned(),
                logical_shape: [rows as u64, columns as u64],
                source_sha256: sha256_bytes(&source),
                value_sha256: sha256_bytes(quantized.values()),
                scale_sha256: sha256_bytes(&scales),
            },
            value_dtype: encoding.value_dtype(),
            value_shape,
            scale_shape: vec![rows as u64, (columns / 32) as u64],
            values: quantized.values().to_vec(),
            scales,
        });
    }
    let mut data = Vec::new();
    let mut header = Map::new();
    for tensor in &converted {
        let value_start = u64::try_from(data.len())
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP payload offset overflows"))?;
        data.extend_from_slice(&tensor.values);
        let value_end = u64::try_from(data.len())
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP payload size overflows"))?;
        let scale_start = value_end;
        data.extend_from_slice(&tensor.scales);
        let scale_end = u64::try_from(data.len())
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP payload size overflows"))?;
        header.insert(
            tensor.record.name.clone(),
            serde_json::json!({
                "dtype": tensor.value_dtype,
                "shape": tensor.value_shape,
                "data_offsets": [value_start, value_end],
            }),
        );
        header.insert(
            format!("{}{SCALE_SUFFIX}", tensor.record.name),
            serde_json::json!({
                "dtype": encoding.scale_dtype(),
                "shape": tensor.scale_shape,
                "data_offsets": [scale_start, scale_end],
            }),
        );
    }
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({
            "format": "pt",
            "sllm_schema": SCHEMA,
            "sllm_encoding": encoding.manifest_name(),
        }),
    );
    let header_bytes = serde_json::to_vec(&header).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("payload header JSON: {error}"))
    })?;
    let header_length = u64::try_from(header_bytes.len())
        .map_err(|_| MtpQuantizedSidecarError::invalid("payload header size overflows"))?;
    if header_length > MAX_HEADER_BYTES {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload header exceeds the bounded size",
        ));
    }
    let mut payload_bytes = Vec::with_capacity(8 + header_bytes.len() + data.len());
    payload_bytes.extend_from_slice(&header_length.to_le_bytes());
    payload_bytes.extend_from_slice(&header_bytes);
    payload_bytes.extend_from_slice(&data);
    let payload_path = directory.join(PAYLOAD_FILE);
    let manifest_path = directory.join(MANIFEST_FILE);
    write_new_file(&payload_path, &payload_bytes)?;
    let payload_digest = sha256_bytes(&payload_bytes);
    let tensors = converted
        .into_iter()
        .map(|tensor| OutputTensor {
            name: tensor.record.name,
            logical_shape: tensor.record.logical_shape,
            source_sha256: tensor.record.source_sha256,
            value_sha256: tensor.record.value_sha256,
            scale_sha256: tensor.record.scale_sha256,
        })
        .collect::<Vec<_>>();
    let base_recipe = artifact.recipe_digest();
    let lock_fingerprint = lock.fingerprint().to_owned();
    let payload_size = u64::try_from(payload_bytes.len())
        .map_err(|_| MtpQuantizedSidecarError::invalid("payload size overflows"))?;
    let manifest_without_fingerprint = OutputManifest {
        schema_version: SCHEMA,
        source: OutputSource {
            repository: UNSLOTH_QWEN38_NVFP4_REPOSITORY,
            resolved_revision: UNSLOTH_QWEN38_NVFP4_REVISION,
            lock_fingerprint: &lock_fingerprint,
            mtp_sha256: UNSLOTH_QWEN38_NVFP4_MTP_SHA256,
        },
        converter: "sllm-qwen38-mtp-quantizer-v1",
        encoding: encoding.manifest_name(),
        base_recipe_digest: base_recipe,
        payload: OutputPayload {
            path: PAYLOAD_FILE,
            size_bytes: payload_size,
            sha256: payload_digest,
        },
        tensors: &tensors,
        fingerprint: String::new(),
    };
    let mut value = serde_json::to_value(&manifest_without_fingerprint).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("manifest serialization: {error}"))
    })?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("manifest is not an object"))?;
    object.remove("fingerprint");
    let fingerprint = sha256_bytes(&serde_json::to_vec(&value).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("manifest canonicalization: {error}"))
    })?);
    let mut final_value = value;
    final_value["fingerprint"] = Value::String(fingerprint);
    let manifest_bytes = serde_json::to_vec(&final_value)
        .map_err(|error| MtpQuantizedSidecarError::invalid(format!("manifest JSON: {error}")))?;
    write_new_file(&manifest_path, &manifest_bytes)
}

fn validate_record(
    record: &TensorRecord,
    value: &SafeTensorMetadata,
    scale: &SafeTensorMetadata,
    logical_shape: [u64; 2],
    encoding: MtpWeightEncoding,
    file: &mut File,
    data_start: u64,
) -> Result<(), MtpQuantizedSidecarError> {
    if logical_shape[0] == 0 || logical_shape[1] == 0 || logical_shape[1] % 32 != 0 {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar matrix shape is not block32 aligned",
        ));
    }
    let values_len = value.data_offsets[1]
        .checked_sub(value.data_offsets[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value range is reversed"))?;
    let scales_len = scale.data_offsets[1]
        .checked_sub(scale.data_offsets[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP scale range is reversed"))?;
    let expected_values =
        match encoding {
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0 => logical_shape[0]
                .checked_mul(logical_shape[1])
                .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value shape overflows"))?,
            MtpWeightEncoding::Mxfp6W6A6Block32E8M0 => logical_shape[0]
                .checked_mul(logical_shape[1])
                .and_then(|elements| elements.checked_mul(3))
                .map(|bytes| bytes / 4)
                .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value shape overflows"))?,
            MtpWeightEncoding::Bf16 => {
                return Err(MtpQuantizedSidecarError::invalid(
                    "BF16 sidecar is unsupported",
                ));
            }
        };
    let expected_scales = logical_shape[0]
        .checked_mul(logical_shape[1] / 32)
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP scale shape overflows"))?;
    if !is_sha256(&record.source_sha256)
        || !is_sha256(&record.value_sha256)
        || !is_sha256(&record.scale_sha256)
        || value.dtype != encoding.value_dtype()
        || scale.dtype != encoding.scale_dtype()
        || value.shape
            != match encoding {
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0 => vec![logical_shape[0], logical_shape[1]],
                MtpWeightEncoding::Mxfp6W6A6Block32E8M0 => {
                    vec![logical_shape[0], expected_values / logical_shape[0]]
                }
                MtpWeightEncoding::Bf16 => unreachable!(),
            }
        || scale.shape != vec![logical_shape[0], logical_shape[1] / 32]
        || values_len != expected_values
        || scales_len != expected_scales
        || data_start.checked_add(value.data_offsets[1]).is_none()
        || data_start.checked_add(scale.data_offsets[1]).is_none()
        || sha256_range(file, data_start, value.data_offsets)? != record.value_sha256
        || sha256_range(file, data_start, scale.data_offsets)? != record.scale_sha256
    {
        return Err(MtpQuantizedSidecarError::invalid(format!(
            "MTP sidecar value/scale contract differs: {}",
            record.name
        )));
    }
    Ok(())
}

fn combined_recipe_digest(base: &str, manifest: &str, encoding: MtpWeightEncoding) -> String {
    let mut digest = Sha256::new();
    digest.update(DIGEST_DOMAIN);
    digest.update(base.as_bytes());
    digest.update([0]);
    digest.update(manifest.as_bytes());
    digest.update([0]);
    digest.update(encoding.manifest_name().as_bytes());
    format!("sha256:{:x}", digest.finalize())
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), MtpQuantizedSidecarError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| MtpQuantizedSidecarError::io("create sidecar file", error))?;
    file.write_all(bytes)
        .map_err(|error| MtpQuantizedSidecarError::io("write sidecar file", error))?;
    file.sync_all()
        .map_err(|error| MtpQuantizedSidecarError::io("sync sidecar file", error))?;
    let mut permissions = file
        .metadata()
        .map_err(|error| MtpQuantizedSidecarError::io("stat sidecar file", error))?
        .permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(path, permissions)
        .map_err(|error| MtpQuantizedSidecarError::io("set sidecar permissions", error))
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, SafeTensorMetadata>), MtpQuantizedSidecarError> {
    let mut file =
        File::open(path).map_err(|error| MtpQuantizedSidecarError::io("open payload", error))?;
    let mut raw_length = [0_u8; 8];
    file.read_exact(&mut raw_length)
        .map_err(|error| MtpQuantizedSidecarError::io("read payload header length", error))?;
    let length = u64::from_le_bytes(raw_length);
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload header length is invalid",
        ));
    }
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| MtpQuantizedSidecarError::invalid(
            "payload header is too large"
        ))?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| MtpQuantizedSidecarError::io("read payload header", error))?;
    let mut value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("payload header JSON: {error}"))
    })?;
    value
        .as_object_mut()
        .and_then(|object| object.remove("__metadata__"));
    let header = serde_json::from_value(value).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("payload header schema: {error}"))
    })?;
    Ok((8 + length, header))
}

fn validate_payload_ranges(
    header: &BTreeMap<String, SafeTensorMetadata>,
    payload_size: u64,
    data_start: u64,
) -> Result<(), MtpQuantizedSidecarError> {
    let data_size = payload_size
        .checked_sub(data_start)
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload data starts past EOF"))?;
    let mut ranges = Vec::with_capacity(header.len());
    for (name, metadata) in header {
        let [start, end] = metadata.data_offsets;
        if end < start || end > data_size {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "payload range is outside the file: {name}"
            )));
        }
        ranges.push((start, end, name));
    }
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    let mut cursor = 0_u64;
    for (start, end, name) in ranges {
        if start != cursor {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "payload ranges have a gap or overlap before {name}"
            )));
        }
        cursor = end;
    }
    if cursor != data_size {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload ranges do not cover the complete data section",
        ));
    }
    Ok(())
}

fn bounded_read(
    path: &Path,
    maximum: u64,
    description: &str,
) -> Result<Vec<u8>, MtpQuantizedSidecarError> {
    let metadata = path
        .metadata()
        .map_err(|error| MtpQuantizedSidecarError::io(&format!("stat {description}"), error))?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err(MtpQuantizedSidecarError::invalid(format!(
            "{description} is absent or too large"
        )));
    }
    let mut file = File::open(path)
        .map_err(|error| MtpQuantizedSidecarError::io(&format!("open {description}"), error))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| MtpQuantizedSidecarError::io(&format!("read {description}"), error))?;
    Ok(bytes)
}

fn read_range(
    file: &File,
    data_start: u64,
    range: [u64; 2],
) -> Result<Vec<u8>, MtpQuantizedSidecarError> {
    let length = range[1]
        .checked_sub(range[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload range is reversed"))?;
    if length > MAX_TENSOR_BYTES {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload tensor is too large",
        ));
    }
    let absolute = data_start
        .checked_add(range[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload range overflows"))?;
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| MtpQuantizedSidecarError::invalid(
            "payload tensor does not fit usize"
        ))?
    ];
    read_exact_at(file, &mut bytes, absolute)?;
    Ok(bytes)
}

fn sha256_range(
    file: &File,
    data_start: u64,
    range: [u64; 2],
) -> Result<String, MtpQuantizedSidecarError> {
    let bytes = read_range(file, data_start, range)?;
    Ok(sha256_bytes(&bytes))
}

fn read_exact_at(
    file: &File,
    mut output: &mut [u8],
    mut offset: u64,
) -> Result<(), MtpQuantizedSidecarError> {
    while !output.is_empty() {
        let count = file
            .read_at(output, offset)
            .map_err(|error| MtpQuantizedSidecarError::io("read payload tensor", error))?;
        if count == 0 {
            return Err(MtpQuantizedSidecarError::invalid(
                "payload tensor is truncated",
            ));
        }
        output = &mut output[count..];
        offset = offset
            .checked_add(
                u64::try_from(count).map_err(|_| {
                    MtpQuantizedSidecarError::invalid("payload read offset overflows")
                })?,
            )
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload read offset overflows"))?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, MtpQuantizedSidecarError> {
    let mut file = File::open(path)
        .map_err(|error| MtpQuantizedSidecarError::io("open hashed file", error))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| MtpQuantizedSidecarError::io("hash file", error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn encoded_value_len(encoding: MtpWeightEncoding, rows: usize, columns: usize) -> usize {
        match encoding {
            MtpWeightEncoding::Bf16 => rows * columns * 2,
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0 => rows * columns,
            MtpWeightEncoding::Mxfp6W6A6Block32E8M0 => rows * columns * 3 / 4,
        }
    }

    fn temporary_test_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sllm-mtp-sidecar-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn validate_fixture(
        encoding: MtpWeightEncoding,
        columns: usize,
        record: TensorRecord,
        value: SafeTensorMetadata,
        scale: SafeTensorMetadata,
        data: &[u8],
    ) -> Result<(), MtpQuantizedSidecarError> {
        let path = temporary_test_path("validate");
        {
            let mut output = File::create(&path).expect("create fixture");
            output.write_all(&[0_u8; 8]).expect("write fixture prefix");
            output.write_all(data).expect("write fixture payload");
            output.sync_all().expect("sync fixture");
        }
        let mut input = File::open(&path).expect("open fixture");
        let result = validate_record(
            &record,
            &value,
            &scale,
            [3, columns as u64],
            encoding,
            &mut input,
            8,
        );
        drop(input);
        std::fs::remove_file(path).expect("remove fixture");
        result
    }

    fn make_validate_fixture(
        encoding: MtpWeightEncoding,
        columns: usize,
    ) -> (
        TensorRecord,
        SafeTensorMetadata,
        SafeTensorMetadata,
        Vec<u8>,
    ) {
        let rows = 3_usize;
        let value_len = encoded_value_len(encoding, rows, columns);
        let scale_len = rows * (columns / 32);
        let values = vec![0x31_u8; value_len];
        let scales = vec![0x7f_u8; scale_len];
        let mut data = values.clone();
        data.extend_from_slice(&scales);
        let record = TensorRecord {
            name: "mtp.test.weight".to_owned(),
            logical_shape: vec![rows as u64, columns as u64],
            source_sha256: sha256_bytes(b"source"),
            value_sha256: sha256_bytes(&values),
            scale_sha256: sha256_bytes(&scales),
        };
        let value = SafeTensorMetadata {
            dtype: encoding.value_dtype().to_owned(),
            shape: match encoding {
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0 => {
                    vec![rows as u64, columns as u64]
                }
                MtpWeightEncoding::Mxfp6W6A6Block32E8M0 => {
                    vec![rows as u64, (columns * 3 / 4) as u64]
                }
                MtpWeightEncoding::Bf16 => vec![rows as u64, columns as u64],
            },
            data_offsets: [0, value_len as u64],
        };
        let scale = SafeTensorMetadata {
            dtype: encoding.scale_dtype().to_owned(),
            shape: vec![rows as u64, (columns / 32) as u64],
            data_offsets: [value_len as u64, (value_len + scale_len) as u64],
        };
        (record, value, scale, data)
    }

    #[test]
    fn matrix_contract_has_exact_eight_names_and_shapes() {
        assert_eq!(MTP_MATRIX_NAMES.len(), 8);
        for name in MTP_MATRIX_NAMES {
            let shape = expected_matrix_shape(name).expect("matrix shape");
            assert!(shape[0] > 0 && shape[1] > 0);
            assert_eq!(shape[1] % 32, 0);
        }
        assert_eq!(expected_matrix_shape("mtp.norm.weight"), None);
    }

    #[test]
    fn encoding_names_and_combined_recipe_identity_are_stable() {
        assert_eq!(MtpWeightEncoding::Bf16.manifest_name(), "bf16");
        assert_eq!(
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0.manifest_name(),
            "mxfp8-w8a8-e4m3-block32-e8m0"
        );
        assert_eq!(
            MtpWeightEncoding::Mxfp6W6A6Block32E8M0.manifest_name(),
            "mxfp6-w6a6-e3m2-block32-e8m0"
        );
        let base = "sha256:base";
        let manifest = "sha256:manifest";
        assert_ne!(
            combined_recipe_digest(base, manifest, MtpWeightEncoding::Mxfp8W8A8Block32E8M0),
            combined_recipe_digest(base, manifest, MtpWeightEncoding::Mxfp6W6A6Block32E8M0)
        );
        assert_ne!(
            combined_recipe_digest(base, manifest, MtpWeightEncoding::Mxfp8W8A8Block32E8M0),
            combined_recipe_digest(
                "sha256:other",
                manifest,
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0
            )
        );
    }

    #[test]
    fn payload_ranges_require_contiguous_complete_data() {
        let mut contiguous = BTreeMap::new();
        contiguous.insert(
            "values".to_owned(),
            SafeTensorMetadata {
                dtype: "U8".to_owned(),
                shape: vec![4],
                data_offsets: [0, 4],
            },
        );
        contiguous.insert(
            "scales".to_owned(),
            SafeTensorMetadata {
                dtype: "U8".to_owned(),
                shape: vec![4],
                data_offsets: [4, 8],
            },
        );
        assert!(validate_payload_ranges(&contiguous, 16, 8).is_ok());

        contiguous.get_mut("scales").expect("scales").data_offsets = [5, 9];
        assert!(validate_payload_ranges(&contiguous, 17, 8).is_err());
    }

    #[test]
    fn sha256_contract_rejects_non_digest_strings() {
        assert!(is_sha256(&sha256_bytes(b"sidecar")));
        assert!(!is_sha256("sha256:short"));
        assert!(!is_sha256(
            "sha256:GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
        ));
    }

    #[test]
    fn validate_record_accepts_n3_for_mxfp8_and_mxfp6_at_block_boundaries() {
        for encoding in [
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0,
            MtpWeightEncoding::Mxfp6W6A6Block32E8M0,
        ] {
            for columns in [32, 64] {
                let (record, value, scale, data) = make_validate_fixture(encoding, columns);
                assert!(
                    validate_fixture(encoding, columns, record, value, scale, &data).is_ok(),
                    "valid {encoding} N=3 K={columns} fixture rejected"
                );
            }
        }
    }

    #[test]
    fn validate_record_rejects_nonaligned_k_dtype_scale_length_and_truncation() {
        let encoding = MtpWeightEncoding::Mxfp8W8A8Block32E8M0;

        let (mut record, value, scale, data) = make_validate_fixture(encoding, 32);
        record.logical_shape = vec![3, 33];
        assert!(validate_fixture(encoding, 33, record, value, scale, &data).is_err());

        let (record, mut value, scale, data) = make_validate_fixture(encoding, 32);
        value.dtype = "BF16".to_owned();
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());

        let (record, value, mut scale, data) = make_validate_fixture(encoding, 32);
        scale.shape = vec![3, 2];
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());

        let (record, value, mut scale, data) = make_validate_fixture(encoding, 32);
        scale.data_offsets[1] -= 1;
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());

        let (record, value, scale, mut data) = make_validate_fixture(encoding, 32);
        data.pop();
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());
    }

    #[test]
    fn read_tensor_bytes_rejects_payload_mutation_after_verification() {
        let path = temporary_test_path("mutation");
        let values = vec![0x31_u8; 96];
        let scales = vec![0x7f_u8; 3];
        let mut payload_bytes = values.clone();
        payload_bytes.extend_from_slice(&scales);
        {
            let mut output = File::create(&path).expect("create mutation fixture");
            output
                .write_all(&payload_bytes)
                .expect("write mutation fixture");
            output.sync_all().expect("sync mutation fixture");
        }
        let payload = Arc::new(File::open(&path).expect("open mutation fixture"));
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "mtp.test.weight".to_owned(),
            MtpQuantizedSidecarTensor {
                name: "mtp.test.weight".to_owned(),
                logical_shape: [3, 32],
                value_range: [0, 96],
                scale_range: [96, 99],
                source_sha256: sha256_bytes(b"source"),
                value_sha256: sha256_bytes(&values),
                scale_sha256: sha256_bytes(&scales),
            },
        );
        let sidecar = VerifiedQwen38MtpQuantizedSidecar {
            manifest_path: PathBuf::from("manifest.json"),
            payload_path: path.clone(),
            payload,
            source_lock_fingerprint: "sha256:lock".to_owned(),
            source_mtp_sha256: "mtp".to_owned(),
            base_recipe_digest: "sha256:recipe".to_owned(),
            manifest_fingerprint: "sha256:manifest".to_owned(),
            encoding: MtpWeightEncoding::Mxfp8W8A8Block32E8M0,
            data_start: 0,
            tensors,
        };
        assert_eq!(
            sidecar
                .read_tensor_bytes("mtp.test.weight")
                .expect("unchanged payload")
                .0,
            values
        );

        {
            let mut input = OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("open mutation fixture for write");
            input.write_all(&[0x32]).expect("mutate value plane");
            input.sync_all().expect("sync mutation");
        }
        assert!(sidecar.read_tensor_bytes("mtp.test.weight").is_err());
        drop(sidecar);
        std::fs::remove_file(path).expect("remove mutation fixture");
    }
}
